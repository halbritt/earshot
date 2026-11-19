//! Run variants on a take and score them.

use crate::stt::Recogniser;
use crate::text::{read_reference, score, words, Score, Section};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

#[derive(Deserialize)]
struct Manifest {
    name: Option<String>,
    recording: PathBuf,
    reference: PathBuf,
    #[serde(default)]
    playback: Vec<PathBuf>,
    chunk_seconds: Option<f64>,
    #[serde(default)]
    region: Vec<Region>,
}

#[derive(Deserialize, Clone, Debug)]
pub struct Region {
    pub name: String,
    pub start: f64,
    pub end: f64,
}

pub struct Take {
    pub name: String,
    pub recording: PathBuf,
    pub reference: Vec<Section>,
    pub playback: Vec<String>,
    pub regions: Vec<Region>,
    pub chunk_seconds: f64,
}

fn read(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

impl Take {
    pub fn load(path: &Path) -> Result<Take, String> {
        let m: Manifest = toml::from_str(&read(path)?).map_err(|e| format!("{}: {}", path.display(), e.message()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        let reference = read_reference(&read(&base.join(&m.reference))?);
        if reference.is_empty() {
            return Err(format!("{}: the reference text has no words", path.display()));
        }
        let mut playback = Vec::new();
        for p in &m.playback {
            playback.extend(words(&read(&base.join(p))?));
        }
        for r in &m.region {
            if !(r.start >= 0.0 && r.end > r.start) {
                return Err(format!("{}: region '{}' must have 0 <= start < end", path.display(), r.name));
            }
        }
        let chunk_seconds = m.chunk_seconds.unwrap_or(25.0);
        if chunk_seconds.is_nan() || chunk_seconds <= 0.0 {
            return Err(format!("{}: chunk_seconds must be greater than 0", path.display()));
        }
        let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Take {
            name: m.name.unwrap_or(stem),
            recording: base.join(m.recording),
            reference,
            playback,
            regions: m.region,
            chunk_seconds,
        })
    }

    pub fn reference_words(&self) -> usize {
        self.reference.iter().map(|(_, w)| w.len()).sum()
    }
}

fn quote(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\\''"))
}

fn children_cpu() -> f64 {
    // SAFETY: getrusage only writes to the struct we pass.
    let mut u: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_CHILDREN, &mut u) };
    let s = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    s(u.ru_utime) + s(u.ru_stime)
}

/// Run a variant command through `sh`. `{in}` and `{out}` are replaced by the recording and
/// the output path. Returns (CPU seconds of the command and its children, wall seconds).
pub fn run_variant(command: &str, recording: &Path, out: &Path) -> Result<(f64, f64), String> {
    if !command.contains("{out}") {
        return Err("a variant command must contain {out}".into());
    }
    let cmd = command.replace("{in}", &quote(recording)).replace("{out}", &quote(out));
    let (before, t0) = (children_cpu(), Instant::now());
    let done = Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .output()
        .map_err(|e| format!("could not start sh: {e}"))?;
    let (cpu, wall) = (children_cpu() - before, t0.elapsed().as_secs_f64());
    if !done.status.success() {
        let err = String::from_utf8_lossy(&done.stderr);
        let tail: String = err.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
        let status = done.status.code().map_or("a signal".to_string(), |c| format!("status {c}"));
        return Err(format!("variant command failed with {status}: {cmd}\n{tail}"));
    }
    if !out.exists() {
        return Err(format!("variant command wrote no output file: {cmd}"));
    }
    Ok((cpu, wall))
}

/// Read one channel of 16-bit audio. Returns (sample rate, samples).
pub fn read_wav(path: &Path) -> Result<(u32, Vec<i16>), String> {
    let mut r = hound::WavReader::open(path).map_err(|e| format!("{}: not a WAV file ({e})", path.display()))?;
    let spec = r.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 || spec.sample_format != hound::SampleFormat::Int {
        return Err(format!(
            "{}: a variant must write one channel of 16-bit samples (found {} channels, {} bits)",
            path.display(),
            spec.channels,
            spec.bits_per_sample
        ));
    }
    let samples = r
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((spec.sample_rate, samples))
}

pub fn wav_bytes(rate: u32, samples: &[i16]) -> Vec<u8> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    {
        let mut w = hound::WavWriter::new(&mut buf, spec).expect("WAV header");
        for s in samples {
            w.write_sample(*s).expect("write to memory");
        }
        w.finalize().expect("finish WAV");
    }
    buf.into_inner()
}

/// The pieces to transcribe: the take's regions, or fixed-length pieces.
pub fn chunks<'a>(take: &Take, rate: u32, samples: &'a [i16]) -> Vec<(String, &'a [i16])> {
    let total = samples.len() as f64 / rate as f64;
    let spans: Vec<(String, f64, f64)> = if take.regions.is_empty() {
        let n = ((total / take.chunk_seconds).ceil() as usize).max(1);
        (0..n)
            .map(|i| {
                let a = i as f64 * take.chunk_seconds;
                (format!("{a:.0}s"), a, (a + take.chunk_seconds).min(total))
            })
            .collect()
    } else {
        take.regions.iter().map(|r| (r.name.clone(), r.start, r.end)).collect()
    };
    let at = |t: f64| ((t * rate as f64) as usize).min(samples.len());
    spans
        .into_iter()
        .map(|(name, a, b)| (name, &samples[at(a)..at(b).max(at(a))]))
        .collect()
}

/// Chunk name to text, in order.
pub type Texts = serde_json::Map<String, serde_json::Value>;

#[derive(Serialize)]
pub struct Outcome {
    #[serde(flatten)]
    pub score: Score,
    pub transcript_words: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_percent_of_real_time: Option<f64>,
    pub texts: Texts,
}

pub fn outcome(take: &Take, texts: Texts, timing: Option<(f64, f64, f64)>) -> Outcome {
    let joined: Vec<&str> = texts.values().filter_map(|v| v.as_str()).collect();
    let heard = words(&joined.join(" "));
    let (cpu, wall, audio) = match timing {
        Some((c, w, a)) => (Some(c), Some(w), Some(a)),
        None => (None, None, None),
    };
    Outcome {
        score: score(&take.reference, &heard, &take.playback),
        transcript_words: heard.len(),
        cpu_seconds: cpu,
        wall_seconds: wall,
        audio_seconds: audio,
        cpu_percent_of_real_time: timing.filter(|t| t.2 > 0.0).map(|(c, _, a)| 100.0 * c / a),
        texts,
    }
}

/// Run one variant and score what the recogniser hears in its output.
pub fn bench_variant(take: &Take, command: &str, out: &Path, recogniser: &Recogniser) -> Result<Outcome, String> {
    let _ = std::fs::remove_file(out);
    let (cpu, wall) = run_variant(command, &take.recording, out)?;
    let (rate, samples) = read_wav(out)?;
    let mut texts = Texts::new();
    for (name, piece) in chunks(take, rate, &samples) {
        let text = if piece.is_empty() {
            String::new()
        } else {
            recogniser.transcribe(&wav_bytes(rate, piece))?
        };
        texts.insert(name, text.into());
    }
    Ok(outcome(take, texts, Some((cpu, wall, samples.len() as f64 / rate as f64))))
}

pub fn table(results: &[(String, Outcome)]) -> String {
    let mut names: Vec<&str> = Vec::new();
    for (_, r) in results {
        for s in &r.score.sections {
            if !names.contains(&s.name.as_str()) {
                names.push(&s.name);
            }
        }
    }
    let by_section = names != ["all"];
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut head: Vec<String> = [
        "variant",
        "WER %",
        "correct",
        "subst",
        "missing",
        "extra",
        "leaked",
        "CPU % of real time",
    ]
    .map(String::from)
    .to_vec();
    if by_section {
        head.extend(names.iter().map(|n| format!("WER % {n}")));
    }
    rows.push(head);
    for (name, r) in results {
        let c = &r.score.counts;
        let mut row = vec![
            name.clone(),
            format!("{:.1}", 100.0 * r.score.wer),
            c.correct.to_string(),
            c.substituted.to_string(),
            c.missing.to_string(),
            c.extra.to_string(),
            r.score.leaked.to_string(),
            r.cpu_percent_of_real_time.map_or("-".into(), |p| format!("{p:.1}")),
        ];
        if by_section {
            row.extend(
                names
                    .iter()
                    .map(|n| r.score.section(n).map_or("-".into(), |s| format!("{:.1}", 100.0 * s.wer))),
            );
        }
        rows.push(row);
    }
    let widths: Vec<usize> = (0..rows[0].len())
        .map(|i| rows.iter().map(|r| r[i].chars().count()).max().unwrap_or(0))
        .collect();
    let mut lines: Vec<String> = rows
        .iter()
        .map(|r| {
            let cells: Vec<String> = r
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    if i == 0 {
                        format!("{c:<w$}", w = widths[i])
                    } else {
                        format!("{c:>w$}", w = widths[i])
                    }
                })
                .collect();
            cells.join("  ")
        })
        .collect();
    for (name, r) in results {
        for run in &r.score.missing_runs {
            lines.push(format!(
                "{name}: {} consecutive words missing in '{}' from reference word {}",
                run.words, run.section, run.first_word
            ));
        }
    }
    lines.join("\n")
}
