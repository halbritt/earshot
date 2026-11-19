use clap::{Parser, Subcommand};
use earshot_bench::run::{bench_variant, outcome, table, Outcome, Take, Texts};
use earshot_bench::stt::Recogniser;
use std::path::PathBuf;
use std::process::ExitCode;

/// Score microphone-array front-end variants on recorded takes.
#[derive(Parser)]
#[command(name = "earshot-bench", version)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run variants on a take and score them against the reference text
    Run {
        /// Take description (TOML)
        take: PathBuf,
        /// A variant: NAME='COMMAND {in} {out}'. The command reads the recording at {in} and
        /// writes one channel of 16-bit WAV to {out}
        #[arg(long, value_name = "NAME=COMMAND")]
        variant: Vec<String>,
        /// Score an existing transcript: NAME=FILE.json, a JSON object of chunk name to text
        #[arg(long, value_name = "NAME=FILE")]
        transcript: Vec<String>,
        /// Address of the recogniser's transcription endpoint
        #[arg(long, env = "EARSHOT_STT_URL", hide_env_values = true)]
        stt_url: Option<String>,
        #[arg(long, default_value = "en")]
        language: String,
        /// Directory for cached transcripts ('' for none)
        #[arg(long, env = "EARSHOT_CACHE", default_value = ".earshot-cache")]
        cache: String,
        /// Keep each variant's output audio in this directory
        #[arg(long, value_name = "DIR")]
        keep: Option<PathBuf>,
        /// Write full results, with transcripts, to this file
        #[arg(long, value_name = "FILE")]
        json: Option<PathBuf>,
    },
    /// Take one channel of a raw interleaved recording and write it as 16-bit WAV
    Pick {
        /// Channels in the recording
        #[arg(long)]
        channels: usize,
        /// Bytes per sample
        #[arg(long, default_value_t = 4)]
        width: usize,
        /// Channel to take, counted from 0
        #[arg(long, default_value_t = 0)]
        channel: usize,
        #[arg(long, default_value_t = 16000)]
        rate: u32,
        /// Keep the level as recorded (default: raise the peak to 0.9 of full scale)
        #[arg(long)]
        no_normalise: bool,
        input: PathBuf,
        output: PathBuf,
    },
}

fn pairs(items: &[String], what: &str) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    for item in items {
        match item.split_once('=') {
            Some((n, v)) if !n.is_empty() && !v.is_empty() => {
                if out.iter().any(|(k, _)| k == n) {
                    return Err(format!("--{what} '{n}' is given twice"));
                }
                out.push((n.to_string(), v.to_string()));
            }
            _ => return Err(format!("--{what} needs NAME=VALUE, got '{item}'")),
        }
    }
    Ok(out)
}

fn run(cmd: Cmd) -> Result<(), String> {
    match cmd {
        Cmd::Pick {
            channels,
            width,
            channel,
            rate,
            no_normalise,
            input,
            output,
        } => {
            if channel >= channels {
                return Err("--channel must be less than --channels".into());
            }
            if width != 2 && width != 4 {
                return Err("--width must be 2 or 4".into());
            }
            let raw = std::fs::read(&input).map_err(|e| format!("{}: {e}", input.display()))?;
            let samples = earshot_bench::pick::pick(&raw, channels, width, channel, !no_normalise);
            std::fs::write(&output, earshot_bench::run::wav_bytes(rate, &samples))
                .map_err(|e| format!("{}: {e}", output.display()))
        }
        Cmd::Run {
            take,
            variant,
            transcript,
            stt_url,
            language,
            cache,
            keep,
            json,
        } => {
            let variants = pairs(&variant, "variant")?;
            let files = pairs(&transcript, "transcript")?;
            if variants.is_empty() && files.is_empty() {
                return Err("give at least one --variant or --transcript".into());
            }
            if let Some((n, _)) = variants.iter().find(|(n, _)| files.iter().any(|(m, _)| m == n)) {
                return Err(format!("'{n}' is the name of both a variant and a transcript"));
            }
            let recogniser = match (variants.is_empty(), stt_url) {
                (true, _) => None,
                (false, Some(url)) => Some(Recogniser {
                    url,
                    language,
                    cache: (!cache.is_empty()).then(|| cache.into()),
                }),
                (false, None) => return Err("no recogniser: give --stt-url or set EARSHOT_STT_URL".into()),
            };
            let take = Take::load(&take)?;
            let tmp;
            let out_dir = match &keep {
                Some(d) => {
                    std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
                    d.clone()
                }
                None => {
                    tmp = std::env::temp_dir().join(format!("earshot-bench-{}", std::process::id()));
                    std::fs::create_dir_all(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
                    tmp.clone()
                }
            };
            let mut results: Vec<(String, Outcome)> = Vec::new();
            let mut failure = None;
            for (name, command) in &variants {
                match bench_variant(
                    &take,
                    command,
                    &out_dir.join(format!("{name}.wav")),
                    recogniser.as_ref().unwrap(),
                ) {
                    Ok(o) => results.push((name.clone(), o)),
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                }
            }
            if keep.is_none() {
                let _ = std::fs::remove_dir_all(&out_dir);
            }
            if let Some(e) = failure {
                return Err(e);
            }
            for (name, path) in &files {
                let data = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                let texts: Texts =
                    serde_json::from_str(&data).map_err(|_| format!("{path}: expected a JSON object of chunk name to text"))?;
                if !texts.values().all(|v| v.is_string()) {
                    return Err(format!("{path}: expected a JSON object of chunk name to text"));
                }
                results.push((name.clone(), outcome(&take, texts, None)));
            }
            println!(
                "take {}: {} reference words, {} playback words",
                take.name,
                take.reference_words(),
                take.playback.len()
            );
            println!("{}", table(&results));
            if let Some(path) = json {
                let map: serde_json::Map<String, serde_json::Value> = results
                    .iter()
                    .map(|(n, o)| (n.clone(), serde_json::to_value(o).expect("results are plain data")))
                    .collect();
                let doc = serde_json::json!({ "take": take.name, "results": map });
                std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap())
                    .map_err(|e| format!("{}: {e}", path.display()))?;
            }
            Ok(())
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse().command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("earshot-bench: {e}");
            ExitCode::from(2)
        }
    }
}
