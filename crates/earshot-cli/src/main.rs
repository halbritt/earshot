//! earshot: command-line front end for the microphone-array core.
mod follow;
mod layout;

use clap::{Parser, Subcommand, ValueEnum};
use earshot_core::beam::{Beam, Kind};
use earshot_core::direction::Finder;
use earshot_core::stft::{Analysis, Synthesis, BINS, HOP};
use earshot_core::Complex32;
use layout::Layout;
use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// One fixed beam over a raw multichannel recording, written as 16-bit mono WAV.
    Beam {
        /// Array description (TOML)
        #[arg(long)]
        array: PathBuf,
        /// Where the beam looks: degrees, TIME:DEGREES,... to turn it at given seconds, or
        /// `auto` to follow the talker
        #[arg(long, default_value = "0")]
        steer: String,
        /// Degrees above the plane of the array
        #[arg(long, default_value_t = 20.0)]
        elevation: f64,
        #[arg(long, value_enum, default_value = "superdirective")]
        kind: BeamKind,
        /// Lowest white-noise gain a superdirective beam may have, dB
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        white_noise_gain: f64,
        /// Gain applied to the output, dB
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gain: f64,
        /// Say on standard error where the beam looks
        #[arg(long)]
        verbose: bool,
        /// Raw interleaved recording
        input: PathBuf,
        /// WAV file to write
        output: PathBuf,
    },
    /// Where the sound in a raw multichannel recording comes from, as one line of JSON per
    /// interval: time in seconds, azimuth in degrees, contrast and level.
    Direction {
        /// Array description (TOML)
        #[arg(long)]
        array: PathBuf,
        /// Seconds of sound for each answer
        #[arg(long, default_value_t = 0.5)]
        every: f64,
        /// Degrees above the plane of the array at which to look
        #[arg(long, default_value_t = 20.0)]
        elevation: f64,
        /// Raw interleaved recording
        input: PathBuf,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum BeamKind {
    DelaySum,
    Superdirective,
}

/// "100" or "0:100,113.5:305": the look direction from each time on.
fn steering(text: &str) -> Result<Vec<(f64, f64)>, String> {
    let mut plan = Vec::new();
    for part in text.split(',') {
        let (at, az) = match part.split_once(':') {
            Some((t, a)) => (t.trim().parse::<f64>().ok(), a.trim().parse::<f64>().ok()),
            None => (Some(0.0), part.trim().parse::<f64>().ok()),
        };
        match (at, az) {
            (Some(t), Some(a)) if t >= 0.0 => plan.push((t, a)),
            _ => return Err(format!("--steer: cannot read '{part}' (want DEGREES or SECONDS:DEGREES)")),
        }
    }
    if plan.windows(2).any(|w| w[1].0 <= w[0].0) {
        return Err("--steer: times must increase".into());
    }
    Ok(plan)
}

#[allow(clippy::too_many_arguments)]
fn beam(
    array: PathBuf,
    steer: &str,
    elevation: f64,
    kind: BeamKind,
    wng: f64,
    gain: f64,
    verbose: bool,
    input: PathBuf,
    output: PathBuf,
) -> Result<(), String> {
    let layout = Layout::load(&array)?;
    let mut raw = Vec::new();
    std::fs::File::open(&input)
        .and_then(|mut f| f.read_to_end(&mut raw))
        .map_err(|e| format!("{}: {e}", input.display()))?;
    let plan = if steer == "auto" {
        follow::follow(&layout, &raw, elevation)
    } else {
        steering(steer)?
    };
    if verbose {
        for (t, az) in &plan {
            eprintln!("from {t:.2} s the beam looks at {az:.0} degrees");
        }
    }
    let kind = match kind {
        BeamKind::DelaySum => Kind::DelaySum,
        BeamKind::Superdirective => Kind::Superdirective {
            min_white_noise_gain_db: wng,
        },
    };
    let mics = layout.array();
    let beams: Vec<Beam> = plan
        .iter()
        .map(|&(_, az)| Beam::design(&mics, layout.rate, az, elevation, kind))
        .collect();

    let step = layout.frame_bytes();
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: layout.rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut wav = hound::WavWriter::create(&output, spec).map_err(|e| format!("{}: {e}", output.display()))?;

    let n = mics.len();
    let mut analysis: Vec<Analysis> = (0..n).map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; n];
    let (mut synthesis, mut mixed) = (Synthesis::default(), [Complex32::new(0.0, 0.0); BINS]);
    let (mut hop, mut out) = ([0.0f32; HOP], [0.0f32; HOP]);
    let scale = 32768.0 * 10f32.powf(gain as f32 / 20.0);
    let frames = raw.len() / step;
    // one silent hop at the end pushes the last samples out of the overlap
    for k in 0..frames.div_ceil(HOP) + 1 {
        for (m, a) in analysis.iter_mut().enumerate() {
            for (i, v) in hop.iter_mut().enumerate() {
                let f = k * HOP + i;
                *v = if f < frames {
                    layout.sample(&raw[f * step..(f + 1) * step], m)
                } else {
                    0.0
                };
            }
            a.push(&hop, &mut spectra[m]);
        }
        let now = (k * HOP) as f64 / layout.rate as f64;
        let which = plan.iter().rposition(|&(t, _)| t <= now).unwrap_or(0);
        beams[which].apply(&spectra, &mut mixed);
        synthesis.pull(&mixed, &mut out);
        // the transform answers one hop late: leave that hop out so times match the recording
        if k == 0 {
            continue;
        }
        let keep = (frames - (k - 1) * HOP).min(HOP);
        for v in &out[..keep] {
            wav.write_sample((v * scale).round().clamp(-32768.0, 32767.0) as i16)
                .map_err(|e| e.to_string())?;
        }
    }
    wav.finalize().map_err(|e| e.to_string())
}

fn direction(array: PathBuf, every: f64, elevation: f64, input: PathBuf) -> Result<(), String> {
    let layout = Layout::load(&array)?;
    let mics = layout.array();
    let mut raw = Vec::new();
    std::fs::File::open(&input)
        .and_then(|mut f| f.read_to_end(&mut raw))
        .map_err(|e| format!("{}: {e}", input.display()))?;
    let step = layout.frame_bytes();
    let n = mics.len();
    let mut finder = Finder::new(&mics, layout.rate, elevation, 72, 300.0, 3500.0);
    let mut analysis: Vec<Analysis> = (0..n).map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; n];
    let mut hop = [0.0f32; HOP];
    let per = ((every * layout.rate as f64 / HOP as f64).round() as usize).max(1);
    let mut energy = 0.0f64;
    for k in 0..raw.len() / step / HOP {
        for (m, a) in analysis.iter_mut().enumerate() {
            for (i, v) in hop.iter_mut().enumerate() {
                let f = k * HOP + i;
                *v = layout.sample(&raw[f * step..(f + 1) * step], m);
            }
            if m == 0 {
                energy += hop.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            }
            a.push(&hop, &mut spectra[m]);
        }
        finder.add(&spectra);
        if (k + 1) % per == 0 {
            let t = ((k + 1) * HOP) as f64 / layout.rate as f64;
            let level = 10.0 * (energy / (per * HOP) as f64).max(1e-12).log10();
            if let Some(found) = finder.found() {
                println!(
                    "{{\"t\": {t:.2}, \"azimuth\": {:.1}, \"contrast\": {:.2}, \"level_db\": {level:.1}}}",
                    found.azimuth, found.contrast
                );
            }
            finder.clear();
            energy = 0.0;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let done = match Cli::parse().command {
        Command::Beam {
            array,
            steer,
            elevation,
            kind,
            white_noise_gain,
            gain,
            verbose,
            input,
            output,
        } => beam(array, &steer, elevation, kind, white_noise_gain, gain, verbose, input, output),
        Command::Direction {
            array,
            every,
            elevation,
            input,
        } => direction(array, every, elevation, input),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("earshot: {e}");
            ExitCode::FAILURE
        }
    }
}
