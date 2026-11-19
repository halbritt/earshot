//! earshot: command-line front end for the microphone-array core.
mod layout;

use clap::{Parser, Subcommand, ValueEnum};
use earshot_core::beam::{Beam, Kind};
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
        /// Where the beam looks: degrees, or TIME:DEGREES,... to turn it at given seconds
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
        /// Raw interleaved recording
        input: PathBuf,
        /// WAV file to write
        output: PathBuf,
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
    input: PathBuf,
    output: PathBuf,
) -> Result<(), String> {
    let layout = Layout::load(&array)?;
    let plan = steering(steer)?;
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

    let mut raw = Vec::new();
    std::fs::File::open(&input)
        .and_then(|mut f| f.read_to_end(&mut raw))
        .map_err(|e| format!("{}: {e}", input.display()))?;
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

fn main() -> ExitCode {
    let done = match Cli::parse().command {
        Command::Beam {
            array,
            steer,
            elevation,
            kind,
            white_noise_gain,
            gain,
            input,
            output,
        } => beam(array, &steer, elevation, kind, white_noise_gain, gain, input, output),
    };
    match done {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("earshot: {e}");
            ExitCode::FAILURE
        }
    }
}
