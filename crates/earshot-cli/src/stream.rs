//! Live streaming array front end: reads continuous raw frames, emits mono PCM16 + gate/DOA JSONL.
use crate::layout::Layout;
use earshot_core::beam::{Beam, Kind};
use earshot_core::direction::{Finder, Found};
use earshot_core::stft::{Analysis, Synthesis, BINS, HOP};
use earshot_core::Complex32;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Write};
use std::path::PathBuf;

const INTERVAL: usize = 16; // hops: 0.256 s at 16 kHz
const AGREE: usize = 3; // measurements that must agree before auto-steering
const NEAR: f64 = 20.0; // degrees within which two DOA measurements agree
const LOUD: f64 = 4.0; // energy factor above noise floor (6 dB)
const ECHO_LOUD: f64 = 8.0; // residual factor during playback (~9 dB)
const STEP: f64 = 10.0; // degrees per precomputed beam
const PLAYING: f32 = 1e-4;
const PLAY_HOLD: usize = 2; // intervals after playback during which room rings
const SPEECH_HOLD: usize = 2; // hangover (~512 ms)

pub struct StreamConfig {
    pub array: PathBuf,
    pub steer: String,
    pub elevation: f64,
    pub kind: Kind,
    pub white_noise_gain: f64,
    pub gain: f64,
    pub input: PathBuf,
}

fn apart(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

fn circular_mean(azimuths: &[f64]) -> f64 {
    let (s, c) = azimuths
        .iter()
        .fold((0.0, 0.0), |(s, c), a| (s + a.to_radians().sin(), c + a.to_radians().cos()));
    s.atan2(c).to_degrees().rem_euclid(360.0)
}

fn read_hop<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<bool, String> {
    let mut filled = 0;
    while filled < buf.len() {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(format!("stream read failed: {e}")),
        }
    }
    Ok(true)
}

fn build_beams(layout: &Layout, cfg: &StreamConfig) -> Result<(bool, Vec<Beam>, Vec<f64>), String> {
    let mics = layout.array();
    let kind = match cfg.kind {
        Kind::DelaySum => Kind::DelaySum,
        Kind::Superdirective { .. } => Kind::Superdirective {
            min_white_noise_gain_db: cfg.white_noise_gain,
        },
    };
    if cfg.steer.trim() == "auto" {
        let count = (360.0 / STEP).round() as usize;
        let degs: Vec<f64> = (0..count).map(|i| i as f64 * STEP).collect();
        let beams = degs
            .iter()
            .map(|&az| Beam::design(&mics, layout.rate, az, cfg.elevation, kind))
            .collect();
        return Ok((true, beams, degs));
    }
    if cfg.steer.contains(':') || cfg.steer.contains(',') {
        return Err("--steer: stream mode supports 'auto' or a single degree angle".into());
    }
    let az = cfg
        .steer
        .trim()
        .parse::<f64>()
        .map(|d| d.rem_euclid(360.0))
        .map_err(|_| format!("--steer: invalid angle '{}'", cfg.steer))?;
    let beam = Beam::design(&mics, layout.rate, az, cfg.elevation, kind);
    Ok((false, vec![beam], vec![az]))
}

struct GateLine {
    t: f64,
    speech_active: bool,
    echo_residual_ok: bool,
    playback_active: bool,
    found: Option<Found>,
    steer_deg: f64,
    level_db: f64,
}

fn write_gate_json<E: Write>(err: &mut E, g: &GateLine) {
    let (az_s, ct_s) = match g.found {
        Some(f) => (format!("{:.1}", f.azimuth), format!("{:.2}", f.contrast)),
        None => ("null".to_string(), "null".to_string()),
    };
    let _ = writeln!(
        err,
        "{{\"t\": {:.3}, \"speech_active\": {}, \"echo_residual_ok\": {}, \"playback_active\": {}, \"azimuth\": {az_s}, \"contrast\": {ct_s}, \"steer_deg\": {:.1}, \"level_db\": {:.1}}}",
        g.t, g.speech_active, g.echo_residual_ok, g.playback_active, g.steer_deg, g.level_db
    );
    let _ = err.flush();
}

pub fn stream_io<R: Read, W: Write, E: Write>(
    cfg: &StreamConfig,
    reader: &mut R,
    out_writer: &mut W,
    err_writer: &mut E,
) -> Result<(), String> {
    let layout = Layout::load(&cfg.array)?;
    let (auto_steer, beams, beam_degs) = build_beams(&layout, cfg)?;
    let mics = layout.array();
    let (n, step, hop_bytes) = (mics.len(), layout.frame_bytes(), HOP * layout.frame_bytes());
    let mut finder = Finder::new(&mics, layout.rate, cfg.elevation, 72, 300.0, 3500.0);
    let mut analysis: Vec<Analysis> = (0..n).map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; n];
    let (mut synthesis, mut mixed) = (Synthesis::default(), [Complex32::new(0.0, 0.0); BINS]);
    let (mut hop, mut out) = ([0.0f32; HOP], [0.0f32; HOP]);
    let scale = 32768.0 * 10f32.powf(cfg.gain as f32 / 20.0);
    let mut raw_hop = vec![0u8; hop_bytes];
    let mut out_bytes = [0u8; HOP * 2];
    let (mut active_idx, mut energy, mut playing) = (0usize, 0.0f64, false);
    let (mut floor, mut rings, mut speech_hold) = (f64::INFINITY, 0usize, 0usize);
    let mut pending: Vec<f64> = Vec::new();
    let mut current: Option<f64> = None;

    for k in 0usize.. {
        if !read_hop(reader, &mut raw_hop)? {
            let _ = out_writer.flush();
            return Ok(());
        }
        for (m, a) in analysis.iter_mut().enumerate() {
            for (i, v) in hop.iter_mut().enumerate() {
                *v = layout.sample(&raw_hop[i * step..(i + 1) * step], m);
            }
            if m == 0 {
                energy += hop.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            }
            a.push(&hop, &mut spectra[m]);
        }
        playing |= (0..HOP).any(|i| layout.reference_peak(&raw_hop[i * step..(i + 1) * step]) > PLAYING);
        finder.add(&spectra);
        beams[active_idx].apply(&spectra, &mut mixed);
        synthesis.pull(&mixed, &mut out);
        if k > 0 {
            for (i, v) in out.iter().enumerate() {
                let s = (v * scale).round().clamp(-32768.0, 32767.0) as i16;
                out_bytes[i * 2..(i + 1) * 2].copy_from_slice(&s.to_le_bytes());
            }
            match out_writer.write_all(&out_bytes) {
                Ok(()) => {}
                Err(e) if e.kind() == ErrorKind::BrokenPipe => return Ok(()),
                Err(e) => return Err(format!("stream write failed: {e}")),
            }
            if k % 10 == 0 {
                match out_writer.flush() {
                    Ok(()) => {}
                    Err(e) if e.kind() == ErrorKind::BrokenPipe => return Ok(()),
                    Err(e) => return Err(format!("stream flush failed: {e}")),
                }
            }
        }
        if (k + 1) % INTERVAL != 0 {
            continue;
        }
        let t = ((k + 1) * HOP) as f64 / layout.rate as f64;
        let level_db = 10.0 * (energy / (INTERVAL * HOP) as f64).max(1e-12).log10();
        let e = energy + 1e-12;
        floor = if e < floor { e } else { floor * 1.02 };
        rings = if playing { PLAY_HOLD } else { rings.saturating_sub(1) };
        let playback_active = playing || rings > 0;
        let loud = e > LOUD * floor;
        let echo_residual_ok = !playback_active || e > ECHO_LOUD * floor;
        speech_hold = if loud && echo_residual_ok {
            SPEECH_HOLD
        } else {
            speech_hold.saturating_sub(1)
        };
        let speech_active = (loud && echo_residual_ok) || speech_hold > 0;
        let found = if loud && rings == 0 { finder.found() } else { None };
        if auto_steer {
            match found {
                Some(f) => {
                    let az = f.azimuth as f64;
                    if current.is_some_and(|c| apart(az, c) <= NEAR) {
                        pending.clear();
                    } else {
                        if pending.last().is_some_and(|&last| apart(az, last) > NEAR) {
                            pending.clear();
                        }
                        pending.push(az);
                        if pending.len() >= AGREE {
                            let idx = ((circular_mean(&pending) / STEP).round() as usize) % beam_degs.len();
                            active_idx = idx;
                            current = Some(beam_degs[idx]);
                            pending.clear();
                        }
                    }
                }
                None => pending.clear(),
            }
        }
        write_gate_json(
            err_writer,
            &GateLine {
                t,
                speech_active,
                echo_residual_ok,
                playback_active,
                found,
                steer_deg: beam_degs[active_idx],
                level_db,
            },
        );
        finder.clear();
        energy = 0.0;
        playing = false;
    }
    Ok(())
}

pub fn stream(cfg: StreamConfig) -> Result<(), String> {
    let stdout = io::stdout();
    let stderr = io::stderr();
    let mut out_lock = stdout.lock();
    let mut err_lock = stderr.lock();
    if cfg.input.as_os_str() == "-" {
        let stdin = io::stdin();
        let mut in_lock = stdin.lock();
        stream_io(&cfg, &mut in_lock, &mut out_lock, &mut err_lock)
    } else {
        let file = File::open(&cfg.input).map_err(|e| format!("{}: {e}", cfg.input.display()))?;
        let mut reader = BufReader::new(file);
        stream_io(&cfg, &mut reader, &mut out_lock, &mut err_lock)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn stream_steers_towards_source_and_emits_pcm16_and_jsonl() {
        let array_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../arrays/respeaker-core-v2.toml");
        let layout = Layout::load(&array_path).unwrap();
        let mics = layout.array();
        let target_az_deg = 100.0f64;
        let el_deg = 20.0f64;
        let total_hops = 96usize; // 6 intervals * 16 hops = 1.536 s
        let total_frames = total_hops * HOP;
        let speed_of_sound = 343.0f64;
        let az = target_az_deg.to_radians();
        let el = el_deg.to_radians();
        let dir = [az.cos() * el.cos(), az.sin() * el.cos(), el.sin()];

        let mut raw = Vec::with_capacity(total_frames * layout.frame_bytes());
        for f in 0..total_frames {
            let t = f as f64 / layout.rate as f64;
            let mut ch = [0i32; 8];
            for mic in &layout.mic {
                let p = mic.position;
                let tau = (p[0] * dir[0] + p[1] * dir[1] + p[2] * dir[2]) / speed_of_sound;
                let phase_t = t + tau;
                // First 16 hops (256 ms) quiet ambient floor, then active 700 + 1400 Hz speech-band tone.
                let amp = if f < INTERVAL * HOP { 0.002 } else { 0.25 };
                let s = amp
                    * ((2.0 * std::f64::consts::PI * 700.0 * phase_t).sin()
                        + 0.7 * (2.0 * std::f64::consts::PI * 1400.0 * phase_t).sin());
                ch[mic.channel] = (s * 2_147_483_647.0).clamp(-2_147_483_648.0, 2_147_483_647.0) as i32;
            }
            let _ = &mics;
            for v in ch {
                raw.extend_from_slice(&v.to_le_bytes());
            }
        }

        let cfg = StreamConfig {
            array: array_path,
            steer: "auto".into(),
            elevation: el_deg,
            kind: Kind::Superdirective {
                min_white_noise_gain_db: 0.0,
            },
            white_noise_gain: 0.0,
            gain: 0.0,
            input: PathBuf::from("-"),
        };
        let (mut pcm_out, mut err_out) = (Vec::new(), Vec::new());
        stream_io(&cfg, &mut Cursor::new(raw), &mut pcm_out, &mut err_out).unwrap();
        assert_eq!(pcm_out.len(), (total_hops - 1) * HOP * 2);

        let lines: Vec<&str> = std::str::from_utf8(&err_out).unwrap().lines().collect();
        assert_eq!(lines.len(), total_hops / INTERVAL);
        let last = lines.last().unwrap();
        assert!(last.contains("\"speech_active\": true"), "expected speech_active in {last}");
        assert!(last.contains("\"steer_deg\": 100.0"), "expected steer_deg 100.0 in {last}");
    }
}
