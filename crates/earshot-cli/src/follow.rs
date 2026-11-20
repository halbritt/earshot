//! Agent-mode stream policy for recordings: follow the one talker.
//!
//! The direction is measured four times a second. A measurement counts when the interval is
//! clearly louder than the room and the device itself is not playing. The beam turns when
//! three measurements in a row agree on a new direction, and it turns from the first of
//! them: a live version must hold the audio back by that long (0.77 s) to do the same.
use crate::layout::Layout;
use earshot_core::direction::Finder;
use earshot_core::stft::{Analysis, BINS, HOP};
use earshot_core::Complex32;

const INTERVAL: usize = 16; // hops: 0.256 s at 16 kHz
const AGREE: usize = 3; // measurements that must agree before the beam turns
const NEAR: f64 = 20.0; // degrees within which two measurements agree
const LOUD: f64 = 4.0; // an interval this many times the room's energy is a sound (6 dB)
const STEP: f64 = 10.0; // beams are designed on this grid, degrees
const PLAYING: f32 = 1e-4; // reference peak above this: the device is playing
const PLAY_HOLD: usize = 2; // intervals after playback during which the room still rings

fn apart(a: f64, b: f64) -> f64 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

fn mean(azimuths: &[f64]) -> f64 {
    let (s, c) = azimuths
        .iter()
        .fold((0.0, 0.0), |(s, c), a| (s + a.to_radians().sin(), c + a.to_radians().cos()));
    s.atan2(c).to_degrees().rem_euclid(360.0)
}

/// The look direction from each time on, in the form `--steer` takes. The first direction
/// found also covers the time before it.
pub fn follow(layout: &Layout, raw: &[u8], elevation: f64) -> Vec<(f64, f64)> {
    let mics = layout.array();
    let (n, step) = (mics.len(), layout.frame_bytes());
    let mut finder = Finder::new(&mics, layout.rate, elevation, 72, 300.0, 3500.0);
    let mut analysis: Vec<Analysis> = (0..n).map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; n];
    let mut hop = [0.0f32; HOP];
    let (mut energy, mut playing, mut floor, mut rings) = (0.0f64, false, f64::INFINITY, 0usize);
    let mut plan: Vec<(f64, f64)> = Vec::new();
    let mut pending: Vec<(f64, f64)> = Vec::new(); // start and azimuth of measurements that agree
    let mut current: Option<f64> = None;
    for k in 0..raw.len() / step / HOP {
        for (m, a) in analysis.iter_mut().enumerate() {
            for (i, v) in hop.iter_mut().enumerate() {
                *v = layout.sample(&raw[(k * HOP + i) * step..(k * HOP + i + 1) * step], m);
            }
            if m == 0 {
                energy += hop.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
            }
            a.push(&hop, &mut spectra[m]);
        }
        playing |= (0..HOP).any(|i| layout.reference_peak(&raw[(k * HOP + i) * step..(k * HOP + i + 1) * step]) > PLAYING);
        finder.add(&spectra);
        if (k + 1) % INTERVAL != 0 {
            continue;
        }
        let start = ((k + 1 - INTERVAL) * HOP) as f64 / layout.rate as f64;
        energy += 1e-12;
        // the room's level: down at once, up slowly
        floor = if energy < floor { energy } else { floor * 1.02 };
        rings = if playing { PLAY_HOLD } else { rings.saturating_sub(1) };
        let heard = if energy > LOUD * floor && rings == 0 {
            finder.found()
        } else {
            None
        };
        match heard {
            Some(found) => {
                let az = found.azimuth as f64;
                if current.is_some_and(|c| apart(az, c) <= NEAR) {
                    pending.clear();
                } else {
                    if pending.last().is_some_and(|&(_, last)| apart(az, last) > NEAR) {
                        pending.clear();
                    }
                    pending.push((start, az));
                    if pending.len() >= AGREE {
                        let to = (mean(&pending.iter().map(|p| p.1).collect::<Vec<_>>()) / STEP).round() * STEP % 360.0;
                        plan.push((if plan.is_empty() { 0.0 } else { pending[0].0 }, to));
                        current = Some(to);
                        pending.clear();
                    }
                }
            }
            None => pending.clear(),
        }
        finder.clear();
        (energy, playing) = (0.0, false);
    }
    if plan.is_empty() {
        plan.push((0.0, 0.0));
    }
    plan
}
