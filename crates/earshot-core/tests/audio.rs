//! The audio path: transform, beam, and back.
use earshot_core::array::{Array, SOUND_SPEED};
use earshot_core::beam::{Beam, Kind};
use earshot_core::stft::{Analysis, Synthesis, BINS, HOP};
use earshot_core::Complex32;

// a repeatable signal with energy at all frequencies
fn noise(n: usize) -> Vec<f32> {
    let mut s = 12345u32;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0
        })
        .collect()
}

#[test]
fn analysis_then_synthesis_returns_the_input_one_hop_late() {
    let x = noise(HOP * 40);
    let (mut a, mut s) = (Analysis::default(), Synthesis::default());
    let mut y = Vec::new();
    let mut spec = [Complex32::new(0.0, 0.0); BINS];
    let mut out = [0.0f32; HOP];
    for hop in x.as_chunks::<HOP>().0 {
        a.push(hop, &mut spec);
        s.pull(&spec, &mut out);
        y.extend_from_slice(&out);
    }
    for i in HOP..x.len() {
        assert!((y[i] - x[i - HOP]).abs() < 1e-4, "sample {i}: {} for {}", y[i], x[i - HOP]);
    }
}

// What each microphone hears of a tone from a direction in the horizontal plane.
fn plane_wave(array: &Array, hz: f64, azimuth: f64, n: usize) -> Vec<Vec<f32>> {
    let (c, s) = (azimuth.to_radians().cos(), azimuth.to_radians().sin());
    array
        .mics
        .iter()
        .map(|p| {
            let lead = (p[0] * c + p[1] * s) / SOUND_SPEED;
            (0..n)
                .map(|i| (2.0 * std::f64::consts::PI * hz * (i as f64 / 16000.0 + lead)).sin() as f32)
                .collect()
        })
        .collect()
}

fn through(beam: &Beam, mics: &[Vec<f32>]) -> Vec<f32> {
    let mut analysis: Vec<Analysis> = mics.iter().map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; mics.len()];
    let (mut synthesis, mut mixed, mut out, mut y) = (
        Synthesis::default(),
        [Complex32::new(0.0, 0.0); BINS],
        [0.0f32; HOP],
        Vec::new(),
    );
    for k in 0..mics[0].len() / HOP {
        for (m, a) in analysis.iter_mut().enumerate() {
            a.push(mics[m][k * HOP..(k + 1) * HOP].try_into().unwrap(), &mut spectra[m]);
        }
        beam.apply(&spectra, &mut mixed);
        synthesis.pull(&mixed, &mut out);
        y.extend_from_slice(&out);
    }
    y
}

fn level_db(x: &[f32]) -> f64 {
    10.0 * (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).log10()
}

#[test]
fn a_tone_from_the_look_direction_comes_through_at_its_own_level_and_one_from_behind_does_not() {
    let ring = Array::ring(6, 0.0463, 90.0);
    let beam = Beam::design(
        &ring,
        16000,
        100.0,
        0.0,
        Kind::Superdirective {
            min_white_noise_gain_db: 0.0,
        },
    );
    let n = HOP * 60;
    // 1000 Hz is the centre of a bin, so the tone's level is not smeared by the transform
    let front = through(&beam, &plane_wave(&ring, 1000.0, 100.0, n));
    let back = through(&beam, &plane_wave(&ring, 1000.0, 280.0, n));
    let tone = level_db(&plane_wave(&ring, 1000.0, 100.0, n)[0][HOP * 4..]);
    assert!(
        (level_db(&front[HOP * 4..]) - tone).abs() < 0.1,
        "front {} dB, tone {} dB",
        level_db(&front[HOP * 4..]),
        tone
    );
    assert!(
        level_db(&back[HOP * 4..]) < tone - 6.0,
        "back {} dB, tone {} dB",
        level_db(&back[HOP * 4..]),
        tone
    );
}
