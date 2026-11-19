//! Direction finding on sound made from the geometry.
use earshot_core::array::{Array, SOUND_SPEED};
use earshot_core::direction::Finder;
use earshot_core::stft::{Analysis, BINS, HOP};
use earshot_core::Complex32;

fn ring() -> Array {
    Array::ring(6, 0.0463, 30.0)
}

// What each microphone hears of a sound from a direction in the horizontal plane: thirty
// tones between 400 and 3300 Hz, so that it is as broad as a voice.
fn sound(array: &Array, azimuth: f64, frames: usize, level: f64) -> Vec<Vec<f32>> {
    let (c, s) = (azimuth.to_radians().cos(), azimuth.to_radians().sin());
    array
        .mics
        .iter()
        .map(|p| {
            let lead = (p[0] * c + p[1] * s) / SOUND_SPEED;
            (0..frames * HOP)
                .map(|i| {
                    let t = i as f64 / 16000.0 + lead;
                    let sum: f64 = (0..30)
                        .map(|k| (2.0 * std::f64::consts::PI * (400.0 + 97.3 * k as f64) * t + k as f64).sin())
                        .sum();
                    (level * sum / 30.0) as f32
                })
                .collect()
        })
        .collect()
}

fn mix(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(p, q)| p + q).collect())
        .collect()
}

fn find(array: &Array, mics: &[Vec<f32>]) -> Option<earshot_core::direction::Found> {
    let mut finder = Finder::new(array, 16000, 0.0, 72, 300.0, 3500.0);
    let mut analysis: Vec<Analysis> = mics.iter().map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; mics.len()];
    for k in 0..mics[0].len() / HOP {
        for (m, a) in analysis.iter_mut().enumerate() {
            a.push(mics[m][k * HOP..(k + 1) * HOP].try_into().unwrap(), &mut spectra[m]);
        }
        if k > 0 {
            finder.add(&spectra);
        }
    }
    finder.found()
}

fn apart(a: f32, b: f32) -> f32 {
    ((a - b + 180.0).rem_euclid(360.0) - 180.0).abs()
}

#[test]
fn finds_a_sound_wherever_it_is_round_the_array() {
    for azimuth in [0.0, 37.0, 100.0, 181.0, 305.0, 358.0] {
        let found = find(&ring(), &sound(&ring(), azimuth, 10, 0.5)).unwrap();
        assert!(
            apart(found.azimuth, azimuth as f32) <= 2.0,
            "sound at {azimuth}, found {found:?}"
        );
        assert!(found.contrast > 2.0, "{found:?}");
    }
}

#[test]
fn the_level_of_the_sound_does_not_matter() {
    let loud = find(&ring(), &sound(&ring(), 100.0, 10, 0.5)).unwrap();
    let quiet = find(&ring(), &sound(&ring(), 100.0, 10, 0.0005)).unwrap();
    assert!(apart(loud.azimuth, quiet.azimuth) < 0.5 && (loud.contrast - quiet.contrast).abs() < 0.05 * loud.contrast);
}

#[test]
fn of_two_sounds_the_stronger_is_found() {
    let both = mix(&sound(&ring(), 100.0, 10, 0.5), &sound(&ring(), 250.0, 10, 0.1));
    let found = find(&ring(), &both).unwrap();
    assert!(apart(found.azimuth, 100.0) <= 5.0, "{found:?}");
}

#[test]
fn silence_has_no_direction() {
    let silent = vec![vec![0.0f32; HOP * 10]; 6];
    assert_eq!(find(&ring(), &silent), None);
}

#[test]
fn clearing_forgets_what_was_heard() {
    let array = ring();
    let mut finder = Finder::new(&array, 16000, 0.0, 72, 300.0, 3500.0);
    let mics = sound(&array, 100.0, 4, 0.5);
    let mut analysis: Vec<Analysis> = mics.iter().map(|_| Analysis::default()).collect();
    let mut spectra = vec![[Complex32::new(0.0, 0.0); BINS]; 6];
    for k in 0..4 {
        for (m, a) in analysis.iter_mut().enumerate() {
            a.push(mics[m][k * HOP..(k + 1) * HOP].try_into().unwrap(), &mut spectra[m]);
        }
        finder.add(&spectra);
    }
    assert!(finder.found().is_some());
    finder.clear();
    assert_eq!(finder.found(), None);
}
