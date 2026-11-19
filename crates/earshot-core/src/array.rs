//! The array as data: where the microphones are.
use alloc::vec::Vec;
use core::f64::consts::PI;
use num_complex::Complex64;

/// Speed of sound in air, metres per second.
pub const SOUND_SPEED: f64 = 343.0;

/// Microphone positions in metres. x points to azimuth 0, y to azimuth 90, z up.
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    pub mics: Vec<[f64; 3]>,
}

/// Unit vector towards a direction given in degrees.
pub fn towards(azimuth: f64, elevation: f64) -> [f64; 3] {
    let (az, el) = (azimuth.to_radians(), elevation.to_radians());
    [libm::cos(el) * libm::cos(az), libm::cos(el) * libm::sin(az), libm::sin(el)]
}

impl Array {
    pub fn new(mics: Vec<[f64; 3]>) -> Array {
        Array { mics }
    }

    /// `count` microphones evenly on a circle in the horizontal plane, the first at
    /// `first_at` degrees.
    pub fn ring(count: usize, radius: f64, first_at: f64) -> Array {
        let mics = (0..count)
            .map(|i| {
                let a = first_at.to_radians() + 2.0 * PI * i as f64 / count as f64;
                [radius * libm::cos(a), radius * libm::sin(a), 0.0]
            })
            .collect();
        Array { mics }
    }

    pub fn len(&self) -> usize {
        self.mics.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mics.is_empty()
    }

    /// What a plane wave of frequency `hz` from the given direction looks like at each
    /// microphone, relative to the origin (phase leads where the wave arrives earlier).
    pub fn steering(&self, hz: f64, azimuth: f64, elevation: f64) -> Vec<Complex64> {
        let u = towards(azimuth, elevation);
        self.mics
            .iter()
            .map(|p| {
                let lead = (p[0] * u[0] + p[1] * u[1] + p[2] * u[2]) / SOUND_SPEED;
                Complex64::from_polar(1.0, 2.0 * PI * hz * lead)
            })
            .collect()
    }

    /// Coherence between each pair of microphones in spherically diffuse noise at `hz`:
    /// sin(kd)/(kd). Row-major, `len()` by `len()`.
    pub fn diffuse(&self, hz: f64) -> Vec<f64> {
        let n = self.len();
        let mut g = Vec::with_capacity(n * n);
        for a in &self.mics {
            for b in &self.mics {
                let d = libm::sqrt((0..3).map(|i| (a[i] - b[i]) * (a[i] - b[i])).sum::<f64>());
                let x = 2.0 * PI * hz * d / SOUND_SPEED;
                g.push(if x < 1e-9 { 1.0 } else { libm::sin(x) / x });
            }
        }
        g
    }
}
