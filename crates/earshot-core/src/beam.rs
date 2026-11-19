//! Fixed beams, calculated once from the geometry.
use crate::array::Array;
use crate::stft::BINS;
use alloc::vec;
use alloc::vec::Vec;
use num_complex::{Complex32, Complex64};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    /// Align the microphones on the look direction and average.
    DelaySum,
    /// Least diffuse noise for an undistorted look direction, with the white-noise gain
    /// kept at or above the limit so that microphone noise and mismatch are not amplified.
    Superdirective { min_white_noise_gain_db: f64 },
}

/// Weights for one frequency. The beam's output is the sum of conj(w[m]) * x[m].
pub fn weights(array: &Array, hz: f64, azimuth: f64, elevation: f64, kind: Kind) -> Vec<Complex64> {
    let n = array.len();
    let d = array.steering(hz, azimuth, elevation);
    let limit = match kind {
        Kind::DelaySum => return d.iter().map(|x| x / n as f64).collect(),
        Kind::Superdirective { min_white_noise_gain_db } => min_white_noise_gain_db,
    };
    let g = array.diffuse(hz);
    let mut w = Vec::new();
    // More loading of the diagonal gives more white-noise gain and less directivity: take
    // the least loading that meets the limit.
    for step in 0..=35 {
        let eps = libm::pow(10.0, -5.0 + 0.2 * step as f64);
        let mut a = g.clone();
        for i in 0..n {
            a[i * n + i] += eps;
        }
        let l = cholesky(&a, n);
        let re = solve(&l, n, &d.iter().map(|x| x.re).collect::<Vec<_>>());
        let im = solve(&l, n, &d.iter().map(|x| x.im).collect::<Vec<_>>());
        let rd: Vec<Complex64> = re.iter().zip(&im).map(|(&r, &i)| Complex64::new(r, i)).collect();
        let gain: Complex64 = d.iter().zip(&rd).map(|(a, b)| a.conj() * b).sum();
        w = rd.iter().map(|x| x / gain).collect();
        if white_noise_gain_db(&w) >= limit {
            break;
        }
    }
    w
}

/// Gain over noise that is independent in every microphone, in dB.
pub fn white_noise_gain_db(w: &[Complex64]) -> f64 {
    -10.0 * libm::log10(w.iter().map(|x| x.norm_sqr()).sum())
}

/// Gain over spherically diffuse noise, in dB, for a beam with unit gain in its look direction.
pub fn directivity_db(array: &Array, hz: f64, w: &[Complex64]) -> f64 {
    let (n, g) = (array.len(), array.diffuse(hz));
    let mut noise = 0.0;
    for i in 0..n {
        for j in 0..n {
            noise += (w[i].conj() * w[j]).re * g[i * n + j];
        }
    }
    -10.0 * libm::log10(noise)
}

/// The beam's power response to a plane wave from a direction, as a ratio.
pub fn response(array: &Array, hz: f64, w: &[Complex64], azimuth: f64, elevation: f64) -> f64 {
    let d = array.steering(hz, azimuth, elevation);
    w.iter().zip(&d).map(|(w, d)| w.conj() * d).sum::<Complex64>().norm_sqr()
}

// Lower-triangular factor of a symmetric positive-definite matrix, row-major.
fn cholesky(a: &[f64], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i * n + k] * l[j * n + k]).sum();
            l[i * n + j] = if i == j {
                libm::sqrt(a[i * n + i] - s)
            } else {
                (a[i * n + j] - s) / l[j * n + j]
            };
        }
    }
    l
}

fn solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let s: f64 = (0..i).map(|k| l[i * n + k] * y[k]).sum();
        y[i] = (b[i] - s) / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let s: f64 = (i + 1..n).map(|k| l[k * n + i] * x[k]).sum();
        x[i] = (y[i] - s) / l[i * n + i];
    }
    x
}

/// One beam for every frequency bin of the STFT, ready for the audio path.
pub struct Beam {
    mics: usize,
    conj: Vec<Complex32>, // bin-major: conj(w) for each microphone
}

impl Beam {
    pub fn design(array: &Array, rate: u32, azimuth: f64, elevation: f64, kind: Kind) -> Beam {
        let mics = array.len();
        let mut conj = Vec::with_capacity(BINS * mics);
        for bin in 0..BINS {
            let hz = bin as f64 * rate as f64 / ((BINS - 1) * 2) as f64;
            // below the voice there is nothing to gain and the matrix is at its worst
            let w = weights(array, hz.max(50.0), azimuth, elevation, kind);
            conj.extend(w.iter().map(|x| Complex32::new(x.re as f32, -x.im as f32)));
        }
        Beam { mics, conj }
    }

    /// `spectra`: one frame of every microphone, microphone-major. Allocates nothing.
    pub fn apply(&self, spectra: &[[Complex32; BINS]], out: &mut [Complex32; BINS]) {
        debug_assert_eq!(spectra.len(), self.mics);
        for (bin, o) in out.iter_mut().enumerate() {
            let w = &self.conj[bin * self.mics..(bin + 1) * self.mics];
            *o = w.iter().zip(spectra).map(|(w, s)| w * s[bin]).sum();
        }
    }
}
