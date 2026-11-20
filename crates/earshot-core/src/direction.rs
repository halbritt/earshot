//! Where a sound comes from: steered response power with phase transform (SRP-PHAT), over
//! azimuth only.
//!
//! Every microphone's spectrum is reduced to its phase, so that loud low frequencies do not
//! decide alone; frequencies with almost nothing in them are left out. For each candidate direction the phases are aligned and summed; the
//! direction in which they agree best is where the sound came from.
use crate::array::Array;
use crate::stft::BINS;
use alloc::vec;
use alloc::vec::Vec;
use num_complex::Complex32;

pub struct Finder {
    mics: usize,
    directions: usize,
    low: usize,
    high: usize,           // bins low..high are used
    align: Vec<Complex32>, // direction-major, then bin, then microphone: conj of the steering vector
    power: Vec<f32>,       // per direction, summed over the frames added
    frames: usize,
}

/// A direction and how much it stands out.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Found {
    /// Degrees, 0 to 360.
    pub azimuth: f32,
    /// Power in this direction over the mean of all directions. 1 = nothing stands out.
    pub contrast: f32,
}

impl Finder {
    /// `directions` candidates evenly round the circle, at one `elevation`; frequencies from
    /// `low_hz` to `high_hz`.
    pub fn new(array: &Array, rate: u32, elevation: f64, directions: usize, low_hz: f64, high_hz: f64) -> Finder {
        let hz_per_bin = rate as f64 / ((BINS - 1) * 2) as f64;
        let low = (libm::ceil(low_hz / hz_per_bin) as usize).clamp(1, BINS - 1);
        let high = (libm::floor(high_hz / hz_per_bin) as usize + 1).clamp(low + 1, BINS);
        let mut align = Vec::with_capacity(directions * (high - low) * array.len());
        for d in 0..directions {
            for bin in low..high {
                let s = array.steering(bin as f64 * hz_per_bin, 360.0 * d as f64 / directions as f64, elevation);
                align.extend(s.iter().map(|x| Complex32::new(x.re as f32, -x.im as f32)));
            }
        }
        Finder {
            mics: array.len(),
            directions,
            low,
            high,
            align,
            power: vec![0.0; directions],
            frames: 0,
        }
    }

    /// Adds one frame of every microphone. Allocates nothing.
    pub fn add(&mut self, spectra: &[[Complex32; BINS]]) {
        debug_assert_eq!(spectra.len(), self.mics);
        let (bins, mics) = (self.high - self.low, self.mics);
        // Bins with next to nothing in them (40 dB under the strongest) hold only what
        // leaked from the others, with their phase and not their own: leave them out.
        let power = |b: usize| spectra[0][self.low + b].norm_sqr();
        let least = (0..bins).map(power).fold(0.0, f32::max) * 1e-4;
        for d in 0..self.directions {
            let mut sum = 0.0;
            for b in (0..bins).filter(|&b| power(b) > least) {
                let w = &self.align[(d * bins + b) * mics..(d * bins + b + 1) * mics];
                let mut s = Complex32::new(0.0, 0.0);
                for (w, x) in w.iter().zip(spectra) {
                    let x = x[self.low + b];
                    let size = x.norm();
                    if size > 0.0 {
                        s += w * x / size;
                    }
                }
                sum += s.norm_sqr();
            }
            self.power[d] += sum;
        }
        self.frames += 1;
    }

    /// Forgets the frames added so far.
    pub fn clear(&mut self) {
        self.power.iter_mut().for_each(|p| *p = 0.0);
        self.frames = 0;
    }

    /// The direction that stands out in the frames added, or `None` when there were none or
    /// they were silent.
    pub fn found(&self) -> Option<Found> {
        let mean = self.power.iter().sum::<f32>() / self.directions as f32;
        if self.frames == 0 || mean <= 0.0 {
            return None;
        }
        let best = (0..self.directions).max_by(|&a, &b| self.power[a].total_cmp(&self.power[b]))?;
        // a parabola through the peak and its neighbours places it between candidates
        let n = self.directions;
        let (l, c, r) = (self.power[(best + n - 1) % n], self.power[best], self.power[(best + 1) % n]);
        let bend = l - 2.0 * c + r;
        let shift = if bend < 0.0 {
            (0.5 * (l - r) / bend).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let turn = (best as f32 + shift) * 360.0 / n as f32;
        let azimuth = if turn < 0.0 {
            turn + 360.0
        } else if turn >= 360.0 {
            turn - 360.0
        } else {
            turn
        };
        Some(Found {
            azimuth,
            contrast: c / mean,
        })
    }
}
