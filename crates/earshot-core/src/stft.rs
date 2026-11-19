//! Short-time Fourier transform: 512-sample frames (32 ms at 16 kHz), half overlapped, with
//! a square-root Hann window on both sides, so that analysis followed by synthesis returns
//! the input, one hop late.
use core::f32::consts::PI;
use num_complex::Complex32;

pub const FRAME: usize = 512;
pub const HOP: usize = FRAME / 2;
pub const BINS: usize = FRAME / 2 + 1;

fn window(i: usize) -> f32 {
    libm::sinf(PI * i as f32 / FRAME as f32)
}

/// One channel's analysis.
pub struct Analysis {
    last: [f32; HOP],
}

impl Default for Analysis {
    fn default() -> Self {
        Analysis { last: [0.0; HOP] }
    }
}

impl Analysis {
    /// Takes the next `HOP` samples, gives the spectrum of the frame that ends with them.
    pub fn push(&mut self, hop: &[f32; HOP], out: &mut [Complex32; BINS]) {
        let mut frame = [0.0f32; FRAME];
        for i in 0..HOP {
            frame[i] = self.last[i] * window(i);
            frame[HOP + i] = hop[i] * window(HOP + i);
        }
        self.last = *hop;
        let half = microfft::real::rfft_512(&mut frame);
        out[..HOP].copy_from_slice(half);
        // the transform packs the real value at the highest frequency into bin 0
        out[HOP] = Complex32::new(out[0].im, 0.0);
        out[0].im = 0.0;
    }
}

/// One channel's synthesis.
pub struct Synthesis {
    tail: [f32; HOP],
}

impl Default for Synthesis {
    fn default() -> Self {
        Synthesis { tail: [0.0; HOP] }
    }
}

impl Synthesis {
    /// Takes one spectrum, gives the next `HOP` samples.
    pub fn pull(&mut self, spectrum: &[Complex32; BINS], out: &mut [f32; HOP]) {
        let mut full = [Complex32::new(0.0, 0.0); FRAME];
        full[..BINS].copy_from_slice(spectrum);
        for i in 1..HOP {
            full[FRAME - i] = spectrum[i].conj();
        }
        let time = microfft::inverse::ifft_512(&mut full);
        for i in 0..HOP {
            out[i] = self.tail[i] + time[i].re * window(i);
            self.tail[i] = time[HOP + i].re * window(HOP + i);
        }
    }
}
