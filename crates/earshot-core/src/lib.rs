//! Microphone-array processing core.
//!
//! No operating-system calls and no allocation after set-up, so the same code can run on a
//! small chip. Designing beams (set-up) uses double precision; the audio path is single
//! precision.
#![no_std]
extern crate alloc;

pub mod array;
pub mod beam;
pub mod direction;
pub mod stft;

pub use num_complex::{Complex32, Complex64};
