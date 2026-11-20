//! The array description file: what is on each channel of a recording.
use earshot_core::array::Array;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layout {
    pub rate: u32,
    pub channels: usize,
    /// Bits per sample, signed little-endian: 16 or 32.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Channels that carry what the device itself plays.
    #[serde(default)]
    pub reference: Vec<usize>,
    pub mic: Vec<Mic>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mic {
    pub channel: usize,
    /// Metres. x points to azimuth 0, y to azimuth 90, z up.
    pub position: [f64; 3],
}

fn default_width() -> u32 {
    32
}

impl Layout {
    pub fn load(path: &Path) -> Result<Layout, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let layout: Layout = toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        layout.check().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(layout)
    }

    fn check(&self) -> Result<(), String> {
        if self.width != 16 && self.width != 32 {
            return Err(format!("width must be 16 or 32, not {}", self.width));
        }
        if self.mic.is_empty() {
            return Err("no microphones".into());
        }
        // the transform has 512-sample frames; its length in time must stay near 32 ms
        if self.rate != 16000 {
            return Err(format!("only 16000 Hz is supported so far, not {}", self.rate));
        }
        for c in self.mic.iter().map(|m| m.channel).chain(self.reference.iter().copied()) {
            if c >= self.channels {
                return Err(format!("channel {c} does not exist: there are {}", self.channels));
            }
        }
        Ok(())
    }

    pub fn array(&self) -> Array {
        Array::new(self.mic.iter().map(|m| m.position).collect())
    }

    pub fn frame_bytes(&self) -> usize {
        self.channels * self.width as usize / 8
    }

    /// The largest reference sample of an interleaved frame, as 0..1; 0 without a reference.
    pub fn reference_peak(&self, frame: &[u8]) -> f32 {
        self.reference
            .iter()
            .map(|&c| self.channel(frame, c).abs())
            .fold(0.0, f32::max)
    }

    /// One microphone's sample from an interleaved frame, as -1..1.
    pub fn sample(&self, frame: &[u8], mic: usize) -> f32 {
        self.channel(frame, self.mic[mic].channel)
    }

    fn channel(&self, frame: &[u8], channel: usize) -> f32 {
        let at = channel * self.width as usize / 8;
        if self.width == 16 {
            i16::from_le_bytes([frame[at], frame[at + 1]]) as f32 / 32768.0
        } else {
            i32::from_le_bytes([frame[at], frame[at + 1], frame[at + 2], frame[at + 3]]) as f32 / 2147483648.0
        }
    }
}
