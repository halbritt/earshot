//! Take one channel from a raw interleaved recording: the simplest variant, one microphone
//! and nothing else.

/// `raw` holds little-endian signed samples of `width` bytes (2 or 4), `channels` per frame.
/// Returns 16-bit samples of `channel`. With `normalise`, the level is raised so that the
/// peak is at 0.9 of full scale; without it the top 16 bits are kept.
pub fn pick(raw: &[u8], channels: usize, width: usize, channel: usize, normalise: bool) -> Vec<i16> {
    assert!(width == 2 || width == 4, "width must be 2 or 4 bytes");
    assert!(channel < channels, "channel must be less than channels");
    let frame = channels * width;
    let one: Vec<i64> = raw
        .chunks_exact(frame)
        .map(|f| {
            let s = &f[channel * width..(channel + 1) * width];
            if width == 2 { i16::from_le_bytes([s[0], s[1]]) as i64 } else { i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as i64 }
        })
        .collect();
    let shift = 8 * (width - 2);
    let peak = one.iter().map(|v| v.abs()).max().unwrap_or(0).max(1);
    let full = 1i64 << (8 * width - 1);
    let gain = if normalise { 0.9 * full as f64 / peak as f64 } else { 1.0 };
    one.iter().map(|&v| (((v as f64 * gain) as i64) >> shift).clamp(-32768, 32767) as i16).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw32(v: &[i32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn takes_the_chosen_channel_and_normalises_to_nine_tenths() {
        let out = pick(&raw32(&[1 << 28, 5, -(1 << 29), 6]), 2, 4, 0, true);
        assert!((out[1] as f64 + 0.9 * 32768.0).abs() <= 2.0, "{out:?}");
        assert!((out[0] as f64 - 0.45 * 32768.0).abs() <= 2.0, "{out:?}");
    }

    #[test]
    fn without_normalising_keeps_the_top_sixteen_bits() {
        assert_eq!(pick(&raw32(&[1000 << 16, 0, -2000 << 16, 0]), 2, 4, 0, false), vec![1000, -2000]);
    }

    #[test]
    fn sixteen_bit_input_passes_through() {
        let raw: Vec<u8> = [7i16, 100, -8, 200].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(pick(&raw, 2, 2, 1, false), vec![100, 200]);
    }

    #[test]
    fn a_trailing_partial_frame_is_ignored() {
        let mut raw = raw32(&[1 << 16, 2 << 16]);
        raw.extend([1, 2, 3]);
        assert_eq!(pick(&raw, 1, 4, 0, false), vec![1, 2]);
    }
}
