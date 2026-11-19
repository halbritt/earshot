//! Beam design checked against the simulation in sim/arrays.py (NumPy), whose results are
//! in sim/arrays.json, and against what a beam is required to do.
use earshot_core::array::Array;
use earshot_core::beam::{directivity_db, response, weights, white_noise_gain_db, Kind};

const SPOT: [f64; 6] = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 7000.0];
const ELEVATION: f64 = 20.0;

fn simulated(array: &str, kind: &str) -> Vec<f64> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../sim/arrays.json");
    let all: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    all[array][kind]["spot"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

// mean directivity over eight look directions, as the simulation reports it
fn spot(array: &Array, kind: Kind) -> Vec<f64> {
    SPOT.iter()
        .map(|&hz| {
            (0..8)
                .map(|i| directivity_db(array, hz, &weights(array, hz, 45.0 * i as f64, ELEVATION, kind)))
                .sum::<f64>()
                / 8.0
        })
        .collect()
}

fn assert_close(got: &[f64], want: &[f64]) {
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 0.051, "directivity {got:?}, the simulation says {want:?}");
    }
}

const SD: Kind = Kind::Superdirective {
    min_white_noise_gain_db: 0.0,
};

#[test]
fn respeaker_ring_matches_the_simulation() {
    let ring = Array::ring(6, 0.0463, 90.0);
    assert_close(
        &spot(&ring, Kind::DelaySum),
        &simulated("ReSpeaker Core v2: 6 on 93 mm", "ds"),
    );
    assert_close(&spot(&ring, SD), &simulated("ReSpeaker Core v2: 6 on 93 mm", "sd0"));
}

#[test]
fn sixteen_microphone_ring_matches_the_simulation() {
    let ring = Array::ring(16, 0.0941, 90.0);
    assert_close(&spot(&ring, Kind::DelaySum), &simulated("Whistler: 16 on 188 mm", "ds"));
    assert_close(&spot(&ring, SD), &simulated("Whistler: 16 on 188 mm", "sd0"));
}

#[test]
fn two_rings_match_the_simulation() {
    let mut mics = Array::ring(4, 0.020, 0.0).mics;
    mics.extend(Array::ring(8, 0.060, 0.0).mics);
    assert_close(
        &spot(&Array::new(mics), SD),
        &simulated("4 on 40 mm + 8 on 120 mm (12)", "sd0"),
    );
}

#[test]
fn every_beam_passes_its_look_direction_unchanged() {
    let ring = Array::ring(6, 0.0463, 90.0);
    for kind in [Kind::DelaySum, SD] {
        for hz in [300.0, 1000.0, 3400.0, 7000.0] {
            let w = weights(&ring, hz, 100.0, ELEVATION, kind);
            assert!((response(&ring, hz, &w, 100.0, ELEVATION) - 1.0).abs() < 1e-9);
        }
    }
}

#[test]
fn superdirective_beam_keeps_the_white_noise_gain_it_was_asked_for() {
    let ring = Array::ring(6, 0.0463, 90.0);
    for limit in [0.0, -10.0] {
        for hz in [200.0, 500.0, 1000.0, 4000.0] {
            let w = weights(
                &ring,
                hz,
                30.0,
                ELEVATION,
                Kind::Superdirective {
                    min_white_noise_gain_db: limit,
                },
            );
            assert!(white_noise_gain_db(&w) >= limit, "{hz} Hz: {} dB", white_noise_gain_db(&w));
        }
    }
}

#[test]
fn superdirective_beam_rejects_a_talker_at_the_side_better_than_delay_and_sum() {
    let ring = Array::ring(6, 0.0463, 90.0);
    let side = |kind| 10.0 * response(&ring, 1000.0, &weights(&ring, 1000.0, 0.0, ELEVATION, kind), 90.0, ELEVATION).log10();
    assert!(
        side(SD) < side(Kind::DelaySum) - 3.0,
        "{} against {}",
        side(SD),
        side(Kind::DelaySum)
    );
}
