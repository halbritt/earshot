//! `earshot beam` end to end: a recording in, a WAV out.
use std::f64::consts::PI;
use std::path::Path;
use std::process::Command;

const ARRAY: &str = r#"
rate = 16000
channels = 4
width = 16
reference = [3]
[[mic]]
channel = 2
position = [0.05, 0.0, 0.0]
[[mic]]
channel = 0
position = [-0.05, 0.0, 0.0]
"#;

// Two microphones 10 cm apart on the x axis, on channels 2 and 0 of four. A tone arrives
// from `azimuth` in the plane of the array; channel 1 is loud noise that belongs to no
// microphone.
fn recording(seconds: f64, azimuth: f64) -> Vec<u8> {
    tone_recording(seconds, azimuth, 1000.0)
}

fn tone_recording(seconds: f64, azimuth: f64, hz: f64) -> Vec<u8> {
    let mut raw = Vec::new();
    let mut s = 1u32;
    for i in 0..(seconds * 16000.0) as usize {
        let lead = 0.05 * azimuth.to_radians().cos() / 343.0;
        let tone = |lead: f64| (8000.0 * (2.0 * PI * hz * (i as f64 / 16000.0 + lead)).sin()) as i16;
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        for v in [tone(-lead), (s >> 17) as i16, tone(lead), 0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
    }
    raw
}

fn run(dir: &Path, raw: &[u8], args: &[&str]) -> (bool, Vec<i16>, String) {
    std::fs::write(dir.join("array.toml"), ARRAY).unwrap();
    std::fs::write(dir.join("in.raw"), raw).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_earshot"))
        .current_dir(dir)
        .args(["beam", "--array", "array.toml", "--elevation", "0"])
        .args(args)
        .args(["in.raw", "out.wav"])
        .output()
        .unwrap();
    let samples = match hound::WavReader::open(dir.join("out.wav")) {
        Ok(mut r) => {
            assert_eq!(
                (r.spec().channels, r.spec().sample_rate, r.spec().bits_per_sample),
                (1, 16000, 16)
            );
            r.samples::<i16>().map(|s| s.unwrap()).collect()
        }
        Err(_) => Vec::new(),
    };
    (
        out.status.success(),
        samples,
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn rms(x: &[i16]) -> f64 {
    (x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / x.len() as f64).sqrt()
}

#[test]
fn a_tone_from_the_look_direction_keeps_its_level_and_the_output_is_as_long_as_the_input() {
    let dir = tempfile::tempdir().unwrap();
    let (ok, y, err) = run(dir.path(), &recording(1.0, 0.0), &["--steer", "0", "--kind", "delay-sum"]);
    assert!(ok, "{err}");
    assert_eq!(y.len(), 16000);
    let level = rms(&y[2000..14000]);
    assert!((level - 8000.0 / 2f64.sqrt()).abs() < 60.0, "{level}");
}

#[test]
fn a_tone_from_the_side_of_the_beam_is_cancelled() {
    // along the axis, 1715 Hz is half a wavelength between the two microphones
    // (0.1 m / 343 m/s * 1715 Hz = 0.5 cycles): they sum to nothing in a beam that looks
    // at 90 degrees, which adds them as they are
    let dir = tempfile::tempdir().unwrap();
    let (ok, y, err) = run(
        dir.path(),
        &tone_recording(1.0, 0.0, 1715.0),
        &["--steer", "90", "--kind", "delay-sum"],
    );
    assert!(ok, "{err}");
    assert!(rms(&y[2000..14000]) < 8000.0 * 0.05, "{}", rms(&y[2000..14000]));
}

#[test]
fn the_beam_turns_at_the_time_it_is_told_to() {
    let dir = tempfile::tempdir().unwrap();
    let (ok, y, err) = run(
        dir.path(),
        &tone_recording(2.0, 0.0, 1715.0),
        &["--steer", "0:90,1:0", "--kind", "delay-sum"],
    );
    assert!(ok, "{err}");
    assert!(
        rms(&y[2000..14000]) < 400.0,
        "cancelled while the beam looks at 90: {}",
        rms(&y[2000..14000])
    );
    assert!(
        rms(&y[18000..30000]) > 5000.0,
        "heard once the beam looks at the tone: {}",
        rms(&y[18000..30000])
    );
}

#[test]
fn gain_is_applied_in_decibels() {
    let dir = tempfile::tempdir().unwrap();
    let (_, y, _) = run(
        dir.path(),
        &recording(1.0, 0.0),
        &["--steer", "0", "--kind", "delay-sum", "--gain", "-6.02"],
    );
    let level = rms(&y[2000..14000]);
    assert!((level - 4000.0 / 2f64.sqrt()).abs() < 40.0, "{level}");
}

#[test]
fn a_channel_the_array_does_not_have_is_an_error_that_names_it() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("bad.toml"), ARRAY.replace("channel = 2", "channel = 7")).unwrap();
    std::fs::write(dir.path().join("in.raw"), recording(0.1, 0.0)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_earshot"))
        .current_dir(dir.path())
        .args(["beam", "--array", "bad.toml", "in.raw", "out.wav"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("channel 7 does not exist"));
}

#[test]
fn the_respeaker_description_in_the_repository_loads() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("in.raw"), vec![0u8; 8 * 4 * 1600]).unwrap();
    let array = concat!(env!("CARGO_MANIFEST_DIR"), "/../../arrays/respeaker-core-v2.toml");
    let out = Command::new(env!("CARGO_BIN_EXE_earshot"))
        .current_dir(dir.path())
        .args(["beam", "--array", array, "--steer", "100", "in.raw", "out.wav"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(hound::WavReader::open(dir.path().join("out.wav")).unwrap().len(), 1600);
}

#[test]
fn direction_reports_where_a_tone_comes_from_twice_a_second() {
    // two microphones on the x axis cannot tell front from back, but they can tell the axis
    // from the side: a tone along the axis is found at 0 or 180 degrees
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("array.toml"), ARRAY).unwrap();
    std::fs::write(dir.path().join("in.raw"), tone_recording(2.0, 90.0, 1000.0)).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_earshot"))
        .current_dir(dir.path())
        .args(["direction", "--array", "array.toml", "--elevation", "0", "in.raw"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4, "{text}");
    for line in &lines[1..] {
        let az: f64 = line
            .split("\"azimuth\": ")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        assert!((az - 90.0).abs() <= 3.0 || (az - 270.0).abs() <= 3.0, "{line}");
    }
    assert!(lines[0].starts_with("{\"t\": 0.50,"), "{}", lines[0]);
}
