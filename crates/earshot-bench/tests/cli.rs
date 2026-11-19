//! End to end: the real binary, real variant commands, a recogniser stand-in over real HTTP.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

const BIN: &str = env!("CARGO_BIN_EXE_earshot-bench");

/// The stand-in "hears" a text chosen by the first sample of the audio it is sent.
fn heard(first: i16) -> &'static str {
    match first {
        1000 => "the quick brown fox",
        2000 => "jumps over the lazy dog",
        3000 => "the quick fox",
        _ => "",
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

struct Stand {
    url: String,
    seen: Arc<Mutex<Vec<i16>>>,
}

fn recogniser() -> Stand {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/x", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut s = stream.unwrap();
            let mut buf = Vec::new();
            let mut tmp = [0u8; 8192];
            let (mut head_end, mut need) = (None, 0usize);
            loop {
                let n = s.read(&mut tmp).unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&tmp[..n]);
                if head_end.is_none() {
                    if let Some(p) = find(&buf, b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..p]).to_lowercase();
                        let len = head.lines().find_map(|l| l.strip_prefix("content-length:")).unwrap().trim().parse::<usize>().unwrap();
                        head_end = Some(p + 4);
                        need = p + 4 + len;
                    }
                }
                if head_end.is_some() && buf.len() >= need {
                    break;
                }
            }
            let body = &buf[head_end.unwrap()..];
            let riff = find(body, b"RIFF").unwrap();
            let data = riff + find(&body[riff..], b"data").unwrap() + 8;
            let first = i16::from_le_bytes([body[data], body[data + 1]]);
            log.lock().unwrap().push(first);
            let reply = format!("{{\"text\": \"{}\"}}", heard(first));
            write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", reply.len(), reply).unwrap();
        }
    });
    Stand { url, seen }
}

/// A raw recording, 2 channels of 32-bit samples at 100 Hz. `per_second[c]` lists channel
/// c's constant sample value for each second.
fn raw_recording(path: &Path, per_second: [&[i32]; 2]) {
    let mut bytes = Vec::new();
    for s in 0..per_second[0].len() {
        for _ in 0..100 {
            for ch in per_second {
                bytes.extend((ch[s] << 16).to_le_bytes());
            }
        }
    }
    std::fs::write(path, bytes).unwrap();
}

/// Channel 0 carries "the quick brown fox" for 2 s then "jumps over the lazy dog" for 2 s;
/// channel 1 carries a worse first half.
fn take() -> tempfile::TempDir {
    let d = tempfile::tempdir().unwrap();
    raw_recording(&d.path().join("rec.raw"), [&[1000, 1000, 2000, 2000], &[3000, 3000, 2000, 2000]]);
    std::fs::write(d.path().join("ref.txt"), "# region: first\nThe quick brown fox\n# region: second\njumps over the lazy dog.\n").unwrap();
    std::fs::write(
        d.path().join("take.toml"),
        "name = \"t\"\nrecording = \"rec.raw\"\nreference = \"ref.txt\"\n\
         [[region]]\nname = \"first\"\nstart = 0\nend = 2\n[[region]]\nname = \"second\"\nstart = 2\nend = 4\n",
    )
    .unwrap();
    d
}

fn pick(channel: usize) -> String {
    format!("{BIN} pick --channels 2 --channel {channel} --rate 100 --no-normalise {{in}} {{out}}")
}

struct Ran {
    code: i32,
    out: String,
    err: String,
    results: Option<serde_json::Value>,
}

fn run(dir: &Path, url: Option<&str>, args: &[String]) -> Ran {
    let json: PathBuf = dir.join("r.json");
    let _ = std::fs::remove_file(&json);
    let mut c = Command::new(BIN);
    c.arg("run").arg(dir.join("take.toml")).arg("--cache").arg(dir.join("cache")).arg("--json").arg(&json).args(args);
    c.env_remove("EARSHOT_STT_URL").env_remove("EARSHOT_CACHE");
    if let Some(u) = url {
        c.arg("--stt-url").arg(u);
    }
    let o = c.output().unwrap();
    let results = std::fs::read_to_string(&json).ok().map(|s| serde_json::from_str::<serde_json::Value>(&s).unwrap()["results"].clone());
    Ran { code: o.status.code().unwrap(), out: String::from_utf8_lossy(&o.stdout).into(), err: String::from_utf8_lossy(&o.stderr).into(), results }
}

fn variant(name: &str, cmd: &str) -> Vec<String> {
    vec!["--variant".into(), format!("{name}={cmd}")]
}

fn section<'a>(r: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    r["sections"].as_array().unwrap().iter().find(|s| s["name"] == name).unwrap()
}

#[test]
fn a_variant_is_run_transcribed_per_region_and_scored() {
    let (d, s) = (take(), recogniser());
    let r = run(d.path(), Some(&s.url), &[variant("mic0", &pick(0)), variant("mic1", &pick(1))].concat());
    assert_eq!(r.code, 0, "{}", r.err);
    let res = r.results.unwrap();
    assert_eq!((res["mic0"]["wer"].as_f64(), res["mic0"]["correct"].as_u64()), (Some(0.0), Some(9)));
    // mic1 heard "the quick fox" for "the quick brown fox": one of nine words missing
    assert_eq!(res["mic1"]["missing"], 1);
    assert!((res["mic1"]["wer"].as_f64().unwrap() - 1.0 / 9.0).abs() < 1e-9);
    assert_eq!(section(&res["mic1"], "first")["missing"], 1);
    assert_eq!(section(&res["mic1"], "second")["wer"].as_f64(), Some(0.0));
    assert_eq!(res["mic0"]["audio_seconds"].as_f64(), Some(4.0));
    assert_eq!(res["mic0"]["texts"]["second"], "jumps over the lazy dog");
}

#[test]
fn the_table_names_each_variant_and_its_error_rate() {
    let (d, s) = (take(), recogniser());
    let r = run(d.path(), Some(&s.url), &variant("mic1", &pick(1)));
    assert!(r.out.contains("9 reference words"), "{}", r.out);
    let line = r.out.lines().find(|l| l.starts_with("mic1")).unwrap();
    assert!(line.contains("11.1"), "{line}");
}

#[test]
fn an_existing_transcript_is_scored_without_a_recogniser() {
    let d = take();
    let live = d.path().join("live.json");
    std::fs::write(&live, r#"{"first": "the quick brown fox", "second": "jumps over"}"#).unwrap();
    let r = run(d.path(), None, &["--transcript".into(), format!("live={}", live.display())]);
    assert_eq!(r.code, 0, "{}", r.err);
    let res = r.results.unwrap();
    assert_eq!(res["live"]["missing"], 3);
    assert!(res["live"].get("cpu_seconds").is_none());
}

#[test]
fn unchanged_audio_is_not_sent_to_the_recogniser_twice() {
    let (d, s) = (take(), recogniser());
    run(d.path(), Some(&s.url), &variant("mic0", &pick(0)));
    assert_eq!(s.seen.lock().unwrap().len(), 2);
    let r = run(d.path(), Some(&s.url), &variant("mic0", &pick(0)));
    assert_eq!(s.seen.lock().unwrap().len(), 2);
    assert_eq!(r.results.unwrap()["mic0"]["wer"].as_f64(), Some(0.0));
}

#[test]
fn the_cpu_time_of_the_variant_command_is_measured() {
    let (d, s) = (take(), recogniser());
    // Spin for 0.4 s of wall time, then do the same work as the fast variant.
    let burn = format!("end=$(( $(date +%s%N) + 400000000 )); while [ $(date +%s%N) -lt $end ]; do :; done; {}", pick(0));
    let r = run(d.path(), Some(&s.url), &[variant("slow", &burn), variant("fast", &pick(0))].concat());
    assert_eq!(r.code, 0, "{}", r.err);
    let res = r.results.unwrap();
    let (slow, fast) = (res["slow"]["cpu_seconds"].as_f64().unwrap(), res["fast"]["cpu_seconds"].as_f64().unwrap());
    assert!(slow - fast >= 0.15, "slow {slow} fast {fast}");
    assert!(fast < 0.1, "fast {fast}");
    assert!(res["slow"]["cpu_percent_of_real_time"].as_f64().unwrap() >= 100.0 * 0.15 / 4.0);
}

#[test]
fn a_failing_variant_stops_the_run_with_its_error() {
    let (d, s) = (take(), recogniser());
    let r = run(d.path(), Some(&s.url), &variant("bad", "echo broken >&2; exit 3 # {out}"));
    assert_eq!(r.code, 2);
    assert!(r.results.is_none());
    assert!(r.err.contains("status 3") && r.err.contains("broken"), "{}", r.err);
}

#[test]
fn a_variant_that_writes_stereo_is_rejected() {
    let (d, s) = (take(), recogniser());
    let stereo = d.path().join("stereo.wav");
    let spec = hound::WavSpec { channels: 2, sample_rate: 100, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut w = hound::WavWriter::create(&stereo, spec).unwrap();
    for _ in 0..200 {
        w.write_sample(0i16).unwrap();
    }
    w.finalize().unwrap();
    let r = run(d.path(), Some(&s.url), &variant("st", &format!("cp {} {{out}}", stereo.display())));
    assert_eq!(r.code, 2);
    assert!(r.err.contains("one channel"), "{}", r.err);
}

#[test]
fn without_regions_the_audio_is_cut_into_fixed_pieces() {
    let (d, s) = (take(), recogniser());
    std::fs::write(d.path().join("take.toml"), "recording = \"rec.raw\"\nreference = \"ref.txt\"\nchunk_seconds = 2\n").unwrap();
    let r = run(d.path(), Some(&s.url), &variant("mic0", &pick(0)));
    let res = r.results.unwrap();
    let names: Vec<&String> = res["mic0"]["texts"].as_object().unwrap().keys().collect();
    assert_eq!(names, ["0s", "2s"]);
    assert_eq!(res["mic0"]["wer"].as_f64(), Some(0.0));
}

#[test]
fn a_missing_recogniser_address_is_an_error() {
    let d = take();
    let r = run(d.path(), None, &variant("mic0", &pick(0)));
    assert_eq!(r.code, 2);
    assert!(r.err.contains("EARSHOT_STT_URL"), "{}", r.err);
}

#[test]
fn an_unreachable_recogniser_is_reported_without_its_address() {
    let d = take();
    let r = run(d.path(), Some("http://127.0.0.1:9/secret-path"), &variant("mic0", &pick(0)));
    assert_eq!(r.code, 2);
    assert!(r.err.contains("could not reach the recogniser"), "{}", r.err);
    assert!(!r.err.contains("secret-path") && !r.err.contains("127.0.0.1"), "{}", r.err);
}

#[test]
fn a_take_without_a_reference_file_is_an_error() {
    let d = take();
    std::fs::remove_file(d.path().join("ref.txt")).unwrap();
    let r = run(d.path(), Some("http://127.0.0.1:9/x"), &variant("mic0", &pick(0)));
    assert_eq!(r.code, 2);
    assert!(r.err.contains("ref.txt"), "{}", r.err);
}
