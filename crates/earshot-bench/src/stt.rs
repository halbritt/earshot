//! Send audio to a speech recogniser and return text.
//!
//! The recogniser is any server with an OpenAI-style transcription endpoint that accepts a
//! multipart upload with a `file` field and answers JSON with a `text` field (whisper.cpp's
//! server and others). Results are cached by the audio's hash, because the same chunk is
//! sent again every time an unchanged variant is scored.

use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Duration;

pub struct Recogniser {
    pub url: String,
    pub language: String,
    pub cache: Option<PathBuf>,
}

impl Recogniser {
    pub fn transcribe(&self, wav: &[u8]) -> Result<String, String> {
        let mut h = Sha256::new();
        h.update(self.language.as_bytes());
        h.update([0]);
        h.update(wav);
        let key: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if let Some(dir) = &self.cache {
            if let Ok(text) = std::fs::read_to_string(dir.join(&key)) {
                return Ok(text);
            }
        }
        let boundary = format!("earshot{}", &key[..24]);
        let mut body = Vec::with_capacity(wav.len() + 600);
        for (name, value) in [("response_format", "json"), ("temperature", "0.0"), ("language", self.language.as_str())] {
            body.extend(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").bytes());
        }
        body.extend(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n")
                .bytes(),
        );
        body.extend(wav);
        body.extend(format!("\r\n--{boundary}--\r\n").bytes());
        // The address is not put in error messages: it can be private.
        let reply = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(300))
            .build()
            .post(&self.url)
            .set("Content-Type", &format!("multipart/form-data; boundary={boundary}"))
            .send_bytes(&body)
            .map_err(|e| match e {
                ureq::Error::Status(code, _) => format!("the recogniser answered with status {code}"),
                ureq::Error::Transport(t) => format!("could not reach the recogniser: {}", t.kind()),
            })?;
        let json: serde_json::Value =
            serde_json::from_reader(reply.into_reader()).map_err(|e| format!("the recogniser's answer is not JSON: {e}"))?;
        let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("").to_string();
        if let Some(dir) = &self.cache {
            std::fs::create_dir_all(dir).and_then(|_| std::fs::write(dir.join(&key), &text)).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        Ok(text)
    }
}
