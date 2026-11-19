"""Send audio to a speech recogniser and return text.

The recogniser is any server with an OpenAI-style transcription endpoint that accepts a
multipart upload with a `file` field and answers JSON with a `text` field (whisper.cpp
server, faster-whisper-server and others). Results are cached by the audio's hash, because
the same chunk is sent again every time an unchanged variant is scored.
"""
import hashlib
import json
import pathlib
import urllib.request
import uuid


class Recogniser:
    def __init__(self, url, cache_dir=None, language="en", timeout=300):
        self.url, self.language, self.timeout = url, language, timeout
        self.cache = pathlib.Path(cache_dir) if cache_dir else None
        self.requests = 0

    def transcribe(self, wav_bytes):
        key = hashlib.sha256(self.language.encode() + b"\0" + wav_bytes).hexdigest()
        if self.cache and (self.cache / key).exists():
            return (self.cache / key).read_text()
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in (("response_format", "json"), ("temperature", "0.0"), ("language", self.language)):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n".encode()
        )
        parts += [wav_bytes, f"\r\n--{boundary}--\r\n".encode()]
        request = urllib.request.Request(
            self.url, b"".join(parts), {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            text = json.loads(response.read()).get("text", "")
        self.requests += 1
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
            (self.cache / key).write_text(text)
        return text
