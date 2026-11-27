(function () {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const token = params.get("token") || "";
  let ws = null;
  let currentState = null;
  let evidenceOpen = false;
  let activeTab = "asr";
  let lastFeed = { segments: [], decisions: [], has_test_audio: false, audio_rx_bytes: 0 };

  const healthBadge = document.getElementById("health-badge");
  const modeBadge = document.getElementById("mode-badge");
  const modeSelect = document.getElementById("mode-select");
  const healthReason = document.getElementById("health-reason");
  const manualBanner = document.getElementById("manual-banner");
  const cardPanel = document.getElementById("card-panel");
  const emptyPanel = document.getElementById("empty-panel");
  const emptyMessage = document.getElementById("empty-message");
  const categoryChip = document.getElementById("category-chip");
  const pinnedBadge = document.getElementById("pinned-badge");
  const originBadge = document.getElementById("origin-badge");
  const cardQuestion = document.getElementById("card-question");
  const evidenceDrawer = document.getElementById("evidence-drawer");
  const evidenceRoute = document.getElementById("evidence-route");
  const evidenceList = document.getElementById("evidence-list");
  const btnEvidence = document.getElementById("btn-evidence");

  const btnMic = document.getElementById("btn-mic");
  const micDeviceSelect = document.getElementById("mic-device-select");
  const btnRespeaker = document.getElementById("btn-respeaker");
  const btnTestAudio = document.getElementById("btn-test-audio");
  const fileAudioInput = document.getElementById("file-audio-input");
  const micMeterFill = document.getElementById("mic-meter-fill");
  const audioStatus = document.getElementById("audio-source-status");
  const liveAudioWrap = document.getElementById("live-audio-wrap");
  const liveAudioEl = document.getElementById("live-audio-el");
  const tabAsr = document.getElementById("tab-asr");
  const tabDecisions = document.getElementById("tab-decisions");
  const inspectorList = document.getElementById("inspector-list");

  function healthClass(health) {
    if (health === "ok") return "badge health-ok";
    if (health === "no_transcript") return "badge health-err";
    return "badge health-warn";
  }

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function fmtMs(ms) {
    const s = Math.max(0, Math.floor((Number(ms) || 0) / 1000));
    const m = Math.floor(s / 60);
    const r = s % 60;
    return String(m).padStart(2, "0") + ":" + String(r).padStart(2, "0");
  }

  function renderState(state) {
    currentState = state;
    const health = String(state.health || "ok");
    const mode = String(state.mode || "Proactive");
    healthBadge.className = healthClass(health);
    healthBadge.textContent = "health: " + health;
    modeBadge.textContent = "mode: " + mode;
    if (modeSelect && modeSelect.value !== mode) modeSelect.value = mode;
    healthReason.textContent = state.health_reason ? String(state.health_reason) : "";

    const ms = state.manual_status;
    if (ms && ms.message) {
      manualBanner.textContent = String(ms.message);
      manualBanner.classList.remove("hidden");
    } else {
      manualBanner.textContent = "";
      manualBanner.classList.add("hidden");
    }

    const card = state.active_card;
    if (!card) {
      cardPanel.classList.add("hidden");
      emptyPanel.classList.remove("hidden");
      emptyMessage.textContent = state.off_record
        ? "Off the record: ingestion halted."
        : health === "no_transcript"
        ? "No transcript stream."
        : "Listening. No card to show.";
      return;
    }

    emptyPanel.classList.add("hidden");
    cardPanel.classList.remove("hidden");
    categoryChip.textContent = String(card.category_chip || card.category || "");
    originBadge.textContent = String(card.origin || "proactive");
    if (card.pinned) pinnedBadge.classList.remove("hidden");
    else pinnedBadge.classList.add("hidden");
    cardQuestion.textContent = String(card.text || "");

    evidenceRoute.textContent = "Route: " + String(card.route || "template");
    clearChildren(evidenceList);
    const evItems = Array.isArray(card.evidence) ? card.evidence : [];
    for (const item of evItems) {
      const li = document.createElement("li");
      const spk = String(item.speaker || "S?");
      const age = String(item.age || "");
      const txt = String(item.text || "");
      li.textContent = "[" + spk + (age ? " | " + age : "") + "] " + txt;
      evidenceList.appendChild(li);
    }
    evidenceDrawer.classList.toggle("hidden", !evidenceOpen);
    btnEvidence.setAttribute("aria-expanded", evidenceOpen ? "true" : "false");
  }

  function renderRespeakerStatus() {
    const rs = lastFeed.respeaker;
    if (!rs || !rs.configured) {
      btnRespeaker.classList.add("hidden");
      return;
    }
    btnRespeaker.classList.remove("hidden");
    const st = String(rs.status || "waiting");
    btnRespeaker.className = "btn-respeaker " + st;
    if (!rs.enabled) {
      btnRespeaker.textContent = "ReSpeaker: Muted";
    } else if (st === "streaming" || st === "paused") {
      btnRespeaker.textContent = "ReSpeaker: " + String(rs.detail || st);
    } else if (st === "connecting") {
      btnRespeaker.textContent = "ReSpeaker: Connecting...";
    } else {
      btnRespeaker.textContent = "ReSpeaker: Waiting for device...";
    }
    if (activeAudioMode === "idle") {
      if (st === "streaming" && rs.enabled) {
        const lvl = typeof rs.level_db === "number" ? rs.level_db : -90;
        const pct = Math.max(0, Math.min(100, Math.round(((lvl + 60) / 50) * 100)));
        micMeterFill.style.width = pct + "%";
        audioStatus.textContent = "ReSpeaker " + String(rs.detail || "Live") + " -> peecee-speech";
      } else if (audioStatus.textContent.startsWith("ReSpeaker ")) {
        micMeterFill.style.width = "0%";
        audioStatus.textContent = "Audio idle";
      }
    }
  }

  function renderInspector() {
    renderRespeakerStatus();
    if (lastFeed.has_test_audio) btnTestAudio.classList.remove("hidden");
    clearChildren(inspectorList);
    if (activeTab === "asr") {
      const segs = Array.isArray(lastFeed.segments) ? lastFeed.segments : [];
      if (!segs.length) {
        const empty = document.createElement("div");
        empty.className = "feed-row";
        empty.textContent = "No live ASR segments yet. Click Start Live Mic or Stream 5-Min Test Audio Live.";
        inspectorList.appendChild(empty);
        return;
      }
      for (let i = segs.length - 1; i >= 0; i--) {
        const seg = segs[i];
        const row = document.createElement("div");
        row.className = "feed-row" + (seg.stability === "interim" ? " feed-interim" : "");
        const meta = document.createElement("div");
        meta.className = "feed-meta";
        const spk = document.createElement("span");
        spk.className = "feed-spk";
        spk.textContent = String(seg.speaker_id || "S?") + (seg.speaker_uncertain ? " [uncertain]" : "");
        const info = document.createElement("span");
        info.textContent = fmtMs(seg.start_ms) + " · " + String(seg.stability || "");
        meta.append(spk, info);
        const body = document.createElement("div");
        body.textContent = String(seg.text || "");
        row.append(meta, body);
        inspectorList.appendChild(row);
      }
    } else {
      const decs = Array.isArray(lastFeed.decisions) ? lastFeed.decisions : [];
      if (!decs.length) {
        const empty = document.createElement("div");
        empty.className = "feed-row";
        empty.textContent = "No coach routing decisions yet.";
        inspectorList.appendChild(empty);
        return;
      }
      for (const d of decs) {
        const row = document.createElement("div");
        row.className = "feed-row";
        const meta = document.createElement("div");
        meta.className = "feed-meta";
        const pill = document.createElement("span");
        pill.className = "feed-spk";
        pill.textContent = String(d.outcome || d.terminal_state || "DROPPED") + (d.category ? " · " + d.category : "");
        const info = document.createElement("span");
        const opp = typeof d.opportunity === "number" ? " · p=" + d.opportunity.toFixed(2) : "";
        info.textContent = String(d.terminal_reason || d.terminal_state || "") + opp;
        meta.append(pill, info);
        const body = document.createElement("div");
        body.textContent = String(d.anchor_text || "");
        row.append(meta, body);
        inspectorList.appendChild(row);
      }
    }
  }

  // --- Live 16 kHz PCM16 streaming (Mic & MediaElement) ---
  let audioCtx = null;
  let micStream = null;
  let micSourceNode = null;
  let mediaSourceNode = null;
  let procNode = null;
  let pcmAccumulator = [];
  let activeAudioMode = "idle"; // "idle" | "mic" | "media"
  const FRAME_SAMPLES_16K = 2560; // 160 ms @ 16 kHz

  function ensureAudioCtx() {
    if (!audioCtx) {
      const Ctx = window.AudioContext || window.webkitAudioContext;
      audioCtx = new Ctx();
    }
    if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
    return audioCtx;
  }

  function pushFloatTo16kPcm(inputFloat, inRate) {
    let peak = 0;
    for (let i = 0; i < inputFloat.length; i++) {
      const av = Math.abs(inputFloat[i]);
      if (av > peak) peak = av;
    }
    micMeterFill.style.width = Math.min(100, Math.round(peak * 180)) + "%";

    const ratio = inRate / 16000;
    const outLen = Math.max(1, Math.floor(inputFloat.length / ratio));
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const idx = Math.floor(pos);
      const frac = pos - idx;
      const s0 = inputFloat[idx] || 0;
      const s1 = inputFloat[Math.min(idx + 1, inputFloat.length - 1)] || 0;
      const sample = s0 + (s1 - s0) * frac;
      const clamped = Math.max(-1, Math.min(1, sample));
      pcmAccumulator.push(clamped < 0 ? Math.round(clamped * 32768) : Math.round(clamped * 32767));
    }
    while (pcmAccumulator.length >= FRAME_SAMPLES_16K) {
      const frame = pcmAccumulator.splice(0, FRAME_SAMPLES_16K);
      const buf = new Int16Array(frame);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(buf.buffer);
      }
    }
  }

  function stopMicStream() {
    if (procNode) {
      procNode.onaudioprocess = null;
      try { procNode.disconnect(); } catch (_e) {}
      procNode = null;
    }
    if (micSourceNode) {
      try { micSourceNode.disconnect(); } catch (_e) {}
      micSourceNode = null;
    }
    if (micStream) {
      for (const tr of micStream.getTracks()) tr.stop();
      micStream = null;
    }
    if (pcmAccumulator.length && ws && ws.readyState === WebSocket.OPEN) {
      const buf = new Int16Array(pcmAccumulator.splice(0));
      ws.send(buf.buffer);
    }
    if (activeAudioMode === "mic") {
      sendAction("commit_audio");
      activeAudioMode = "idle";
      btnMic.textContent = "Start Live Mic";
      btnMic.className = "btn-primary";
      audioStatus.textContent = "Audio idle";
      micMeterFill.style.width = "0%";
    }
  }

  async function refreshMicDevices() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const devs = await navigator.mediaDevices.enumerateDevices();
      const inputs = devs.filter((d) => d.kind === "audioinput");
      if (!inputs.length) return;
      const prev = micDeviceSelect.value;
      clearChildren(micDeviceSelect);
      const defOpt = document.createElement("option");
      defOpt.value = "";
      defOpt.textContent = "Default Mic";
      micDeviceSelect.appendChild(defOpt);
      let respeakerId = "";
      for (const dev of inputs) {
        const opt = document.createElement("option");
        opt.value = dev.deviceId || "";
        opt.textContent = dev.label || ("Mic " + micDeviceSelect.length);
        micDeviceSelect.appendChild(opt);
        if (!respeakerId && /respeaker/i.test(opt.textContent) && opt.value) respeakerId = opt.value;
      }
      if (prev && Array.from(micDeviceSelect.options).some((o) => o.value === prev)) {
        micDeviceSelect.value = prev;
      } else if (respeakerId) {
        micDeviceSelect.value = respeakerId;
      }
      micDeviceSelect.classList.remove("hidden");
    } catch (_e) {}
  }

  async function startMicStream() {
    if (activeAudioMode === "mic") { stopMicStream(); return; }
    if (!liveAudioEl.paused) liveAudioEl.pause();
    const ctx = ensureAudioCtx();
    const chosenId = micDeviceSelect ? micDeviceSelect.value : "";
    const audioCfg = chosenId
      ? { deviceId: { exact: chosenId }, echoCancellation: true, noiseSuppression: true, channelCount: 1 }
      : { echoCancellation: true, noiseSuppression: true, channelCount: 1 };
    try {
      micStream = await navigator.mediaDevices.getUserMedia({ audio: audioCfg });
      refreshMicDevices();
    } catch (err) {
      audioStatus.textContent = "Mic denied: " + String(err.message || err);
      return;
    }
    pcmAccumulator = [];
    micSourceNode = ctx.createMediaStreamSource(micStream);
    procNode = ctx.createScriptProcessor(4096, 1, 1);
    procNode.onaudioprocess = (ev) => {
      if (activeAudioMode === "mic") pushFloatTo16kPcm(ev.inputBuffer.getChannelData(0), ctx.sampleRate);
    };
    micSourceNode.connect(procNode);
    procNode.connect(ctx.destination);
    activeAudioMode = "mic";
    btnMic.textContent = "Stop Mic";
    btnMic.className = "btn-active";
    audioStatus.textContent = "Streaming live mic -> peecee-speech";
  }

  function ensureMediaPipeline() {
    const ctx = ensureAudioCtx();
    if (!mediaSourceNode) {
      mediaSourceNode = ctx.createMediaElementSource(liveAudioEl);
      mediaSourceNode.connect(ctx.destination);
    }
    if (!procNode) {
      procNode = ctx.createScriptProcessor(4096, 1, 1);
      procNode.onaudioprocess = (ev) => {
        if (activeAudioMode === "media" && !liveAudioEl.paused) {
          pushFloatTo16kPcm(ev.inputBuffer.getChannelData(0), ctx.sampleRate);
        }
      };
      mediaSourceNode.connect(procNode);
      procNode.connect(ctx.destination);
    }
  }

  function loadMediaUrl(url, label) {
    stopMicStream();
    liveAudioWrap.classList.remove("hidden");
    liveAudioEl.src = url;
    audioStatus.textContent = "Loaded: " + label;
    ensureMediaPipeline();
    liveAudioEl.play().catch(() => {});
  }

  liveAudioEl.addEventListener("play", () => {
    stopMicStream();
    ensureMediaPipeline();
    activeAudioMode = "media";
    audioStatus.textContent = "Streaming audio -> peecee-speech";
  });
  liveAudioEl.addEventListener("pause", () => {
    if (activeAudioMode === "media") {
      if (pcmAccumulator.length && ws && ws.readyState === WebSocket.OPEN) {
        ws.send(new Int16Array(pcmAccumulator.splice(0)).buffer);
      }
      sendAction("commit_audio");
      activeAudioMode = "idle";
      audioStatus.textContent = "Audio paused";
      micMeterFill.style.width = "0%";
    }
  });
  liveAudioEl.addEventListener("ended", () => {
    if (activeAudioMode === "media") sendAction("commit_audio");
    activeAudioMode = "idle";
    audioStatus.textContent = "Audio finished";
    micMeterFill.style.width = "0%";
  });

  function sendAction(action, extra) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const payload = Object.assign({ type: "action", action: action }, extra || {});
    if (
      (action === "use" || action === "dismiss" || action === "pin") &&
      currentState &&
      currentState.active_card
    ) {
      payload.suggestion_id = currentState.active_card.suggestion_id;
    }
    ws.send(JSON.stringify(payload));
  }

  function toggleEvidence() {
    evidenceOpen = !evidenceOpen;
    evidenceDrawer.classList.toggle("hidden", !evidenceOpen);
    btnEvidence.setAttribute("aria-expanded", evidenceOpen ? "true" : "false");
  }

  document.getElementById("btn-use").addEventListener("click", () => sendAction("use"));
  document.getElementById("btn-pin").addEventListener("click", () => sendAction("pin"));
  document.getElementById("btn-dismiss").addEventListener("click", () => sendAction("dismiss"));
  btnEvidence.addEventListener("click", toggleEvidence);
  document.getElementById("btn-manual").addEventListener("click", () => sendAction("manual"));
  document.getElementById("btn-off-record").addEventListener("click", () => sendAction("off_record"));
  modeSelect.addEventListener("change", () => sendAction("mode", { mode: modeSelect.value }));

  btnMic.addEventListener("click", () => { startMicStream(); });
  btnRespeaker.addEventListener("click", () => {
    const cur = !lastFeed.respeaker || lastFeed.respeaker.enabled !== false;
    sendAction("toggle_respeaker", { enabled: !cur });
  });
  if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
    navigator.mediaDevices.addEventListener("devicechange", () => { refreshMicDevices(); });
  }
  refreshMicDevices();
  btnTestAudio.addEventListener("click", () => {
    loadMediaUrl("./test-audio?token=" + encodeURIComponent(token), "5-Min Test Conversation");
  });
  fileAudioInput.addEventListener("change", (ev) => {
    const f = ev.target && ev.target.files && ev.target.files[0];
    if (f) loadMediaUrl(URL.createObjectURL(f), f.name);
  });
  for (const cueBtn of document.querySelectorAll(".seek-cue-btn")) {
    cueBtn.addEventListener("click", () => {
      liveAudioEl.currentTime = Number(cueBtn.getAttribute("data-seek") || 0);
      liveAudioEl.play().catch(() => {});
    });
  }
  tabAsr.addEventListener("click", () => {
    activeTab = "asr";
    tabAsr.classList.add("active");
    tabDecisions.classList.remove("active");
    renderInspector();
  });
  tabDecisions.addEventListener("click", () => {
    activeTab = "decisions";
    tabDecisions.classList.add("active");
    tabAsr.classList.remove("active");
    renderInspector();
  });

  const feedbackMap = { 1: "too_late", 2: "already_answered", 3: "not_relevant", 4: "badly_phrased" };
  for (const [k, reason] of Object.entries(feedbackMap)) {
    document.getElementById("btn-fb-" + k).addEventListener("click", () => sendAction("dismiss", { feedback: reason }));
  }

  document.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const tag = ev.target && ev.target.tagName ? ev.target.tagName.toLowerCase() : "";
    if (tag === "input" || tag === "textarea" || tag === "select") return;
    if (ev.key === "u") sendAction("use");
    else if (ev.key === "d") sendAction("dismiss");
    else if (ev.key === "p") sendAction("pin");
    else if (ev.key === "m") sendAction("manual");
    else if (ev.key === "o") sendAction("off_record");
    else if (ev.key === "e") toggleEvidence();
    else if (feedbackMap[ev.key]) sendAction("dismiss", { feedback: feedbackMap[ev.key] });
  });

  async function pollFeed() {
    try {
      const resp = await fetch("./feed?token=" + encodeURIComponent(token));
      if (resp.ok) {
        lastFeed = await resp.json();
        renderInspector();
      }
    } catch (_err) {}
  }

  function connectWs() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = proto + "//" + window.location.host + "/ws?token=" + encodeURIComponent(token);
    ws = new WebSocket(url);
    ws.binaryType = "arraybuffer";
    ws.addEventListener("message", (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg.type === "state") {
          renderState(msg);
          pollFeed();
        }
      } catch (_err) {}
    });
    ws.addEventListener("close", () => { setTimeout(connectWs, 1000); });
  }

  connectWs();
  pollFeed();
  setInterval(pollFeed, 600);
})();
