(function () {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const token = params.get("token") || "";
  let ws = null;
  let currentState = null;
  let evidenceOpen = false;

  const healthBadge = document.getElementById("health-badge");
  const modeBadge = document.getElementById("mode-badge");
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

  function healthClass(health) {
    if (health === "ok") return "badge health-ok";
    if (health === "no_transcript") return "badge health-err";
    return "badge health-warn";
  }

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function renderState(state) {
    currentState = state;
    const health = String(state.health || "ok");
    const mode = String(state.mode || "proactive");
    healthBadge.className = healthClass(health);
    healthBadge.textContent = "health: " + health;
    modeBadge.textContent = "mode: " + mode;
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
    if (card.pinned) {
      pinnedBadge.classList.remove("hidden");
    } else {
      pinnedBadge.classList.add("hidden");
    }
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

  const feedbackMap = {
    1: "too_late",
    2: "already_answered",
    3: "not_relevant",
    4: "badly_phrased",
  };
  for (const [k, reason] of Object.entries(feedbackMap)) {
    document
      .getElementById("btn-fb-" + k)
      .addEventListener("click", () => sendAction("dismiss", { feedback: reason }));
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

  function connectWs() {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = proto + "//" + window.location.host + "/ws?token=" + encodeURIComponent(token);
    ws = new WebSocket(url);
    ws.addEventListener("message", (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg.type === "state") {
          renderState(msg);
        }
      } catch (_err) {
        // Ignore malformed frame
      }
    });
    ws.addEventListener("close", () => {
      setTimeout(connectWs, 1000);
    });
  }

  connectWs();
})();
