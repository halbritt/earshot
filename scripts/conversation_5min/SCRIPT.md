# 5-Minute Conversation Script: Checkout Edge-Cache Cutover Readiness Review

- **Session ID:** `conv-5min-cutover-001`
- **Mode:** `technical_working_discussion`
- **Objective:** Finalize owners, cutover dates, canary success criteria, scope boundaries, and benchmark evidence for the checkout edge-cache rollout.
- **Turns / Words:** 42 turns, 604 spoken words (~5 min with natural pauses)
- **TTS Stack:** `~/git/announce` (`gemini-3.8-flash-tts`, render-only via `announce --out`)

## Characters

| Speaker ID | Name | Role | Gemini Voice | Azimuth (deg) | Style Direction |
|---|---|---|---|---:|---|
| `ME` | Alex | `engineering_lead` | `Charon` | 0 | Calm, thoughtful male engineering lead in a live technical meeting. Natural conversational pace, clear diction. |
| `S1` | Maya | `platform_engineer` | `Kore` | 95 | Direct, articulate female platform engineer in a technical working discussion. Brisk, clear conversational pace. |
| `S2` | Sam | `product_manager` | `Puck` | 200 | Upbeat, engaged male product manager in a working meeting. Natural, slightly energetic conversational delivery. |
| `S3` | Priya | `sre_lead` | `Sulafat` | 290 | Warm, measured female site reliability engineer in a technical meeting. Composed, clear conversational pace. |

## Dialogue

### Act 1

- **Turn 01 — ME (Alex, `Charon`)** [recipient_speech_context, pause 900 ms]: Thanks for joining. Let us lock down the remaining pieces for the checkout edge-cache rollout.
- **Turn 02 — S1 (Maya, `Kore`)** [below_t_low_status_update, pause 800 ms]: Staging dual-write handled overnight burst traffic cleanly with zero dropped cache invalidations.
- **Turn 03 — S2 (Sam, `Puck`)** [prefilter_backchannel, pause 600 ms]: Yeah, totally.
- **Turn 04 — S3 (Priya, `Sulafat`)** [below_t_low_status_update, pause 800 ms]: Cluster memory stayed under seventy percent on our Grafana dashboards throughout the soak test.
- **Turn 05 — S3 (Priya, `Sulafat`)** [prefilter_too_short, pause 600 ms]: Makes sense.
- **Turn 06 — ME (Alex, `Charon`)** [recipient_question, pause 900 ms]: Good to hear staging is steady. What operational item is still unresolved?
- **Turn 07 — S1 (Maya, `Kore`)** [below_t_low_context, pause 700 ms]: Most services are ready, though we still have a gap around payment notification retries.
- **Turn 08 — S1 (Maya, `Kore`)** [display_CONFIRM_OWNER, pause 2300 ms pinned]: Before we flip production traffic, we still need someone to own the webhook replay backfill runbook for the legacy orders table.
- **Turn 09 — S2 (Sam, `Puck`)** [cooldown_active, pause 900 ms]: Finance will definitely want an audit trail for every replayed order state during warmup.
- **Turn 10 — S1 (Maya, `Kore`)** [cooldown_active, pause 900 ms]: The ops script is merged, it just needs dry-run checks with anonymized payloads.
- **Turn 11 — S3 (Priya, `Sulafat`)** [resolves_confirm_owner, pause 900 ms]: Priya will own the webhook replay backfill runbook and test it in staging today.
- **Turn 12 — ME (Alex, `Charon`)** [recipient_transition, pause 900 ms]: Thanks Priya. Let us check customer-facing readiness next.
### Act 2

- **Turn 13 — S2 (Sam, `Puck`)** [self_resolving_within_settle_ms, pause 250 ms pinned]: Someone should also update the merchant status page maintenance banner prior to launch.
- **Turn 14 — S2 (Sam, `Puck`)** [ownership_resolved_before_settle, pause 900 ms]: Actually, Sam will handle updating the merchant status page banner right after this call.
- **Turn 15 — S3 (Priya, `Sulafat`)** [below_t_low_context, pause 900 ms]: Support asked for twenty-four hours of notice on that banner so tickets stay predictable.
- **Turn 16 — S1 (Maya, `Kore`)** [below_t_low_context, pause 900 ms]: Sandbox partners also pull checkout schemas from the older staging endpoint for webhook verification.
- **Turn 17 — ME (Alex, `Charon`)** [recipient_question, pause 900 ms]: We cannot break sandbox integrations during key rotation. Where do those partner updates stand?
- **Turn 18 — S2 (Sam, `Puck`)** [display_CONFIRM_TIME, pause 2300 ms pinned]: For the partner sandbox migration, we need to ship the token rotation patch before launch, and we still have no target date or deadline on the calendar.
- **Turn 19 — S1 (Maya, `Kore`)** [cooldown_active, pause 900 ms]: DevRel drafted the note, though engineering has not frozen the deprecation window for older HMAC headers.
- **Turn 20 — ME (Alex, `Charon`)** [recipient_transition, pause 900 ms]: Let us freeze that sandbox date with partner engineering right after we check reliability drills.
### Act 3

- **Turn 21 — S3 (Priya, `Sulafat`)** [below_t_low_context, pause 900 ms]: We simulated a cold cache restart yesterday and recovery finished in forty seconds.
- **Turn 22 — S3 (Priya, `Sulafat`)** [equivalent_question_asked_trigger, pause 350 ms pinned]: We need someone to run the failover chaos drill on the secondary Redis cluster.
- **Turn 23 — ME (Alex, `Charon`)** [equivalent_question_asked_by_recipient, pause 800 ms]: Who will own running that failover chaos drill on Redis?
- **Turn 24 — S3 (Priya, `Sulafat`)** [ownership_resolved, pause 900 ms]: Priya can run the failover chaos drill tomorrow morning with Maya shadowing on call.
- **Turn 25 — S2 (Sam, `Puck`)** [prefilter_backchannel, pause 600 ms]: Awesome, thank you.
- **Turn 26 — S2 (Sam, `Puck`)** [suppressed_do_not_suggest, pause 2200 ms pinned]: Separately, we still need someone to own renegotiating the vendor contract pricing before next quarter.
- **Turn 27 — ME (Alex, `Charon`)** [recipient_redirect, pause 900 ms]: Let us keep commercial contract discussions separate and track that offline with procurement.
### Act 4

- **Turn 28 — S1 (Maya, `Kore`)** [below_t_low_context, pause 900 ms]: Our edge nodes are configured for regional sticky routing with automatic fallback to origin.
- **Turn 29 — ME (Alex, `Charon`)** [recipient_question, pause 900 ms]: What initial rollout stage are we recommending for the midday traffic window?
- **Turn 30 — S3 (Priya, `Sulafat`)** [display_SUCCESS_CRITERION, pause 2300 ms pinned]: I propose we adopt the five percent canary rollout for checkout traffic first, though we have not defined the success criteria or pass condition to promote it.
- **Turn 31 — S1 (Maya, `Kore`)** [cooldown_active, pause 900 ms]: We should watch both checkout error rate and stale-hit ratio before expanding traffic.
- **Turn 32 — S3 (Priya, `Sulafat`)** [resolves_success_criterion, pause 900 ms]: If error rate stays under zero point one percent for two hours with zero stale prices, we promote to twenty-five percent.
- **Turn 33 — ME (Alex, `Charon`)** [recipient_ack, pause 900 ms]: Zero point one percent error ceiling gives us a clear promotion gate.
- **Turn 34 — S1 (Maya, `Kore`)** [clarify_scope_or_attention_budget_cap, pause 2100 ms pinned]: On the cache invalidation worker, our rollout scope covers catalog reads, except we have not decided whether regional tax quotes are in scope or out of scope for phase one.
- **Turn 35 — S2 (Sam, `Puck`)** [scope_boundary_stated, pause 900 ms]: Tax quotes depend on postal rules, so keeping regional tax quotes out of scope for phase one is safer.
### Act 5

- **Turn 36 — ME (Alex, `Charon`)** [recipient_question, pause 900 ms]: Catalog reads are in scope and tax quotes stay on origin. How do latency numbers look?
- **Turn 37 — S2 (Sam, `Puck`)** [request_evidence_or_attention_budget_cap, pause 2100 ms pinned]: The edge network team claims checkout tail latency drops by sixty percent under peak load, though nobody has shared what benchmark shows that or what measured data proves it.
- **Turn 38 — S1 (Maya, `Kore`)** [evidence_follow_up, pause 900 ms]: I can attach Tuesday's load report and ask the edge team for their raw p99 histogram.
- **Turn 39 — S2 (Sam, `Puck`)** [prefilter_stock_phrase, pause 600 ms]: Sounds good.
- **Turn 40 — ME (Alex, `Charon`)** [recipient_recap, pause 900 ms]: Quick recap: Priya has the webhook replay runbook and tomorrow's Redis failover drill.
- **Turn 41 — S1 (Maya, `Kore`)** [summary_update, pause 900 ms]: Maya will post the edge p99 benchmark traces and coordinate the sandbox token rotation schedule.
- **Turn 42 — ME (Alex, `Charon`)** [session_closing, pause 500 ms pinned]: Great, let us reconvene at four o'clock once the staging checklist is green. Thanks everyone.

## Reproduce Audio & Coach Replay

```sh
uv run python scripts/conversation_5min/render_announce.py --write-markdown scripts/conversation_5min/SCRIPT.md
uv run coach replay --session takes/coach_conversation_5min/events.jsonl --policy policies/policy-v0-a.yaml --db takes/coach_conversation_5min/replay.db --run-id conv5m-ground-truth --json
uv run coach report --db takes/coach_conversation_5min/replay.db --session conv-5min-cutover-001 --run-id conv5m-ground-truth --json
```
