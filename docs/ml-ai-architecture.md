# UploadM8 ML & AI architecture

> Replaces the deleted `CONTENT-GENERATION-ARCHITECTURE.md`. Operator playbook UI: `frontend/guide.html#feat-ml-ai-hub`. Brand: `stages/m8_engine_brand.py` (**M8_ENGINE** + **M8_ENGINE AI**).

## Four runtime systems

| System | Kind | Key modules | Customer surface |
|--------|------|-------------|------------------|
| Promo + content loops | Classical ML (sklearn) | `ml_feature_registry`, `ml_engine`, `promo_targeting_*`, `content_success_*` | Marketing targeting, Smart Insights rankings |
| Publish-hour priors | Classical ML | `m8_publish_hour_model`, `smart_schedule_insights` | Smart schedule / presign slots |
| M8 caption brain | Generative AI | `stages/m8_engine`, `pipeline_ai_trace`, `ai_service_costs` | Upload captions/titles, coach |
| Pikzels + Thumbnail Studio | External + studio | `pikzels_v2*`, `thumbnail_studio*` | Studio UI, pipeline thumbs |

Customer brand is **Smart** (`smart-insights.html`, `smart-coach.js`). Legacy **AI** URLs redirect; backend modules may still use `ai_*` names.

## Compliance contracts (do not break)

### Data / leakage

- Feature columns come **only** from `services/ml_feature_registry.py`.
- Promo leakage columns `channel` / `delivery_status` are `deprecated` — never re-add as `active`.
- Prefer `is_snapshot` + nulls over fake categoricals when attribution is missing.

### Auth

- Admin ML routes (`/api/admin/ml/*`, upload AI/smart trace) use `require_admin`.
- Public `GET /api/features/ml-hub` is **link-only** (Hub URLs / docs). Never expose `HF_TOKEN` or write credentials.

### Billing

- Upload pipeline AIC: `stages/ai_service_costs.py` → wallet reserve/capture.
- `dashcam_osd` is included in `resolve_enabled_ai_services` when Vision is on and `aiServiceDashcamOSD` is true (matches worker stage).
- Studio / feature debits: `core.wallet.atomic_debit_tokens` inside a DB transaction.
- Studio estimates may keep local base/variant math but must debit via `atomic_debit_tokens` and reference `SERVICE_WEIGHTS` where mapped.

### Pipeline / Hub promotion

- Engine cycle: `services/ml_engine.run_ml_engine_cycle`.
- Cold / insufficient data → `status=blocked_on_data` (cycle still `ok`).
- **Seeded** bootstrap models → `trained_not_published` (never Hub promote).
- ROC below `publish_min_roc_auc` or champion gate fail → `trained_not_published`.
- **Content near-miss**: ROC within `publish_near_miss_epsilon` of the floor **and**
  `ndcg_at_10 >= publish_min_ndcg_at_10` → allow Hub promote (ranking-quality rescue).
- **Promo small-n / overfit**: require `n_user_groups >= publish_min_user_groups`;
  withhold if ROC ≥ `publish_overfit_roc_auc` while groups &lt; `publish_overfit_max_user_groups`.
- **Publish-hour priors**: full learn ≥800 rows; cold blend 120–799 (regularized + static mix);
  static fallback below 120. Optional auto-widen PCI→PCI+uploads when thin.

### Dual-repo

| Scope | Repo |
|-------|------|
| `services/`, `stages/`, `routers/`, `scripts/ml_*`, `docs/ml-ai-architecture.md` | `uploadm8-auth` (backend) |
| `frontend/smart-*`, `admin-ml-observability.html`, coach JS | `uploadm8-frontend` |

### Smart schedule repair

- Complete `schedule_metadata` must not force a full slot rebuild when only `scheduled_time` is null — derive the anchor instead.
- Top-level `scheduled_time` is `min(pending platform slots)` — **exclude** already-successful platforms so partial publishes do not rewind the anchor into the past.
- Retry for both `smart` and `scheduled` modes fails closed if schedule repair fails.
- `is_retryable_upload` mirrors `classify_retry_error` hard-blocks (no UI/API mismatch).

## Adding a classical ML loop

1. Register features in `ml_feature_registry.py`
2. Feature module deriving `active_num` / `active_cat` / `label`
3. `scripts/build_*` + `scripts/train_*` (+ optional HF Jobs UV script)
4. Wire into `ml_engine` + `ml_engine_config` + `ml_hub_config`
5. Admin observability + unit tests in `tests/test_ml_features.py`

## Adding a generative AI feature

1. Stage under `stages/` (or extend `m8_engine`)
2. Register AIC in `ai_service_costs.SERVICE_WEIGHTS` + tier allowlist
3. Attribution via `core/content_attribution.py`; optional `pipeline_ai_trace`
4. Surface via `growth_intelligence` / `ai_insights_hub` + thin `routers/me.py` handler
5. Frontend on Smart surfaces only (legacy AI = redirect)

## Video accuracy ladder (Tier 1–4)

| Hook | Module | Role |
|------|--------|------|
| Weak Vision detector | `core/vision_labels.vision_labels_are_weak` | Generic outdoor/vehicle/person → deepen |
| Depth router | `services/multimodal_depth_router.py` | Clip kind + `forceTwelveLabs` when Vision thin |
| TL skip invert | `stages/twelvelabs_stage.py` | Do not skip TL when Vision weak |
| Place without `.map` | `services/place_evidence.py` | Landmarks → Nominatim; OCR beaches/monuments/plates/teams |
| Auto Whisper on speech | `stages/audio_stage.py` | RMS energy → Whisper when STT not explicitly off |
| Paid baselines | `core/upload_baseline_defaults.py` | Paid active: Whisper + captions + Vision + TL + VI + Studio on; free/trial opt-in |
| Shot list | `worker.py` → `output_artifacts.shot_list_v1` | Temporal spine for M8 digest |
| Grounding score | `services/grounding_eval.py` | Deterministic overlap on `hydration_report` |
| M8 pass 2 claims | `services/m8_grounding_pass.py` | `claims[]` + evidence catalog; strip/inject must_use |
| Quality rollup | `services/ml_scoring_job.py` | `upload_quality_scores_daily.mean_grounding` |
| Coach / insights | `growth_intelligence` + `content_insights` | STT + low-grounding suggestions |
| Evidence features | `ml_feature_registry` (experimental) | `grounding_score`, `evidence_lane_count`, `transcript_chars` |
| Gold-set CI | `tests/fixtures/grounding_gold_set.json` | Min grounding gate on gold vs bad captions |
| Upload Q&A | `POST /api/uploads/{id}/ask` → `services/upload_qa.py` | Answers only from hydration evidence + citations |
| Whisper AIC | `stages/ai_service_costs` | `audio_whisper` weight 0 / `AIC_BILLING_EXEMPT` |

Eval: `python scripts/agent/eval_loop.py --mode grounding --json` or `python run_tests.py grounding`.

Env:

- `MULTIMODAL_DEPTH_FORCE_TL=false` — disable force-TL
- `MULTIMODAL_AUTO_WHISPER_ON_SPEECH=false` — disable speech-energy Whisper
- `M8_GROUNDING_PASS2=false` — disable claim critique pass
- Existing `TWELVELABS_SKIP_WHEN_VI_RICH` still applies when Vision is strong

Q&A example:

```http
POST /api/uploads/{upload_id}/ask
{"question": "Where was this filmed?"}
```

Response includes `answer`, `evidence_ids`, `citations[]`, `grounding_ok`.

## AV read data engine (P0–P9)

Owns cheaper **sense+align** over time without replacing the trust layer.

| Piece | Role |
|-------|------|
| Consent | `aiServiceRecognitionTraining` (default OFF, no AIC / no `SERVICE_WEIGHTS`) |
| Pack | `services/av_training_pack.py` → R2 `ml-packs/{user}/{upload}/pack.json` + ≤8 keyframes |
| Writer | `stages/av_training_pack_stage.py` after `enforce_hydration` (fail-soft) |
| Labels | `tl_status`, `fusion_source`, `needs_deep_teacher` (Layer B TL backup cohort) |
| Quality filter | `services/av_training_pack_quality.py` — refuse low grounding / thin lanes / TL-failed-without-fusion from distill dataset (no engagement labels) |
| Distill | `scripts/train_av_read_distill.py`; `UM8_ML_ENGINE_RUN_AV_READ` default **false**; always `trained_not_published` |
| Eval | `python run_tests.py hero-fact` + grounding gold |

**Trust invariant:** Student may propose later; `content_identity` + M8 grounding always judge. Never skip fusion as TL outage backup.

**TL backup layers**

- **A (today):** `scene_fusion` fills `video_understanding` when TL empty — do not regress
- **B (packs):** record `tl_status` / `fusion_source`
- **C (P6):** when `AV_READ_DEPTH_SKIP_TL` + `UM8_AV_READ_SKIP_TL_FLOORS_OK` are on, worker runs **vision-first then conditional TL** via `deep_clf` feature matrix (`services/av_read_features.py` + `av_read_distill_infer.py`); skip only if conf ≥ 0.85 and not vision_weak/force. Flags off → unchanged parallel TL. If VU empty after skip → fusion still runs.

**Same-upload soft bias:** `AV_READ_SOFT_BIAS` may boost hero ranking from pack meta, or **identity_fallback** (`content_identity` hero class) when pack is not written yet. Pack stage stays after hydration; attribution refresh after pack remains.

**Publish pack (`publish_pack_v1`):** Built after identity + hydration, before thumbnail/caption. One subject/hook/`paint_policy` for covers + captions + hashtags. Lone Vision logos/OCR (e.g. roadside false brands) are not paintable unless they agree with subject/hook; dashcam/automotive defaults to hook-only (speed/place/music) or no-text.

**Packaging aftermath (P3–P5)**

| Flag / surface | Default | Behavior |
|----------------|---------|----------|
| Pack features | experimental | `pack_present`, `pack_tl_status`, … → content_success; coarse flags on attribution snapshot **excluded from strategy_key hash** (no free text). Same-upload: pack stage refreshes attribution snap after write. |
| Soft bias / skip-TL / Hub | Operator one-button on **Admin → ML Observability** (`POST /api/admin/ml/av-read-flags`) stores `av_read_operator_bundle_v1` in `admin_settings`; env=1 still overrides. End users never see these flags. |
| Style/persona | recommend-only | Insights never fake `pack_present`; hard prefs only via apply-optimized `confirm=true` |
| Recognition training | consent OFF | No AIC / not in `SERVICE_WEIGHTS` / not clamped by tier AIC map |

**HuggingFace**

- Auto-default AV dataset/model from promo owner (`{owner}/uploadm8-av-read-v1`, `…-model-v1`) like content
- `scripts/init_hf_ml_hub_repos.py --av-dataset-repo` / `--av-model-repo`
- Engine: dataset `--push-to` when repo set; distill stays `trained_not_published` unless floors + `UM8_AV_READ_HUB_PROMOTE` → upload `av_read_distill_model.joblib` + eval/metrics to `av_model_repo` (`published`); push failure → `promote_eligible_local`

**Retention / promote (P7–P9)**

- TTL: `AV_TRAINING_PACK_TTL_DAYS` (default 90); `scripts/purge_expired_av_training_packs.py` dry-run by default; `--apply` deletes R2 keys from DB pack meta and sets `purged_at`
- Account delete → `purge_user_ml_packs`
- Propose-only: `AV_READ_PROPOSE_ONLY` (default off) → `av_read_proposals_v1` artifact; judge still required

Eval: `python scripts/agent/eval_loop.py --mode hero-fact --json` (includes pack + ultracode wave + runtime flag tests)


## Admin & scripts

- Observability: `frontend/admin-ml-observability.html` → `/api/admin/ml/*`
- Manual cycle: `python scripts/ml_engine_run.py` or `POST /api/admin/ml/engine-run`
- Hub bootstrap: `scripts/init_hf_ml_hub_repos.py`, `scripts/verify_ml_hub_wiring.py`
- AV dataset: `scripts/build_av_training_dataset.py`
- AV distill: `scripts/train_av_read_distill.py`
