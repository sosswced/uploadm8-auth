"""
Admin Marketing AI Strategist — selectable objectives, tones, offer styles, and channel mixes.
Drives deterministic fallbacks and OpenAI system prompts (single source of truth).
"""

from __future__ import annotations

from typing import Any, Dict, List

# Each option: id, label, hint (UI), playbook (LLM + deterministic copy guidance)
OBJECTIVES: List[Dict[str, str]] = [
    {
        "id": "revenue_growth",
        "label": "Revenue Growth",
        "hint": "Maximize MRR, token top-ups, and same-session upgrades from active uploaders.",
        "playbook": "Prioritize PUT/AIC bundles, Creator Lite→Pro ladder, and wallet nudges when CTR is healthy.",
    },
    {
        "id": "upsell_conversion",
        "label": "Upsell Conversion",
        "hint": "Move free and Lite users to paid tiers with proof from their upload volume.",
        "playbook": "Target high-intent free uploaders; anchor on upload caps, multi-platform limits, and AI credits.",
    },
    {
        "id": "retention_save",
        "label": "Retention Save",
        "hint": "Win back churn-risk users — low tokens, declining uploads, or dismiss-heavy nudge behavior.",
        "playbook": "Empathy-first save offers, token relief, and 'what you are missing' coaching — avoid hard discount spam.",
    },
    {
        "id": "enterprise_expansion",
        "label": "Enterprise Expansion",
        "hint": "Agency / Studio / multi-seat motion for 3+ platform accounts and high enterprise-fit scores.",
        "playbook": "High-touch consult CTAs, team workflow positioning, Thumbnail Studio + queue scale story.",
    },
    {
        "id": "thumbnail_studio",
        "label": "Thumbnail Studio Adoption",
        "hint": "Drive Pikzels / Thumbnail Studio usage and template-render wins on upload pipeline.",
        "playbook": "Spotlight studio ops, persona pipelines, and before/after packaging lift from engagement_truth.",
    },
    {
        "id": "platform_expansion",
        "label": "Platform Expansion",
        "hint": "Get creators publishing to more destinations — cross-post lift from analytics.",
        "playbook": "Nudge under-used platforms where engagement_rate_pct beats account baseline on other surfaces.",
    },
]

TONES: List[Dict[str, str]] = [
    {
        "id": "executive_clear",
        "label": "Executive Clear",
        "hint": "Crisp, outcome-led copy — numbers first, no fluff.",
        "playbook": "Short sentences, metrics in subject lines, decisive CTAs.",
    },
    {
        "id": "premium_confident",
        "label": "Premium Confident",
        "hint": "Studio-grade positioning — quality, craft, and professional results.",
        "playbook": "Confident voice, avoid discount language unless offer_style demands it.",
    },
    {
        "id": "creator_friendly",
        "label": "Creator Friendly",
        "hint": "Peer-to-peer creator voice — encouraging, practical, platform-native.",
        "playbook": "You/your channel, casual but credible; reference uploads and real workflows.",
    },
    {
        "id": "urgency_scarcity",
        "label": "Urgency & Scarcity",
        "hint": "Time-bound windows and limited slots — use sparingly with real schedule hooks.",
        "playbook": "Deadline language only when metrics support a cohort window; never fake countdowns.",
    },
    {
        "id": "data_driven",
        "label": "Data Driven",
        "hint": "Lead with engagement_truth, funnel CTR, and cohort comparisons.",
        "playbook": "Cite aggregated metrics from the truth bundle; no invented percentages.",
    },
    {
        "id": "community_warm",
        "label": "Community Warm",
        "hint": "Belonging and momentum — leaderboard, milestones, shared wins.",
        "playbook": "Celebrate progress; soft ask; good for retention_save and Trill-adjacent cohorts.",
    },
]

OFFER_STYLES: List[Dict[str, str]] = [
    {
        "id": "value_first",
        "label": "Value First",
        "hint": "Lead with outcomes (reach, saves, time) before price.",
        "playbook": "Offers emphasize workflow ROI, not % off.",
    },
    {
        "id": "discount_forward",
        "label": "Discount Forward",
        "hint": "Explicit savings — trials, coupons, limited promo codes.",
        "playbook": "Clear monetary hook; pair with require_no_revenue_7d targeting when appropriate.",
    },
    {
        "id": "roi_anchored",
        "label": "ROI Anchored",
        "hint": "Compare spend to views, hours saved, or revenue attributed in window.",
        "playbook": "Anchor price to measurable upload/engagement outcomes from metrics.",
    },
    {
        "id": "bundle_upgrade",
        "label": "Bundle Upgrade",
        "hint": "Tier + token pack combos — Creator Pro + PUT bundle, etc.",
        "playbook": "Name concrete SKUs (PUT/AIC packs, tier step-ups) from UploadM8 catalog mental model.",
    },
    {
        "id": "trial_first",
        "label": "Trial First",
        "hint": "Low-friction taste of paid — time-boxed tier or credit trial.",
        "playbook": "Risk reversal; strong for upsell_conversion on active free users.",
    },
    {
        "id": "social_proof",
        "label": "Social Proof",
        "hint": "Cohort benchmarks — 'creators like you' from segment_signals.",
        "playbook": "Reference aggregate peer behavior, never individual user stories in outbound.",
    },
]

CHANNEL_MIXES: List[Dict[str, str]] = [
    {
        "id": "in_app",
        "label": "In-App (wallet nudges)",
        "hint": "Wallet opportunities + dashboard tips — fastest path, no outbound approval.",
        "playbook": "suggested_campaign.channel must be in_app; activate immediately on deploy.",
    },
    {
        "id": "mixed",
        "label": "Mixed",
        "hint": "In-app plus email/Discord follow-up — draft until templates + master approve.",
        "playbook": "Layer channels in execution_plan; primary channel mixed.",
    },
    {
        "id": "email",
        "label": "Email",
        "hint": "Newsletter-style; needs template_subject/html and approval ticket.",
        "playbook": "Rich subject_lines in newsletter; draft campaign until approved.",
    },
    {
        "id": "discount",
        "label": "Discount",
        "hint": "Promo-forward wallet + billing CTAs — active nudges on deploy.",
        "playbook": "Discount channel; pair with discount_forward or trial_first offer_style.",
    },
    {
        "id": "discord",
        "label": "Discord / in-app prompt",
        "hint": "Short community line — uses discord_message_text on save.",
        "playbook": "Concise conversational hook; draft until approved for outbound.",
    },
]

FORCE_DEPLOY_OPTIONS = [
    {"id": "false", "label": "No", "hint": "Respect confidence guardrails on Deploy AI Campaign."},
    {"id": "true", "label": "Yes", "hint": "Override deploy threshold — master admin judgment."},
]

ALLOW_PII_OPTIONS = [
    {"id": "false", "label": "No (sanitized)", "hint": "Strip PII from metrics sent to OpenAI — recommended."},
    {"id": "true", "label": "Yes", "hint": "Send full truth bundle to LLM — only for locked-down admin use."},
]


def _by_id(options: List[Dict[str, str]], key: str) -> Dict[str, str]:
    for row in options:
        if row["id"] == key:
            return row
    return options[0] if options else {"id": key, "label": key, "hint": "", "playbook": ""}


def strategist_presets_payload() -> Dict[str, Any]:
    return {
        "objectives": OBJECTIVES,
        "tones": TONES,
        "offer_styles": OFFER_STYLES,
        "channel_mixes": CHANNEL_MIXES,
        "force_deploy": FORCE_DEPLOY_OPTIONS,
        "allow_pii_in_llm": ALLOW_PII_OPTIONS,
        "defaults": {
            "objective": "revenue_growth",
            "tone": "executive_clear",
            "offer_style": "value_first",
            "channel_mix": "in_app",
            "force_deploy": "false",
            "allow_pii_in_llm": "false",
        },
    }


def build_strategist_system_prompt(payload: Dict[str, Any]) -> str:
    obj = _by_id(OBJECTIVES, str(payload.get("objective") or "revenue_growth"))
    tone = _by_id(TONES, str(payload.get("tone") or "executive_clear"))
    offer = _by_id(OFFER_STYLES, str(payload.get("offer_style") or "value_first"))
    channel = _by_id(CHANNEL_MIXES, str(payload.get("channel_mix") or "in_app"))
    rk = str(payload.get("range") or payload.get("range_key") or "30d")

    return (
        "You are UploadM8's principal growth strategist — B2C SaaS for multi-platform video creators.\n"
        "Given JSON metrics (aggregated truth bundle) and request knobs, output ONE JSON object with keys:\n"
        "game_plan (north_star with platform_views + platform_engagement_rate_pct, data_quality with "
        "platform_coverage_pct, confidence_score 0-100),\n"
        "newsletter (subject_lines array, 3-6 lines),\n"
        "offers (array of {name, value_prop}, 3-5 items),\n"
        "execution_plan (string array, 6-10 concrete ops steps tied to metrics),\n"
        "suggested_campaign (name, objective, channel, range, min_uploads_30d int, "
        "min_enterprise_fit_score int, min_nudge_ctr_pct float, tiers string array, "
        "require_no_revenue_7d bool, notes string).\n\n"
        f"WINDOW: {rk}\n"
        f"OBJECTIVE — {obj['label']}: {obj['playbook']}\n"
        f"TONE — {tone['label']}: {tone['playbook']}\n"
        f"OFFER STYLE — {offer['label']}: {offer['playbook']}\n"
        f"CHANNEL MIX — {channel['label']}: {channel['playbook']}\n"
        "suggested_campaign.channel MUST match the channel mix id when possible "
        f"({channel['id']}).\n"
        "Confidence_score must reflect real signal strength in metrics — do not inflate.\n"
        "Never invent user emails, names, or PII. Use only aggregated metrics provided.\n"
        "Write like a marketing genius: specific, actionable, tied to UploadM8 features "
        "(uploads, wallet PUT/AIC, Thumbnail Studio, Smart insights, Trill, multi-platform)."
    )


def _subject_prefix(tone_id: str) -> str:
    return {
        "executive_clear": "",
        "premium_confident": "Pro workflow: ",
        "creator_friendly": "Quick win: ",
        "urgency_scarcity": "This week: ",
        "data_driven": "By the numbers: ",
        "community_warm": "You're on a roll — ",
    }.get(tone_id, "")


def deterministic_copy_variants(
    payload: Dict[str, Any], metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Objective/tone/offer-specific copy layered onto the numeric skeleton."""
    rk = str(payload.get("range") or payload.get("range_key") or "30d")
    objective = str(payload.get("objective") or "revenue_growth")
    tone_id = str(payload.get("tone") or "executive_clear")
    offer_id = str(payload.get("offer_style") or "value_first")
    channel = str(payload.get("channel_mix") or "in_app")
    kpis = metrics.get("kpis") or {}
    seg = metrics.get("segment_signals") or {}
    et = metrics.get("engagement_truth") or {}
    eng_win = et.get("window") or {}
    er_pct = float(eng_win.get("avg_engagement_rate_pct") or 0)
    prefix = _subject_prefix(tone_id)

    obj_row = _by_id(OBJECTIVES, objective)
    offer_row = _by_id(OFFER_STYLES, offer_id)

    if objective == "retention_save":
        subjects = [
            f"{prefix}We saved your best upload workflow",
            f"{prefix}Don't lose momentum on {rk} uploads",
            f"Token relief for your next batch week",
        ]
        offers = [
            {"name": "PUT safety net", "value_prop": "Top up before the next queue run stalls"},
            {"name": "Comeback coaching", "value_prop": "Smart insights recap for your last posts"},
            {"name": "Lite win-back", "value_prop": "Short trial on AI captions + thumbnails"},
        ]
        tiers = ["free", "creator_lite"]
        min_uploads = 1
        require_no_rev = False
        campaign_name = "Auto: retention save"
    elif objective == "enterprise_expansion":
        subjects = [
            f"{prefix}Scale your agency upload pipeline",
            f"{prefix}Studio workflow for {seg.get('expansion_ready_accounts', 0)} multi-platform accounts",
            "Enterprise fit: team queue + Thumbnail Studio",
        ]
        offers = [
            {"name": "Studio / Agency consult", "value_prop": "Multi-seat workflow and billing review"},
            {"name": "Flex tier path", "value_prop": "3+ platform destinations without friction"},
            {"name": "Thumbnail Studio at scale", "value_prop": "Persona pipelines for client brands"},
        ]
        tiers = ["creator_pro", "studio", "agency"]
        min_uploads = 6
        require_no_rev = False
        campaign_name = "Auto: enterprise expansion"
    elif objective == "upsell_conversion":
        subjects = [
            f"{prefix}Upgrade path for active uploaders",
            f"{prefix}{seg.get('free_high_intent_uploaders', 0)} creators ready for Creator Lite",
            "Unlock AI credits + multi-platform on your next upload",
        ]
        offers = [
            {"name": "Creator Lite step-up", "value_prop": "More uploads, AI captions, and scheduling"},
            {"name": "Creator Pro bundle", "value_prop": "Thumbnail Studio + higher PUT allowance"},
            {"name": "AIC starter pack", "value_prop": "Try AI packaging on your best-performing niche"},
        ]
        tiers = ["free", "creator_lite"]
        min_uploads = 3
        require_no_rev = True
        campaign_name = "Auto: upsell conversion"
    elif objective == "thumbnail_studio":
        subjects = [
            f"{prefix}Thumbnail Studio wins for your niche",
            "Pikzels-powered frames that match your engagement peak",
            f"{prefix}Packaging lift on {rk} uploads",
        ]
        offers = [
            {"name": "Studio trial sprint", "value_prop": "3 persona variants on your next YouTube drop"},
            {"name": "Template render boost", "value_prop": "Auto thumbs on upload when CTR is rising"},
            {"name": "Smart insights packaging", "value_prop": "Apply winning combo from attribution"},
        ]
        tiers = ["creator_lite", "creator_pro", "studio"]
        min_uploads = 2
        require_no_rev = False
        campaign_name = "Auto: Thumbnail Studio push"
    elif objective == "platform_expansion":
        subjects = [
            f"{prefix}Post where your audience reacts most",
            f"{prefix}Add a destination — engagement_truth says you have room",
            "Cross-post without re-uploading manually",
        ]
        offers = [
            {"name": "Multi-platform onboarding", "value_prop": "Connect the platform with highest ER gap"},
            {"name": "Scheduler bundle", "value_prop": "One video, optimized per destination"},
            {"name": "Platform playbook", "value_prop": "Per-platform meta from Smart insights"},
        ]
        tiers = ["free", "creator_lite", "creator_pro"]
        min_uploads = 2
        require_no_rev = False
        campaign_name = "Auto: platform expansion"
    else:
        subjects = [
            f"{prefix}Your upload funnel — {rk} pulse",
            f"{prefix}More reach: thumbnails + multi-platform",
            f"Ops note: {seg.get('token_pressure_accounts', 0)} accounts near token floor",
        ]
        offers = [
            {"name": "PUT + AIC refill bundle", "value_prop": "Micro-transaction for heavy batch weeks"},
            {
                "name": "Creator Lite trial nudge",
                "value_prop": f"Target {seg.get('free_high_intent_uploaders', 0)} active free uploaders",
            },
            {
                "name": "Studio expansion",
                "value_prop": f"{seg.get('expansion_ready_accounts', 0)} multi-platform accounts",
            },
        ]
        tiers = ["free", "creator_lite"]
        min_uploads = 4 if seg.get("free_high_intent_uploaders", 0) > 6 else 2
        require_no_rev = True
        campaign_name = "Auto: data-backed revenue push"

    if offer_id == "discount_forward":
        offers = [
            {"name": "Limited upgrade credit", "value_prop": "Save on first month of Creator Lite/Pro"},
            {"name": "PUT pack promo", "value_prop": "Extra tokens for batch upload weekends"},
            {"name": "Win-back coupon", "value_prop": offer_row["playbook"]},
        ]
    elif offer_id == "roi_anchored":
        offers = [
            {
                "name": "Engagement ROI pack",
                "value_prop": f"Avg ER {er_pct:.2f}% — scale what already works",
            },
            {"name": "Time saved bundle", "value_prop": "Multi-platform upload vs manual cross-post"},
            {"name": "Revenue-attributed nudge", "value_prop": "For clicked-but-not-converted cohort"},
        ]
    elif offer_id == "bundle_upgrade":
        offers = [
            {"name": "Pro + PUT 500", "value_prop": "Tier step-up with upload headroom"},
            {"name": "Lite + AIC 250", "value_prop": "AI packaging for your next 10 posts"},
            {"name": "Studio stack", "value_prop": "Thumbnail Studio + queue for teams"},
        ]
    elif offer_id == "trial_first":
        offers = [
            {"name": "7-day Pro taste", "value_prop": "Full AI + studio without annual lock-in"},
            {"name": "AIC trial credits", "value_prop": "Test Smart insights apply on real uploads"},
            {"name": "Lite trial", "value_prop": "Prove multi-platform lift before commit"},
        ]
    elif offer_id == "social_proof":
        offers = [
            {
                "name": "Cohort benchmark offer",
                "value_prop": f"Peers in your tier avg {kpis.get('nudge_ctr_pct', 0):.1f}% nudge CTR",
            },
            {"name": "Top quartile playbook", "value_prop": "What high-upload accounts do differently"},
            {"name": "Community milestone", "value_prop": "Join creators shipping 4+ uploads / month"},
        ]

    execution = [
        f"Objective ({obj_row['label']}): {obj_row['playbook']}",
        f"Tone ({tone_id}): {offer_row['playbook']}",
        f"Channel ({channel}): prioritize {channel} touchpoints in week 1.",
        "Prioritize in-app nudges for low-PUT cohort while CTR holds.",
        "Email cohort: clicked a nudge but no revenue in 7d.",
        "Discord: spotlight Thumbnail Studio workflows using live Pikzels usage data.",
        "Surface views vs cohort coaching on dashboard for underperforming uploaders.",
        (
            f"Engagement truth ({rk}): avg (likes+comments+shares)/views ~{er_pct:.2f}% "
            f"on {int(eng_win.get('sample_uploads') or 0)} uploads."
        ),
    ]

    return {
        "subjects": subjects,
        "offers": offers,
        "execution_plan": execution,
        "suggested_campaign": {
            "name": campaign_name,
            "objective": objective,
            "channel": channel,
            "range": rk,
            "min_uploads_30d": min_uploads,
            "min_enterprise_fit_score": 55 if objective == "enterprise_expansion" else 45,
            "min_nudge_ctr_pct": max(
                5.0, min(12.0, float(kpis.get("nudge_ctr_pct", 8) or 8) * 0.6)
            ),
            "tiers": tiers,
            "require_no_revenue_7d": require_no_rev,
            "notes": (
                f"{obj_row['label']} · {offer_row['label']} · {tone_id} · "
                "Generated from marketing_events, revenue_tracking, uploads, studio_usage_events, "
                "and engagement_truth."
            ),
        },
    }
