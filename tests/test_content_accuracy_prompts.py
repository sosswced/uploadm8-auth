"""Content-accuracy guardrails for caption + thumbnail prompt layers.

Locks the Phase 1 anti-clickbait defaults so hook templates and prompt rules
cannot silently regress back to false-scarcity framing.
"""
import inspect
import re

from stages.caption_stage import CONTENT_CATEGORIES, _build_narrative_prompt
from stages.context import THUMBNAIL_BRIEF_PROMPT

# False-scarcity / unverifiable-claim shapes. The model must describe what is
# actually on screen instead of implying hidden knowledge.
BANNED_HOOK_PATTERNS = (
    r"nobody (expected|teaches|talks about|tells you|is talking)",
    r"the secret (to|nobody)",
    r"you need to see this",
    r"you won'?t believe",
    r"the \d+% of people who know",
)


def test_caption_prompt_carries_accuracy_over_engagement_rule():
    src = inspect.getsource(_build_narrative_prompt)
    assert "ACCURACY OVER ENGAGEMENT" in src
    assert "clickbait" in src.lower()


def test_no_category_hook_template_uses_false_scarcity():
    offenders = []
    for category, data in CONTENT_CATEGORIES.items():
        for hook in data.get("hook_templates", []):
            for pattern in BANNED_HOOK_PATTERNS:
                if re.search(pattern, hook, re.IGNORECASE):
                    offenders.append((category, hook, pattern))
    assert not offenders, f"clickbait hook templates present: {offenders}"


def test_every_category_still_has_hooks_and_seeds():
    """Accuracy fixes must not flatten a category into having no vocabulary."""
    for category, data in CONTENT_CATEGORIES.items():
        assert data.get("hook_templates"), f"{category} lost its hook templates"
        assert data.get("hashtag_seeds"), f"{category} lost its hashtag seeds"


def test_thumbnail_brief_prompt_requires_truthful_headlines_and_badges():
    assert "ACCURACY:" in THUMBNAIL_BRIEF_PROMPT
    assert "TOP 5" in THUMBNAIL_BRIEF_PROMPT
    # Badge selection is constrained by prompt instruction (no hard filter),
    # so the conditional wording is the guardrail under test.
    assert "badge only when it accurately fits" in THUMBNAIL_BRIEF_PROMPT
    assert "FAST only if speed/telemetry present" in THUMBNAIL_BRIEF_PROMPT
