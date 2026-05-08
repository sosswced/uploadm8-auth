"""
Canonical naming for UploadM8's **M8_ENGINE** product line and its unified **M8_ENGINE AI** layer.

- **M8_ENGINE**: multimodal publishing core (captions, scene graph, ranking).
- **M8_ENGINE AI** (slug ``M8_ENGINE_AI``): single brand for machine learning *and* generative AI —
  learned priors, coach, growth intel, marketing automation. Same family; use ``M8_ENGINE_AI_*``
  in new code.

``M8_ENGINE_MLAI_*`` names remain as aliases equal to ``M8_ENGINE_AI_*`` for backward
compatibility (imports and legacy JSON keys).
"""

from __future__ import annotations

# Family / umbrella (matches env and docs; no spaces for slugs)
M8_ENGINE_SLUG = "M8_ENGINE"

# Combined ML + generative AI learning layer (coach, campaigns, priors, touchpoints)
M8_ENGINE_AI_SLUG = "M8_ENGINE_AI"
M8_ENGINE_AI_DISPLAY = "M8_ENGINE AI"

# Backward-compatible aliases — same values as M8_ENGINE_AI_*
M8_ENGINE_MLAI_SLUG = M8_ENGINE_AI_SLUG
M8_ENGINE_MLAI_DISPLAY = M8_ENGINE_AI_DISPLAY

M8_ENGINE_FAMILY_TAGLINE = (
    "M8_ENGINE multimodal publishing brain with M8_ENGINE AI — machine learning and AI in one stack "
    "(quality priors, creator coach, growth and campaign intelligence)."
)
