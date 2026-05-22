"""
Shared visual-entity buckets for Google Vision + Video Intelligence rollups.

Every upload type (dashcam, vlog, cooking, camping, fishing, art, etc.) maps
detected phrases into recall-friendly buckets so M8, hydration, and the user
entity catalog can cite concrete nouns instead of generic labels.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from core.vision_labels import is_generic_vision_label, vision_label_slug

# Canonical bucket keys stored in user_visual_entity_catalog and recognition_flat.
RECOGNITION_BUCKETS: Sequence[str] = (
    "vehicles",
    "brands",
    "food",
    "plants",
    "animals",
    "outdoors",
    "sports",
    "art",
    "products",
    "places",
    "colors",
    "signage",
    "restaurants",
    "people",
    "text_on_screen",
    "objects",
    "web_matches",
    "all_entities",
)

# Niche → buckets to foreground in narratives and must-use shortlists.
NICHE_BUCKET_PRIORITY: Dict[str, List[str]] = {
    "general": ["objects", "web_matches", "brands", "food", "places", "outdoors"],
    "gaming": ["products", "brands", "objects", "people"],
    "finance": ["objects", "text_on_screen", "brands"],
    "education": ["objects", "text_on_screen", "places"],
    "automotive": ["vehicles", "colors", "brands", "signage", "text_on_screen"],
    "lifestyle": ["objects", "places", "people", "brands"],
    "comedy": ["objects", "people", "text_on_screen"],
    "podcast": ["people", "text_on_screen", "brands"],
    "music": ["people", "brands", "objects"],
    "sports": ["sports", "people", "outdoors", "objects"],
    "tech": ["products", "brands", "objects"],
    "beauty": ["products", "colors", "people"],
    "food": ["food", "plants", "restaurants", "brands", "colors"],
    "travel": ["places", "outdoors", "plants", "objects"],
    "fitness": ["sports", "people", "objects"],
    "true_crime": ["places", "text_on_screen", "objects"],
    "real_estate": ["places", "objects", "text_on_screen"],
    "business": ["brands", "text_on_screen", "objects"],
    "news": ["text_on_screen", "places", "brands"],
}

_COLOR_WORDS = frozenset(
    {
        "red", "blue", "green", "yellow", "orange", "purple", "pink",
        "black", "white", "silver", "gray", "grey", "gold", "brown",
        "beige", "teal", "navy", "maroon", "bronze", "copper",
    }
)

_FOOD_RE = re.compile(
    r"\b("
    r"goulash|goulashes|cilantro|parsley|basil|oregano|thyme|rosemary|"
    r"pasta|pizza|burger|taco|sushi|ramen|pho|curry|steak|salmon|shrimp|"
    r"chicken|pork|beef|lamb|seafood|soup|salad|sandwich|burrito|"
    r"plating|garnish|appetizer|entree|dessert|pastry|bread|cake|"
    r"cooking|baking|grill|fry|saute|kitchen|chef|recipe|ingredient|"
    r"vegetable|fruit|tomato|onion|garlic|pepper|cheese|butter|cream|"
    r"coffee|espresso|latte|wine|beer|cocktail|stew|broth|noodle|rice|"
    r"mcdonald|starbucks|chipotle|wendy|taco\s*bell"
    r")\b",
    re.I,
)

_VEHICLE_RE = re.compile(
    r"\b("
    r"mustang|viper|corvette|camaro|charger|challenger|corolla|civic|accord|"
    r"wrangler|f-150|silverado|ram\s*1500|model\s*[sxy3]|type\s*r|"
    r"bmw|mercedes|audi|porsche|ferrari|lamborghini|mclaren|bugatti|"
    r"tesla|rivian|lucid|honda|toyota|ford|chevrolet|chevy|nissan|mazda|"
    r"subaru|jeep|dodge|gmc|cadillac|lincoln|lexus|infiniti|acura|"
    r"sports?\s*car|supercar|hypercar|sedan|coupe|hatchback|suv|pickup|truck"
    r")\b",
    re.I,
)

YEAR_MAKE_MODEL_RE = re.compile(
    r"\b((?:19|20)\d{2})\s+"
    r"([A-Za-z][A-Za-z0-9\-]*(?:\s+[A-Za-z0-9\-]+){0,4})"
    r"(?:\s+(ACR|GT|RS|SS|ST|Type\s*R|SRT|Hellcat))?\b",
    re.I,
)

_PRODUCT_RE = re.compile(
    r"\b(phone|laptop|tablet|watch|sneaker|shoe|handbag|backpack|camera|drone|gopro)\b",
    re.I,
)

_PLANTS_RE = re.compile(
    r"\b("
    r"flower|flowers|floral|bloom|blooms|rose|roses|sunflower|tulip|daisy|"
    r"lily|orchid|lavender|cherry\s*blossom|tree|palm|cactus|succulent|garden|"
    r"meadow|wildflower|fern|moss|ivy|bonsai|houseplant|house\s*plant|"
    r"cilantro|parsley|basil|mint|sage|dill|thyme|rosemary|lettuce|kale|spinach"
    r")\b",
    re.I,
)

_ANIMALS_RE = re.compile(
    r"\b("
    r"dog|puppy|cat|kitten|bird|eagle|hawk|owl|deer|moose|elk|bear|"
    r"fox|rabbit|squirrel|horse|cow|goat|sheep|pig|chicken|duck|goose|"
    r"fish|salmon|trout|bass|shark|whale|dolphin|seal|otter|"
    r"insect|butterfly|bee|spider|snake|lizard|turtle|frog"
    r")\b",
    re.I,
)

_OUTDOORS_RE = re.compile(
    r"\b("
    r"camping|campfire|tent|hiking|trail|backpacking|kayak|canoe|"
    r"mountain|summit|lake|river|waterfall|beach|coast|forest|woods|"
    r"desert|canyon|valley|glacier|snow|ski|snowboard|sunset|sunrise|"
    r"national\s*park|state\s*park|wilderness|scenic|overlook"
    r")\b",
    re.I,
)

_SPORTS_RE = re.compile(
    r"\b("
    r"fishing|angler|bass|trout|fly\s*fishing|reel|rod|lure|bait|"
    r"golf|basketball|football|soccer|baseball|tennis|surf|skate|"
    r"workout|gym|yoga|run|running|marathon|cycling|bike|climb|climbing"
    r")\b",
    re.I,
)

_ART_RE = re.compile(
    r"\b("
    r"drawing|sketch|painting|canvas|watercolor|illustration|"
    r"pottery|sculpture|craft|knitting|sewing|digital\s*art|"
    r"pencil|charcoal|marker|brush|easel|studio\s*art"
    r")\b",
    re.I,
)

_SIGNAGE_RE = re.compile(
    r"\b("
    r"stop\s*sign|yield|speed\s*limit|highway\s*sign|road\s*sign|street\s*sign|"
    r"exit\s*sign|billboard|traffic\s*light|signal|interstate|route\s*\d+|"
    r"us\s*\d+|i-?\d{1,3}|sr-?\d+|mph|km/h"
    r")\b",
    re.I,
)

_RESTAURANT_RE = re.compile(
    r"\b("
    r"restaurant|cafe|coffee\s*shop|diner|grill|bar\s*&\s*grill|"
    r"mcdonald|mcdonalds|burger\s*king|wendy|taco\s*bell|chipotle|"
    r"starbucks|dunkin|in-?n-?out|chick-?fil-?a|subway|kfc|pizza\s*hut|"
    r"domino|five\s*guys|shake\s*shack|whataburger|penelope"
    r")\b",
    re.I,
)

_BRAND_HINT_RE = re.compile(
    r"\b(mcdonald|starbucks|nike|adidas|apple|google|amazon|costco|walmart|target)\b",
    re.I,
)


def empty_catalog() -> Dict[str, List[Dict[str, Any]]]:
    return {k: [] for k in RECOGNITION_BUCKETS}


def niche_bucket_order(niche: Optional[str]) -> List[str]:
    key = (niche or "general").strip().lower().replace(" ", "_").replace("-", "_")
    return list(NICHE_BUCKET_PRIORITY.get(key) or NICHE_BUCKET_PRIORITY["general"])


def classify_phrase(
    phrase: str,
    *,
    catalog: Dict[str, List[Dict[str, Any]]],
    source: str,
    score: float,
) -> None:
    """Route a detected phrase into all applicable buckets."""
    low = phrase.lower()
    if _VEHICLE_RE.search(phrase) or YEAR_MAKE_MODEL_RE.search(phrase):
        _append_unique(catalog["vehicles"], phrase, source=source, score=score)
    if _FOOD_RE.search(phrase):
        _append_unique(catalog["food"], phrase, source=source, score=score)
    if _PLANTS_RE.search(phrase):
        _append_unique(catalog["plants"], phrase, source=source, score=score)
    if _ANIMALS_RE.search(phrase):
        _append_unique(catalog["animals"], phrase, source=source, score=score)
    if _OUTDOORS_RE.search(phrase):
        _append_unique(catalog["outdoors"], phrase, source=source, score=score)
    if _SPORTS_RE.search(phrase):
        _append_unique(catalog["sports"], phrase, source=source, score=score)
    if _ART_RE.search(phrase):
        _append_unique(catalog["art"], phrase, source=source, score=score)
    if _PRODUCT_RE.search(phrase):
        _append_unique(catalog["products"], phrase, source=source, score=score)
    if _SIGNAGE_RE.search(phrase):
        _append_unique(catalog["signage"], phrase, source=source, score=score)
    if _RESTAURANT_RE.search(phrase):
        _append_unique(catalog["restaurants"], phrase, source=source, score=score)
    if _BRAND_HINT_RE.search(phrase):
        _append_unique(catalog["brands"], phrase, source=source, score=score)
    for cw in _COLOR_WORDS:
        if re.search(rf"\b{re.escape(cw)}\b", low):
            _append_unique(catalog["colors"], cw, source=source, score=score)
    slug = vision_label_slug(phrase)
    if slug in _COLOR_WORDS:
        _append_unique(catalog["colors"], phrase, source=source, score=score)
    if not is_generic_vision_label(phrase, min_specific_len=3):
        _append_unique(catalog["objects"], phrase, source=source, score=score, limit=48)
    if len(phrase) > 3 and not is_generic_vision_label(phrase, min_specific_len=4):
        _append_unique(catalog["all_entities"], phrase, source=source, score=score, limit=64)


def _append_unique(
    bucket: List[Dict[str, Any]],
    phrase: str,
    *,
    source: str,
    score: float = 0.0,
    limit: int = 24,
) -> None:
    p = re.sub(r"\s+", " ", str(phrase or "").strip())
    if not p or len(p) < 2:
        return
    key = p.lower()
    if any(x.get("name", "").lower() == key for x in bucket):
        return
    if len(bucket) >= limit:
        return
    bucket.append({"name": p[:120], "source": source, "score": round(float(score or 0), 3)})


def narrative_bucket_labels() -> Dict[str, str]:
    return {
        "vehicles": "Vehicles",
        "brands": "Brands/logos",
        "food": "Food/cooking",
        "plants": "Plants/flora",
        "animals": "Animals/wildlife",
        "outdoors": "Outdoors/camping/travel",
        "sports": "Sports/fitness/fishing",
        "art": "Art/crafts",
        "colors": "Colors",
        "products": "Products/gear",
        "places": "Places/landmarks",
        "signage": "Signage/roads",
        "restaurants": "Restaurants/cafes",
        "objects": "Objects/scene",
        "text_on_screen": "On-screen text",
    }
