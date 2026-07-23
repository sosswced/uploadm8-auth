"""
Shared Google Vision label filtering for hashtags, M8 scoring, and hydration.

Vision LABEL_DETECTION often returns coarse object tags (windshield, boat, blue)
that are poor discovery hashtags when stronger signals exist (geo, VI text, landmarks).

``AMBIENT_REDUNDANT_SLUGS_BY_PROFILE`` holds labels that are almost always present
for a given shoot type (dashcam POV furniture, kitchen utensils, garden tools) and
should not become hashtags, catalog entries, or thumbnail brief noise. Profiles are
resolved dynamically from category + filename + label hints — not dashcam-only.
"""

from __future__ import annotations

import re
from typing import Any, FrozenSet, Iterable, List, Optional, Set

# Slugs (alphanumeric, lower) for labels we never want as hashtag evidence.
GENERIC_VISION_LABEL_SLUGS: frozenset[str] = frozenset(
    {
        "sky",
        "outdoor",
        "outdoors",
        "indoor",
        "indoors",
        "vehicle",
        "car",
        "transport",
        "transportation",
        "modeoftransport",
        "asphalt",
        "road",
        "roadway",
        "street",
        "boulevard",
        "tree",
        "trees",
        "plant",
        "grass",
        "land",
        "landscape",
        "nature",
        "scenery",
        "scenic",
        "horizon",
        "atmosphere",
        "environment",
        "weather",
        "daytime",
        "nighttime",
        "infrastructure",
        "cloud",
        "clouds",
        "person",
        "people",
        "object",
        "view",
        "scene",
        "blue",
        "green",
        "red",
        "white",
        "black",
        "color",
        "colour",
        "light",
        "dark",
        "day",
        "night",
        "water",
        "wheel",
        "tire",
        "tyre",
        "glass",
        "windshield",
        "windscreen",
        "windscreenwiper",
        "windshieldwiper",
        "wiper",
        "mirror",
        "hood",
        "bonnet",
        "dashboard",
        "steering",
        "steeringwheel",
        "seat",
        "motor",
        "motorvehicle",
        "landvehicle",
        "automotive",
        "automobile",
        "driving",
        "drive",
        "boat",
        "ship",
        "watercraft",
        "lensflare",
        "lens",
        "flare",
        "reflection",
        "shadow",
        "texture",
        "pattern",
        "material",
        "metal",
        "plastic",
        "wood",
        "concrete",
        "building",
        "structure",
        "architecture",
        "furniture",
        "room",
        "wall",
        "floor",
        "ceiling",
        "window",
        "door",
        "sign",
        "symbol",
        "logo",
        "text",
        "font",
        "line",
        "shape",
        "circle",
        "rectangle",
        # Colors — never discovery tags / never category titles
        "yellow",
        "orange",
        "purple",
        "pink",
        "gray",
        "grey",
        "brown",
        "cyan",
        "magenta",
        "beige",
        "gold",
        "golden",
        "silver",
        "navy",
        "teal",
        "maroon",
        "violet",
        "indigo",
        "turquoise",
        "coral",
        "cream",
        "ivory",
        "crimson",
        "scarlet",
        "azure",
        "amber",
        "pastel",
        "neon",
        "monochrome",
        "colorful",
        "colourful",
        "vibrant",
        "multicolor",
        "multicolour",
        # Vague nature / scenery taxonomy
        "flora",
        "fauna",
        "wildlife",
        "wilderness",
        "countryside",
        "meadow",
        "forest",
        "woods",
        "mountain",
        "mountains",
        "hill",
        "hills",
        "valley",
        "field",
        "fields",
        "earth",
        "planet",
        "natural",
        "naturalbeauty",
        "mothernature",
        "outdoorphotography",
        "naturephotography",
        "naturelovers",
        "naturelover",
        "getoutside",
        "optoutside",
        # Vague category / mood filler
        "aesthetic",
        "vibes",
        "vibe",
        "mood",
        "ambiance",
        "ambience",
        "ambient",
        "beautiful",
        "beauty",
        "amazing",
        "awesome",
        "cool",
        "nice",
        "pretty",
        "epic",
        "legendary",
        "photography",
        "photooftheday",
        "nofilter",
        "instagood",
        "picoftheday",
        "lifestyle",
        "travel",
        "wanderlust",
        "adventure",
        "explore",
        "inspiration",
        "motivational",
        "background",
        "foreground",
        "perspective",
        "composition",
        "blur",
        "bokeh",
        "focus",
        "motion",
        "still",
        "image",
        "photo",
        "picture",
        "footage",
        "clip",
        "recording",
        "category",
        "general",
        "misc",
        "miscellaneous",
        "other",
        "unknown",
        "unspecified",
    }
)

# Pure color slugs — always junk for hashtags even if somehow missing above.
COLOR_HASHTAG_SLUGS: frozenset[str] = frozenset(
    {
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
        "white", "gray", "grey", "brown", "cyan", "magenta", "beige", "gold",
        "golden", "silver", "navy", "teal", "maroon", "violet", "indigo",
        "turquoise", "coral", "cream", "ivory", "crimson", "scarlet", "azure",
        "amber", "pastel", "neon", "color", "colour", "colorful", "colourful",
        "vibrant", "multicolor", "multicolour", "monochrome",
    }
)

# Filename / HUD OCR / taxonomy dumps that look specific but are useless discovery tags.
_JUNK_HASHTAG_RE = re.compile(
    r"(?ix)"
    r"("
    r"^\d{8,}"                          # timestamp / epoch dumps
    r"|\d{6,}(?:am|pm)"                 # 20250303111931am…
    r"|\d{1,3}[o0l]?\s*mph"             # 66mph / 7omph (OCR O/l confusion)
    r"|mph[a-z]{2,}"                    # mphcwalker mashups
    r"|(?:lat|lon|gps)\d"
    r"|\d{2,3}\.\d{3,}"                 # coordinate fragments
    r")"
)

# Always-on scene furniture for a profile — merged with GENERIC when profile is active.
AMBIENT_REDUNDANT_SLUGS_BY_PROFILE: dict[str, frozenset[str]] = {
    "automotive": frozenset(
        {
            "automotiveexterior",
            "automotivemirror",
            "automotivemirrors",
            "automotivesideviewmirror",
            "automotivesideviewmirrors",
            "rearviewmirror",
            "rearmirror",
            "sideviewmirror",
            "sideviewmirrors",
            "exterior",
            "automobilemake",
            "tires",
            "tyres",
            "wheels",
            "rims",
            "gas",
            "gasoline",
            "gasstation",
            "fuel",
            "petrol",
            "fuelpump",
            "carwindow",
            "carwindows",
            "windows",
            "bumper",
            "fender",
            "fenders",
            "headlight",
            "headlights",
            "taillight",
            "taillights",
            "grille",
            "exhaust",
            "muffler",
            "carpart",
            "carparts",
            "tar",
            "pavement",
            "curb",
            "lane",
            "lanes",
            "roadtrip",
            "commuting",
            "driver",
            "controlledaccesshighway",
            "freeway",
            "expressway",
            "kitcar",
            "racecar",
            "cruise",
            "cruising",
        }
    ),
    "dashcam": frozenset(
        {
            "travel",
            "trip",
            "highway",
            "instrumentpanel",
            "speedometer",
            "odometer",
            "dashcam",
            "actioncamera",
            "gopro",
        }
    ),
    "gardening": frozenset(
        {
            "dirt",
            "soil",
            "mud",
            "ground",
            "earth",
            "mulch",
            "compost",
            "glove",
            "gloves",
            "gardeninggloves",
            "workgloves",
            "rake",
            "shovel",
            "spade",
            "hoe",
            "trowel",
            "shears",
            "pruner",
            "pruners",
            "planting",
            "gardener",
            "gardening",
            "lawn",
            "backyard",
            "yard",
            "hose",
            "wateringcan",
            "wheelbarrow",
            "seed",
            "seeds",
            "planter",
            "planters",
        }
    ),
    "cooking": frozenset(
        {
            "plate",
            "plates",
            "dish",
            "dishes",
            "bowl",
            "bowls",
            "fork",
            "knife",
            "knives",
            "spoon",
            "spoons",
            "cutlery",
            "cuttingboard",
            "choppingboard",
            "countertop",
            "counter",
            "stove",
            "oven",
            "range",
            "cooktop",
            "microwave",
            "pan",
            "pans",
            "pot",
            "pots",
            "skillet",
            "wok",
            "spatula",
            "ladle",
            "whisk",
            "tongs",
            "apron",
            "chefhat",
            "kitchen",
            "cookware",
            "tableware",
            "servingdish",
            "cooking",
            "baking",
        }
    ),
    "camping": frozenset(
        {
            "tent",
            "tentpole",
            "sleepingbag",
            "campfire",
            "matches",
            "backpack",
            "hikingboot",
            "hikingboots",
            "trekkingpole",
            "campground",
            "campsite",
            "camping",
        }
    ),
    "fishing": frozenset(
        {
            "fishingrod",
            "fishingline",
            "reel",
            "hook",
            "hooks",
            "bait",
            "lure",
            "lures",
            "tacklebox",
            "net",
            "fishingnet",
            "fishing",
            "angler",
        }
    ),
    "fitness": frozenset(
        {
            "gym",
            "gymnasium",
            "fitness",
            "fitnesscenter",
            "workout",
            "exercise",
            "exercising",
            "training",
            "dumbbell",
            "dumbbells",
            "barbell",
            "kettlebell",
            "kettlebells",
            "treadmill",
            "elliptical",
            "exercisebike",
            "stationarybike",
            "spinbike",
            "rowingmachine",
            "squat",
            "squatrack",
            "benchpress",
            "weightbench",
            "weight",
            "weights",
            "weightplate",
            "weightplates",
            "weightlifting",
            "weightlifter",
            "crossfit",
            "yogamat",
            "exercisemat",
            "foamroller",
            "resistanceband",
            "pullupbar",
            "chinupbar",
            "locker",
            "lockerroom",
            "gymfloor",
            "gymequipment",
            "fitnessequipment",
            "sportswear",
            "activewear",
            "athleticwear",
            "leggings",
            "sweatband",
            "waterbottle",
        }
    ),
    "podcast": frozenset(
        {
            "podcast",
            "podcasting",
            "microphone",
            "microphones",
            "mic",
            "headphone",
            "headphones",
            "headset",
            "earbuds",
            "boomarm",
            "microphonearm",
            "popfilter",
            "windscreen",
            "shockmount",
            "audiomixer",
            "mixingconsole",
            "mixer",
            "audiointerface",
            "soundboard",
            "soundbooth",
            "recordingstudio",
            "broadcaststudio",
            "studiolight",
            "studiolighting",
            "xlrcable",
            "cable",
            "cables",
            "desk",
            "officechair",
            "chair",
            "webcam",
            "camera",
            "monitor",
            "computer",
            "laptop",
            "keyboard",
        }
    ),
    "beauty": frozenset(
        {
            "makeup",
            "cosmetics",
            "cosmetic",
            "beauty",
            "lipstick",
            "mascara",
            "foundation",
            "concealer",
            "blush",
            "eyeshadow",
            "eyeliner",
            "makeupbrush",
            "makeupbrushes",
            "brush",
            "brushes",
            "palette",
            "makeuppalette",
            "compact",
            "powder",
            "vanity",
            "vanitymirror",
            "dressingtable",
            "hairdryer",
            "curlingiron",
            "flatiron",
            "hairstraightener",
            "salon",
            "beautysalon",
            "hairsalon",
            "barber",
            "barbershop",
            "nailpolish",
            "manicure",
            "pedicure",
            "skincare",
            "serum",
            "moisturizer",
            "lotion",
            "perfume",
            "fragrance",
            "beautyproduct",
            "beautyproducts",
        }
    ),
}

_DASHCAM_FILENAME_TOKENS = ("CAM_", "DASHCAM", "_EVNT", "BLACKVU", "THINKWARE", "GOPRO", "DRIFT")
_DASHCAM_LABEL_MARKERS = (
    "windshield",
    "windscreen",
    "rear-view mirror",
    "rearview mirror",
    "automotive exterior",
    "automotive mirror",
    "automotive side-view mirror",
    "hood",
)
_GARDENING_LABEL_MARKERS = ("gardening", "rake", "shovel", "planting", "garden glove", "wheelbarrow")
_COOKING_LABEL_MARKERS = ("cooking", "baking", "recipe", "kitchen", "chef")
_CAMPING_LABEL_MARKERS = ("camping", "tent", "campfire", "campsite")
_FISHING_LABEL_MARKERS = ("fishing", "angler", "fishing rod", "fishingrod")
_FITNESS_LABEL_MARKERS = (
    "gym",
    "workout",
    "dumbbell",
    "barbell",
    "treadmill",
    "kettlebell",
    "crossfit",
    "yoga mat",
    "squat rack",
    "weight training",
    "fitness",
)
_PODCAST_LABEL_MARKERS = (
    "podcast",
    "microphone",
    "headphones",
    "recording studio",
    "pop filter",
    "boom arm",
    "audio mixer",
    "sound booth",
    "broadcast",
)
_BEAUTY_LABEL_MARKERS = (
    "makeup",
    "cosmetics",
    "lipstick",
    "mascara",
    "foundation",
    "vanity",
    "beauty salon",
    "hair salon",
    "makeup brush",
    "eyeshadow",
    "skincare",
)


def vision_label_slug(raw: Any) -> str:
    """Normalize a Vision label to a lowercase alphanumeric slug."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(raw or "")).lower()


def is_generic_vision_label(raw: Any, *, min_specific_len: int = 4) -> bool:
    """True when the label is too generic for hashtag / must-use use."""
    slug = vision_label_slug(raw)
    if not slug or len(slug) < min_specific_len:
        return True
    if slug in GENERIC_VISION_LABEL_SLUGS:
        return True
    try:
        from services.generic_hard_ban import is_hard_banned_slug

        if is_hard_banned_slug(slug):
            return True
    except Exception:
        pass
    return False


def is_junk_hashtag_body(raw: Any) -> bool:
    """True for HUD OCR mashups, filename timestamps, or taxonomy filler tags.

    Used as the last gate before publishing hashtags so M8 / OCR / VI never
    ship half-assed discovery noise like ``66mphcwalker`` or ``modeoftransport``.
    Colors, nature/scenery taxonomy, and vague category mood tags are always junk.
    Dynamic admin hard-ban registry is consulted on every check.
    """
    text = str(raw or "").strip().lstrip("#")
    if not text:
        return True
    slug = vision_label_slug(text)
    if not slug:
        return True
    try:
        from services.generic_hard_ban import is_hard_banned_slug

        if is_hard_banned_slug(slug):
            return True
    except Exception:
        if slug in GENERIC_VISION_LABEL_SLUGS or slug in COLOR_HASHTAG_SLUGS:
            return True
    if slug in GENERIC_VISION_LABEL_SLUGS or slug in COLOR_HASHTAG_SLUGS:
        return True
    # Color-* compounds (bluesky, greentrees, redcar) when the stem is a color.
    for color in COLOR_HASHTAG_SLUGS:
        if len(color) >= 3 and (slug.startswith(color) or slug.endswith(color)):
            # Keep real brands that merely contain a color syllable (e.g. RedBull
            # is a logo — logos are pushed before scrub; still allow known brands
            # only when longer and mixed). Bare color+noun taxonomy stays junk.
            if slug == color or slug in {
                f"{color}sky",
                f"{color}car",
                f"{color}road",
                f"{color}tree",
                f"{color}trees",
                f"{color}horizon",
                f"{color}nature",
                f"{color}scape",
                f"{color}vibes",
                f"{color}aesthetic",
            }:
                return True
    if slug.endswith(("vibes", "aesthetic", "mood", "scenery", "landscape")):
        return True
    digit_n = sum(ch.isdigit() for ch in slug)
    if digit_n >= 8:
        return True
    if _JUNK_HASHTAG_RE.search(slug) or _JUNK_HASHTAG_RE.search(text.replace(" ", "")):
        return True
    # Pure numeric / date-like bodies
    if slug.isdigit() and len(slug) >= 6:
        return True
    return False


def is_vague_taxonomy_copy(raw: Any) -> bool:
    """True when title/caption is basically a Vision category label, not a story.

    Catches publishable copy like ``Nature``, ``Horizon views``, ``Mode of transport``,
    ``Blue skies`` — useless without place/speed/brand substance.
    """
    text = re.sub(r"\s+", " ", str(raw or "").strip())
    if not text:
        return True
    core = text.rstrip(".!?…").strip()
    if not core:
        return True
    words = re.findall(r"[A-Za-z0-9]+", core)
    if not words:
        return True
    if len(words) <= 4:
        slug = vision_label_slug(core)
        if slug in GENERIC_VISION_LABEL_SLUGS or slug in COLOR_HASHTAG_SLUGS:
            return True
        try:
            from services.generic_hard_ban import is_hard_banned_slug

            if is_hard_banned_slug(slug):
                return True
        except Exception:
            pass
        if is_junk_hashtag_body(core):
            return True
        # Every token is weak taxonomy / color filler.
        if all(
            vision_label_slug(w) in GENERIC_VISION_LABEL_SLUGS
            or vision_label_slug(w) in COLOR_HASHTAG_SLUGS
            or is_junk_hashtag_body(w)
            for w in words
        ):
            return True
    if re.match(
        r"(?i)^(nature|horizon|scenery|landscape|outdoors?|travel|adventure|"
        r"lifestyle|automotive|vehicle|transport|mode of transport|beautiful|"
        r"blue skies|green trees|open skies)\b",
        core,
    ) and len(words) <= 6 and not re.search(r"\d", core):
        return True
    return False

def vision_labels_are_weak(
    labels: Optional[Iterable[Any]] = None,
    *,
    landmark_names: Optional[Iterable[Any]] = None,
    logo_names: Optional[Iterable[Any]] = None,
    ocr_text: str = "",
    min_specific: int = 2,
) -> bool:
    """
    True when Vision mostly returned coarse labels (outdoor/vehicle/person)
    without landmarks, logos, or meaningful OCR — signal to deepen multimodal
    (force Twelve Labs / keep VI) before M8 writes captions.
    """
    if landmark_names and any(str(x).strip() for x in landmark_names):
        return False
    if logo_names and any(str(x).strip() for x in logo_names):
        return False
    ocr = (ocr_text or "").strip()
    if len(ocr) >= 12:
        return False
    specific = [
        str(x).strip()
        for x in (labels or [])
        if str(x).strip() and not is_generic_vision_label(x)
    ]
    return len(specific) < max(0, int(min_specific))


def resolve_ambient_profiles(
    *,
    category: str = "general",
    filename: str = "",
    vision_label_names: Optional[Iterable[Any]] = None,
) -> frozenset[str]:
    """
    Detect which ambient-redundancy profiles apply to this upload.

    Profiles stack (automotive + dashcam for POV clips; cooking for food niche, etc.).
    """
    profiles: set[str] = set()
    cat = (category or "general").strip().lower().replace(" ", "_").replace("-", "_")

    _CATEGORY_PROFILES: dict[str, frozenset[str]] = {
        "automotive": frozenset({"automotive"}),
        "travel": frozenset({"automotive"}),
        "dashcam": frozenset({"automotive", "dashcam"}),
        "food": frozenset({"cooking"}),
        "camping": frozenset({"camping"}),
        "fishing": frozenset({"fishing"}),
        "sports": frozenset({"fitness"}),
        "fitness": frozenset({"fitness"}),
        "podcast": frozenset({"podcast"}),
        "beauty": frozenset({"beauty"}),
    }
    profiles.update(_CATEGORY_PROFILES.get(cat, frozenset()))

    fname = (filename or "").upper()
    if any(tok in fname for tok in _DASHCAM_FILENAME_TOKENS):
        profiles.update({"automotive", "dashcam"})

    labels_blob = " ".join(str(x).lower() for x in (vision_label_names or []))

    if sum(1 for m in _DASHCAM_LABEL_MARKERS if m in labels_blob) >= 2:
        profiles.update({"automotive", "dashcam"})

    if cat in ("lifestyle", "general", "education", "home", "garden") and any(
        m in labels_blob for m in _GARDENING_LABEL_MARKERS
    ):
        profiles.add("gardening")

    if cat in ("food", "lifestyle", "general") and any(m in labels_blob for m in _COOKING_LABEL_MARKERS):
        profiles.add("cooking")

    if cat in ("travel", "lifestyle", "general", "outdoors") and any(
        m in labels_blob for m in _CAMPING_LABEL_MARKERS
    ):
        profiles.add("camping")

    if any(m in labels_blob for m in _FISHING_LABEL_MARKERS):
        profiles.add("fishing")

    if cat in ("fitness", "sports", "lifestyle", "general") and any(
        m in labels_blob for m in _FITNESS_LABEL_MARKERS
    ):
        profiles.add("fitness")

    if cat in ("podcast", "music", "lifestyle", "general", "education", "business") and any(
        m in labels_blob for m in _PODCAST_LABEL_MARKERS
    ):
        profiles.add("podcast")

    if cat in ("beauty", "lifestyle", "general") and any(m in labels_blob for m in _BEAUTY_LABEL_MARKERS):
        profiles.add("beauty")

    return frozenset(profiles)


def merged_ambient_redundant_slugs(profiles: Optional[Iterable[str]] = None) -> FrozenSet[str]:
    """Union of global generic slugs plus profile-specific ambient furniture."""
    out: set[str] = set(GENERIC_VISION_LABEL_SLUGS)
    for profile in profiles or ():
        out.update(AMBIENT_REDUNDANT_SLUGS_BY_PROFILE.get(str(profile), frozenset()))
    return frozenset(out)


def is_ambient_redundant_vision_label(
    raw: Any,
    *,
    ambient_profiles: Optional[Iterable[str]] = None,
    min_specific_len: int = 4,
) -> bool:
    """True when the label is generic or ambient-always-present for active profiles."""
    slug = vision_label_slug(raw)
    if not slug or len(slug) < min_specific_len:
        return True
    if slug in GENERIC_VISION_LABEL_SLUGS:
        return True
    if not ambient_profiles:
        return False
    return slug in merged_ambient_redundant_slugs(ambient_profiles)


def is_redundant_vision_label(
    raw: Any,
    *,
    ambient_profiles: Optional[Iterable[str]] = None,
    min_specific_len: int = 4,
) -> bool:
    """Alias: generic OR profile-ambient redundant."""
    return is_ambient_redundant_vision_label(
        raw,
        ambient_profiles=ambient_profiles,
        min_specific_len=min_specific_len,
    )


def filter_vision_labels_for_hashtags(
    labels: Iterable[Any],
    *,
    min_specific_len: int = 4,
    ambient_profiles: Optional[Iterable[str]] = None,
    category: str = "",
    filename: str = "",
) -> List[str]:
    """Return original label strings that are safe to consider for hashtags."""
    profiles = ambient_profiles
    if profiles is None and (category or filename):
        profiles = resolve_ambient_profiles(
            category=category or "general",
            filename=filename,
            vision_label_names=labels,
        )
    out: List[str] = []
    seen: Set[str] = set()
    for raw in labels or []:
        text = str(raw or "").strip()
        if not text:
            continue
        slug = vision_label_slug(text)
        if not slug or slug in seen:
            continue
        if is_redundant_vision_label(
            text,
            ambient_profiles=profiles,
            min_specific_len=min_specific_len,
        ):
            continue
        seen.add(slug)
        out.append(text)
    return out


def filter_vision_labels_for_context(
    labels: Iterable[Any],
    *,
    category: str = "general",
    filename: str = "",
    min_specific_len: int = 4,
) -> List[str]:
    """Filter labels using dynamically resolved ambient profiles for this upload."""
    profiles = resolve_ambient_profiles(
        category=category,
        filename=filename,
        vision_label_names=labels,
    )
    return filter_vision_labels_for_hashtags(
        labels,
        min_specific_len=min_specific_len,
        ambient_profiles=profiles,
    )


def vision_labels_for_m8_scene_graph(
    labels: Iterable[Any],
    *,
    limit: int = 24,
    category: str = "general",
    filename: str = "",
) -> List[str]:
    """
    Labels exposed to the M8 scene graph: prefer landmarks/logos/OCR elsewhere;
    only pass non-generic label strings so the model does not copy detector noise.
    """
    return filter_vision_labels_for_context(
        labels,
        category=category,
        filename=filename,
    )[: max(0, int(limit or 24))]


def evidence_pool_has_strong_hashtag_signals(pool: Any) -> bool:
    """
    True when geo, VI, landmarks, logos, music, or highway OCR exist — vision
    object labels should not compete with those signals in evidence hashtags.
    """
    if pool is None:
        return False

    def _nonempty_str(val: Any) -> bool:
        return bool(str(val or "").strip())

    if _nonempty_str(getattr(pool, "road", None)):
        return True
    if _nonempty_str(getattr(pool, "city", None)) or _nonempty_str(getattr(pool, "state", None)):
        return True
    if _nonempty_str(getattr(pool, "gazetteer_place", None)):
        return True
    if _nonempty_str(getattr(pool, "protected_area", None)):
        return True
    if getattr(pool, "near_protected_land", False):
        return True
    if _nonempty_str(getattr(pool, "music_artist", None)) or _nonempty_str(getattr(pool, "music_title", None)):
        return True
    if getattr(pool, "vision_landmarks", None):
        return True
    if getattr(pool, "vision_logos", None):
        return True
    if getattr(pool, "vision_highways", None):
        return True
    if getattr(pool, "vi_logos", None):
        return True
    # Object tracks / detector label lists alone are NOT strong — those are the
    # weak taxonomy path we want to suppress when geo/brands/music exist.
    if getattr(pool, "vi_text_detections", None):
        # Only count text detections that look brand/route-like, not HUD dumps.
        for row in list(getattr(pool, "vi_text_detections", None) or [])[:8]:
            txt = ""
            if isinstance(row, dict):
                txt = str(row.get("text") or "")
            else:
                txt = str(row or "")
            if txt and not is_junk_hashtag_body(txt) and not is_generic_vision_label(txt):
                return True
    for kind in ("places", "products", "organizations", "people"):
        ents = (getattr(pool, "transcript_entities", None) or {}).get(kind) or []
        if ents:
            return True
    return False


def penalize_generic_vision_hashtags(
    tags: Iterable[Any],
    *,
    ambient_profiles: Optional[Iterable[str]] = None,
) -> float:
    """M8 scoring penalty: count generic / ambient-redundant vision slugs in hashtag list."""
    penalty = 0.0
    for t in tags or []:
        if is_redundant_vision_label(t, ambient_profiles=ambient_profiles):
            penalty += 4.0
    return penalty
