"""Super AI Ads — prompt templates for global product commercials."""
from __future__ import annotations

ADS_REALISM_SUFFIX = (
    "photorealistic, cinematic lighting, natural smooth motion, sharp focus, "
    "realistic textures, lifelike, high detail, professional color grading"
)

ADS_NEGATIVE_EXTRA = (
    "cartoon, illustration, messy background, cluttered scene, shaky camera, "
    "low budget, amateur, blurry product, distorted logo, unreadable text, watermark, "
    "deformed hands, extra fingers, uncanny face"
)

SCENE_STYLE_SUFFIX: dict[str, str] = {
    "product": (
        "professional product commercial, studio lighting, soft shadows, clean background, "
        "slow elegant camera motion, sharp product focus, premium advertising look"
    ),
    "lifestyle": (
        "authentic lifestyle commercial, real-world setting, natural human energy, "
        "warm inviting mood, relatable everyday moment, cinematic storytelling"
    ),
    "enjoy": (
        "satisfying consumption moment, refreshing drink pour or sip, thirst-quenching appeal, "
        "appetizing close-up, condensation and cold fizz, viewer desires the product, "
        "joyful indulgence, mouth-watering presentation"
    ),
    "benefits": (
        "compelling product benefits showcase, clear value for the customer, persuasive marketing "
        "storytelling, viewer understands why to buy, highlight key features and emotional payoff, "
        "before-and-after feeling, problem solved, satisfied user reaction"
    ),
    "on_model": (
        "real person wearing or using the exact product, full body or natural close-up, "
        "authentic human model, confident natural movement, product clearly visible on the person, "
        "fashion editorial commercial, believable lifestyle moment"
    ),
}

CATEGORY_PREFIX: dict[str, str] = {
    "perfume": "Luxury perfume commercial, glass bottle hero shot,",
    "tech": "Premium tech product commercial, sleek device showcase,",
    "app": "Mobile app commercial, phone screen showcase, clean UI presentation,",
    "drink": "Refreshing beverage commercial, cold drink bottle hero shot,",
    "shoes": "Sneaker commercial, athletic footwear hero shot, dynamic product focus,",
    "fashion": "High-end fashion commercial, elegant apparel presentation,",
    "food": "Gourmet food commercial, appetizing product shot,",
    "watch": "Luxury watch commercial, precision timepiece close-up,",
    "skincare": "Beauty skincare commercial, clean cosmetic product shot,",
    "auto": "Premium product commercial, hero product centered,",
}

DRINK_ENJOY_DEFAULT = (
    "cold cola bottle with ice, slow pour into glass, fizzy bubbles rising, "
    "person takes a refreshing sip and smiles with satisfaction, summer vibe"
)

APP_BENEFITS_DEFAULT = (
    "smartphone showing the app interface, user discovers key features one by one, "
    "happy customer saves time and achieves their goal, clear benefit messaging, "
    "modern app promo commercial"
)

SHOES_ON_MODEL_DEFAULT = (
    "athletic person lacing up and wearing these exact sneakers, walking then light jogging "
    "on city street, full body shot, confident natural stride, product clearly on feet"
)

SHOES_PRODUCT_DEFAULT = (
    "white sneakers floating, soft shadow, elegant slow 360 orbit, premium studio lighting"
)

FASHION_ON_MODEL_DEFAULT = (
    "elegant model wearing this outfit, confident walk toward camera, golden hour street, "
    "fabric flowing naturally, fashion editorial look"
)

WATCH_ON_MODEL_DEFAULT = (
    "professional person glances at luxury watch on wrist, subtle satisfied smile, "
    "modern office lobby, product detail sharp on wrist"
)

PERFUME_ON_MODEL_DEFAULT = (
    "sophisticated person sprays perfume on wrist, slow motion mist, evening attire, "
    "luxurious intimate mood, bottle and skin detail sharp"
)

FOOD_ENJOY_DEFAULT = (
    "chef presents hot dish, steam rising, appetizing close-up bite, warm kitchen lighting, "
    "satisfied smile after tasting"
)

SKINCARE_BENEFITS_DEFAULT = (
    "person applies skincare product gently, glowing healthy skin, mirror reflection, "
    "fresh confident smile, clean bathroom light"
)

TECH_BENEFITS_DEFAULT = (
    "user unboxes sleek device, discovers key features, productive happy workflow, "
    "modern desk setup, premium tech commercial"
)

# (category, scene_style) -> default prompt when user leaves the field empty.
DEFAULT_PROMPTS: dict[tuple[str, str], str] = {
    ("drink", "enjoy"): DRINK_ENJOY_DEFAULT,
    ("food", "enjoy"): FOOD_ENJOY_DEFAULT,
    ("app", "benefits"): APP_BENEFITS_DEFAULT,
    ("tech", "benefits"): TECH_BENEFITS_DEFAULT,
    ("skincare", "benefits"): SKINCARE_BENEFITS_DEFAULT,
    ("shoes", "on_model"): SHOES_ON_MODEL_DEFAULT,
    ("shoes", "product"): SHOES_PRODUCT_DEFAULT,
    ("fashion", "on_model"): FASHION_ON_MODEL_DEFAULT,
    ("watch", "on_model"): WATCH_ON_MODEL_DEFAULT,
    ("perfume", "on_model"): PERFUME_ON_MODEL_DEFAULT,
    ("shoes", "lifestyle"): SHOES_ON_MODEL_DEFAULT,
    ("fashion", "lifestyle"): FASHION_ON_MODEL_DEFAULT,
    ("fashion", "benefits"): FASHION_ON_MODEL_DEFAULT,
}


def normalize_category(category: str | None) -> str:
    key = (category or "auto").strip().lower()
    return key if key in CATEGORY_PREFIX else "auto"


def normalize_scene_style(style: str | None) -> str:
    key = (style or "product").strip().lower()
    return key if key in SCENE_STYLE_SUFFIX else "product"


def build_ads_prompt(
    user_prompt: str,
    category: str | None = None,
    *,
    scene_style: str | None = "product",
    has_product_image: bool = False,
) -> str:
    p = user_prompt.strip()
    cat = normalize_category(category)
    style = normalize_scene_style(scene_style)

    if not p:
        p = DEFAULT_PROMPTS.get((cat, style), "")
    if not p:
        return p

    prefix = CATEGORY_PREFIX[cat]
    scene = SCENE_STYLE_SUFFIX[style]
    image_note = (
        " animate the uploaded product faithfully, keep logo shape and brand colors accurate,"
        if has_product_image
        else ""
    )
    lower = p.lower()
    if "commercial" in lower or "product shot" in lower or "advertising" in lower:
        return f"{p},{image_note} {scene}, {ADS_REALISM_SUFFIX}"
    return f"{prefix} {p},{image_note} {scene}, {ADS_REALISM_SUFFIX}"


def ads_negative_prompt(base_negative: str) -> str:
    return f"{base_negative}, {ADS_NEGATIVE_EXTRA}"
