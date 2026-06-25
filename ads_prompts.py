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

    if not p and cat == "drink" and style == "enjoy":
        p = DRINK_ENJOY_DEFAULT
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
