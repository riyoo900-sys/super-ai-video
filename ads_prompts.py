"""Super AI Ads — prompt templates for global product commercials."""
from __future__ import annotations

ADS_REALISM_SUFFIX = (
    "professional product commercial, studio lighting, soft shadows, clean background, "
    "slow elegant camera motion, sharp product focus, premium advertising look, "
    "4K commercial quality, photorealistic, cinematic color grading"
)

ADS_NEGATIVE_EXTRA = (
    "cartoon, illustration, messy background, cluttered scene, shaky camera, "
    "low budget, amateur, blurry product, distorted logo, unreadable text, watermark"
)

CATEGORY_PREFIX: dict[str, str] = {
    "perfume": "Luxury perfume commercial, glass bottle hero shot,",
    "tech": "Premium tech product commercial, sleek device showcase,",
    "app": "Mobile app commercial, phone screen showcase, clean UI presentation,",
    "drink": "Premium beverage commercial, cold drink bottle hero shot, condensation,",
    "shoes": "Sneaker commercial, athletic footwear hero shot, dynamic product focus,",
    "fashion": "High-end fashion commercial, elegant apparel presentation,",
    "food": "Gourmet food commercial, appetizing product shot,",
    "watch": "Luxury watch commercial, precision timepiece close-up,",
    "skincare": "Beauty skincare commercial, clean cosmetic product shot,",
    "auto": "Premium product commercial, hero product centered,",
}


def normalize_category(category: str | None) -> str:
    key = (category or "auto").strip().lower()
    return key if key in CATEGORY_PREFIX else "auto"


def build_ads_prompt(user_prompt: str, category: str | None = None, *, has_product_image: bool = False) -> str:
    p = user_prompt.strip()
    if not p:
        return p
    cat = normalize_category(category)
    prefix = CATEGORY_PREFIX[cat]
    lower = p.lower()
    image_note = (
        " keep the uploaded product appearance faithful, logo and colors accurate,"
        if has_product_image
        else ""
    )
    if "commercial" in lower or "product shot" in lower or "advertising" in lower:
        return f"{p},{image_note} {ADS_REALISM_SUFFIX}"
    return f"{prefix} {p},{image_note} {ADS_REALISM_SUFFIX}"


def ads_negative_prompt(base_negative: str) -> str:
    return f"{base_negative}, {ADS_NEGATIVE_EXTRA}"
