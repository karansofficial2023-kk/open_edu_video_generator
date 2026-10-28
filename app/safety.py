"""School-student safety gate: no nudity, bare bodies, sexual or graphic content may reach the screen, whatever the lesson topic.

One place for the rules so pictures, motion clips and the final video are judged the same way. Every check fails CLOSED: when the
vision model is unavailable or answers nonsense, the image counts as unsafe (a person must look at it)."""
from __future__ import annotations

from . import vision
from .config import AppConfig

SAFETY_NEGATIVE = ("nude, nudity, naked, topless, shirtless, bare chest, exposed genitals, breasts, nipples, buttocks, underwear, lingerie, bikini, "
                   "sexual, erotic, suggestive, adult content, pornographic, nsfw, fetish, gore, graphic injury, blood splatter, bare skin, "
                   "bare shoulders, midriff, human torso, body outline")

SAFETY_ASK = ("Look carefully at this image from a video for school students. Answer with JSON keys: "
              "adult_content (bool: TRUE if it shows nudity, a bare or partly undressed human body, breasts, buttocks, genitals, underwear or swimwear, "
              "anything sexual, erotic or suggestive, or an adult-content / nsfw style image), "
              "body_visible (bool: TRUE if any human body part other than a face or hands is visible, including a torso, abdomen, back, legs, bare skin, "
              "a body silhouette or outline around drawn organs, or a statue / engraving / painting of a bare body), "
              "body_outline (bool: TRUE if drawn or rendered organs sit inside, or are surrounded by, a translucent or solid human body shape - hips, thighs, "
              "waist, legs, torso or buttocks outline - instead of standing alone on a plain background), "
              "clothed_only (bool: TRUE only if every visible person is fully covered by clothing with shoulders, arms and legs covered), "
              "graphic (bool: TRUE if blood, wounds or gore are visible). Isolated organs, cells, molecules, plants, animals, diagrams on plain "
              "backgrounds and objects are fine. Be strict.")


def with_safety_negative(negative: str) -> str:
    """The configured negative prompt plus the safety terms (added once, whatever the YAML says)."""
    base = (negative or "").strip().rstrip(",")
    return f"{base}, {SAFETY_NEGATIVE}" if base else SAFETY_NEGATIVE


def _flag(value) -> bool:
    return value is True or str(value).strip().lower() in {"true", "yes", "1"}


def unsafe_reason(image, config: AppConfig, purpose: str = "safety", passes: int = 1) -> str | None:
    """Reason the image must not be shown to students, or None when it is fine. Fails closed.
    The vision model is not deterministic, so `passes` > 1 asks again and any single unsafe answer rejects the image."""
    for attempt in range(max(1, passes)):
        reason = _unsafe_once(image, config, purpose if attempt == 0 else f"{purpose}-recheck{attempt}")
        if reason:
            return reason
    return None


def _unsafe_once(image, config: AppConfig, purpose: str) -> str | None:
    try:
        result = vision.ask(image, SAFETY_ASK, config, purpose=purpose)
    except Exception as exc:                        # model down, invalid JSON, unreadable file: unknown means unsafe
        return f"unsafe: safety check could not run ({str(exc)[:80]})"
    if not isinstance(result, dict) or not any(k in result for k in ("adult_content", "body_visible", "graphic")):
        return "unsafe: safety check returned no usable answer"
    if _flag(result.get("body_outline")):
        return "unsafe: organs drawn inside a human body outline"
    if _flag(result.get("adult_content")):
        return "unsafe: nudity or adult content"
    if _flag(result.get("graphic")):
        return "unsafe: visible blood or graphic content"
    if _flag(result.get("body_visible")) and not _flag(result.get("clothed_only")):
        return "unsafe: human body that is not fully clothed"
    return None
