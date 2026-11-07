"""Primary-school storybook look used by both render paths."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import Shot, Storyboard


PRIMARY_ILLUSTRATION_STYLE = (
    "high-quality 2D hand-painted children's storybook illustration, polished educational animation still, "
    "expressive but natural characters, rich layered scenery, clear foreground and background, warm cinematic light, "
    "vivid balanced colours, carefully rendered details, clean composition, appealing classroom-safe picture-book art, "
    "wide 16:9 frame, no words, no letters, no captions, no logos, no watermark"
)
PRIMARY_ILLUSTRATION_NEGATIVE = (
    "photograph, photorealistic, live action, documentary photo, 3D render, plastic texture, low resolution, "
    "blurry, muddy colours, deformed faces, extra limbs, duplicate people, text, letters, labels, captions, watermark, logo"
)

PRIMARY_GUIDE = (
    "When a guide character is useful, include the recurring friendly Indian schoolgirl guide: "
    "young child, warm brown skin, large expressive dark eyes, black hair in one braid, "
    "blue-and-white school uniform; keep her design, face, braid, and clothes consistent, "
    "place her at the left edge, and keep the lesson subject unobstructed."
)


@dataclass(frozen=True)
class TopicStyle:
    key: str
    prompt: str
    accent: str
    highlight: str


_TOPIC_STYLES = {
    "ecology": TopicStyle("ecology", "lush habitat storybook scene with layered leaves, gentle sunlight, and clear examples of living things", "39733F", "FFD65A"),
    "plants": TopicStyle("plants", "bright botanical picture-book scene with clearly shaped leaves, flowers, seeds, and plant forms", "43834A", "FFCF55"),
    "animals": TopicStyle("animals", "warm animal-adventure storybook scene with expressive, anatomically clear animals in their natural habitats", "168A91", "FFB36B"),
    "body": TopicStyle("body", "friendly health and human-body learning illustration with simple, gentle, non-scary visual explanations", "D76F65", "8EE0C0"),
    "space": TopicStyle("space", "wonder-filled sky and space picture-book scene with clean celestial shapes and deep blue gradients", "5268B7", "FFD45A"),
    "math": TopicStyle("math", "playful mathematics picture-book scene using large clear shapes, counters, and colorful classroom objects", "7655A6", "FFD45A"),
    "chemistry": TopicStyle("chemistry", "safe child-friendly chemistry storybook scene with colorful glassware and clearly separated materials", "8056A8", "B7E36A"),
    "physics": TopicStyle("physics", "hands-on physics picture-book scene with familiar objects, simple motion, and clear cause-and-effect", "3977B5", "FFB64F"),
    "earth": TopicStyle("earth", "bright earth-science picture-book scene with clear land, water, weather, and environment details", "39858A", "FFD45A"),
    "general": TopicStyle("general", "cheerful classroom storybook scene with a clear central learning idea and uncluttered composition", "347DA0", "FFD45A"),
}


def get_topic_style(key: str = "") -> TopicStyle:
    return _TOPIC_STYLES.get((key or "").strip().lower(), _TOPIC_STYLES["general"])

_TOPIC_RULES = (
    ("ecology", re.compile(r"\b(biodivers\w*|habitat\w*|ecosystem\w*|conservation|food chain|food web|environment)\b", re.I)),
    ("plants", re.compile(r"\b(plant\w*|tree\w*|shrub\w*|herb\w*|flower\w*|seed\w*|root\w*|leaf|leaves|photosynthesis|pollination)\b", re.I)),
    ("animals", re.compile(r"\b(animal\w*|mammal\w*|bird\w*|fish|insect\w*|reptile\w*|amphibian\w*|movement|habitat)\b", re.I)),
    ("body", re.compile(r"\b(body|health|heart|lung\w*|brain|digest\w*|skeleton|muscle\w*|sense\w*|nutrition)\b", re.I)),
    ("space", re.compile(r"\b(space|planet\w*|solar system|moon|star\w*|sun|orbit|universe|constellation)\b", re.I)),
    ("math", re.compile(r"\b(math\w*|number\w*|addition|subtract\w*|multipli\w*|divi\w*|fraction\w*|geometry|shape\w*|measure\w*)\b", re.I)),
    ("chemistry", re.compile(r"\b(chem\w*|mixture\w*|solution\w*|solid\w*|liquid\w*|gas\w*|matter|material\w*|reaction\w*)\b", re.I)),
    ("physics", re.compile(r"\b(force\w*|motion|speed|light|sound|electric\w*|magnet\w*|energy|heat|simple machine\w*)\b", re.I)),
    ("earth", re.compile(r"\b(earth|weather|cloud\w*|rain\w*|water cycle|rock\w*|soil|ocean\w*|river\w*|landform\w*)\b", re.I)),
)


def topic_style_for(subject: str = "", *texts: str) -> TopicStyle:
    """Choose a stable child-friendly art direction from the current scene's words and subject."""
    # The scene title gets extra weight; a broad subject such as Biology must not override a specific topic.
    title = texts[0] if texts else ""
    body = " ".join((subject, *texts))
    scored = []
    for order, (key, pattern) in enumerate(_TOPIC_RULES):
        title_hits = len(pattern.findall(title or ""))
        all_hits = len(pattern.findall(body))
        if all_hits:
            scored.append((title_hits * 3 + all_hits, title_hits, -order, key))
    if scored:
        return _TOPIC_STYLES[max(scored)[3]]
    subject_text = (subject or "").lower()
    for key, pattern in _TOPIC_RULES:
        if pattern.search(subject_text):
            return _TOPIC_STYLES[key]
    return _TOPIC_STYLES["general"]

_PHOTO_LANGUAGE = re.compile(
    r"(?i)\b(?:photorealistic|photo[- ]realistic|photographic|photograph|documentary\s+(?:photo|photograph|footage|still))\b"
)

_LAYOUT_PROMPT = re.compile(
    r"(?i)\b(?:content visual with|centered text|text\s+['\"“]|white text|white font|black band|dark background|"
    r"bold sans[- ]serif|subtitle band|transparent overlay|title card|title-card|display(?:ing)? text)\b"
)
_PROMPT_SCAFFOLDING = re.compile(
    r"(?i)\b(?:sharp\s+1920x1080\s+realistic documentary image|exact shot requirement|premium educational)\b"
)
_CAMERA_MOTION = re.compile(
    r"(?i)[,;.]?\s*camera\s+(?:slowly\s+)?(?:pans?|moves?|zooms?|tracks?)\b[^.;]*"
)


def _primary_subject(segment) -> str:
    """Choose a source-grounded still subject and discard text/layout or camera-motion instructions."""
    candidates = (segment.visual, segment.image_requirement, segment.image_prompt)
    for candidate in candidates:
        text = re.sub(r"\s+", " ", candidate or "").strip(" .;,:")
        if not text or _LAYOUT_PROMPT.search(text):
            continue
        text = _CAMERA_MOTION.sub("", text).strip(" .;,:")
        if len(text.split()) >= 5:
            return text
    return re.sub(r"\s+", " ", segment.narration or "").strip()


def illustration_subject_prompt(prompt: str) -> str:
    """Keep the storyboard's scene facts while removing directions that fight the illustration profile."""
    clean = _PHOTO_LANGUAGE.sub("", prompt or "")
    return re.sub(r"\s+", " ", clean).strip(" ,.;")


def apply_primary_storybook(board: Storyboard) -> bool:
    """Apply topic-aware storybook art and subtitle theme to primary lessons."""
    if not board.is_primary():
        return False
    for scene, segment in board.all_segments():
        style = topic_style_for(board.subject, scene.title, segment.heading, segment.narration,
                                segment.image_requirement, segment.visual)
        segment.topic_style = style.key
        segment.subtitle_style = "primary_karaoke"
        source_fields = (segment.visual or "", segment.image_requirement or "", segment.image_prompt or "")
        invalid_visual_contract = (
            any(_LAYOUT_PROMPT.search(value) or _CAMERA_MOTION.search(value) for value in source_fields)
            or any(_PROMPT_SCAFFOLDING.search(value) for value in source_fields[:2])
        )
        subject_prompt = _primary_subject(segment) if invalid_visual_contract else (
            segment.visual or segment.image_requirement or segment.image_prompt or segment.narration
        )
        if invalid_visual_contract and subject_prompt:
            segment.image_requirement = subject_prompt
            if (_LAYOUT_PROMPT.search(segment.visual or "") or _PROMPT_SCAFFOLDING.search(segment.visual or "")
                    or _CAMERA_MOTION.search(segment.visual or "")):
                segment.visual = subject_prompt
        shot = segment.shot
        if shot is None:
            continue
        if shot.template in {"process", "comparison"} and not segment.formula_lines:
            # For young learners, show the narrated scene as an illustration rather than
            # replacing the whole picture with abstract cards. The narration remains the source of truth.
            segment.steps = segment.steps or list(shot.steps)
            subject_prompt = illustration_subject_prompt(subject_prompt or segment.narration)
            segment.image_prompt = f"{subject_prompt}. {style.prompt}. {PRIMARY_GUIDE}"
            segment.image_prompt += (
                " Show the lesson's examples as two to four visually distinct, balanced picture areas; "
                "keep important objects in the upper two thirds so the renderer's exact caption cards remain readable below."
            )
            segment.visual_type = "realistic_image"
            segment.shot = Shot(template="photo", heading=shot.heading, asset_path=shot.asset_path)
        if segment.shot.template == "photo" and not segment.motion:
            segment.motion = "camera=slow_zoom_in"
        if segment.shot.template in {"photo", "video"}:
            if invalid_visual_contract or not segment.image_prompt.strip():
                segment.image_prompt = f"{illustration_subject_prompt(subject_prompt or segment.narration)}. {style.prompt}. {PRIMARY_GUIDE}"
            elif style.prompt not in segment.image_prompt:
                segment.image_prompt = f"{illustration_subject_prompt(segment.image_prompt)}. {style.prompt}. {PRIMARY_GUIDE}"
    return True
