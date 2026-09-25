from __future__ import annotations

import re

from .schema import Shot, Storyboard


REPLACEMENTS = {
    "glystogamy": "cleistogamy",
    "gystogamy": "cleistogamy",
    "gynostegy": "cleistogamy",
    "Glystogamy": "Cleistogamy",
    "Gystogamy": "Cleistogamy",
    "Gynostegy": "Cleistogamy",
    "protogenic": "protogynous",
    "Protogenic": "Protogynous",
    "hetrostyle": "heterostyly",
    "Hetrostyle": "Heterostyly",
    "Xora": "Ixora",
}

UNSAFE_CURL_CLAIM = re.compile(
    r"(?:the\s+)?(?:male\s+(?:part|anther)|anther)(?:\s+of\s+the\s+flower)?\s+"
    r"(?:curls?|bends?)\s+(?:inward|in\s+on\s+itself)\s+"
    r"to\s+(?:transfer|deposit)\s+pollen\s+(?:on|onto)\s+"
    r"(?:the\s+)?(?:sticky\s+)?(?:female\s+(?:part|structure|stigma)|stigma)",
    re.IGNORECASE,
)


def normalize_storyboard(storyboard: Storyboard) -> None:
    """Make imported legacy storyboards renderable without silently executing source instructions."""
    storyboard.title = _clean_text(storyboard.title)
    storyboard.source = _clean_text(storyboard.source)
    for scene, segment in storyboard.all_segments():
        scene.title = _clean_text(scene.title)
        scene.narration = _clean_text(scene.narration)
        segment.narration = _clean_text(segment.narration)
        segment.visual = _clean_text(segment.visual)
        segment.image_prompt = _clean_text(segment.image_prompt)
        segment.keywords = [
            cleaned for keyword in segment.keywords
            if (cleaned := _clean_text(keyword).lower()) not in {"curl", "curls", "curling", "inward"}
        ]
        if not segment.source_references:
            segment.source_references = ["Imported storyboard; verify against curriculum before publication."]
        if segment.shot is None:
            segment.shot = _infer_shot(scene.title, segment.narration)
        else:
            segment.shot.heading = _clean_text(segment.shot.heading)
            if re.search(r"\b(?:curl|curls|curling)\b", segment.shot.heading, re.IGNORECASE):
                segment.shot.heading = _heading_from_text(segment.narration)
            segment.shot.learning_objective = _clean_text(segment.shot.learning_objective)
            segment.shot.steps = [_clean_text(step) for step in segment.shot.steps]
            segment.shot.cues = [_clean_text(cue) for cue in segment.shot.cues]
            segment.shot.motion = {
                key: _clean_text(value) if isinstance(value, str) else value
                for key, value in segment.shot.motion.items()
            }
            segment.shot = _refine_imported_shot(scene.title, segment)
            segment.shot = _ensure_transfer_labels(segment)
    _ground_ambiguous_visuals(storyboard)



def _refine_imported_shot(scene_title: str, segment) -> Shot:
    """Replace generic imported card shots with safer subject-specific visuals."""
    shot = segment.shot
    if shot and shot.template in {
        "title_card", "labeled_image", "realistic_labeled_image",
        "realistic_background_with_labels", "diagram_overlay",
        "split_screen", "formula", "video_broll",
    }:
        return shot
    text = " ".join([
        scene_title or "",
        segment.narration or "",
        segment.visual or "",
        segment.image_prompt or "",
        " ".join(segment.keywords or []),
        shot.heading if shot else "",
        shot.learning_objective if shot else "",
        " ".join(shot.steps if shot else []),
    ]).lower()

    if "self-pollination" in text and "cross-pollination" in text and ("summary" in text or "comparing" in text):
        return Shot(
            template="classification",
            heading="Types Of Pollination",
            learning_objective="Separate self-pollination from cross-pollination.",
            steps=["Self-pollination", "Cross-pollination"],
            stage_fractions=[0, 0.5],
        )
    if shot and shot.template == "process_steps":
        return shot.model_copy(update={"template": "realistic_labeled_image"})
    if "cleistogamy" in text or "unopened flower" in text or "closed flower" in text:
        return Shot(
            template="photo",
            heading="Cleistogamy",
            learning_objective="Show a real closed cleistogamous flower clearly.",
        )
    if "protandry" in text or "anther matures first" in text or "anthers mature before" in text:
        return Shot(
            template="photo",
            heading="Protandry",
            learning_objective="Show the real flower structures used to explain protandry.",
        )
    if "protogyn" in text or "stigma matures first" in text or "stigma maturation first" in text:
        return Shot(
            template="photo",
            heading="Protogyny",
            learning_objective="Show the real flower structures used to explain protogyny.",
        )
    if "bee orchid" in text or "female bee mimic" in text:
        segment.image_prompt = (
            "Macro documentary photograph of a bee orchid flower that resembles a female bee, "
            "with a male bee nearby, realistic botanical detail, natural daylight, no text, no labels."
        )
        return Shot(
            template="photo",
            heading="Bee Orchid Mimicry",
            learning_objective="Show how deceptive mimicry can attract a pollinator.",
        )
    return shot


REAL_IMAGE_TEMPLATES = {"pollination", "insect", "wind", "water", "agents", "life_cycle"}
VIDEO_TEMPLATES = {"insect", "wind"}
VIDEO_KEYWORDS = {
    "bee": 7,
    "bees": 7,
    "insect": 7,
    "insects": 7,
    "butterfly": 6,
    "butterflies": 6,
    "bird": 6,
    "birds": 6,
    "bat": 6,
    "bats": 6,
    "animal": 5,
    "animals": 5,
    "pollinator": 7,
    "pollinators": 7,
    "entomophily": 7,
    "anemophily": 6,
    "hydrophily": 6,
    "wind": 5,
    "water": 5,
    "aquatic": 5,
    "flower": 3,
    "flowers": 3,
    "pollen": 3,
    "nectar": 4,
    "orchid": 5,
    "meadow": 4,
    "sunflower": 4,
    "cross-pollination": 5,
}
DIAGRAM_FIRST_TERMS = {
    "classification",
    "classified",
    "compare",
    "advantages",
    "disadvantages",
    "protandry",
    "protogyny",
    "cleistogamy",
    "autogamy",
    "geitonogamy",
    "sequence",
    "earlier",
    "later",
}


def promote_real_image_shots(storyboard: Storyboard, config) -> None:
    """Use HD generated photos for real-world biology scenes from imported storyboards."""
    comfy = getattr(config, "comfyui", None)
    if not comfy or not comfy.enabled or not getattr(comfy, "real_image_auto_promote", True):
        return
    for scene, segment in storyboard.all_segments():
        shot = segment.shot
        if shot is None or shot.template in {"photo", "video"} or shot.asset_path:
            continue
        if shot.template not in REAL_IMAGE_TEMPLATES:
            continue
        score = _real_image_candidate_score(scene.title, segment)
        if score <= 0:
            continue
        segment.image_prompt = _photo_prompt_from_segment(scene.title, segment)
        segment.shot = Shot(
            template="photo",
            heading=shot.heading or _heading_from_text(segment.narration),
            learning_objective=shot.learning_objective or "Show this concept with a realistic educational visual.",
        )


def promote_video_shots(storyboard: Storyboard, config) -> None:
    """Use Wan only for a small number of simple motion shots.

    Wan text-to-video can look impressive, but it is unreliable for exact labelled
    science. Real HD images and deterministic diagrams are safer for most educational
    scenes. Automatic Wan promotion is limited to biology/pollination-style natural
    motion; other subjects use Wan only if the source storyboard explicitly requests it.
    """
    _keep_only_explicit_video_shots_for_non_nature_topic(storyboard)
    comfy = getattr(config, "comfyui", None)
    if not comfy or not comfy.enabled or not comfy.video_enabled or not comfy.video_auto_promote:
        return
    max_segments = max(0, int(comfy.video_max_segments or 0))
    if max_segments == 0 or not comfy.video_workflow_path:
        return
    if not _allows_automatic_wan(storyboard):
        return

    candidates = []
    for order, (scene, segment) in enumerate(storyboard.all_segments()):
        shot = segment.shot
        if shot is None or shot.template == "video" or shot.asset_path:
            continue
        score = _video_candidate_score(scene.title, segment)
        if score > 0:
            candidates.append((score, order, segment, shot))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, _, segment, shot in candidates[:max_segments]:
        prompt = _video_prompt_from_segment(segment)
        if prompt:
            segment.image_prompt = prompt
        segment.shot = Shot(
            template="video",
            heading=shot.heading or _heading_from_text(segment.narration),
            learning_objective=shot.learning_objective or "Show the idea as a short cinematic educational clip.",
        )


def _demote_unreliable_generated_video(scene_title: str, segment) -> None:
    """Prefer stable photo visuals where local T2V is likely to hallucinate anatomy."""
    shot = segment.shot
    if shot is None or shot.template != "video" or shot.asset_path:
        return
    text = " ".join([
        scene_title or "",
        segment.narration or "",
        segment.visual or "",
        segment.image_prompt or "",
        shot.heading or "",
        shot.learning_objective or "",
    ]).lower()
    risky_terms = {
        "template: video", "bee", "bees", "insect", "insects", "butterfly", "pollinator", "pollinators",
        "orchid", "mimicry", "diagram", "label", "labeled", "anther", "stigma",
    }
    if not any(term in text for term in risky_terms):
        return
    segment.image_prompt = _photo_prompt_from_segment(scene_title, segment)
    segment.shot = Shot(
        template="photo",
        heading=shot.heading or _heading_from_text(segment.narration),
        learning_objective=shot.learning_objective or "Show this concept with a realistic educational visual.",
    )
def _contains_term(text: str, term: str) -> bool:
    pattern = r"(?<![a-z])" + re.escape(term.lower()).replace(r"\ ", r"\s+") + r"(?![a-z])"
    return re.search(pattern, text.lower()) is not None


def _keep_only_explicit_video_shots_for_non_nature_topic(storyboard: Storyboard) -> None:
    if _allows_automatic_wan(storyboard):
        return
    for _, segment in storyboard.all_segments():
        if segment.shot and segment.shot.template == "video" and not _source_explicitly_requested_video(segment):
            segment.shot = Shot(
                template="photo",
                heading=segment.shot.heading or _heading_from_text(segment.narration),
                learning_objective=segment.shot.learning_objective or "Show this segment with a realistic educational visual.",
            )


def _allows_automatic_wan(storyboard: Storyboard) -> bool:
    text = " ".join([storyboard.title or "", storyboard.source or ""]).lower()
    return any(_contains_term(text, term) for term in [
        "pollination", "flower", "bee", "insect", "butterfly", "wind pollination", "plant"
    ])


def _source_explicitly_requested_video(segment) -> bool:
    text = " ".join([segment.visual or "", segment.image_prompt or ""]).lower()
    return "wan video" in text or "template: video" in text


def _real_image_candidate_score(scene_title: str, segment) -> int:
    shot = segment.shot
    text = " ".join([
        scene_title or "",
        segment.narration or "",
        segment.visual or "",
        segment.image_prompt or "",
        " ".join(segment.keywords or []),
        shot.heading if shot else "",
        shot.learning_objective if shot else "",
    ]).lower()
    score = 0
    if shot and shot.template in REAL_IMAGE_TEMPLATES:
        score += 4
    for word, weight in VIDEO_KEYWORDS.items():
        if _contains_term(text, word):
            score += max(1, weight // 2)
    for term in DIAGRAM_FIRST_TERMS:
        if term in text:
            score -= 6
    return score


def _photo_prompt_from_segment(scene_title: str, segment) -> str:
    text = " ".join([scene_title or "", segment.narration or "", segment.visual or "", segment.image_prompt or ""]).lower()
    if "hydrophily" in text or "water" in text or "aquatic" in text:
        subject = "macro documentary photograph of aquatic flowering plants at the water surface, small pollen grains floating across calm water toward another flower"
    elif "anemophily" in text or "wind" in text:
        subject = "documentary photograph of grass flowers releasing fine pollen into a light breeze, natural outdoor daylight, shallow depth of field"
    elif "entomophily" in text or "bee" in text or "insect" in text or "butterfl" in text:
        subject = "macro documentary photograph of a bee visiting a bright flower, pollen grains visible on its body, realistic petals and anthers"
    elif "bird" in text or "ornithophily" in text:
        subject = "documentary photograph of a sunbird or hummingbird feeding from a tubular flower, pollen transfer suggested naturally"
    elif "bat" in text or "chiropterophily" in text:
        subject = "night documentary photograph of a bat pollinating a pale night-blooming flower, realistic natural history lighting"
    elif "cross-pollination" in text:
        subject = "realistic educational photograph of two flowering plants of the same species, pollinator moving between flowers, pollen transfer implied"
    elif "life cycle" in text or "seed" in text or "fruit" in text:
        subject = "realistic photograph of a flowering plant with developing fruit and seeds nearby, natural daylight, educational biology context"
    else:
        subject = segment.image_prompt or segment.visual or segment.narration
    return (
        f"{subject}. High resolution realistic educational biology photograph, accurate natural flower structures, "
        "single clear scene, no labels, no text, no arrows, no diagram, no collage."
    )


def _video_candidate_score(scene_title: str, segment) -> int:
    shot = segment.shot
    text = " ".join([
        scene_title or "",
        segment.narration or "",
        segment.visual or "",
        segment.image_prompt or "",
        " ".join(segment.keywords or []),
        shot.heading if shot else "",
        shot.learning_objective if shot else "",
    ]).lower()
    score = 0
    if shot and shot.template in VIDEO_TEMPLATES:
        score += 3
    if shot and shot.template == "photo" and any(_contains_term(text, word) for word in ["bee", "insect", "butterfly", "butterflies", "wind"]):
        score += 4
    if any(_contains_term(text, word) for word in ["hydrophily", "aquatic", "floating pollen", "water surface", "water pollination"]):
        score -= 20
    for word, weight in VIDEO_KEYWORDS.items():
        if _contains_term(text, word):
            score += weight
    for term in DIAGRAM_FIRST_TERMS:
        if term in text:
            score -= 5
    if shot and shot.template in {"classification", "comparison", "process", "pros_cons", "protandry", "protogyny"}:
        score -= 4
    return score


def _video_prompt_from_segment(segment) -> str:
    text = " ".join([segment.narration or "", segment.visual or "", segment.image_prompt or ""]).lower()
    if "bee" in text or "insect" in text or "entomophily" in text or "butterfl" in text:
        return (
            "Macro nature video of a bee slowly visiting a single bright flower, pollen dust visible on the bee body, "
            "realistic petals and anthers, shallow depth of field, gentle natural movement, no text."
        )
    if "wind" in text or "anemophily" in text:
        return (
            "Realistic nature video of grass flowers in a light breeze releasing fine pollen, soft daylight, "
            "stable close camera, gentle plant movement, no text."
        )
    return (
        "Realistic macro nature video of a flower with subtle natural movement, soft daylight, stable camera, "
        "accurate plant details, no text."
    )


def _clean_text(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("\ufffd", "-")
    for old, new in REPLACEMENTS.items():
        cleaned = re.sub(rf"\b{re.escape(old)}\b", new, cleaned)
    cleaned = UNSAFE_CURL_CLAIM.sub(
        "pollen grains from the anther are transferred to the receptive stigma",
        cleaned,
    )
    teaser_pattern = re.compile(
        r"while\s+([^.!?]+?),\s+(?:they|it|these|those)\s+have\s+(?:a\s+)?(?:clever\s+)?"
        r"strategy\s+to\s+[^.!?]+[.!?]?",
        flags=re.IGNORECASE,
    )
    def replace_teaser(match: re.Match[str]) -> str:
        statement = match.group(1).strip().rstrip(".")
        return statement[:1].upper() + statement[1:] + "."
    cleaned = teaser_pattern.sub(replace_teaser, cleaned)
    return cleaned


AMBIGUOUS_VISUAL_TEXT = re.compile(
    r"^(?:at\s+this|this|these|those|they|them|it|here|there)\b|"
    r"source-faithful\s+visible\s+subject|show\s+the\s+narrated\s+concept|"
    r"educational\s+(?:image|visual)$",
    re.IGNORECASE,
)


def _ground_ambiguous_visuals(storyboard: Storyboard) -> None:
    previous_narration = storyboard.title
    for _scene, segment in storyboard.all_segments():
        narration = (segment.narration or "").strip()
        prompt = (segment.image_prompt or "").strip()
        visual = (segment.visual or "").strip()
        if segment.shot and segment.shot.template != "title_card":
            narration_is_ambiguous = bool(AMBIGUOUS_VISUAL_TEXT.search(narration))
            metadata_is_ambiguous = bool(
                AMBIGUOUS_VISUAL_TEXT.search(prompt) or AMBIGUOUS_VISUAL_TEXT.search(visual)
            )
            if narration_is_ambiguous:
                subject_context = (
                    "Continue the exact physical subject established by the immediately preceding narration: "
                    f"{previous_narration}. "
                )
            else:
                subject_context = ""
            if narration_is_ambiguous or metadata_is_ambiguous:
                relationship_guard = ""
                lowered = narration.lower()
                if "self-pollination" in lowered and "anther" in lowered and "stigma" in lowered:
                    relationship_guard = (
                        " Show one complete flower with both the anther and stigma visible; "
                        "no bee, insect, animal, hand, or other external transfer agent."
                    )
                segment.image_prompt = (
                    f"{subject_context}Show the current narration literally: {narration}. "
                    f"{prompt}{relationship_guard} No unrelated substitute subject, no symbolic stand-in, "
                    "no text, no labels."
                )
            if segment.shot.labels:
                names = [
                    str(item.get("text") or item.get("name") or "").strip()
                    for item in segment.shot.labels
                ]
                names = [name for name in names if name]
                if names:
                    segment.image_prompt = (
                        f"{segment.image_prompt} Stable macro scientific reference photograph of one complete "
                        f"subject with every requested physical structure simultaneously visible and sharply "
                        f"distinguishable: {', '.join(names)}. Keep the full subject inside the frame and reserve "
                        "clear side margins for later composited labels; do not generate label text or arrows."
                    )
        if narration and not AMBIGUOUS_VISUAL_TEXT.search(narration):
            previous_narration = narration


def _ensure_transfer_labels(segment) -> Shot:
    shot = segment.shot
    if shot is None or shot.template in {"title_card", "video", "video_broll", "short_motion_clip"}:
        return shot
    if shot.labels:
        return shot
    match = re.search(
        r"\bfrom\s+(?:the\s+)?([a-z][a-z -]{1,45}?)\s+(?:is|are)\s+"
        r"transferred\s+to\s+(?:the\s+)?([a-z][a-z -]{1,45}?)(?:[,.;]|$)",
        segment.narration or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return shot
    source = match.group(1).strip()
    receiver = match.group(2).strip()
    labels = [
        {"text": source.title(), "placement": f"exact visible {source} named as the transfer source"},
        {"text": receiver.title(), "placement": f"exact visible {receiver} named as the transfer receiver"},
    ]
    return shot.model_copy(update={
        "template": "realistic_labeled_image",
        "labels": labels,
        "motion": {**shot.motion, "type": "arrow_draw_then_label_fade"},
    })


def _infer_shot(scene_title: str, narration: str) -> Shot:
    text = f"{scene_title} {narration}".lower()
    if "life cycle" in text or "seed" in text or "fruit" in text or "fertilization" in text:
        return Shot(
            template="life_cycle",
            heading="Pollination In The Plant Life Cycle",
            learning_objective="Connect pollination to fertilization, seeds, and fruits.",
            stage_fractions=[0, 0.52],
        )
    if "classified" in text or "classification" in text or ("self-pollination" in text and "cross-pollination" in text):
        return Shot(
            template="classification",
            heading="Types Of Pollination",
            learning_objective="Separate self-pollination from cross-pollination.",
            steps=["Self-pollination", "Cross-pollination"],
            stage_fractions=[0, 0.5],
        )
    if "autogamy" in text or "geitonogamy" in text or "same flower" in text or "same plant" in text:
        return Shot(
            template="classification",
            heading="Self-Pollination",
            learning_objective="Compare autogamy and geitonogamy.",
            steps=["Autogamy: same flower", "Geitonogamy: same plant"],
            stage_fractions=[0, 0.5],
        )
    if "cross-pollination" in text or "different plant" in text or "different flowers" in text:
        return Shot(
            template="pollination",
            heading="Cross-Pollination",
            learning_objective="Show pollen transfer between flowers on different plants.",
            stage_fractions=[0, 0.52],
        )
    if "anemophily" in text or "wind" in text or "feathery" in text or "airborne" in text:
        return Shot(
            template="wind",
            heading="Anemophily",
            learning_objective="Show how wind carries light pollen to a feathery stigma.",
            stage_fractions=[0, 0.52],
        )
    if "hydrophily" in text or "vallisneria" in text or "hydrilla" in text or "water" in text:
        return Shot(
            template="water",
            heading="Hydrophily",
            learning_objective="Show pollen transfer through water in aquatic plants.",
            stage_fractions=[0, 0.52],
        )
    if "entomophily" in text or "insect" in text or "bee" in text or "butterfl" in text:
        return Shot(
            template="insect",
            heading="Entomophily",
            learning_objective="Show insect attraction and pollen attachment.",
            stage_fractions=[0, 0.52],
        )
    if any(word in text for word in ["ornithophily", "chiropterophily", "zoophily", "agents", "birds", "bats", "animals"]):
        return Shot(
            template="agents",
            heading="Pollination Agents",
            learning_objective="Compare major agents of cross-pollination.",
            steps=["Wind", "Water", "Insects", "Birds, bats, animals"],
            stage_fractions=[0, 0.25, 0.5, 0.75],
        )
    if "advantages" in text or "disadvantages" in text or "genetic diversity" in text or "pure lines" in text:
        return Shot(
            template="pros_cons",
            heading="Advantages And Limits",
            learning_objective="Compare benefits and limitations.",
            steps=["Advantages", "Disadvantages"],
            stage_fractions=[0, 0.5],
        )
    if "protandry" in text or "anther matures first" in text:
        return Shot(
            template="protandry",
            heading="Protandry",
            learning_objective="Show that the male part matures before the female part.",
            stage_fractions=[0, 0.52],
        )
    if "protogyn" in text or "stigma matures first" in text:
        return Shot(
            template="protogyny",
            heading="Protogyny",
            learning_objective="Show that the female part matures before the male part.",
            stage_fractions=[0, 0.52],
        )
    if "pollination" in text or "pollen" in text or "anther" in text or "stigma" in text:
        return Shot(
            template="pollination",
            heading="Pollination",
            learning_objective="Show pollen transfer to a receptive stigma.",
            stage_fractions=[0, 0.55],
        )
    if "unisexual" in text or ("male" in text and "female" in text):
        return Shot(
            template="comparison",
            heading="Flower Types",
            learning_objective="Compare male and female flower roles.",
            steps=["Male flower produces pollen", "Female flower bears stigma and ovary"],
            stage_fractions=[0, 0.5],
        )
    if "cleistogamy" in text or "closed" in text:
        return Shot(
            template="process",
            heading="Cleistogamy",
            learning_objective="Show self-pollination inside a closed flower.",
            steps=["Flower remains closed", "Self-pollination occurs inside"],
            stage_fractions=[0, 0.52],
        )
    if "bee orchid" in text or "mimics" in text:
        return Shot(
            template="process",
            heading="Pollinator Mimicry",
            learning_objective="Show how mimicry can attract pollinators.",
            steps=["Flower resembles a pollinator signal", "Visitor transfers pollen"],
            stage_fractions=[0, 0.52],
        )
    if "water" in text:
        return Shot(
            template="process",
            heading="Water And Pollination",
            learning_objective="Separate pollination agents from nectar attraction.",
            steps=["Some pollen moves by water", "Many flowers attract animals with nectar"],
            stage_fractions=[0, 0.52],
        )
    return Shot(
        template="process",
        heading=_heading_from_text(narration),
        learning_objective="Explain the key idea in this segment.",
        steps=_two_steps(narration),
        stage_fractions=[0, 0.52],
    )


def _heading_from_text(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", text)
    return " ".join(words[:4]) or "Key Idea"


def _two_steps(text: str) -> list[str]:
    clauses = [part.strip(" .") for part in re.split(r",|\band\b|\bbut\b|\bthen\b", text) if part.strip()]
    if len(clauses) >= 2:
        return [_short_step(clauses[0]), _short_step(clauses[1])]
    return [_short_step(text), "Connect it to the main concept"]


def _short_step(text: str) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z-]{1,}", text)
    return " ".join(words[:8]) or "Observe the idea"




