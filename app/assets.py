"""Still-image production: prompt repair, FLUX generation, technical + semantic QA, retry rounds, provenance.

Memory policy (12 GB): generation and vision review run in separate phases so only one heavy model is resident.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter

from . import progress
from .resilience import atomic_write_text, read_json_or_none

from . import commons, ocr, vision
from .comfyui_client import ComfyUIClient
from .config import AppConfig
from .schema import Segment, Shot, Storyboard

TARGET = (1920, 1080)
GENERIC_MARKERS = ("subject-appropriate visual medium", "production asset in", "sharp 1920x1080", "1920x1080",
                   "premium educational documentary frame", "full-frame visual coverage", "no blank card", "foreground-background separation",
                   "sharp native")
# Words that make text-to-image models draw lettering, posters or diagrams.
TEXT_INDUCING = re.compile(r"(?i)\b(title[- ]?cards?|titles?|headings?|captions?|labell?ed|labell?ing|labels?|text(?:ual)?|typography|"
                           r"infographics?|diagrams?|posters?|banners?|slides?|logos?|watermarks?|words?|letters?|arrows?)\b")
STYLE_SUFFIX = ("photorealistic high-resolution educational documentary photograph, natural lighting, sharp focus, "
                "rich natural colour, clean uncluttered composition, plain unmarked surfaces with no writing of any kind, no text, no letters, no labels, no arrows, "
                "no captions, no watermark, no logo")
NEGATIVE = "text, letters, words, labels, captions, arrows, watermark, logo, signature, infographic, collage, blurry, distorted, deformed"
_STOP = set("the a an of and to in on with for from is are was were that this these those it its by as at or be can which their".split())


def _words(text: str) -> set[str]:
    """Content-word stems in any script: split on whitespace/punctuation so combining marks stay inside Indic words."""
    strip = " \t\n\"'.,;:!?()[]{}*_-/।۔"
    tokens = (w.strip(strip) for w in (text or "").lower().split())
    return {w[:6] for w in tokens if len(w) >= (3 if any(ord(c) >= 0x0590 for c in w) else 4) and w not in _STOP and not w.isdigit()}


def is_generic_prompt(segment: Segment) -> bool:
    prompt = (segment.image_prompt or "").lower()
    if not prompt.strip() or any(m in prompt for m in GENERIC_MARKERS):
        return True
    basis = (segment.narration or "") + " " + " ".join(segment.labels)
    letters = [c for c in basis if c.isalpha()]
    if letters and sum(1 for c in letters if ord(c) >= 0x0590) / len(letters) > 0.5:
        return False                    # native-script narration shares no stems with an English prompt; do not rewrite every prompt
    return len(_words(prompt) & _words(basis)) < 2


_QUOTED = re.compile(r"""(['"\u2018\u201c])[^'"\u2019\u201d]{2,80}\1""")
_READS = re.compile(r"(?i)\b(?:displaying|displays|showing|shows|reading|reads|bearing|inscribed|printed|written|marked|titled|named|called|saying)\b[^.;,]*")


_UI_MOMENT = re.compile(
    r"(?i)\b(drag(?:ging)?(?:\s+and\s+drop)?\s+(?:the|it|them|a|an|each|every)\b|drop\s+(?:it|them|the)\s+(?:into|onto|in|on)\b|"
    r"click(?:ing)?\s+(?:on|the|a|here)\b|tap(?:ping)?\s+(?:on|the\s+(?:button|icon|screen|option|answer)|here)\b|"
    r"swipe\s+(?:left|right|up|down|to)\b|move\s+the\s+slider|slider\s+(?:to|at)\b|"
    r"on\s+the\s+screen,?\s+(?:you\s+can\s+)?(?:click|tap|drag|select|choose|press)|"
    r"highlighted\s+(?:piece|part|apparatus|button)|press\s+the\s+(?:\w+\s+)?button|check\s*box|"
    r"in\s+this\s+(?:simulation|simulator|activity)|(?:this|the)\s+simulation\b|"
    r"(?:take|answer|attempt|start)\s+the\s+quiz|interact(?:ive)?\s+with)\b")


def is_screen_interaction(text: str) -> bool:
    """Narration that describes operating a software simulation/quiz cannot be photographed; it is shown as concept cards."""
    return bool(_UI_MOMENT.search(text or ""))


def strip_ui_clause(text: str) -> str:
    """Remove software-operation wording ("Let's drag the labels into their boxes:") and keep the subject matter that follows it."""
    text = (text or "").strip()
    head, sep, tail = text.partition(":")
    if sep and tail.strip() and is_screen_interaction(head):
        return tail.strip()
    kept = [part for part in re.split(r"(?<=[.!?])\s+", text) if not is_screen_interaction(part)]
    return " ".join(kept).strip() or text


_FILLER = re.compile(
    r"(?i)^\W*(?:(?:so|now|okay|ok|alright|well|and)[,\s]+)*(?:let'?s\s+(?:answer|solve|begin|start|move\s+on|summari[sz]e|recap)|"
    r"keep\s+(?:learning|exploring|observing|practi[sc]ing)|thank\s+you|thanks\s+for|see\s+you|that'?s\s+(?:all|it)|until\s+next\s+time|stay\s+tuned|"
    r"(?:don'?t\s+forget\s+to\s+)?(?:like|subscribe|share)|we\s+believe\s+in\s+you|good\s+luck|i\s+hope\s+you|welcome\s+(?:back\s+)?to|"
    # rhetorical checks and encouragement: spoken over the current picture, never a screen of their own
    r"(?:is\s+(?:that|it)\s+)?clear\b|right\s*\?|got\s+it|understood|any\s+questions|fascinating|interesting,?\s+isn'?t\s+it|amazing|"
    r"isn'?t\s+(?:it|that)\s+(?:fascinating|interesting|amazing)|keep\s+(?:it\s+up|going)|you\s+can\s+do\s+it|"
    r"(?:we|i)\s+believe\s+in\s+your|happy\s+learning)")


# Hindi/Marathi sign-offs and transitions (no \b: Devanagari vowel signs are not word characters to the regex engine)
_FILLER_INDIC = re.compile(
    r"^[\W]*(?:तो\s+|अब\s+|चलिए\s+|चलो\s+)*(?:धन्यवाद|शुक्रिया|फिर\s+मिलेंगे|अगले\s+वीडियो|सीखते\s+रहिए|सीखते\s+रहो|"
    r"देखते\s+रहिए|चलिए\s+शुरू|चलिए\s+शुरू\s+करते|शुरू\s+करते\s+हैं|आगे\s+बढ़ते\s+हैं|सवाल\s+हल\s+करते|लाइक|सब्सक्राइब|शेयर\s+करना)")


def is_filler_narration(text: str) -> bool:
    """Short greetings, sign-offs and transitions carry nothing to photograph (and would repeat an earlier frame)."""
    text = (text or "").strip()
    return len(text.split()) <= 12 and bool(_FILLER.search(text) or _FILLER_INDIC.search(text))


_NAMING = re.compile(r"(?i)^\W*(?:this|these|that|those|it|they)\b.*?\b(?:is|are|was|were)\s+(?:also\s+|often\s+|then\s+|commonly\s+)?"
                     r"(?:called|known\s+as|termed|referred\s+to\s+as|named|described\s+as)\b")


def is_continuation(text: str) -> bool:
    """A sentence that only NAMES what the previous picture already shows ('This process is known as protandry.', 'Cross-pollination.'):
    an image model cannot draw a term, and the right visual is the picture just shown."""
    text = (text or "").strip()
    words = re.findall(r"[^\W\d_][\w'-]*", text)
    if not words or len(words) > 12:
        return False
    return bool(_NAMING.search(text)) or len(words) <= 2


# Things an image model covers in writing unless told otherwise (packaged products, medicines) and things that ARE writing (documents).
_PRODUCTS = re.compile(r"(?i)\b(pills?|tablets?|capsules?|medicines?|medications?|drugs?|bottles?|jars?|vials?|packag\w*|boxe?s|cartons?|"
                       r"cans?|packets?|blister\w*|products?|brands?|containers?|tubes?|sachets?|syringes?|labels?)\b")
_PRODUCT_NAME = re.compile(r"\b(?:[A-Z][a-z]{2,}(?:\s+[A-Z])?)\b(?=\s*(?:,|$|\s+(?:pills?|tablets?|bottles?|capsules?)))")
_WRITING = re.compile(r"(?i)\b(reads?|labell?ed|named|says?|written|printed|inscribed|marked|titled|captioned)\b")
_DOCUMENTS = re.compile(r"(?i)\b(textbook|book|page|notebook|worksheet|chart|whiteboard|blackboard|chalkboard|poster|sign(?:board)?|"
                        r"screen|monitor|form|prescription|document|newspaper|menu|certificate|"
                        r"graph|plotted|axes|x-axis|y-axis)s?\b")     # a graph must be plotted from data, never drawn by an image model


def text_free_prompt(prompt: str) -> str:
    """Keep the picture free of lettering at the source (teacher's view: a picture of a medicine shows the medicine, not its label).
    Sentences about documents (a textbook page, a chart, a whiteboard) are dropped when the prompt has another subject, because an image
    model fills them with fake writing; packaged products and medicines are asked for as plain, unbranded, unprinted items."""
    text = (prompt or "").strip()
    if not text:
        return text
    sentences = [x for x in re.split(r"(?<=[.;])\s+", text) if x.strip()]
    sentences = [x for x in sentences if not _WRITING.search(x)] or sentences       # "The labels read cyclosporine A, statin ..."
    kept = [x for x in sentences if not _DOCUMENTS.search(x)]
    if not kept:
        return ""          # the picture would only be a page, chart or screen full of writing: the shot is explained by cards instead
    text = " ".join(kept)
    if _PRODUCTS.search(text) and "unbranded" not in text.lower():
        text = text.rstrip(" .") + (". Every item is plain and unbranded with smooth unprinted surfaces: no label, no imprint, no printing, "
                                    "no writing or numbers on anything")
    return text


def clean_prompt(prompt: str) -> str:
    """Drop mechanical template sentences, quoted strings and 'displaying ...' clauses that provoke embedded lettering."""
    prompt = _READS.sub("", _QUOTED.sub("", prompt or ""))
    kept = []
    for sentence in re.split(r"(?<=[.;])\s+", prompt or ""):
        lowered = sentence.lower()
        if any(marker in lowered for marker in GENERIC_MARKERS) or re.match(r"(?i)\s*no (generated|embedded)", sentence):
            continue
        kept.append(TEXT_INDUCING.sub("", sentence))
    text = re.sub(r"\s{2,}", " ", " ".join(kept)).strip(" ,;")
    text = re.sub(r"(?i)\b(with|and|of|featuring|showing|including)\s*(?=[.,;]|$)", "", text)
    text = re.sub(r"\s+([.,;])", r"\1", text)
    return text


def director_prompt(segment: Segment, scene_title: str, lesson: str, config: AppConfig, feedback: list[str] | None = None) -> str:
    """Rewrite a weak image prompt from approved narration only. No new facts; text-free by construction."""
    system = ("You write image-generation prompts for a photorealistic educational video. Use ONLY facts stated in the "
              "narration or required visual; never introduce species, objects or settings that they do not mention. "
              "Describe objects by shape, material and colour only; never mention names, brands, titles, screens, labels or any text on objects. "
              "Never show pages, books, charts, whiteboards, blackboards, screens, posters, signs or forms (they always carry writing): show the "
              "real thing they are about instead. Medicines and products are plain and unbranded: blank tablets, unprinted bottles and boxes. "
              "If the narration is abstract, choose a neutral real-world scene that fits the lesson title. Describe one concrete subject, its visible action, spatial layout, camera "
              "distance/angle, natural lighting and colours in 40-70 words. Never ask for text, labels, arrows, diagrams, "
              "collages or watermarks. Return JSON {\"image_prompt\": \"...\"}.")
    payload = {"lesson": lesson, "scene": scene_title, "narration": segment.narration,
               "required_visual": segment.image_requirement, "visible_structures_to_include": segment.labels,
               "visual_type": segment.visual_type}
    if feedback:
        payload["previous_attempt_problems"] = feedback
        system += (" A previous image failed review for the listed problems: write a simpler prompt with ONE clearly visible "
                   "subject, no comparisons or multiple species side by side, and avoid what failed.")
    try:
        response = requests.post(f"{config.ollama_url}/api/generate", json={
            "model": config.production.director_model, "system": system, "prompt": json.dumps(payload),
            "stream": False, "format": "json", "think": False, "keep_alive": "5m",
            "options": {"temperature": 0.3, "num_predict": 400}}, timeout=600)
        response.raise_for_status()
        text = vision.extract_json(response.json()["response"]).get("image_prompt", "").strip()
        return text if len(text.split()) >= 12 else segment.image_prompt
    except (requests.RequestException, ValueError, KeyError):
        return segment.image_prompt


def key_ideas(segment: Segment, config: AppConfig) -> list[str]:
    """2-3 takeaway cards ('Term: short explanation') built only from the approved narration. LLM first, deterministic fallback."""
    narration = strip_ui_clause(segment.narration)
    for temperature in (0.1, 0.5):
        try:
            response = requests.post(f"{config.ollama_url}/api/generate", json={
                "model": config.production.director_model, "stream": False, "format": "json", "think": False, "keep_alive": "2m",
                "system": "Turn the narration into 2-3 takeaway cards. Each card has a TERM (1-3 words copied from the narration) and "
                          "a TEXT (a complete short statement of 4-9 words reusing the narration's own words). Each card covers a "
                          "DIFFERENT part of the narration, never repeats its term inside its text, and the cards follow the "
                          "narration's order; add nothing new. Return JSON {\"cards\": [{\"term\": \"...\", \"text\": \"...\"}]}.",
                "prompt": narration, "options": {"temperature": temperature, "num_predict": 300}}, timeout=300)
            ideas: list[str] = []
            for card in vision.extract_json(response.json()["response"]).get("cards", []):
                term, text = str(card.get("term", "")).strip(" .:"), str(card.get("text", "")).strip(" .")
                if not (term and text and len(term.split()) <= 4 and 3 <= len(text.split()) <= 10
                        and _words(term + " " + text) & _words(narration)):
                    continue
                if not _faithful(term + " " + text, narration):
                    continue      # the model added facts that the narration does not contain
                if text.lower().startswith(term.lower()) or any(len(_words(text) & _words(x)) > 0.6 * max(1, len(_words(text))) for x in ideas):
                    continue      # restates its own term or an earlier card
                ideas.append(f"{term[:1].upper()}{term[1:]}: {text}")
            if 2 <= len(ideas) <= 4:
                return ideas
        except (requests.RequestException, ValueError, KeyError, AttributeError):
            pass
    return complete_cards(narration, (segment.shot.heading if segment.shot else "") or segment.heading)


def picture_helps(narration: str, subject: str, lesson: str, config: AppConfig) -> bool | None:
    """Would a real photograph help a student understand this sentence? None when the model cannot be asked (keep the photo)."""
    system = (f"You are an experienced {subject} teacher planning the slide for ONE sentence of a lesson titled {lesson!r}. "
              "Decide whether a real photograph would help a student understand this sentence. Answer true only when the sentence is "
              "about something concrete that can be seen in the real world: an apparatus, instrument, material, substance, organism, "
              "place, industrial process or everyday phenomenon. Answer false when it states a definition, a law, a relationship "
              "between quantities, a calculation step, a symbol, a question to the student or an abstract property; a clear "
              "explanation card is better then. Return JSON {\"photograph\": true or false}.")
    try:
        response = requests.post(f"{config.ollama_url}/api/generate", json={
            "model": config.production.director_model, "system": system, "prompt": narration, "stream": False, "format": "json",
            "think": False, "keep_alive": "2m", "options": {"temperature": 0, "num_predict": 40}}, timeout=300)
        response.raise_for_status()
        answer = vision.extract_json(response.json()["response"]).get("photograph")
        return answer if isinstance(answer, bool) else None
    except (requests.RequestException, ValueError, KeyError, AttributeError):
        return None


WHOLE_STATEMENT_MAX_WORDS = 30        # a card can hold a sentence this long (the card text shrinks to fit); longer ones are cut at a clause


def complete_cards(narration: str, heading: str = "") -> list[str]:
    """Cards a teacher would write on the board: every card is a complete idea, never half of a sentence.
    - several sentences: one card per sentence (up to 3), each a full statement;
    - one sentence: its key term, then the whole statement;
    - one very long sentence: two statements cut where the sentence itself pauses (each must stand on its own)."""
    text = re.sub(r"\s+", " ", narration or "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    written = [s for s in sentences if not is_filler_narration(s)]           # remarks are spoken, never written on the board
    if written and len(written) < len(sentences):
        sentences, text = written, " ".join(written)
    if 2 <= len(sentences) <= 3 and all(3 <= len(s.split()) <= WHOLE_STATEMENT_MAX_WORDS for s in sentences):
        return [s[:1].upper() + s[1:] for s in sentences]
    words = text.rstrip(".?!").split()
    if not words:
        return ["Key idea", "Listen to the explanation"]
    whole = text[:1].upper() + text[1:]
    term = _key_term((sentences[0] if sentences else text).rstrip(".?!").split())     # from the main sentence, not a remark after it
    action_word = len(term.split()) == 1 and term.lower().endswith(("ed", "ing", "es")) and len(term) > 4
    if (action_word or not re.search(r"[^\W\d_]{3,}", term)) and heading.strip():
        term = heading.strip()         # "1.1" or "Simplifies" names nothing: the shot's own topic title does
    if len(words) <= WHOLE_STATEMENT_MAX_WORDS:
        return [term, whole]
    parts = _split_clauses(text, strong_only=True)
    subordinate = re.match(r"(?i)\s*(if|when|whenever|although|though|because|since|unless|while|as long as|once|after|before)\b",
                           parts[0]) if parts else None
    if parts and all(len(p.split()) >= 6 for p in parts) and not subordinate:      # "If we know X," alone is not a statement
        return parts
    return [term, whole]


def _llm_split(narration: str, config: AppConfig) -> list[str] | None:
    """Extractive fallback: the model only chooses WHERE to cut at a phrase boundary; the two parts must rejoin to the exact narration."""
    words = narration.strip().rstrip(".?!").split()
    if len(words) < 8 or not getattr(config, "use_ollama", True):       # same floor as _split_clauses: shorter sentences get a key-term card
        return None
    try:
        response = requests.post(f"{config.ollama_url}/api/generate", json={
            "model": config.production.director_model, "stream": False, "format": "json", "think": False, "keep_alive": "2m",
            "system": "Split the sentence into two parts at a natural phrase boundary near the middle (never inside a noun phrase, never "
                      "after a word like of/and/or/the/whether). Copy the original words exactly, in order, adding and removing nothing. "
                      "Return JSON {\"first\": \"...\", \"second\": \"...\"}.",
            "prompt": " ".join(words), "options": {"temperature": 0, "num_predict": 200}}, timeout=300)
        parts = vision.extract_json(response.json()["response"])
        first, second = str(parts.get("first", "")).strip(" ,;:"), str(parts.get("second", "")).strip(" ,;:")
    except (requests.RequestException, ValueError, KeyError, AttributeError):
        return None
    norm = lambda t: re.sub(r"[^\w ]", "", t.lower()).split()
    if not (len(norm(first)) >= 2 and len(norm(second)) >= 2 and norm(first) + norm(second) == norm(" ".join(words))):
        return None
    if first.split()[-1].lower().strip(",;:") in _DANGLING or second.split()[0].lower() in {"or", "of", "as", "to", "than"}:
        return None
    return [first[:1].upper() + first[1:], second[:1].upper() + second[1:]]


def _faithful(card: str, narration: str, minimum: float = 0.7) -> bool:
    """True when most content words of a generated card come from the approved narration (no invented facts)."""
    card_words = _words(card)
    return not card_words or len(card_words & _words(narration)) >= minimum * len(card_words)


# A cut must not leave one of these dangling at the end of the first card or open the second card.
_DANGLING = set("a an the of and or to in on at for from with by as is are was were be whether that which than if so not "
                "this these those its their his her our your can could will would may might should must do does did has have had "
                "about some more most all each every other another into onto over under between among after before during through "
                "let let's us we you they he she it i very also just only such many much few".split())
# never a card title on their own: question words, pointers and conversational words ("What draws", "Here", "Now")
_NOT_A_TERM = set("what why how when where who whom whose here there now then okay ok yes no right clear isn't aren't don't "
                  "means mean say says said given called known like".split())


def _key_term(words: list[str]) -> str:
    """The longest run (max 3) of content words in the sentence, as written: 'Let's turn on the power supply' -> 'Power supply'."""
    runs: list[list[str]] = []
    run: list[str] = []
    for word in words:
        bare = word.lower().strip(",;:.!?\"")
        if bare in _DANGLING or bare in _STOP or bare in _NOT_A_TERM or len(bare) < 3 or not re.search(r"[^\W\d_]", bare):
            if run:
                runs.append(run)
            run = []
        else:
            run.append(word.strip(",;:.!?\""))
            if word.endswith((",", ";", ":")):
                runs.append(run)
                run = []
    if run:
        runs.append(run)
    verb_like = lambda r: bool(r) and len(r) == 1 and r[-1].lower().endswith(("ed", "ing", "es")) and len(r[-1]) > 4   # "asked", "showing"
    best = max(runs, key=lambda r: (not verb_like(r), min(len(r), 3), len(r) and sum(map(len, r))), default=words[:3])   # a naming run, longest
    best = list(best)
    while len(best) > 1 and best[-1].lower().endswith(("ed", "ing")) and len(best[-1]) > 4:
        best.pop()                    # "Kohlrausch discovered" -> "Kohlrausch": a card title names a thing, not an action
    term = " ".join(best[-3:])
    return term[:1].upper() + term[1:]


def _split_clauses(narration: str, strong_only: bool = False) -> list[str] | None:
    """Two complete clauses split near the middle: at a comma/colon/conjunction when there is one, otherwise at a word
    boundary that does not leave a connective dangling ("whether strong or | weak ...").
    strong_only=True returns None unless the sentence has such a natural break."""
    words = narration.strip().rstrip(".?!").split()
    if len(words) < 8:            # too short to cut: a key-term card (its own words) followed by the whole sentence, never a filler card
        return None if strong_only else [_key_term(words), " ".join(words)]
    middle = len(words) // 2
    clean = lambda w: w.lower().strip(",;:")
    inner = range(3, len(words) - 2)
    open_ok = lambda i: clean(words[i - 1]) not in _DANGLING and clean(words[i]) not in {"or", "of", "as", "to", "than", "is", "are", "was", "were", "be", "been"}
    strong = [i for i in inner if open_ok(i) and (words[i - 1].endswith((",", ";", ":")) or clean(words[i]) in {"and", "while", "which", "but", "based", "with", "by"})]
    if strong_only and not strong:
        return None
    weak = [i for i in inner if open_ok(i)]
    pool = strong or weak or list(inner)
    cut = min(pool, key=lambda i: (abs(i - middle), -i))
    first, second = " ".join(words[:cut]).rstrip(",;:"), " ".join(words[cut:]).lstrip(",;:")
    return [first[:1].upper() + first[1:], second[:1].upper() + second[1:]]


def technical_qa(image: Image.Image, config: AppConfig) -> dict:
    gray = np.asarray(image.convert("L").resize((1024, round(1024 * image.height / image.width))))
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    spread = float(gray.std())
    reasons = []
    if spread < 12:
        reasons.append("blank_or_flat")
    if sharp < config.production.min_sharpness:
        reasons.append("blurry")
    if image.width < 1024:
        reasons.append("too_small")
    return {"ok": not reasons, "sharpness": round(sharp, 1), "contrast": round(spread, 1), "reasons": reasons}


def finish_image(image: Image.Image) -> Image.Image:
    """Cover-crop to 16:9, upscale to 1920x1080 and restore edge detail (open-source Lanczos + unsharp)."""
    image = image.convert("RGB")
    scale = max(TARGET[0] / image.width, TARGET[1] / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left, top = (resized.width - TARGET[0]) // 2, (resized.height - TARGET[1]) // 2
    cropped = resized.crop((left, top, left + TARGET[0], top + TARGET[1]))
    return cropped.filter(ImageFilter.UnsharpMask(radius=1.4, percent=55, threshold=3))


def semantic_qa(path: Path, segment: Segment, scene_title: str, config: AppConfig, prompt: str = "") -> dict:
    structures = "; ".join(segment.labels) or "none"
    if segment.shot and segment.shot.template == "title_card":
        prompt = (
            f"This image is the background for a title card of the lesson topic: {segment.shot.heading or scene_title}.\n"
            f"Lesson context: {segment.narration}\n"
            "Return JSON: matches (bool: the scene is plausibly related to the topic AND the central third of the frame is calm and uncluttered so a large title can be read over it), "
            "score (0..1), contains_text (bool: any letters/words visible), missing (list), issues (list of visual defects).")
        result = vision.ask(path, prompt, config, purpose=f"semantic:{segment.shot_id}")
        return {"matches": _is_true(result.get("matches")), "score": _as_float(result.get("score")),
                "contains_text": _is_true(result.get("contains_text")),
                "missing": [], "issues": [str(x) for x in result.get("issues", [])][:6]}
    # The reviewer reads English: for a lesson in another script it gets the planned English visual requirement, not the native narration.
    native = bool(segment.narration) and sum(1 for c in segment.narration if c.isalpha() and ord(c) >= 0x0590) > len(segment.narration) * 0.3
    narration_line = "" if native else f"Narration: {segment.narration}\n"
    required = segment.image_requirement or segment.image_prompt or ("" if native else segment.narration)
    prompt = (
        "You are a strict reviewer of educational video imagery.\n"
        f"{narration_line}Required visual: {required}\n"
        f"Structures that must be visible: {structures}\n"
        "Judge the image against what is required, not against the topic in general. Calibration: score 0.9-1.0 when everything concrete "
        "that is required is clearly visible; an image that merely looks related to the subject scores at most 0.4. "
        "A camera cannot show ideas: when the narration describes an invisible process (gas exchange, energy conversion, chemical change), "
        "judge whether the image shows the visible setting of that process (the plant, the leaf, the apparatus, the materials) and do not penalise "
        "the absence of the invisible part itself. "
        "In `checks` list at most 4 CONCRETE, VISIBLE things that the required visual names: objects, organisms, apparatus, materials, colours or "
        "clearly visible movement, and mark each one visible or not. NEVER list abstract ideas, processes, energy, quantities, times, or invisible "
        "substances and molecules (carbon dioxide, oxygen, glucose, photosynthesis, 'converting', 'forming'). "
        "Return JSON with keys: checks (list of {\"item\": str, \"visible\": bool}), matches (bool: the image visibly supports the narration), "
        "score (0..1), contains_text (bool: true only if words or letters that a viewer would notice at normal viewing size are visible - "
        "titles, captions, signs, labels, logos, watermarks or clearly garbled lettering; IGNORE tiny incidental markings, scale ticks, "
        "dial graduations, +/- symbols on a battery and textures), "
        "missing (list of required things not visible), issues (list of visual defects such as distorted anatomy)."
    )
    result = vision.ask(path, prompt, config, purpose=f"semantic:{segment.shot_id}")
    return apply_checklist({"matches": _is_true(result.get("matches")), "score": _as_float(result.get("score")),
                            "contains_text": _is_true(result.get("contains_text")),
                            "missing": [str(x) for x in result.get("missing", [])][:6],
                            "issues": [str(x) for x in result.get("issues", [])][:6]}, result.get("checks"),
                           grounding=grounding_text(segment, structures, prompt))


def grounding_text(segment: Segment, structures: str, prompt: str = "") -> str:
    """What the shot truly requires: the narration, the labels, the shot heading and the director's refined prompt (specific to the
    shot). NOT the raw image_requirement / image_prompt: the contract fills those with template boilerplate ("depict the concrete
    subject", "principal subject large and unobscured"); the generic photo words are also dropped when the stems are compared."""
    heading = segment.shot.heading if segment.shot and segment.shot.heading else ""
    return f"{segment.narration} {structures} {heading} {prompt}"


# Words that describe photographs in general, not a subject: they must never make a checklist item look "required".
_GENERIC_PHOTO = set("""large small big tiny concrete subject principal visible image photo picture frame shot scene sharp detail detailed
documentary realistic natural lighting light color colour contrast stable professional composition margin margins overlay safe camera
distance angle eye level evidence structure structures background foreground neutral clean clear uncluttered focus focal point separation
coverage educational premium native resolution education element elements object objects item items white black dark bright surface
text label labels words letters font sans serif bold centered border thin colored coloured""".split())


def title_relevant(title: str, grounding: str) -> bool:
    """True when a stock photo's file title shares at least one non-generic subject word with what the shot requires."""
    name = re.sub(r"^file:", "", title or "", flags=re.I).rsplit(".", 1)[0].replace("_", " ")
    return bool(_stems(name, generic=True) & _stems(grounding, generic=True))


def _stems(text: str, generic: bool = False) -> set[str]:
    """Singular 5-letter stems of the content words (so rods/rod and anodes/anode match). generic=True also drops generic photo words."""
    return {w.rstrip("s")[:5] for w in re.findall(r"[^\W\d_]{3,}", (text or "").lower())
            if w not in _STOP and not (generic and w in _GENERIC_PHOTO)}


def apply_checklist(semantic: dict, checks, grounding: str = "") -> dict:
    """A reviewer's overall verdict is not trusted over its own itemised answers.

    With three or more concrete items, a picture fails when at least half of them are missing; with fewer items it fails only when none
    is visible. A single missing item lowers the score but does not sink an otherwise good picture (a leaf photograph for a sentence about
    photosynthesis is a fair illustration even though the process itself cannot be photographed)."""
    items = [c for c in (checks or []) if isinstance(c, dict) and str(c.get("item", "")).strip()]
    if not items:
        return semantic
    if grounding.strip():
        # The checklist must come from what the shot REQUIRES. A reviewer that lists what it sees in the picture instead ("industrial
        # machinery", "concrete floor" for a silver rod in a solution) can pass any photograph, so its verdict is not trusted.
        wanted = _stems(grounding, generic=True)
        grounded = [c for c in items if _stems(str(c.get("item", "")), generic=True) & wanted]
        if len(grounded) * 2 < len(items):
            semantic["matches"] = False
            semantic["score"] = min(semantic["score"], 0.3)
            semantic["missing"] = list(dict.fromkeys(semantic["missing"] + ["reviewer checklist not based on the required visual"]))[:6]
            semantic["checks"] = [{"item": str(c["item"]).strip(), "visible": _is_true(c.get("visible"))} for c in items]
            return semantic
        items = grounded
    absent =[str(c["item"]).strip() for c in items if not _is_true(c.get("visible"))]
    semantic["checks"] = [{"item": str(c["item"]).strip(), "visible": _is_true(c.get("visible"))} for c in items]
    if absent:
        fraction = len(absent) / len(items)
        fails = fraction >= 0.5 if len(items) >= 3 else fraction >= 1.0
        if fails:
            semantic["matches"] = False
        semantic["score"] = min(semantic["score"], round(1 - 0.5 * fraction if not fails else 1 - fraction, 2))
        semantic["missing"] = list(dict.fromkeys(semantic["missing"] + absent))[:6]
    return semantic


def _is_true(value) -> bool:
    """Model output may carry booleans as text ("false"); bool("false") would be True."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "visible"}
    return bool(value)


def _as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


REVIEWER_VERSION = 8  # 8: small printed labels are painted out instead of rejecting the picture. Bump whenever the semantic reviewer's criteria change: older accepted stills are judged again (no regeneration)


def _accepts(semantic: dict, config: AppConfig) -> bool:
    return semantic["matches"] and semantic["score"] >= config.production.min_semantic_score and not semantic["contains_text"]


@dataclass
class Job:
    key: str                # unique asset key, e.g. "1_2" or "3_1_p2"
    segment: Segment        # the segment used for prompts and QA (a panel gets its own view)
    owner: Segment          # the contract segment that receives the finished asset
    scene_title: str
    panel: int | None = None
    caption: str = ""


class StillProducer:
    def __init__(self, storyboard: Storyboard, output_dir: Path, config: AppConfig):
        self.board, self.config = storyboard, config
        self.dir = output_dir / "assets"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.client = ComfyUIClient(config)
        self.issues: list[dict] = []
        self._downgraded: set[int] = set()

    def _plan_continuations(self) -> None:
        """Shots that only name or close what the previous picture shows are not generated: they reuse the nearest earlier photo."""
        self._continuations = []
        for scene, seg in self.board.all_segments():
            shot = seg.shot
            if shot is None or shot.asset_path or shot.template not in {"photo", "video"} or seg.labels or seg.formula_lines:
                continue
            if is_continuation(seg.narration) or is_filler_narration(seg.narration):
                self._continuations.append((scene, seg))
        self._continuation_ids = {id(seg) for _scene, seg in self._continuations}

    def _previous_picture(self, target: Segment) -> str | None:
        """The picture shown just before this shot: nearest earlier segment that has a real photograph."""
        previous = None
        for _scene, seg in self.board.all_segments():
            if seg is target:
                return previous
            shot = seg.shot
            if shot and shot.template in {"photo", "video"} and shot.asset_path and Path(shot.asset_path).is_file():
                previous = shot.asset_path
        return previous

    def _apply_continuations(self) -> None:
        for scene, seg in self._continuations:
            picture = self._previous_picture(seg)
            if picture:
                seg.shot.asset_path = picture
                seg.continuation = True
                self.issues.append({"shot_id": seg.shot_id, "field": "visual", "severity": "info", "renderer": "continuation",
                                    "reason": "narration only names or closes what the previous picture shows; that picture stays on screen",
                                    "repair": "none needed; supply an asset_path to show something else"})
            else:                                   # nothing earlier to continue from: fall back to cards as before
                job = Job(seg.shot_id or f"{scene.scene_number}_{seg.segment_number}", seg, seg, scene.title)
                if self._downgrade(job):
                    self.issues.append({"shot_id": seg.shot_id, "field": "visual", "severity": "info", "renderer": "concept_cards",
                                        "reason": "greeting, sign-off or naming sentence with no earlier picture to continue; shown as a card",
                                        "repair": "none needed"})

    def _jobs(self) -> list[Job]:
        jobs: list[Job] = []
        for scene, seg in self.board.all_segments():
            shot = seg.shot
            if shot is None or shot.asset_path:
                continue
            if id(seg) in getattr(self, "_continuation_ids", ()):
                continue                                    # reuses the previous picture (see _plan_continuations)
            base = seg.shot_id.replace(".", "_") or f"{scene.scene_number}_{seg.segment_number}"
            if shot.template == "title_card" and not self.config.production.generate_title_backgrounds:
                continue            # the renderer draws a deterministic gradient behind the title; no image model is needed
            if shot.template in {"photo", "title_card", "video"}:
                jobs.append(Job(base, seg, seg, scene.title))
            elif shot.template == "split_screen":
                seg.panel_assets = []
                for i, column in enumerate(seg.columns[:3], start=1):
                    caption, _, prompt = column.partition("|")
                    view = seg.model_copy(deep=True)
                    view.narration = f"{seg.narration} Panel: {caption.strip()}."
                    view.image_requirement = prompt.strip() or f"{caption.strip()} - {seg.image_requirement or seg.narration}"
                    view.image_prompt = prompt.strip()
                    view.labels, view.label_placement = [], []
                    view.visual_type = "realistic_image"
                    view.shot = shot.model_copy(update={"template": "photo"})
                    jobs.append(Job(f"{base}_p{i}", view, seg, scene.title, panel=i, caption=caption.strip()))
        return jobs

    def _record_path(self, job: Job) -> Path:
        return self.dir / f"shot_{job.key}.json"

    def _signature(self, prompt: str) -> str:
        workflow = Path(self.config.production.image_workflow_path)
        blob = json.dumps([prompt, self.config.production.image_checkpoint, self.config.production.image_width,
                           self.config.production.image_height, workflow.read_text(encoding="utf-8") if workflow.exists() else ""])
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _cards_for_screen_moments(self) -> None:
        for scene, seg in self.board.all_segments():
            shot = seg.shot
            if shot is None or shot.asset_path or shot.template not in {"photo", "video"} or seg.labels:
                continue
            if id(seg) in getattr(self, "_continuation_ids", ()):
                continue                                    # a sign-off / naming sentence continues the previous picture instead of a card
            filler = is_filler_narration(seg.narration)
            if not filler and not is_screen_interaction(seg.narration):
                continue
            job = Job(seg.shot_id or f"{scene.scene_number}_{seg.segment_number}", seg, seg, scene.title)
            if self._downgrade(job):
                self.issues.append({"shot_id": seg.shot_id, "field": "visual", "severity": "info", "renderer": "concept_cards",
                                    "reason": ("narration is a greeting, sign-off or transition with nothing to photograph; shown as a card"
                                               if filler else "narration describes operating a software simulation or quiz; shown as concept cards instead of a photograph"),
                                    "repair": "none needed; supply a screen capture as asset_path if the interface itself must be shown"})

    def _cards_for_abstract_statements(self) -> None:
        """Subject-aware (see subject.py): in mathematics, physics and chemistry a photograph earns its place only when the sentence is
        about something that can be seen; a law, definition, relationship, calculation or question is explained better by a clear card.
        Biology and unknown subjects keep photographs first. Decided once per sentence by the director model and cached."""
        from .subject import photos_first, subject_of
        subject = subject_of(self.board)
        if photos_first(subject) or not self.config.use_ollama:
            return
        cache_path = self.dir / "visual_need.json"
        cache = read_json_or_none(cache_path) or {}
        changed = False
        for scene, seg in self.board.all_segments():
            shot = seg.shot
            if shot is None or shot.asset_path or shot.template not in {"photo", "video"} or seg.labels or seg.animate:
                continue
            if id(seg) in getattr(self, "_continuation_ids", ()):
                continue
            key = hashlib.sha1(f"need1|{subject}|{seg.narration}".encode()).hexdigest()[:16]
            if key not in cache:
                cache[key] = picture_helps(seg.narration, subject, self.board.title, self.config)
                changed = True
            if cache[key] is not False:
                continue
            job = Job(seg.shot_id or f"{scene.scene_number}_{seg.segment_number}", seg, seg, scene.title)
            if self._downgrade(job):
                self.issues.append({"shot_id": seg.shot_id, "field": "visual", "severity": "info", "renderer": "concept_cards",
                                    "reason": f"{subject}: the sentence states a law, definition, relationship or question with nothing "
                                              "to photograph; shown as explanation cards",
                                    "repair": "none needed; give the shot labels or an asset_path if a real picture is required"})
        if changed:
            atomic_write_text(cache_path, json.dumps(cache, indent=1))

    def run(self) -> list[dict]:
        self._plan_continuations()
        self._cards_for_screen_moments()
        self._cards_for_abstract_statements()
        jobs = self._jobs()
        records: dict[str, dict] = {}
        for job in jobs:
            record = self._load(job)
            if record is not None and not record["accepted"] and not record.get("rereview") \
                    and not (record.get("exhausted") and not self.config.production.retry_rejected):
                record = None          # an unfinished/rejected shot gets a fresh set of attempts (exhausted ones are kept unless retry_rejected)
            if record is None:
                seg = job.segment
                prompt = seg.image_prompt
                if self.config.production.prompt_director and is_generic_prompt(seg):
                    prompt = director_prompt(seg, job.scene_title, self.board.title, self.config)
                if seg.visual_type == "title_card" or (seg.shot and seg.shot.template == "title_card"):
                    prompt = f"{clean_prompt(prompt)}. Wide establishing scene, main subject placed off to one side, soft out-of-focus open space in the centre"
                else:
                    prompt = text_free_prompt(clean_prompt(prompt) or seg.image_prompt)
                    if not prompt and self.config.production.prompt_director:     # the planned picture was a page or chart: ask for the real thing
                        prompt = text_free_prompt(clean_prompt(director_prompt(seg, job.scene_title, self.board.title, self.config)))
                    if not prompt and job.panel is None and self._downgrade(job):
                        self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image_prompt", "severity": "info",
                                            "renderer": "concept_cards", "reason": "the only picture for this sentence would be a page, "
                                            "chart or label full of writing; shown as explanation cards", "repair": "none needed"})
                        continue
                    prompt = prompt or seg.image_prompt
                record = {"shot_id": job.key, "narration": seg.narration, "prompt": prompt, "original_prompt": seg.image_prompt,
                          "attempts": [], "accepted": False, "path": ""}
            records[job.key] = record
        jobs = [job for job in jobs if job.key in records]          # shots turned into cards above need no picture
        if not jobs:
            self._apply_continuations()
            return self.issues             # keep what the pre-pass recorded (greeting / screen-operation shots shown as cards)
        vision.unload(self.config)
        if self.config.production.retrieval_enabled:
            for done, job in enumerate(jobs, start=1):
                record = records[job.key]
                if done % 5 == 1 or done == len(jobs):
                    progress.step("stock photo search", done, len(jobs), f"shot {job.owner.shot_id or job.key}")
                if not record["accepted"] and not record["attempts"] \
                        and not (job.segment.shot and job.segment.shot.template == "title_card"):
                    self._retrieve(job, record)
        self._review_pending(jobs, records)         # also judges stills whose approval predates the current reviewer
        for round_no in range(1, self.config.production.max_attempts + 1):
            pending = [j for j in jobs if not records[j.key]["accepted"] and self._generated(records[j.key]) < round_no]
            if not pending:
                break
            print(f"Image round {round_no}: generating {len(pending)} still(s)...", flush=True)
            if round_no > 1 and self.config.production.prompt_director:
                for job in pending:     # corrected prompt from the reviewer's findings, still bound to the approved narration
                    last = (records[job.key]["attempts"] or [{}])[-1].get("semantic") or {}
                    problems = list(last.get("missing") or []) + list(last.get("issues") or [])
                    if problems:
                        revised = director_prompt(job.segment, job.scene_title, self.board.title, self.config, feedback=problems[:5])
                        if revised and revised != records[job.key]["prompt"]:
                            records[job.key]["prompt"] = text_free_prompt(clean_prompt(revised) or revised) or records[job.key]["prompt"]      # a rewrite must not bring charts or labels back
                vision.unload(self.config)
            for done, job in enumerate(pending, start=1):
                progress.step(f"round {round_no} picture", done, len(pending), f"shot {job.owner.shot_id or job.key}")
                try:
                    self._generate(job, records[job.key], round_no)
                except Exception as exc:     # ComfyUI gave up on this picture (after its own retries): the other shots go on
                    print(f"Picture for shot {job.owner.shot_id or job.key} failed ({str(exc)[:120]}); continuing.", flush=True)
                    self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image_prompt", "severity": "warning",
                                        "renderer": "flux", "reason": f"picture generation failed: {str(exc)[:160]}",
                                        "repair": "check ComfyUI; the shot falls back to explanation cards"})
            self.client.free_memory()
            time.sleep(1.5)
            self._review_pending(jobs, records)
        for job in jobs:
            if id(job.owner) in self._downgraded:
                continue
            record = records[job.key]
            if not record["accepted"]:
                record["exhausted"] = True       # remembered so reruns do not burn GPU time on the same failing shot
                self._save(job, record)
            best = self._best(record)
            if best is None and job.panel is None and self._downgrade(job):
                self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image_prompt", "severity": "warning", "renderer": "flux",
                                    "reason": "no picture could be made; the shot shows explanation cards instead",
                                    "repair": "check ComfyUI health and rerun to get a picture"})
                continue
            if best is None:
                self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image_prompt", "severity": "error", "renderer": "flux",
                                    "reason": "no image could be generated", "repair": "check ComfyUI health and prompt"})
                if job.panel is not None:
                    self._downgrade(job)        # a split screen with a missing panel would show the later pictures under the wrong captions
                continue
            if job.panel is None:
                job.owner.shot.asset_path = best["path"]
            else:
                job.owner.panel_assets.append(best["path"])
            if not record["accepted"] and self._downgrade(job):
                self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image", "severity": "warning", "renderer": "concept_cards",
                                    "reason": f"no photograph passed QA after {len(record['attempts'])} attempts; shown as deterministic concept "
                                              f"cards built from the approved narration (best score {best.get('score', 0):.2f})",
                                    "repair": "SME: supply a reviewed asset_path or a more specific image_requirement if a photograph is required"})
            elif not record["accepted"]:
                self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image", "severity": "error", "renderer": "flux",
                                    "reason": "no candidate passed semantic/technical QA after "
                                              f"{len(record['attempts'])} attempts (best score {best.get('score', 0):.2f}; "
                                              f"missing={best.get('missing')}, issues={best.get('issues')})",
                                    "repair": "revise image_requirement / image_prompt so the required subject is explicit"})
        self._apply_continuations()
        return self.issues

    def _note_unreviewed(self, job: Job, why: str) -> None:
        """A still accepted on technical checks alone must be visible in the report, never silent (policy: no unverified picture)."""
        self.issues.append({"shot_id": job.owner.shot_id or job.key, "field": "image", "severity": "warning", "renderer": "flux",
                            "reason": f"picture accepted without a content review: {why}",
                            "repair": "start the visual review model (Ollama) and re-run, or have a person check this picture"})

    @staticmethod
    def _generated(record: dict) -> int:
        return sum(1 for a in record["attempts"] if a.get("source") != "commons")

    def _review_pending(self, jobs: list[Job], records: dict) -> None:
        todo = [j for j in jobs if records[j.key]["attempts"] and not records[j.key]["attempts"][-1].get("reviewed")
                and not records[j.key]["accepted"]]
        if not todo:
            return
        if self.config.production.semantic_qa and vision.available(self.config):
            print(f"Semantic review of {len(todo)} candidate(s)...", flush=True)
            for done, job in enumerate(todo, start=1):
                progress.step("review", done, len(todo), f"shot {job.owner.shot_id or job.key}")
                self._review(job, records[job.key])
            vision.unload(self.config)
        else:
            for job in todo:
                record = records[job.key]
                record["attempts"][-1]["semantic"] = None
                record["attempts"][-1]["reviewed"] = True
                record["accepted"] = record["attempts"][-1]["technical"]["ok"]
                record.pop("rereview", None)
                self._note_unreviewed(job, "the visual review model is not available" if self.config.production.semantic_qa
                                      else "semantic_qa is switched off")
                self._save(job, record)

    def _retrieve(self, job: Job, record: dict) -> None:
        """Try the best open-licensed Wikimedia Commons image first; it must still pass QA like any candidate."""
        try:
            query = job.caption or commons.search_query(job.segment.narration, self.config)
            grounding = grounding_text(job.segment, "; ".join(job.segment.labels), record.get("prompt", ""))
            # A stock photo whose own title shares no subject word with the shot ("Anaconda Copper converters" for a silver rod in a
            # solution) is an accident of the search engine, and a vision reviewer can still be fooled by it: skip it unseen.
            relevant = [c for c in commons.find(query, self.config, limit=6) if title_relevant(c.get("title", ""), grounding)]
            for candidate in relevant[:2]:
                raw = self.dir / f"shot_{job.key}_commons_raw.jpg"
                commons.download(candidate, raw)
                with Image.open(raw) as image:
                    done = finish_image(image)
                final = self.dir / f"shot_{job.key}_commons.png"
                done.save(final)
                tech = technical_qa(done, self.config)
                raw.unlink(missing_ok=True)
                if tech["ok"]:
                    record["attempts"].append({"round": 0, "source": "commons", "path": str(final), "technical": tech,
                                               "provenance": candidate})
                    self._save(job, record)
                    return
        except Exception as exc:   # network/parse problems must never stop generation
            print(f"Commons retrieval skipped for {job.key}: {exc}", flush=True)

    def _card_ideas(self, job: Job, owner: Segment) -> list[str]:
        """Concept-card text is stored per shot. The language model words the cards differently on every run, which changed the clips (so
        nothing stayed cached) and the text viewers saw; a re-run now reuses the cards unless the narration changed."""
        narration = strip_ui_clause(owner.narration)
        key = hashlib.sha1(f"cards6|{narration}".encode()).hexdigest()[:16]        # bump "cards6" when the card logic changes
        path = self.dir / f"cards_{job.key}.json"
        if path.exists():
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved.get("key") == key and isinstance(saved.get("ideas"), list) and 2 <= len(saved["ideas"]) <= 4:
                    return [str(idea) for idea in saved["ideas"]]
            except (ValueError, OSError):
                pass
        ideas = key_ideas(owner, self.config)
        path.write_text(json.dumps({"key": key, "narration": narration, "ideas": ideas}, ensure_ascii=False, indent=1), encoding="utf-8")
        return ideas

    def _downgrade(self, job: Job) -> bool:
        """Replace a photograph that could not be verified with a deterministic explanation (spec: never ship an unverified picture)."""
        owner, shot = job.owner, job.owner.shot
        if shot.template == "title_card":
            shot.asset_path = None          # the renderer draws a deterministic gradient background behind the title
            self._downgraded.add(id(owner))
            return True
        if shot.template not in {"photo", "split_screen", "video"}:
            return False
        owner.labels, owner.label_placement = [], []       # labels need a verified picture; cards need none
        if job.panel is not None and owner.shot.template != "split_screen":
            return False
        ideas = self._card_ideas(job, owner)
        vision.unload(self.config)
        heading = shot.heading if shot.heading and not is_screen_interaction(shot.heading) and not is_filler_narration(shot.heading) else job.scene_title
        owner.shot = Shot(template="process", heading=heading, steps=ideas[:4],
                          stage_fractions=[round(i / len(ideas[:4]), 3) for i in range(len(ideas[:4]))])
        owner.panel_assets = []
        self._downgraded.add(id(owner))
        return True

    def _content_key(self, job: Job) -> str:
        """What the picture must show: if the approved text changes, an old still was judged against something else."""
        segment = job.segment
        blob = json.dumps([segment.narration, segment.image_prompt, segment.image_requirement, list(segment.labels or []), STYLE_SUFFIX],
                          ensure_ascii=False)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def _load(self, job: Job) -> dict | None:
        path = self._record_path(job)
        if not path.exists():
            return None
        record = read_json_or_none(path)
        if not isinstance(record, dict) or "attempts" not in record:
            return None                 # corrupt or half-written record: treat as not cached and rebuild it
        if record.get("signature") != self._signature(record.get("prompt", "")) or not all(Path(a["path"]).exists() for a in record["attempts"]):
            return None
        if record.get("content") not in (None, self._content_key(job)):
            return None                 # narration, prompt, requirement or labels were edited since this still was approved
        last = record["attempts"][-1] if record["attempts"] else {}
        if record.get("accepted") and last.get("source") == "commons" and not title_relevant(
                (last.get("provenance") or {}).get("title", ""),
                grounding_text(job.segment, "; ".join(job.segment.labels), record.get("prompt", ""))):
            return None                 # an accepted stock photo whose title has nothing to do with the shot: fetch/generate again
        stale_reviewer = record.get("reviewer") != REVIEWER_VERSION
        stale_threshold = record.get("min_score") not in (None, self.config.production.min_semantic_score)
        if (stale_reviewer or stale_threshold) and record["attempts"] and record["attempts"][-1].get("semantic") is not None:
            record["accepted"] = False                      # approved OR rejected under older criteria: judge the same picture again, do not regenerate
            record.pop("exhausted", None)
            record["rereview"] = True
            record["attempts"][-1]["reviewed"] = False
        return record

    def _save(self, job: Job, record: dict) -> None:
        record["signature"] = self._signature(record["prompt"])
        record["reviewer"] = REVIEWER_VERSION
        record["content"] = self._content_key(job)
        record["min_score"] = self.config.production.min_semantic_score
        atomic_write_text(self._record_path(job), json.dumps(record, indent=2, ensure_ascii=False))

    def _generate(self, job: Job, record: dict, round_no: int) -> None:
        seed = random.Random(f"{job.key}:{round_no}:{len(record['attempts'])}:{record['prompt']}").randint(1, 2**31 - 1)   # a retry never repeats a seed
        base = text_free_prompt(record["prompt"]) or record["prompt"]
        prompt = f"{base.rstrip('. ')}. {STYLE_SUFFIX}"
        if record["attempts"] and record["attempts"][-1].get("correction"):
            prompt = f"{base.rstrip('. ')}. {record['attempts'][-1]['correction']}. {STYLE_SUFFIX}"
        raw = self.dir / f"shot_{job.key}_a{round_no}_raw.png"
        final = self.dir / f"shot_{job.key}_a{round_no}.png"
        p = self.config.production
        self.client.generate_still(prompt, NEGATIVE, raw, seed, p.image_workflow_path, p.image_checkpoint,
                                   p.image_width, p.image_height, p.image_steps)
        with Image.open(raw) as image:
            done = finish_image(image)
            done.save(final)
            tech = technical_qa(done, self.config)
        raw.unlink(missing_ok=True)
        record["attempts"].append({"round": round_no, "seed": seed, "prompt": prompt, "path": str(final), "technical": tech})
        if not tech["ok"]:
            record["attempts"][-1]["correction"] = "sharp focus, high detail, well-lit, clear subject"
        self._save(job, record)

    def _review(self, job: Job, record: dict) -> None:
        attempt = record["attempts"][-1]
        attempt["reviewed"] = True
        try:
            semantic = semantic_qa(Path(attempt["path"]), job.segment, job.scene_title, self.config, record.get("prompt", ""))
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:          # also malformed model output (e.g. a text score)
            attempt["semantic"] = {"error": str(exc)}
            record["accepted"] = attempt["technical"]["ok"]
            record.pop("rereview", None)
            self._note_unreviewed(job, f"the visual review failed ({str(exc)[:80]})")
            self._save(job, record)
            return
        if self.config.production.ocr_text_check and not (job.segment.shot and job.segment.shot.template == "title_card"):
            words = ocr.stray_text(Path(attempt["path"]))
            if ocr.has_stray_text(words) and semantic.get("score", 0) >= self.config.production.min_semantic_score:
                # a good picture spoiled only by a small printed label (a medicine box, a bottle): paint the label out, verify again
                erased = ocr.erase_text(Path(attempt["path"]))
                if erased:
                    attempt["text_erased"] = erased
                    words = []
                    semantic["contains_text"] = False
                    semantic["issues"] = [i for i in semantic.get("issues", []) if not re.search(r"(?i)text|letter|label|word|writing", i)]
            if ocr.has_stray_text(words):
                semantic["contains_text"] = True
                semantic["issues"] = list(dict.fromkeys(semantic["issues"] + [f"lettering drawn in the picture: {', '.join(words[:4])}"]))[:6]
            elif semantic.get("contains_text") and words is not None and not ocr.stray_text(Path(attempt["path"]), min_conf=0.6, min_height=0.02):
                # the reviewer "saw text" but a sensitive OCR pass finds not even small words: texture or shapes, not lettering
                semantic["contains_text"] = False
                semantic["text_overruled_by_ocr"] = True
        attempt["semantic"] = semantic
        attempt["score"] = semantic["score"]
        record.pop("rereview", None)
        record["accepted"] = attempt["technical"]["ok"] and _accepts(semantic, self.config)
        if not record["accepted"]:
            fixes = []
            missing = list(semantic["missing"])
            if _PRODUCTS.search(f"{record.get('prompt', '')} {job.segment.narration}"):
                # "must clearly show Cyclosporine A, Statin" asks the image model to WRITE product names: keep only the visible things
                missing = [m for m in missing if not re.search(r"\b[A-Z][a-z]{2,}", str(m)) or _PRODUCTS.search(str(m))]
                missing = [_PRODUCT_NAME.sub("", str(m)).strip(" ,") or str(m) for m in missing]
            if missing:
                fixes.append("must clearly show " + ", ".join(missing))
            if semantic["contains_text"]:
                fixes.append("absolutely no text or lettering anywhere")
            if semantic["issues"]:
                fixes.append("avoid " + ", ".join(semantic["issues"]))
            attempt["correction"] = "; ".join(fixes) or "make the required subject large, central and unambiguous"
        self._save(job, record)

    @staticmethod
    def _best(record: dict) -> dict | None:
        usable = [a for a in record["attempts"] if a["technical"]["ok"]] or record["attempts"]
        clean = [a for a in usable if not (a.get("semantic") or {}).get("contains_text")]
        usable = clean or usable
        if not usable:
            return None
        chosen = max(usable, key=lambda a: (a.get("score") or 0))
        return {"path": chosen["path"], "score": chosen.get("score") or 0,
                "missing": (chosen.get("semantic") or {}).get("missing"), "issues": (chosen.get("semantic") or {}).get("issues")}
