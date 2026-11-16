"""English-to-Hindi replacement of on-screen text inside source-video frames (pipeline v4). Works for ANY subject.

Principles
  * No subject vocabulary is in this code. Reviewed wording lives in data packs (app/data/visual_glossary*.json, plus an optional
    <video>.visual_glossary.json beside a video); everything else is a general safeguard that does not depend on the lesson.
  * Whole text BLOCKS (a bullet, a sentence, a label) are translated as one sentence, never line fragments.
  * Text drawn ON an object in the scene (tilted or in perspective), tiny text, shouted product lettering, formulas, units and
    numbers are left exactly as they are.
  * A translation is accepted only after general checks: Devanagari present, no foreign script, sane length, numbers and symbols
    preserved, no entity swap (packs), and a back-translation that keeps the meaning. Rejected blocks stay English and are listed
    in the report with the reason.
  * OCR run-ons are repaired from the lesson's own narration vocabulary; whatever cannot be repaired stays English.
  * Hindi is drawn only with a real complex-script shaping engine (unshaped Devanagari is misspelt), in the sampled text colour,
    wrapped inside the original block, verified by OCR every few frames with the position held in between.

Translation engine: the local MADLAD-400 model (no Ollama needed); Qwen is only a fallback. A reviewed
``<video>.visual_translations.json`` always overrides the automatic plan.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .renderer import _tool_path
from .translation_qa import _formula_tokens, _SCIENCE_TERMS_HI

PIPELINE_VERSION = 6
DISCOVERY_STEP_SECONDS = 0.25
WINDOW_GAP_SECONDS = 0.60
MATCH_RATIO = 0.90
SCENE_CUT_DIFFERENCE = 25.0       # mean grey difference (0-255, 64x36 thumbnails) that means a hard cut: nothing may be carried over it
FADE_FULL_RATIO = 0.60           # a label whose contrast has fallen under this share of its peak is repainted fading, then dropped
FADE_DROP_OPACITY = 0.12
TRACKING_CONFIDENCE = 0.55       # verifying a block we already decided to translate may accept fading text the discovery pass would not
MISS_TOLERANCE_SCANS = 4         # OCR scans (3 frames each) a held block may be missed before the English is shown again
WINDOW_PAD_SECONDS = 0.4         # text fades in before its first clear sighting and out after the last (painting still needs an OCR match)

_PROTECTED_UNIT = re.compile(r"\b(?:cm|mm|km|kg|mg|ml|mol|hz|pa)\b|[=+*/×÷^<>≤≥→⇌]", re.I)
_CANDIDATE = re.compile(r"[A-Za-z][A-Za-z0-9 ,.:;'’\"&()/%!?-]{1,200}")
_BULLET = re.compile(r"^[•●○◦▪■□·\-–—*]\s*|^[oO0]\s+(?=[A-Z])")

# ---- vocabulary lives in data packs, never in this code ----------------------------------------------------------------
# app/data/visual_glossary*.json (one pack per subject or topic; every pack is loaded) and an optional <video>.visual_glossary.json
# beside the video. A pack may hold: entries (English label -> reviewed Hindi), abbreviations, ocr_fixes, entity_pairs (an entity named
# in the English must survive in the Hindi, and its "never" words must not appear) and wording_fixes (replace a known wrong model
# wording when a pattern matches the English). The code below contains no subject vocabulary at all.
_GLOSSARY: dict = {}
_ABBREVIATIONS: dict = {}
_OCR_FIXES: dict = {}
_ENTITY_PAIRS: dict = {}
_WORDING_FIXES: list = []


def _key(text) -> str:
    """Same normalisation as normalize(); defined here because the packs are loaded at import time."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def _merge_pack(data: dict) -> None:
    _GLOSSARY.update({_key(k): str(v) for k, v in data.get("entries", {}).items()})
    _ABBREVIATIONS.update({str(k): str(v) for k, v in data.get("abbreviations", {}).items()})
    _OCR_FIXES.update({str(k): str(v) for k, v in data.get("ocr_fixes", {}).items()})
    for entity, rule in data.get("entity_pairs", {}).items():
        _ENTITY_PAIRS[_key(entity)] = {"hindi": tuple(rule.get("hindi", ())), "never": tuple(rule.get("never", ()))}
    _WORDING_FIXES.extend(data.get("wording_fixes", []))


def load_glossary_packs(extra_files=()) -> list:
    """(Re)load every pack in app/data plus any extra files. Returns the files read."""
    for container in (_GLOSSARY, _ABBREVIATIONS, _OCR_FIXES, _ENTITY_PAIRS):
        container.clear()
    _WORDING_FIXES.clear()
    files = sorted((Path(__file__).parent / "data").glob("visual_glossary*.json")) + [Path(f) for f in extra_files]
    read = []
    for path in files:
        try:
            _merge_pack(json.loads(path.read_text(encoding="utf-8-sig")))
            read.append(str(path))
        except (OSError, ValueError):
            continue
    return read


load_glossary_packs()


@dataclass
class Line:
    text: str
    box: list
    confidence: float
    x0: int = 0
    y0: int = 0
    x1: int = 0
    y1: int = 0
    keep_left: int = 0                    # pixels left of this x belong to a bullet that stays untouched
    erase_only: bool = False              # erased with the label but never used to place or size the Hindi (title fragments)

    def __post_init__(self):
        points = np.array(self.box, dtype=np.int32)
        (self.x0, self.y0), (self.x1, self.y1) = points.min(axis=0), points.max(axis=0)

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass
class Block:
    lines: list = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(line.text.strip() for line in self.lines).strip()

    @property
    def bbox(self) -> tuple:
        return (min(l.x0 for l in self.lines), min(l.y0 for l in self.lines),
                max(l.x1 for l in self.lines), max(l.y1 for l in self.lines))


_COMMON = set("the a an and or of in on at to for with from by as is are was were be been it its this that these those their there "
              "then than which who whom what when where while will can could should would may might must not no yes if so such into "
              "over under between about after before during each other more most much many some any all both few very also only just "
              "up out off per use used make made real world real".split())
_VOCAB: set = set(_COMMON)


def set_vocabulary(words) -> None:
    """Words the lesson itself uses (from the spoken transcript) help repair OCR errors such as 'thereal' -> 'the real'."""
    _VOCAB.clear()
    _VOCAB.update(_COMMON)
    _VOCAB.update(w.lower() for w in words if re.fullmatch(r"[A-Za-z]{2,}", w))
    _VOCAB.update(w for key in _GLOSSARY for w in key.split())


def _segment(token: str, vocabulary=None):
    """Split a run-together token into known words (dynamic programming, longer words preferred); None when impossible."""
    word = token.lower()
    words = _VOCAB if vocabulary is None else vocabulary
    n = len(word)
    best: list = [None] * (n + 1)
    best[0] = (0, [])
    for i in range(n):
        if best[i] is None:
            continue
        for j in range(i + 1, n + 1):
            piece = word[i:j]
            if piece in words and (len(piece) > 1 or piece in {"a", "i"}):
                # longer pieces score higher, but a word the lesson/common English already knows beats a fragment found only in
                # the general list ("or"+"an" over "oran")
                score = best[i][0] + len(piece) ** 2 + (4 if piece in _VOCAB else -2)
                if best[j] is None or score > best[j][0]:
                    best[j] = (score, best[i][1] + [(i, j)])
    return best[n][1] if best[n] and len(best[n][1]) >= 2 else None


_WORDLIST: set = set()
_WORDLIST_TRIED = False


# English word-PARTS (affixes and combining forms). They are language morphology, not subject vocabulary. A split that produces one of
# them ("respira|tion", "photo|synth|esis") cut a real long word apart, so it is refused.
_WORD_PARTS = set("""tion sion ation ition ment ness ing ings ology logy ologies able ible ence ance ity ous ive ical ally ize ise ism ist
esis osis asis ysis sis thesis synth photo electro micro macro bio chem thermo hydro poly mono tri anti auto meta para pseudo syn
inter intra trans super hyper hypo semi multi omni geo cyto histo neuro psycho astro aero""".split())


def _plausible_split(parts, core: str) -> bool:
    """A general-word-list split is accepted for 3+ pieces of real words, or 2 words of 4+ letters in a 12+ letter token; never a word-part."""
    pieces = [core[i:j].lower() for i, j in parts]
    if any(p in _WORD_PARTS for p in pieces):
        return False
    if any(p not in _VOCAB and len(p) < 4 for p in pieces):
        return False                                  # a 3-letter fragment that is neither common English nor in the lesson ("uns|atura|ted")
    if len(pieces) == 2:                              # "roomtemperature": two whole words of 4+ letters in a token of 12+ letters
        return all(len(p) >= 4 for p in pieces) and len(core) >= 12
    # a one-letter word ("a", "i") is believable only BETWEEN two words, never at either end ("mito|chon|dri|a")
    return len(pieces) >= 3 and all(len(p) >= 2 or (p in {"a", "i"} and 0 < n < len(pieces) - 1) for n, p in enumerate(pieces))


def general_wordlist() -> set:
    """Whole-word pieces (letters only) of the local MADLAD tokenizer: a broad, subject-neutral word list for OCR run-on repair.

    Plain JSON, no extra library. Empty when the model folder is absent, in which case only the lesson vocabulary is used.
    """
    global _WORDLIST_TRIED
    if not _WORDLIST and not _WORDLIST_TRIED:
        _WORDLIST_TRIED = True
        path = Path(os.environ.get("MADLAD_MODEL_DIR", r"D:\AI\models\madlad400-3b-mt")) / "tokenizer.json"
        try:
            for piece, score in json.loads(path.read_text(encoding="utf-8"))["model"]["vocab"]:
                word = piece[1:] if piece.startswith("\u2581") else ""
                if len(word) >= 3 and word.isascii() and word.isalpha() and score >= -14.6:
                    _WORDLIST.add(word.lower())
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return _WORDLIST


def fix_spacing(text: str) -> str:
    """Repair words the OCR glued together ('wordsrunintogether' -> 'words run into together') using the lesson's own vocabulary."""
    def repair(match) -> str:
        core = match.group()
        if len(core) < 6 or core.lower() in _VOCAB:
            return core
        parts = _segment(core)
        if not parts and len(core) >= 10:                           # long run-on: try the general word list as well
            general = general_wordlist()
            if general:
                parts = _segment(core, general | _VOCAB)
                if parts and not _plausible_split(parts, core):
                    parts = None
        return " ".join(core[i:j] for i, j in parts) if parts else core

    # each run of letters is repaired on its own, so "Addition reactions:Additionofamolecule" (glued across a colon) works too
    fixed = re.sub(r"[A-Za-z]+", repair, str(text))
    return re.sub(r"([:;,])(?=[A-Za-z])", lambda m: m.group(1) + " ", fixed)


def has_garbled_token(text: str) -> bool:
    """A very long token the lesson never uses is almost always several words glued together; translating it yields nonsense."""
    for raw in str(text).split():
        core = re.sub(r"[^A-Za-z]", "", raw).lower()
        if len(core) >= 14 and core not in _VOCAB and core not in _GLOSSARY:
            return True
    return False


def apply_ocr_fixes(text: str) -> str:
    """Known OCR confusions (HCI -> HCl) and abbreviations the translator would otherwise transliterate (dil. -> dilute)."""
    value = str(text)
    for wrong, right in _OCR_FIXES.items():
        value = re.sub(rf"(?<![A-Za-z0-9]){re.escape(wrong)}(?![A-Za-z0-9])", right, value)
    for short, full in _ABBREVIATIONS.items():
        value = re.sub(rf"(?<![A-Za-z]){re.escape(short)}\s*", full + " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", value).strip()


def block_text(block) -> str:
    text = apply_ocr_fixes(fix_spacing(clean_line_text(block.text)))
    return re.sub(r"(?<=[A-Za-z]{3})\s*\+\s*(?=[A-Za-z]{3})", " + ", text)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def clean_line_text(text: str) -> str:
    return _BULLET.sub("", str(text).strip()).strip()


_LABEL_START = re.compile(r"^[A-Z][A-Za-z ]{1,25}:")


def group_blocks(lines: list) -> list:
    """Group OCR lines into text blocks: consecutive lines of the same size, evenly spaced, left-aligned or centred.

    A gap clearly larger than the block's own line pitch (paragraph spacing), or a line that starts with a "Label:" begins a
    new block, so two bullets are never merged into one sentence.
    """
    usable = [l for l in lines if len(clean_line_text(l.text)) >= 2 and re.search(r"[A-Za-z]", l.text)]
    blocks: list = []
    for line in sorted(usable, key=lambda l: (l.y0, l.x0)):
        for block in blocks:
            last = block.lines[-1]
            height = min(line.height, last.height)
            gap = line.y0 - last.y1
            pitch = line.y0 - last.y0
            same_size = 0.70 <= line.height / last.height <= 1.45
            left = abs(line.x0 - last.x0) <= 1.7 * height or abs(line.x0 - block.lines[0].x0) <= 1.7 * height
            centred = abs(line.center_x - last.center_x) <= 0.8 * height
            if len(block.lines) >= 2:
                pitches = [b.y0 - a.y0 for a, b in zip(block.lines, block.lines[1:])]
                near = pitch <= 1.30 * float(np.median(pitches))
            else:
                near = gap <= 0.60 * height
            starts_new_item = bool(_LABEL_START.match(clean_line_text(line.text))) or bool(getattr(line, "keep_left", 0))   # "Label:" or a bullet
            if -0.35 * height <= gap and near and same_size and (left or centred) and not starts_new_item:
                block.lines.append(line)
                break
        else:
            blocks.append(Block([line]))
    return blocks


MAX_CAPTION_TILT_DEGREES = 3.0
MAX_CAPTION_SKEW = 0.08
MIN_CAPTION_HEIGHT_FRACTION = 0.020


def scene_object_reason(block, frame_height: int):
    """Why a block is text on an OBJECT in the scene (a label on a bottle, a sign), or None for an overlay caption.

    Captions and slide text are axis-aligned; text printed on 3-D objects is tilted or in perspective, and tiny text cannot be
    replaced legibly. Measured on real frames: captions 0.0-0.35 deg, labels on bottles 3.2-5.3 deg. Such text is left untouched.
    """
    for line in block.lines:
        points = np.array(line.box, dtype=float)
        top, bottom = points[1] - points[0], points[2] - points[3]
        if abs(float(np.degrees(np.arctan2(top[1], top[0])))) > MAX_CAPTION_TILT_DEGREES:
            return "tilted_text_on_object"
        longest = max(float(np.hypot(*top)), float(np.hypot(*bottom)), 1.0)
        if abs(float(np.hypot(*top)) - float(np.hypot(*bottom))) / longest > MAX_CAPTION_SKEW:
            return "perspective_text_on_object"
    if frame_height and float(np.median([l.height for l in block.lines])) < MIN_CAPTION_HEIGHT_FRACTION * frame_height:
        return "too_small_to_replace"
    return None


def _translation_valid(english: str, hindi: str) -> bool:
    """An entity named in the English (from the packs) must survive in the Hindi; its 'never' words must not appear."""
    source = english.casefold()
    for entity, rule in _ENTITY_PAIRS.items():
        if re.search(rf"\b{re.escape(entity)}\b", source):
            if rule["hindi"] and not any(word in hindi for word in rule["hindi"]):
                return False
            if any(word in hindi for word in rule["never"]):
                return False
    return True


def _safe_translation(english: str, hindi: str) -> str:
    """Apply the packs' wording fixes: a pattern on the English selects which replacements are applied to the Hindi."""
    value = str(hindi).strip()
    lower = english.casefold()
    for fix in _WORDING_FIXES:
        when = fix.get("when", [])
        when = [when] if isinstance(when, str) else list(when)
        if not when or not all(re.search(pattern, lower) for pattern in when):
            continue
        if fix.get("unless") and re.search(fix["unless"], lower):
            continue
        for wrong, right in fix.get("replace", []):
            value = value.replace(wrong, right)
    return value


def _protected_tokens_ok(english: str, hindi: str) -> bool:
    """Numbers and formulas of the English block must come back unchanged."""
    numbers = re.findall(r"\d+(?:\.\d+)?", english)
    flat = hindi.translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    if any(number not in flat for number in numbers):
        return False
    # Symbols that must come back verbatim: pH and chemical-looking tokens such as HCl, NaOH (also the OCR's HCI).
    symbols = set(re.findall(r"\b[a-z]{1,3}[A-Z][A-Za-z]*\b", english)) | {t for t in re.findall(r"\b[A-Z][A-Za-z]?[A-Z][A-Za-z]{0,3}\d*\b", english)
                                                      if len(t) <= 6}
    if any(symbol not in hindi for symbol in symbols):
        return False
    return all(token in hindi for token, _a, _b in _formula_tokens(english))


_ELEMENT_SYMBOLS = set("""H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru
Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U""".split())
_OPERATOR_TOKENS = {"+", "=", "→", "⇌", "⟶", "->", "-->", "–", "-"}
_STATE_TOKEN = re.compile(r"^\((?:g|l|s|aq)\)$")
_FORMULA_TOKEN = re.compile(r"^(?:\d*(?:[A-Z][a-z]?\d*|\([A-Za-z0-9]+\)\d*))+(?:\((?:g|l|s|aq)\))?$")
_NUMBER_TOKEN = re.compile(r"^\d+(?:\.\d+)?$")


def _is_notation(token: str) -> str:
    """'operator', 'formula', 'number', 'state' or '' for ordinary prose."""
    t = token.strip(",;:")
    if t in _OPERATOR_TOKENS:
        return "operator"
    if _NUMBER_TOKEN.match(t):
        return "number"
    if _STATE_TOKEN.match(t):
        return "state"
    if _FORMULA_TOKEN.match(t):
        letters = re.findall(r"[A-Z][a-z]?", t)
        if any(c.isdigit() for c in t) or len(letters) >= 2 or (len(letters) == 1 and letters[0] in _ELEMENT_SYMBOLS and len(t) <= 2):
            return "formula"
    return ""


def mixed_runs(value: str):
    """For a line made only of plain words plus formula/operator tokens ("Carbon + Oxygen -> CO2"): the list of (is_prose, text) runs,
    else None. The prose runs can be translated on their own while every formula, operator and number is kept verbatim, in order."""
    tokens = value.split()
    if len(tokens) < 3:
        return None
    runs, words, notation_kinds = [], 0, set()
    for token in tokens:
        kind = _is_notation(token)
        if kind:
            notation_kinds.add(kind)
        elif re.fullmatch(r"[A-Za-z][A-Za-z'’-]*[,.;:]?", token):
            words += 1
        else:
            return None
        is_prose = not kind
        if runs and runs[-1][0] == is_prose:
            runs[-1] = (is_prose, runs[-1][1] + " " + token)
        else:
            runs.append((is_prose, token))
    if words < 2 or not notation_kinds & {"operator", "formula"}:
        return None
    if any(prose and len(text.replace(" ", "")) < 3 for prose, text in runs):
        return None                                   # a lone one- or two-letter "word" between symbols is a label, not prose
    return runs


def classify_block(text: str, extra: list) -> str:
    """'translate' or the reason a block must stay English."""
    value = clean_line_text(text)
    if value and mixed_runs(value):
        return "mixed_prose_formula"                 # words plus formulas: the words are translated, the notation stays
    if not value or not _CANDIDATE.fullmatch(value):
        return "not_plain_english"
    if len(re.findall(r"[A-Za-z]{2,}", value)) == 0:
        return "no_words"
    words = re.findall(r"[A-Za-z]+", value)
    if len(words) == 1 and len(words[0]) <= 4 and words[0].lower() not in _VOCAB and normalize(value) not in _GLOSSARY:
        return "short_unlisted_word"                 # typed-in / half-faded fragments such as "Bug"
    if has_garbled_token(value):
        return "ocr_garbled_text"                    # glued words the vocabulary cannot repair: leave the English alone
    if _PROTECTED_UNIT.search(value):
        return "contains_unit_or_operator"
    letters = re.sub(r"[^A-Za-z]", "", value)
    if len(letters) >= 4 and letters.isupper() and normalize(value) not in _GLOSSARY:
        return "all_caps_label"                      # bottle/package lettering (SOAP, VANILLA ESSENCE ...) is part of the scene
    if value.casefold() in {t.casefold() for t in extra}:
        return "protected_term"
    return "translate"


def _flag(name: str, yaml_value: bool) -> bool:
    """The bat sets the environment variable to 1 (or 0) and that wins; when it is not set the YAML key decides."""
    value = os.environ.get(name, "").strip()
    return (value == "1") if value in {"0", "1"} else bool(yaml_value)


def is_enabled(config) -> bool:
    """On-screen text translation is OFF unless VIDEO_TEXT_TRANSLATION=1 or production.video_text_translation is true."""
    return _flag("VIDEO_TEXT_TRANSLATION", config.production.video_text_translation)


def is_auto(config) -> bool:
    return _flag("VIDEO_TEXT_TRANSLATION_AUTO", config.production.video_text_translation_auto)


def ocr_every(config) -> int:
    raw = os.environ.get("VIDEO_TEXT_OCR_EVERY", "").strip()
    return max(1, int(raw) if raw.isdigit() else int(config.production.video_text_translation_ocr_every or 3))


def engine_name(config) -> str:
    value = (os.environ.get("VIDEO_TEXT_ENGINE", "") or config.production.video_text_translation_engine or "madlad").strip().lower()
    return value if value in {"madlad", "qwen", "glossary"} else "madlad"


def _alt_python() -> str:
    candidates = [os.environ.get("MADLAD_PYTHON", ""), r"D:\AI\ComfyUI_windows_portable\python_embeded\python.exe"]
    return next((c for c in candidates if c and Path(c).is_file()), "")


def madlad_translate(texts: list, config=None, target: str = "hi") -> list | None:
    """Translate with the local MADLAD-400 worker (same one the Transcribe stage uses). None when it cannot run."""
    python = _alt_python()
    sibling = Path(__file__).resolve().parents[2] / "transcribe" / "translate_text.py"
    script = Path(os.environ.get("MADLAD_SCRIPT") or sibling)
    model_dir = Path(os.environ.get("MADLAD_MODEL_DIR", r"D:\AI\models\madlad400-3b-mt"))
    if not python or not script.is_file() or not model_dir.is_dir() or not texts:
        return None
    request = json.dumps({"target": target, "texts": texts}, ensure_ascii=False) + "\n" + json.dumps({"command": "quit"}) + "\n"
    try:
        run = subprocess.run([python, str(script), "--model-dir", str(model_dir), "--device", os.environ.get("MADLAD_DEVICE", "cuda")],
                             input=request, text=True, encoding="utf-8", capture_output=True, timeout=1800, check=False)
        for line in reversed(run.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{"):
                answer = json.loads(line)
                result = answer.get("translations")
                return result if isinstance(result, list) and len(result) == len(texts) else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return None


def _term_hints(english: str) -> str:
    """'sugar = चीनी; salt = नमक' for every pack term the text uses (entity pairs and one-word glossary entries)."""
    lower = str(english).casefold()
    parts = []
    for entity, rule in _ENTITY_PAIRS.items():
        if re.search(rf"\b{re.escape(entity)}\b", lower) and rule.get("hindi"):
            parts.append(f"{entity} = {rule['hindi'][0]}")
    for word in dict.fromkeys(re.findall(r"[a-z]+", lower)):
        if word in _GLOSSARY and f"{word} =" not in " ".join(parts):
            parts.append(f"{word} = {_GLOSSARY[word]}")
    return "; ".join(parts)


def qwen_translate(texts: list, config, hints: list | None = None) -> list | None:
    """Fallback only: a local Qwen when MADLAD cannot start. Skipped quietly when Ollama is not running."""
    if os.environ.get("OLLAMA_AVAILABLE", "") == "0":
        return None
    try:
        import requests
        schema = {"type": "object", "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
                  "required": ["translations"]}
        if hints and any(hints):
            payload = [{"text": t, "required_terms": h} if h else {"text": t} for t, h in zip(texts, hints)]
            extra = " Where required_terms is given, use exactly those Hindi words for those English words (a school textbook's terms, never a transliteration)."
        else:
            payload, extra = texts, ""
        prompt = ("Translate each classroom screen text from English to natural, concise Hindi (Devanagari), one output per input, "
                  "same order. Keep numbers, formulas and chemical symbols unchanged." + extra + " Input: " + json.dumps(payload, ensure_ascii=False))
        reply = requests.post(config.ollama_url.rstrip("/") + "/api/generate", json={
            "model": config.production.director_model, "prompt": prompt, "stream": False, "format": schema,
            "options": {"temperature": 0}}, timeout=180)
        reply.raise_for_status()
        result = json.loads(reply.json().get("response", "{}")).get("translations")
        return result if isinstance(result, list) and len(result) == len(texts) else None
    except Exception:
        return None


def translate_texts(texts: list, config) -> tuple:
    """(translations, engine). The reviewed glossary wins; MADLAD does the rest; Qwen is the fallback."""
    out: list = [None] * len(texts)
    pending = []
    for index, text in enumerate(texts):
        known = _GLOSSARY.get(normalize(clean_line_text(text)))
        if known:
            out[index] = known
        else:
            pending.append(index)
    engine = "glossary" if not pending else "none"
    if pending:
        batch = [clean_line_text(texts[i]) for i in pending]
        wanted = engine_name(config)
        if wanted == "glossary":
            translated, engine = None, "glossary_only"
        else:
            order = [("qwen", qwen_translate), ("madlad", madlad_translate)] if wanted == "qwen" else                     [("madlad", madlad_translate), ("qwen", qwen_translate)]
            translated, engine = None, "unavailable"
            for name, function in order:
                translated = function(batch, config)
                if translated is not None:
                    engine = name
                    break
        if translated is not None:
            for index, hindi in zip(pending, translated):
                out[index] = hindi
    return out, engine


_STOP = set("the a an and or of in on at to for with from by as is are was were be been it its this that these those their there "
            "then than which who what when where while will can all any some most more not no yes into over under each other".split())
MIN_SAMPLES = 3                    # a block must stay on screen at least 0.75 s; shorter ones are animation fragments ("Buga")
MIN_BACK_OVERLAP = 0.40


def _stems(text: str) -> set:
    return {w[:4] for w in re.findall(r"[a-z]{3,}", str(text).casefold()) if w not in _STOP}


def content_overlap(english: str, back_translation: str) -> float:
    """Share of the English content words that survive English -> Hindi -> English (4-letter stems)."""
    wanted = _stems(english)
    return 1.0 if not wanted else len(wanted & _stems(back_translation)) / len(wanted)


def _load_plan(source: Path) -> tuple:
    path = source.with_suffix(".visual_translations.json")
    if not path.is_file():
        return path, []
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = data if isinstance(data, list) else data.get("replacements", [])
    if not isinstance(rows, list):
        raise ValueError(f"Visual translation plan must contain a replacements list: {path}")
    return path, [row for row in rows if isinstance(row, dict)]


_LEADING_BULLET = re.compile(r"^\s*[•●○◦▪■□·oO0]\s+(?=\S)")
_GLUED_BULLET = re.compile(r"^[oO0](?=[A-Z][a-z]{2})")      # "OSteel:" - an unspaced bullet; a real word never starts O + Capital + lowercase


def _bullet_split(frame, x0: int, y0: int, x1: int, y1: int):
    """(bullet_right, text_left) in frame x, measured from the pixels: a small blob, a clear gap, then the lettering. None if unsure."""
    height = y1 - y0
    if frame is None or height < 8:
        return None
    crop = frame[max(0, y0):y1, max(0, x0):max(0, min(x1, x0 + int(height * 3.5)))].astype(np.int16)
    if crop.shape[0] < 6 or crop.shape[1] < 6:
        return None
    background = np.median(np.concatenate([crop[0], crop[-1]]), axis=0)
    distance = np.abs(crop - background).sum(axis=2)
    ink = distance > max(60.0, 0.5 * float(np.percentile(distance, 99)))
    columns = ink.sum(axis=0) > 0
    runs, start = [], None
    for index, flag in enumerate(columns):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(columns)))
    if len(runs) < 2:
        return None
    (a, b), (c, _d) = runs[0], runs[1]
    if (b - a) <= 1.3 * height and (c - b) >= 0.10 * height:
        return x0 + b, x0 + c
    return None


def _ocr_lines(ocr, frame, minimum: float) -> list:
    """OCR lines at or above `minimum`. A bullet the OCR read as a letter ("O Item: ...") is cut off the line's box, so it is
    neither erased nor translated and the Hindi starts where the original text starts."""
    found, _ = ocr(frame)
    return _lines_from_found(found, frame, minimum)


def _lines_from_found(found, frame, minimum: float) -> list:
    """OCR results (box, text, confidence) in the coordinates of `frame` -> Line objects (bullet handling included)."""
    lines = []
    for box, text, conf in found or []:
        if float(conf) < minimum:
            continue
        text = str(text)
        bullet = _LEADING_BULLET.match(text) or _GLUED_BULLET.match(text)
        keep_left = 0
        if bullet and len(text) > len(bullet.group()):
            points = np.array(box, dtype=np.int32)
            (x0, y0), (x1, y1) = points.min(axis=0), points.max(axis=0)
            split = _bullet_split(frame, int(x0), int(y0), int(x1), int(y1))
            if split:                                  # the real gap between the bullet and the first letter
                keep_left, cut = split[0] + 1, split[1] - int(x0)
            else:                                      # fallback: proportional width of the bullet token plus a margin
                cut = int((x1 - x0) * len(bullet.group()) / len(text)) + int(0.3 * (y1 - y0))
                keep_left = int(x0) + max(0, cut - int(0.3 * (y1 - y0)))
            box = [[x0 + cut, y0], [x1, y0], [x1, y1], [x0 + cut, y1]]
            text = text[bullet.end():]
        line = Line(text, box, float(conf))
        line.keep_left = keep_left
        lines.append(line)
    return lines


def _merge_near_duplicates(hits: dict) -> dict:
    """OCR reads the same block slightly differently from frame to frame ("two words" / "twowords"). Merge those
    variants into the most frequent reading so one Hindi wording is used for the whole stretch (no flicker between phrasings)."""
    merged: dict = {}
    for key, item in sorted(hits.items(), key=lambda kv: -len(kv[1]["times"])):
        for rep_key, rep in merged.items():
            squeezed = lambda t: t.replace(" ", "")
            if re.findall(r"\d+", key) != re.findall(r"\d+", rep_key):
                continue                                   # "4 sugar" and "sugar" (or "10 tin" and "0 5 tin") are not the same text
            if (difflib.SequenceMatcher(None, key, rep_key).ratio() >= 0.92
                    or difflib.SequenceMatcher(None, squeezed(key), squeezed(rep_key)).ratio() >= (0.88 if min(len(key), len(rep_key)) >= 6 else 0.96)):
                rep["times"].extend(item["times"])
                rep["object_sightings"] = rep.get("object_sightings", 0) + item.get("object_sightings", 0)
                break
        else:
            merged[key] = {**item, "times": list(item["times"])}
    return merged


_SECOND_CHANCE = ("meaning_not_preserved", "no_hindi_returned", "implausible_length", "entity_mismatch")


def qwen_judge(pairs: list, config) -> list | None:
    """For each (english, hindi): does the Hindi say the same thing as on-screen lesson text? None when Qwen is unavailable."""
    if os.environ.get("OLLAMA_AVAILABLE", "") == "0" or not pairs:
        return None
    try:
        import requests
        schema = {"type": "object", "properties": {"faithful": {"type": "array", "items": {"type": "boolean"}}}, "required": ["faithful"]}
        prompt = ("For each pair, answer true only if the Hindi is a correct, natural translation of the English screen text of a school "
                  "lesson: same meaning, nothing added or lost, correct school-science terminology. A synonym is fine; a different meaning "
                  "is false. One boolean per pair, same order. Pairs: " + json.dumps([{"english": e, "hindi": h} for e, h in pairs], ensure_ascii=False))
        reply = requests.post(config.ollama_url.rstrip("/") + "/api/generate", json={
            "model": config.production.director_model, "prompt": prompt, "stream": False, "format": schema,
            "options": {"temperature": 0}}, timeout=180)
        reply.raise_for_status()
        result = json.loads(reply.json().get("response", "{}")).get("faithful")
        return result if isinstance(result, list) and len(result) == len(pairs) else None
    except Exception:
        return None


def _second_opinion(rejected_items: list, config) -> dict:
    """english -> Hindi for rejected blocks that Qwen can translate AND a separate Qwen check judges faithful. Same hard checks apply."""
    if not rejected_items:
        return {}
    candidates = qwen_translate([clean_line_text(i["english"]) for i in rejected_items], config,
                                [_term_hints(i["english"]) for i in rejected_items])
    if not candidates:
        return {}
    ready = []
    for item, hindi in zip(rejected_items, candidates):
        hindi = _safe_translation(item["english"], str(hindi or ""))
        if (re.search(r"[\u0900-\u097f]", hindi) and not re.search(r"[\u3000-\u9fff\uac00-\ud7ff\u0400-\u04ff]", hindi)
                and _translation_valid(item["english"], hindi) and _protected_tokens_ok(item["english"], hindi)
                and len(hindi) <= 4.0 * len(item["english"]) + 12):
            ready.append((item["english"], hindi))
    verdicts = qwen_judge(ready, config) or []
    return {english: hindi for (english, hindi), ok in zip(ready, verdicts) if ok is True}


def _is_subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(ch in it for ch in small)


def _absorb_reveal_fragments(hits: dict) -> None:
    """A title that is revealed letter by letter is read by OCR as "R Y", "I STRY", "E MISTRY"... Those readings are parts of the
    finished word that sits in the same place a moment later. Fold them into it: the word's window starts with the first fragment and
    the row remembers where the finished word is, so the Hindi can replace the animation from its first letter.

    Strict on purpose (a loose rule once swallowed unrelated labels): the finished word is a single short line; a fragment is NOT itself
    a translatable text, has at least two letters in the word's order, sits inside the word's box on the same line, and was seen only
    in the moments just before the finished word (never at unrelated times)."""
    letters = lambda text: re.sub(r"[^a-z]", "", str(text).casefold())
    for key, item in list(hits.items()):
        full = letters(key)
        if item.get("reason") != "translate" or not 6 <= len(full) <= 20 or not item.get("bbox") or key not in hits:
            continue
        x0, y0, x1, y1 = item["bbox"]
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        first, last = min(item["times"]), max(item["times"])
        for other_key, other in list(hits.items()):
            part = letters(other_key)
            if other_key == key or not other.get("bbox") or other.get("reason") == "translate":
                continue
            if not 2 <= len(part) < len(full) or not _is_subsequence(part, full):
                continue
            ox0, oy0, ox1, oy1 = other["bbox"]
            inside = ox0 >= x0 - 0.2 * width and ox1 <= x1 + 0.2 * width and abs((oy0 + oy1) - (y0 + y1)) / 2.0 <= 0.5 * height
            just_before = all(first - 2.5 <= t <= last + 0.3 for t in other["times"])
            if inside and just_before:
                item.setdefault("fragments", []).append(other_key)
                item["times"] = sorted(item["times"] + other["times"])
                del hits[other_key]


def _graphics_beside(block, blocks: list) -> bool:
    """True when any other OCR block (an arrow label, a catalyst, a note) sits within a line height above/below or beside this one."""
    x0, y0, x1, y1 = block.bbox
    height, width = max(1, y1 - y0), max(1, x1 - x0)
    for other in blocks:
        if other is block:
            continue
        ox0, oy0, ox1, oy1 = other.bbox
        small_note = len(re.sub(r"[^A-Za-z]", "", other.text)) <= 12 or (oy1 - oy0) < 0.75 * height
        if small_note and ox0 < x1 + 0.1 * width and ox1 > x0 - 0.1 * width and oy0 < y1 + height and oy1 > y0 - height:
            return True                                # a neighbouring full-size text line (the next bullet) is not a graphic
    return False


def _keep_equation_halves_together(hits: dict) -> None:
    """An equation or mixed label that must stay English is often wrapped over two lines. Translating only one of the halves leaves a
    half-English, half-Hindi box, so a translatable block stacked directly against such a block (same time, same column) stays English."""
    stay = [v for v in hits.values() if v["reason"] in {"not_plain_english", "contains_unit_or_operator", "ocr_garbled_text", "equation_with_graphics"}
            and len(re.findall(r"[A-Za-z]{3,}", v["english"])) >= 1 and v.get("bbox")]
    for item in hits.values():
        if item["reason"] != "translate" or not item.get("bbox"):
            continue
        ax0, ay0, ax1, ay1 = item["bbox"]
        line_height = max(1, ay1 - ay0)
        for other in stay:
            bx0, by0, bx1, by1 = other["bbox"]
            vertical_gap = max(by0 - ay1, ay0 - by1)
            overlap_x = min(ax1, bx1) - max(ax0, bx0)
            shared = len(set(item["times"]) & set(other["times"]))
            if vertical_gap <= 1.2 * line_height and overlap_x > 0.3 * min(ax1 - ax0, bx1 - bx0) \
                    and shared >= 0.5 * min(len(item["times"]), len(other["times"])):
                item["reason"] = "half_of_untranslated_equation"
                break


def _auto_plan(source: Path, config, report: dict) -> tuple:
    """Sample the video, group OCR lines into blocks, translate each distinct block once and persist the plan."""
    import cv2
    from rapidocr_onnxruntime import RapidOCR

    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = count / fps if fps > 0 else 0
    plan_path = source.with_suffix(".visual_translations.auto.json")
    if fps <= 0 or duration <= 0:
        capture.release()
        return plan_path, []
    ocr, hits = RapidOCR(), {}
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    minimum = float(config.production.video_text_translation_min_confidence)
    extra = config.production.video_text_translation_protected_terms
    for second in np.arange(0, duration, DISCOVERY_STEP_SECONDS):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(second) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        blocks = group_blocks(_ocr_lines(ocr, frame, minimum))
        for block in blocks:
            key = normalize(block_text(block))
            if len(key) < 3:
                continue
            item = hits.setdefault(key, {"english": block_text(block), "times": [], "object_sightings": 0,
                                         "reason": classify_block(block_text(block), extra), "bbox": block.bbox})
            item["times"].append(float(second))
            if scene_object_reason(block, frame_height):
                item["object_sightings"] += 1
            if item["reason"] == "mixed_prose_formula" and _graphics_beside(block, blocks):
                item["decor_sightings"] = item.get("decor_sightings", 0) + 1
    capture.release()
    hits = _merge_near_duplicates(hits)
    for item in hits.values():
        # an equation drawn with an arrow, a catalyst above it and a note below is a graphic: redrawing only its words would break it
        if item["reason"] == "mixed_prose_formula" and item.get("decor_sightings", 0) * 2 >= len(item["times"]):
            item["reason"] = "equation_with_graphics"
    for item in hits.values():                      # mostly seen tilted/in perspective -> text on an object, not a caption
        if item["reason"] == "translate" and item.get("object_sightings", 0) * 2 >= len(item["times"]):
            item["reason"] = "scene_object_text"
    _absorb_reveal_fragments(hits)
    _keep_equation_halves_together(hits)
    brief = {k: v for k, v in hits.items() if len(v["times"]) < MIN_SAMPLES and normalize(v["english"]) not in _GLOSSARY}
    hits = {k: v for k, v in hits.items() if k not in brief}
    report["blocks_too_brief"] = sorted(v["english"] for v in brief.values())[:60]
    translatable = {k: v for k, v in hits.items() if v["reason"] == "translate"}
    for key, item in list(hits.items()):
        if item["reason"] != "mixed_prose_formula":
            continue
        item["runs"] = []
        for number, (prose, text) in enumerate(mixed_runs(item["english"]) or []):
            sub = f"{key}§{number}"
            if prose:
                translatable[sub] = {"english": text, "times": item["times"], "reason": "translate", "hidden": True}
            item["runs"].append((prose, text, sub if prose else None))
    report["blocks_found"] = len(hits)
    report["blocks_not_translatable"] = {k: v["reason"] for k, v in hits.items() if v["reason"] != "translate"}
    if not translatable:
        return plan_path, []
    keys = list(translatable)
    translated, engine = translate_texts([translatable[k]["english"] for k in keys], config)
    report["translation_engine"] = engine
    checks = [i for i, (k, h) in enumerate(zip(keys, translated))
              if h and normalize(translatable[k]["english"]) not in _GLOSSARY and re.search(r"[ऀ-ॿ]", h)]
    back = madlad_translate([_safe_translation(translatable[keys[i]]["english"], translated[i]) for i in checks], config, target="en")         if checks and engine == "madlad" else None
    back_by_index = dict(zip(checks, back)) if back else {}
    def windows(item: dict, hindi: str) -> list:
        out = []
        times = sorted(item["times"])
        start = previous = times[0]
        for current in times[1:] + [None]:
            if current is not None and current - previous <= WINDOW_GAP_SECONDS:
                previous = current
                continue
            row = {"english": item["english"], "hindi": hindi, "start": max(0, round(start - 0.25, 2)),
                   "end": round(min(duration, previous + 0.25), 2), "source": f"auto_ocr_{engine}"}
            if item.get("fragments") and item.get("bbox"):
                row["reveal_box"] = [int(v) for v in item["bbox"]]        # where the finished word sits (letter-by-letter reveal)
            out.append(row)
            if current is not None:
                start = previous = current
        return out

    def build(second: dict) -> tuple:
        rows, rejected, accepted = [], {}, {}
        for index, (key, hindi) in enumerate(zip(keys, translated)):
            item = translatable[key]
            hindi = second.get(item["english"]) or _safe_translation(item["english"], hindi or "")
            if not re.search(r"[\u0900-\u097f]", hindi):
                rejected[item["english"]] = "no_hindi_returned"
            elif re.search(r"[\u3000-\u9fff\uac00-\ud7ff\u0400-\u04ff]", hindi):
                rejected[item["english"]] = "foreign_script"
            elif not _translation_valid(item["english"], hindi):
                rejected[item["english"]] = "entity_mismatch"
            elif not _protected_tokens_ok(item["english"], hindi):
                rejected[item["english"]] = "numbers_or_formula_changed"
            elif len(hindi) > 4.0 * len(item["english"]) + 12:
                rejected[item["english"]] = "implausible_length"
            elif (index in back_by_index and not second.get(item["english"])
                  and content_overlap(item["english"], back_by_index[index]) < MIN_BACK_OVERLAP):
                rejected[item["english"]] = f"meaning_not_preserved (back-translation: {str(back_by_index[index])[:60]})"
            elif item.get("hidden"):
                accepted[item["english"]] = hindi           # a prose run of a mixed line: used only to assemble its parent below
            else:
                rows.extend(windows(item, hindi))
        for item in hits.values():
            if item["reason"] != "mixed_prose_formula":
                continue
            pieces, complete = [], True
            for prose, text, sub in item.get("runs", []):
                if not prose:
                    pieces.append(text)
                elif translatable[sub]["english"] in accepted:
                    pieces.append(accepted[translatable[sub]["english"]])
                else:
                    complete = False
            hindi = " ".join(pieces)
            if not complete:
                rejected[item["english"]] = "mixed_line_words_not_translated"
            elif not _protected_tokens_ok(item["english"], hindi) or not _translation_valid(item["english"], hindi):
                rejected[item["english"]] = "numbers_or_formula_changed"
            else:
                rows.extend(windows(item, hindi))
        return rows, rejected

    rows, rejected = build({})
    retry = [translatable[k] for k in keys if str(rejected.get(translatable[k]["english"], "")).startswith(_SECOND_CHANCE)]
    second = _second_opinion(retry, config)
    if second:
        rows, rejected = build(second)
    report["second_opinion_accepted"] = second
    planned = {row["english"] for row in rows}
    report["blocks_not_translatable"] = {k: v["reason"] for k, v in hits.items()
                                         if v["reason"] != "translate" and v["english"] not in planned}
    report["blocks_rejected"] = rejected
    plan_path.write_text(json.dumps({"source_video": str(source), "engine": engine, "replacements": rows,
                                     "rejected": rejected}, indent=2, ensure_ascii=False), encoding="utf-8")
    return plan_path, rows


def _require_shaping():
    """Devanagari drawn without a shaping engine is misspelt (matras and half-letters break). Refuse instead."""
    from .typography import _is_shaped, font
    if not _is_shaped(font(24, True, True)):
        raise RuntimeError(
            "No Hindi text-shaping engine (PySide6/Qt) in this Python, so on-screen Hindi would be drawn misspelt. Run the video "
            "text translation with a Python that has PySide6 (run_video.ps1 picks one automatically when VIDEO_TEXT_TRANSLATION=1).")


def _text_colour(frame, bbox, lines) -> tuple:
    """Colour of the original lettering: the pixels that differ most from the surrounding background."""
    x0, y0, x1, y1 = bbox
    pad = max(4, int((y1 - y0) * 0.25))
    h, w = frame.shape[:2]
    ring = np.concatenate([frame[max(0, y0 - pad):y0, max(0, x0):x1].reshape(-1, 3),
                           frame[y1:min(h, y1 + pad), max(0, x0):x1].reshape(-1, 3)]) if y1 > y0 else frame[y0:y1, x0:x1].reshape(-1, 3)
    inside = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    if ring.size == 0 or inside.size == 0:
        return (255, 255, 255)
    background = np.median(ring, axis=0).astype(np.int32)
    distance = np.abs(inside - background).sum(axis=1)
    far = inside[distance >= np.percentile(distance, 85)]
    colour = np.median(far, axis=0) if len(far) else np.array([255, 255, 255])
    # On a busy, multicoloured backdrop (a label on a tile) the ring says little about the background: it can contain a border or a
    # neighbouring tile and yield the TILE colour. Thin lettering is the minority inside its own box, so the median is the background
    # and the lettering is what differs most from it. The candidate that contrasts most with the box background wins.
    if len(inside) >= 40:
        own = np.median(inside, axis=0)
        gap = np.abs(inside - own).sum(axis=1)
        ink = inside[gap >= np.percentile(gap, 92)]
        if len(ink):
            candidate = np.median(ink, axis=0)
            if np.abs(candidate - own).sum() > 1.3 * np.abs(colour - own).sum():
                colour = candidate
    return int(colour[2]), int(colour[1]), int(colour[0])          # BGR -> RGB


def _bullet_before(frame, line) -> bool:
    """True when a small bullet-like blob sits just left of this line (so the line is a list item, not a centred label)."""
    height = line.y1 - line.y0
    xa, xb = max(0, line.x0 - int(2.5 * height)), line.x0 - 1
    if height < 8 or xb - xa < 6:
        return False
    crop = frame[max(0, line.y0):line.y1, xa:xb].astype(np.int16)
    if crop.size == 0:
        return False
    background = np.median(crop.reshape(-1, 3), axis=0)
    ink = np.abs(crop - background).sum(axis=2) > 90
    columns = np.where(ink.sum(axis=0) > 0)[0]
    if len(columns) == 0:
        return False
    last = [int(columns[-1])]                       # the blob nearest to the text: walk left while columns stay inked
    for column in columns[::-1][1:]:
        if last[-1] - int(column) > 1:
            break
        last.append(int(column))
    left, right = min(last), max(last)
    rows = np.where(ink[:, left:right + 1].sum(axis=1) > 0)[0]
    blob_h = (rows.max() - rows.min() + 1) if len(rows) else 0
    gap = (xb - xa) - right
    return 0.15 * height <= (right - left + 1) <= 1.3 * height and 0.15 * height <= blob_h <= 0.85 * height and gap >= 0.05 * height


def _flat_fill(frame, mask, pad: int, line_h: int = 30, text_rgb=None):
    """Erase lettering on a plain or smoothly shaded backdrop (a caption pill, a slide) without smearing: the background under the strokes
    is estimated by a morphological opening (light text) or closing (dark text), which removes thin strokes and keeps the gradient.
    None for a busy backdrop (a scene), which gets inpainted instead."""
    import cv2
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return frame.copy()
    k = max(5, int(0.40 * line_h)) | 1
    h, w = mask.shape
    y0, y1 = max(0, ys.min() - k), min(h, ys.max() + k + 1)
    x0, x1 = max(0, xs.min() - k), min(w, xs.max() + k + 1)
    crop, local = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 0
    ring = (cv2.dilate(local.astype(np.uint8), np.ones((2 * pad + 3, 2 * pad + 3), np.uint8)) > 0) & ~local
    if ring.sum() < 30 or local.sum() < 30:
        return None
    ring_px = crop[ring].astype(np.int32)
    background = np.median(ring_px, axis=0)
    if float(np.percentile(np.abs(ring_px - background).sum(axis=1), 50)) > 35.0:     # half the surroundings must match the background
        return None                                     # the surroundings are not a plain backdrop
    if text_rgb is not None:                           # the lettering's own colour decides: light strokes are opened away, dark ones closed
        bright_text = (0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]) > float(background.mean())
    else:
        bright_text = float(crop[local].astype(np.int32).mean()) > float(background.mean())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    estimate = cv2.morphologyEx(crop, cv2.MORPH_OPEN if bright_text else cv2.MORPH_CLOSE, kernel)
    estimate = cv2.GaussianBlur(estimate, (0, 0), 1.2)
    # Replace only the lettering itself (what differs from the estimated backdrop, plus its anti-aliased halo). Every other pixel of the
    # erased box keeps its true value, so a gradient or a bright patch beside the text is never darkened into a visible rectangle.
    difference = np.abs(crop.astype(np.int32) - estimate.astype(np.int32)).sum(axis=2)
    strokes = ((difference > 40) & local).astype(np.uint8)
    strokes = (cv2.dilate(strokes, np.ones((3, 3), np.uint8), iterations=2) > 0) & local
    out = frame.copy()
    region = out[y0:y1, x0:x1]
    region[strokes] = estimate[strokes]
    return out


def _stroke_inpaint(frame, mask, line_h: int, text_rgb):
    """For a backdrop that is not plain (a gradient, a scene): find the lettering pixels (what differs from a morphologically estimated
    backdrop) and inpaint ONLY those. Inpainting the whole erased box smeared a dark rectangle into the picture. None when the lettering
    cannot be isolated (then the whole box is inpainted as before)."""
    import cv2
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    k = max(5, int(0.40 * line_h)) | 1
    h, w = mask.shape
    y0, y1 = max(0, ys.min() - k), min(h, ys.max() + k + 1)
    x0, x1 = max(0, xs.min() - k), min(w, xs.max() + k + 1)
    crop, local = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 0
    if local.sum() < 30:
        return None
    ring = (cv2.dilate(local.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & ~local
    background = np.median(crop[ring].astype(np.int32), axis=0) if ring.sum() >= 30 else np.median(crop[local].astype(np.int32), axis=0)
    bright_text = (0.299 * text_rgb[0] + 0.587 * text_rgb[1] + 0.114 * text_rgb[2]) > float(background.mean())
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    estimate = cv2.morphologyEx(crop, cv2.MORPH_OPEN if bright_text else cv2.MORPH_CLOSE, kernel)
    difference = np.abs(crop.astype(np.int32) - estimate.astype(np.int32)).sum(axis=2)
    strokes = ((difference > 45) & local).astype(np.uint8)
    if strokes.sum() == 0 or strokes.sum() > 0.55 * local.sum():
        return None
    strokes = (cv2.dilate(strokes, np.ones((3, 3), np.uint8), iterations=2) > 0) & local
    patch = cv2.inpaint(np.ascontiguousarray(crop), strokes.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
    out = frame.copy()
    out[y0:y1, x0:x1] = patch
    return out


def _paint_block(frame, lines: list, hindi: str, alpha: float = 1.0):
    import cv2
    from PIL import Image, ImageDraw
    from .typography import font, put_text, text_width, wrap_text

    height, width = frame.shape[:2]
    erase_lines = lines
    lines = [l for l in lines if not getattr(l, "erase_only", False)] or lines
    x0, y0 = min(l.x0 for l in lines), min(l.y0 for l in lines)
    x1, y1 = max(l.x1 for l in lines), max(l.y1 for l in lines)
    line_h = int(np.median([l.height for l in lines]))
    pad = max(3, round(line_h * 0.20))
    colour = _text_colour(frame, (max(0, x0), max(0, y0), min(width, x1), min(height, y1)), lines)
    mask = np.zeros((height, width), np.uint8)
    for line in erase_lines:
        cv2.fillPoly(mask, [np.array(line.box, dtype=np.int32)], 255)
    # Wide horizontally, but only a little vertically: stacked lines sit a few pixels apart, and an erase that reaches into the next line
    # damages its lettering (the white wisps between a heading and its bullets).
    pad_y = max(2, min(pad, int(0.10 * line_h)))
    mask = cv2.dilate(mask, np.ones((pad_y * 2 + 1, pad * 2 + 1), np.uint8))
    for line in erase_lines:
        if getattr(line, "keep_left", 0):
            mask[max(0, line.y0 - pad):line.y1 + pad + 1, :line.keep_left] = 0       # a bullet stays exactly as drawn
    cleaned = _flat_fill(frame, mask, pad, line_h, colour)
    if cleaned is None:
        cleaned = _stroke_inpaint(frame, mask, line_h, colour)
    if cleaned is None:
        cleaned = cv2.inpaint(frame, mask, max(3, pad), cv2.INPAINT_TELEA)
    image = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
    base = image.copy() if alpha < 0.999 else None            # the erased picture, without Hindi, for a fade
    draw = ImageDraw.Draw(image)
    # A single label sits centred in its pill; a wrapped paragraph keeps its left edge unless its lines share one centre.
    x_spread = max(l.x0 for l in lines) - min(l.x0 for l in lines)
    c_spread = max(l.center_x for l in lines) - min(l.center_x for l in lines)
    centred = len(lines) == 1 or (x_spread > 0.6 * line_h and c_spread <= 0.5 * line_h)
    if centred and len(lines) == 1 and (lines[0].keep_left or _bullet_before(frame, lines[0])):
        centred = False                              # a list item stays left-aligned beside its bullet
    widen = False
    if centred and len(lines) == 1 and " " not in hindi.strip():
        probe = font(max(14, int(line_h * 0.88)), True, True)
        if text_width(draw, hindi, probe) > 1.15 * (x1 - x0):
            centred = False                          # wider than the English it replaces: keep its left edge instead of spilling both ways
            widen = True
    centre_x = (x0 + x1) / 2.0
    box_w = max(40, int((x1 - x0) * 1.0))
    if widen:
        box_w = int(box_w * 1.8)
    if centred and " " not in hindi.strip():
        box_w = int(box_w * 1.8)                      # one word is never broken over two lines: it may be wider than a tiny label
    box_w = min(box_w, int(2 * min(centre_x, width - centre_x)) - 8) if centred else min(box_w, width - x0 - 4)
    # Left-aligned paragraphs start where the FIRST line's text starts (right of its bullet), also for hanging-indent bullets.
    left_edge = centre_x - box_w / 2.0 if centred else float(lines[0].x0) + 0.2 * line_h      # small gap after the bullet
    if not centred:
        box_w = min(box_w, width - int(left_edge) - 4)
    size = max(14, int(line_h * 0.88))
    while True:
        shaped = font(size, True, True)
        rows = wrap_text(draw, hindi, shaped, box_w)
        ascent, descent = shaped.getmetrics()
        step = ascent + descent + max(2, size // 9)
        if (len(rows) * step <= (y1 - y0) * 1.30 + pad * 2 or size <= 12) and all(text_width(draw, r, shaped) <= box_w + 2 for r in rows):
            break
        size -= 1
        if size < 11:
            break
    luminance = 0.299 * colour[0] + 0.587 * colour[1] + 0.114 * colour[2]
    outline = (12, 12, 12) if luminance > 128 else (245, 245, 245)
    # A centred label is centred vertically in its box; a left-aligned paragraph starts at the top, beside its bullet.
    top = y0 + (max(0, ((y1 - y0) - len(rows) * step) // 2) if centred else 0)
    if not centred and rows:
        # Devanagari carries its head-line above and its marks below, so top-aligning drops the Hindi under the bullet. Put the
        # first row's ink centre on the centre of the original first line, where the bullet sits.
        try:
            ink = shaped.render(rows[0], colour).getchannel("A").getbbox()
            if ink:
                top = int(round(((lines[0].y0 + lines[0].y1) / 2.0) - (ink[1] + ink[3]) / 2.0 + 2))
        except Exception:
            pass
    for index, row in enumerate(rows):
        row_x = left_edge + (max(0.0, (box_w - text_width(draw, row, shaped)) / 2) if centred else 0.0)
        put_text(draw, (row_x, top + index * step), row, shaped, colour, stroke=max(1, size // 22), stroke_fill=outline)
    if base is not None:
        image = Image.blend(base, image, max(0.0, min(1.0, alpha)))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------------------------------------------------------------
# Repair of approved labels the first render missed (runs only inside translate_video_text, i.e. only when the feature is on).
# The first pass is the normal OCR on the original frames. A label it never painted (or painted for far too few frames) gets a second
# OCR pass on enlarged, contrast-enhanced frames around its own timestamps, then the render is replayed from the paint log.
# ---------------------------------------------------------------------------------------------------------------------------------
REPAIR_MIN_COVERAGE = 0.35        # painted frames / frames in the label's window below this = the label needs a repair pass
REPAIR_MAX_GAP_FRAMES = 8         # detections this close are joined by interpolating the box (no flicker between them)
REPAIR_EDGE_FRAMES = 3            # a detection is held this many frames beyond its first and last sighting (fades), never more
REPAIR_MAX_FRAMES_PER_ROW = 60    # cap on frames re-read per label
REPAIR_CONFIDENCE = 0.50
# (scale, enhancement, tiles per side). Attempt 1: the whole frame enlarged. Attempt 2: the frame cut into overlapping tiles so small
# text is really magnified (the OCR detector caps the image size), and an inverted high-contrast reading.
REPAIR_VARIANTS = [
    [(2.0, "clahe_unsharp", 1), (3.0, "clahe_unsharp", 1)],
    [(3.0, "clahe_unsharp", 2), (3.0, "invert_gray", 2)],
]


def repair_enabled(config) -> bool:
    return _flag("VIDEO_TEXT_REPAIR", getattr(config.production, "video_text_translation_repair", True))


def repair_attempts(config) -> int:
    raw = os.environ.get("VIDEO_TEXT_REPAIR_ATTEMPTS", "").strip()
    value = int(raw) if raw.isdigit() else int(getattr(config.production, "video_text_translation_repair_attempts", 2) or 0)
    return max(0, min(len(REPAIR_VARIANTS), value))


def _timecode(seconds) -> str:
    total = max(0.0, float(seconds or 0.0))
    return f"{int(total // 60):02d}:{total - 60 * int(total // 60):05.2f}"


def _needs_repair(row: dict, hits: int, fps: float) -> bool:
    window = max(1.0, (float(row.get("end", 0)) - float(row.get("start", 0))) * fps)
    return hits < max(1, int(REPAIR_MIN_COVERAGE * window))


def _enhanced(frame, scale: float, mode: str):
    """Enlarged, contrast-enhanced copy of a frame (CLAHE on lightness, optional unsharp mask, or an inverted grey reading)."""
    import cv2
    big = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if mode == "invert_gray":
        grey = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(cv2.cvtColor(big, cv2.COLOR_BGR2GRAY))
        return cv2.cvtColor(255 - grey, cv2.COLOR_GRAY2BGR)
    lightness, a, b = cv2.split(cv2.cvtColor(big, cv2.COLOR_BGR2LAB))
    lightness = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lightness)
    out = cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)
    if "unsharp" in mode:
        out = cv2.addWeighted(out, 1.8, cv2.GaussianBlur(out, (0, 0), 2), -0.8, 0)
    return out


def _rescue_lines(ocr, frame, scale: float, mode: str, tiles: int) -> list:
    """OCR on enlarged tiles of the frame; boxes are mapped back to the frame's own pixel coordinates."""
    height, width = frame.shape[:2]
    found = []
    overlap = 0.15 if tiles > 1 else 0.0
    for row in range(tiles):
        for column in range(tiles):
            x0 = max(0, int(width * (column / tiles - (overlap if column else 0))))
            x1 = min(width, int(width * ((column + 1) / tiles + (overlap if column < tiles - 1 else 0))))
            y0 = max(0, int(height * (row / tiles - (overlap if row else 0))))
            y1 = min(height, int(height * ((row + 1) / tiles + (overlap if row < tiles - 1 else 0))))
            result, _ = ocr(_enhanced(frame[y0:y1, x0:x1], scale, mode))
            for box, text, conf in result or []:
                found.append(([[x0 + p[0] / scale, y0 + p[1] / scale] for p in box], text, conf))
    return _lines_from_found(found, frame, REPAIR_CONFIDENCE)


def _reveal_lines(row: dict) -> list:
    """One line covering the finished word of a letter-by-letter title (a fixed place, so the Hindi does not jump as letters appear)."""
    if "_reveal_lines" not in row:
        x0, y0, x1, y1 = row["reveal_box"]
        row["_reveal_lines"] = [Line(row["english"], [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], 1.0)]
    return row["_reveal_lines"]


def _fragment_fits(block, row: dict, loose: bool = False) -> bool:
    """True when a block's letters are a part of the row's word and sit where the finished word sits (loose: anywhere on that line,
    because an animated title moves while it is being revealed)."""
    letters = re.sub(r"[^a-z]", "", block_text(block).casefold())
    full = re.sub(r"[^a-z]", "", row["key"])
    if len(letters) < 2 or len(letters) >= len(full) + 1 or not _is_subsequence(letters, full):
        return False
    x0, y0, x1, y1 = row["reveal_box"]
    bx0, by0, bx1, by1 = block.bbox
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    if loose:
        return abs((by0 + by1) - (y0 + y1)) / 2.0 <= 0.5 * height and bx1 - bx0 <= 1.2 * width
    return bx0 >= x0 - 0.2 * width and bx1 <= x1 + 0.2 * width and abs((by0 + by1) - (y0 + y1)) / 2.0 <= 0.5 * height


def _reveal_paint_lines(row: dict, ocr_lines: list, matched=None) -> list:
    """The finished word's place (for the Hindi) plus every visible fragment of the animated title (erased, wherever it currently sits).
    Works on the raw OCR lines, so a lone letter ("H") that grouping would drop is erased too."""
    full = re.sub(r"[^a-z]", "", row["key"])
    x0, y0, x1, y1 = row["reveal_box"]
    width, height = max(1, x1 - x0), max(1, y1 - y0)
    taken = {id(l) for l in (matched.lines if matched is not None else [])}
    extra = []
    for line in ocr_lines:
        letters = re.sub(r"[^a-z]", "", line.text.casefold())
        if id(line) in taken or not letters or not _is_subsequence(letters, full):
            continue
        if abs((line.y0 + line.y1) - (y0 + y1)) / 2.0 > 0.5 * height or line.x1 - line.x0 > 1.2 * width:
            continue
        if not 0.5 * height <= line.y1 - line.y0 <= 1.3 * height:
            continue                                   # a logo's own letter is much taller than the title's lettering
        if line.x1 < x0 - 0.35 * width or line.x0 > x1 + 0.2 * width:
            continue
        copy = Line(line.text, line.box, line.confidence)
        copy.erase_only = True
        extra.append(copy)
    return _reveal_lines(row) + extra


def _label_contrast(frame, lines: list) -> float:
    """How strongly the lettering stands out from the background inside its own box (falls to ~0 while a label fades out)."""
    x0, y0 = max(0, min(l.x0 for l in lines)), max(0, min(l.y0 for l in lines))
    x1, y1 = max(l.x1 for l in lines), max(l.y1 for l in lines)
    inside = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
    if len(inside) < 20:
        return 0.0
    gap = np.abs(inside - np.median(inside, axis=0)).sum(axis=1)
    return float(np.percentile(gap, 92))


def _match_label(text: str, key: str, hindi: str):
    """The Hindi to paint when an OCR block IS the approved label, else None. Besides a near-exact match, the label may sit inside a
    block together with numbers ("4% Sugar"): the numbers stay, in place, around the Hindi."""
    if re.findall(r"\d+", text) == re.findall(r"\d+", key) and difflib.SequenceMatcher(None, normalize(text), key).ratio() >= 0.85:
        return hindi
    words, key_words = text.split(), key.split()
    count = len(key_words)
    if not count or len(words) <= count:
        return None
    normal = [normalize(w) for w in words]
    for start in range(len(words) - count + 1):
        if all(difflib.SequenceMatcher(None, normal[start + k], key_words[k]).ratio() >= 0.85 for k in range(count)):
            extras = words[:start] + words[start + count:]
            if all(re.fullmatch(r"[\d.,%°()+\-–/:]+", w) for w in extras):
                return " ".join(words[:start] + [hindi] + words[start + count:])
            return None
    return None


def _lerp_lines(first: list, second: list, t: float) -> list:
    if len(first) != len(second):
        return first
    out = []
    for a, b in zip(first, second):
        box = [[p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t] for p, q in zip(a.box, b.box)]
        line = Line(a.text, box, a.confidence)
        line.keep_left = a.keep_left
        out.append(line)
    return out


def _fill_detections(found: dict, total: int) -> dict:
    """Detections -> a continuous run of (lines, text, opacity). Gaps up to REPAIR_MAX_GAP_FRAMES are interpolated (box and contrast),
    each end is held a few frames, and the opacity follows the label's own contrast so a fading label is repainted fading, not
    suddenly solid (no lingering overlay when the original has already faded out)."""
    frames = sorted(found)
    contrast = {n: found[n][2] for n in frames}
    out = {n: (found[n][0], found[n][1], contrast[n]) for n in frames}
    for a, b in zip(frames, frames[1:]):
        if 1 < b - a <= REPAIR_MAX_GAP_FRAMES:
            for n in range(a + 1, b):
                t = (n - a) / (b - a)
                out[n] = (_lerp_lines(found[a][0], found[b][0], t), found[a][1] if t < 0.5 else found[b][1],
                          contrast[a] + (contrast[b] - contrast[a]) * t)
    for edge, step in ((frames[0], -1), (frames[-1], 1)):
        for k in range(1, REPAIR_EDGE_FRAMES + 1):
            n = edge + step * k
            if 0 <= n < total and n not in out:
                out[n] = (found[edge][0], found[edge][1], contrast[edge] * (1.0 - k / (REPAIR_EDGE_FRAMES + 1)))
    reference = max(0.75 * max(v[2] for v in out.values()), 1.0)
    faded = {n: (lines, text, min(1.0, c / reference)) for n, (lines, text, c) in out.items()}
    return {n: v for n, v in faded.items() if v[2] >= 0.12}        # a label that has all but faded out is not repainted


def _rescue_row(capture, ocr, row: dict, already: set, fps: float, total: int, variants: list) -> dict:
    """{frame number: (lines, Hindi text)} for the frames of this label's window that the first pass left unpainted."""
    import cv2
    first = max(0, int((float(row.get("start", 0)) - WINDOW_PAD_SECONDS) * fps))
    last = min(total - 1, int((float(row.get("end", 0)) + WINDOW_PAD_SECONDS) * fps))
    frames = [n for n in range(first, last + 1) if n not in already]
    if len(frames) > REPAIR_MAX_FRAMES_PER_ROW:
        frames = frames[::-(-len(frames) // REPAIR_MAX_FRAMES_PER_ROW)]
    found = {}
    print(f"VISUAL_TEXT_REPAIR: re-reading '{row['english'][:40]}' {_timecode(row.get('start'))}-{_timecode(row.get('end'))} "
          f"({len(frames)} frames)", flush=True)
    for n in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, n)
        ok, frame = capture.read()
        if not ok:
            continue
        for scale, mode, tiles in variants:
            hit = None
            for block in group_blocks(_rescue_lines(ocr, frame, scale, mode, tiles)):
                painted = _match_label(block_text(block), row["key"], row["hindi"])
                if painted:
                    hit = (block.lines, painted)
                    break
            if hit:
                found[n] = (hit[0], hit[1], _label_contrast(frame, hit[0]))
                break
    return found


def _painted_frames(paint_log: dict, rows: int) -> list:
    painted = [set() for _ in range(rows)]
    for n, entries in paint_log.items():
        for entry in entries:
            painted[entry[0]].add(n)
    return painted


def _replay(source: Path, output: Path, paint_log: dict, fps: float, size: tuple, config, preset: str = "slow",
            progress: bool = False) -> None:
    """Render the picture ONCE from the source, applying the paint log (no OCR here). Silent video, same encoder settings as before."""
    import cv2
    capture = cv2.VideoCapture(str(source))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    began = time.time()
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{size[0]}x{size[1]}", "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264",
               "-preset", preset, "-crf", "17", "-pix_fmt", "yuv420p", str(output)]
    writer = subprocess.Popen(command, stdin=subprocess.PIPE)
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for _index, lines, text, *rest in paint_log.get(frame_number, ()):
                frame = _paint_block(frame, lines, text, rest[0] if rest else 1.0)
            writer.stdin.write(frame.tobytes())
            frame_number += 1
            if progress and frame_number % 1000 == 0:
                print(f"VISUAL_TEXT_RENDER: frame {frame_number}/{total} | {int(time.time() - began)}s", flush=True)
    finally:
        capture.release()
        if writer.stdin:
            writer.stdin.close()
        code = writer.wait()
    if code != 0 or not output.is_file():
        raise RuntimeError("FFmpeg failed while encoding the visually translated video")


def _repair_labels(source: Path, approved: list, paint_log: dict, row_hits: list, ocr, fps: float, config, report: dict) -> None:
    """Up to repair_attempts() passes over the labels pass 1 missed. Each pass re-reads only those labels' own frames with enlarged,
    contrast-enhanced OCR and ADDS what it finds to the paint log. Nothing already found is touched, so every earlier replacement is
    preserved by construction, and since nothing has been rendered yet the picture is still encoded just once, afterwards."""
    import cv2
    log = {"enabled": True, "mode": "rescue_before_render", "attempts": [], "unresolved_labels": []}
    report["repair"] = log
    capture = cv2.VideoCapture(str(source))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    painted = _painted_frames(paint_log, len(approved))
    for attempt in range(1, repair_attempts(config) + 1):
        targets = [i for i, (row, hits) in enumerate(zip(approved, row_hits)) if _needs_repair(row, hits, fps)]
        if not targets:
            break
        started = time.time()
        entry = {"attempt": attempt, "variants": [f"{scale}x {mode}" + (f" {tiles}x{tiles} tiles" if tiles > 1 else "")
                                                  for scale, mode, tiles in REPAIR_VARIANTS[attempt - 1]],
                 "targets": [{"english": approved[i]["english"], "start": approved[i].get("start"), "end": approved[i].get("end"),
                              "painted_frames_before": row_hits[i]} for i in targets]}
        log["attempts"].append(entry)
        print(f"VISUAL_TEXT_REPAIR: attempt {attempt}, {len(targets)} label(s) to re-read with enlarged frames", flush=True)
        recovered = []
        detected = {}
        for index in targets:
            found = _rescue_row(capture, ocr, approved[index], painted[index], fps, total, REPAIR_VARIANTS[attempt - 1])
            if not found:
                continue
            frames = _fill_detections(found, total)
            for n, (lines, text, opacity) in frames.items():
                if n not in painted[index]:
                    paint_log.setdefault(n, []).append((index, lines, text, opacity))
                    painted[index].add(n)
                    row_hits[index] += 1
            detected[approved[index]["english"] + f"@{_timecode(approved[index].get('start'))}"] = len(frames)
            if row_hits[index]:
                recovered.append(approved[index]["english"] + f"@{_timecode(approved[index].get('start'))}")
        entry.update(detected_in_frames=detected, recovered=recovered, rendered=False, kept=bool(detected),
                     seconds=round(time.time() - started, 1))
    capture.release()


def translate_video_text(source_video: Path, output_video: Path, config, vocabulary=None) -> tuple:
    """Create an edited silent video stream, or return the source when nothing can be replaced."""
    source, output = Path(source_video).resolve(), Path(output_video).resolve()
    packs = load_glossary_packs([source.with_suffix(".visual_glossary.json")])      # shared packs + this video's own, if any
    set_vocabulary(vocabulary or ())
    plan_path, rows = _load_plan(source)
    automatic = is_auto(config)
    report = {"enabled": True, "pipeline_version": PIPELINE_VERSION, "plan": str(plan_path), "requested": 0, "applied_frames": 0,
              "unmatched_active_scans": 0, "ocr_engine": "RapidOCR (PP-OCR ONNX)",
              "interval_policy": f"{DISCOVERY_STEP_SECONDS}s block discovery; OCR verification every "
                                 f"{ocr_every(config)} frames with position hold",
              "glossary_packs": packs, "status": "processing", "replacements": []}
    if not rows and automatic:
        plan_path, rows = _auto_plan(source, config, report)
        report["plan"] = str(plan_path)
    report["requested"] = len(rows)
    if not rows:
        report["status"] = "no_plan"
        return source, report
    protected_extra = config.production.video_text_translation_protected_terms
    approved = []
    for row in rows:
        english = clean_line_text(str(row.get("english", "")))
        hindi = _safe_translation(english, str(row.get("hindi", "")).strip())
        reason = ("protected_or_invalid" if classify_block(english, protected_extra) not in ("translate", "mixed_prose_formula") and not row.get("force")
                  or not re.search(r"[\u0900-\u097f]", hindi) or not _translation_valid(english, hindi) else "approved")
        report["replacements"].append({"english": english, "hindi": hindi, "start": row.get("start"), "end": row.get("end"),
                                       "decision": reason})
        if reason == "approved":
            approved.append({**row, "english": english, "hindi": hindi, "key": normalize(english)})
    if not approved:
        report["status"] = "nothing_approved"
        return source, report

    _require_shaping()
    import cv2
    try:
        from rapidocr_onnxruntime import RapidOCR
        ocr = RapidOCR()
    except Exception as error:
        raise RuntimeError("video_text_translation requires rapidocr_onnxruntime") from error
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Could not read source video for visual text translation")
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    started_at = time.time()
    # Pass 1 only ANALYSES the picture (OCR on the original frames, which label is where, how it fades) and writes a paint log. Nothing is
    # painted or encoded yet, so the repair below can still add missed labels and the picture is rendered exactly ONCE afterwards.
    every = ocr_every(config)
    minimum = float(config.production.video_text_translation_min_confidence)
    row_hits = [0] * len(approved)
    paint_log: dict = {}                              # frame number -> [(row index, lines, Hindi text)] : lets a repair replay the render
    held: dict = {}                                   # row index -> (lines, frame number last seen)
    misses: dict = {}                                 # row index -> consecutive OCR scans that did not find it
    peak: dict = {}                                   # row index -> strongest contrast its lettering has had (to follow fades)
    previous_thumb = None
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            second = frame_number / fps
            thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.int16)
            cut = previous_thumb is not None and float(np.mean(np.abs(thumb - previous_thumb))) > SCENE_CUT_DIFFERENCE
            previous_thumb = thumb
            if cut:                                   # a new shot: Hindi must never be carried into it (the "ghost in the next frame")
                held.clear()
                misses.clear()
                peak.clear()
            active = [(i, row) for i, row in enumerate(approved)
                      if float(row.get("start", 0)) - WINDOW_PAD_SECONDS <= second <= float(row.get("end", 1e12)) + WINDOW_PAD_SECONDS]
            if active and (frame_number % every == 0 or cut):
                seen = set()
                ocr_lines = _ocr_lines(ocr, frame, min(minimum, TRACKING_CONFIDENCE))
                blocks = group_blocks(ocr_lines)
                for block in blocks:
                    text = normalize(block_text(block))
                    for index, row in active:
                        if index not in seen and difflib.SequenceMatcher(None, text, row["key"]).ratio() >= MATCH_RATIO:
                            lines = _reveal_paint_lines(row, ocr_lines, block) if row.get("reveal_box") else block.lines
                            held[index] = (lines, frame_number)
                            peak[index] = max(peak.get(index, 0.0), _label_contrast(frame, lines))
                            seen.add(index)
                            break
                for index, row in active:                 # an animated title: any of its letters in place counts
                    if index not in seen and row.get("reveal_box") and any(_fragment_fits(b, row) for b in blocks):
                        lines = _reveal_paint_lines(row, ocr_lines)
                        held[index] = (lines, frame_number)
                        peak[index] = max(peak.get(index, 0.0), _label_contrast(frame, lines))
                        seen.add(index)
                for index, _row in active:
                    if index in seen:
                        misses.pop(index, None)
                    elif index in held:
                        # a fade, an animation frame or one noisy OCR read must not bring the English back for a moment
                        misses[index] = misses.get(index, 0) + 1
                        if misses[index] > MISS_TOLERANCE_SCANS:
                            held.pop(index, None)
                            misses.pop(index, None)
                if not seen:
                    report["unmatched_active_scans"] += 1
            for index in list(held):
                lines, last_seen = held[index]
                row = approved[index]
                if not (float(row.get("start", 0)) - WINDOW_PAD_SECONDS <= second <= float(row.get("end", 1e12)) + WINDOW_PAD_SECONDS) \
                        or frame_number - last_seen > every * (MISS_TOLERANCE_SCANS + 1) + 1:
                    held.pop(index, None)
                    continue
                # While the original lettering fades out its contrast falls; the Hindi fades with it and is dropped once it is gone,
                # instead of staying behind as a light, blurred ghost.
                contrast = _label_contrast(frame, lines)
                opacity = min(1.0, contrast / max(FADE_FULL_RATIO * peak.get(index, 0.0), 1.0))
                if opacity < FADE_DROP_OPACITY and last_seen != frame_number:
                    held.pop(index, None)
                    continue
                paint_log.setdefault(frame_number, []).append((index, lines, row["hindi"], opacity))
                row_hits[index] += 1
            frame_number += 1
            if frame_number % 1000 == 0:
                print(f"VISUAL_TEXT_ANALYSIS: frame {frame_number}/{total_frames} | {int(time.time() - started_at)}s", flush=True)
    finally:
        capture.release()
    print(f"VISUAL_TEXT_ANALYSIS_DONE: {frame_number} frames in {int(time.time() - started_at)}s | labels found in "
          f"{sum(1 for h in row_hits if h)}/{len(approved)} planned", flush=True)
    report["approved_count"] = len(approved)
    if repair_enabled(config) and repair_attempts(config) > 0 and any(_needs_repair(r, h, fps) for r, h in zip(approved, row_hits)):
        try:
            _repair_labels(source, approved, paint_log, row_hits, ocr, fps, config, report)
        except Exception as error:                       # a repair problem keeps what pass 1 found
            report["repair"] = {**report.get("repair", {}), "enabled": True, "error": str(error)[:300]}
    if not paint_log:                                    # nothing to paint: keep the original picture, no re-encode
        report.update(matched_count=0, applied_frames=0, status="words_not_found")
        report["unmatched_replacements"] = [{"english": r["english"], "start": r.get("start"), "end": r.get("end")} for r in approved]
        report.setdefault("repair", {"enabled": False})["unresolved_labels"] = [
            {"english": r["english"], "hindi": r["hindi"], "start": r.get("start"), "end": r.get("end"),
             "start_timecode": _timecode(r.get("start")), "end_timecode": _timecode(r.get("end"))} for r in approved]
        report["needs_teacher_review"] = True
        return source, report
    render_started = time.time()
    print(f"VISUAL_TEXT_RENDER_START: painting {sum(len(v) for v in paint_log.values())} label-frames and encoding once", flush=True)
    _replay(source, output, paint_log, fps, (width, height), config, preset="slow", progress=True)
    report["render_seconds"] = round(time.time() - render_started, 1)
    report["analysis_seconds"] = round(render_started - started_at, 1)
    report["matched_count"] = sum(count > 0 for count in row_hits)
    report["applied_frames"] = sum(row_hits)
    report["unmatched_replacements"] = [{"english": row["english"], "start": row.get("start"), "end": row.get("end")}
                                        for row, count in zip(approved, row_hits) if not count]
    unresolved = [{"english": row["english"], "hindi": row["hindi"], "start": row.get("start"), "end": row.get("end"),
                   "start_timecode": _timecode(row.get("start")), "end_timecode": _timecode(row.get("end"))}
                  for row, count in zip(approved, row_hits) if not count]
    report.setdefault("repair", {"enabled": False})["unresolved_labels"] = unresolved
    report["needs_teacher_review"] = bool(unresolved)
    report["status"] = ("completed" if not unresolved
                        else "completed_needs_review" if report["matched_count"] else "words_not_found")
    return output, report
