"""Ball-and-stick molecules for chemistry lessons: a compound named in the narration is shown as a slowly rotating 3D model.

Name -> structure comes from PubChem (open data) and is cached in `cache/molecules.json`, so a lesson needs the network only the first
time a compound appears; formulas written in the narration (NaCl, CH3COOH) are looked up the same way. The 3D shape comes from RDKit
(open source); drawing is our own: shaded spheres sorted by depth, bonds as rods, element symbols on the atoms. If RDKit or the lookup
is unavailable the board simply shows no molecule.
"""
from __future__ import annotations

import json
import math
import re
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .typography import font

CACHE = Path(__file__).resolve().parent.parent / "cache" / "molecules.json"
_LOCK = threading.Lock()

# CPK-style colours (bright enough for a dark board) and relative radii
COLORS = {"H": (235, 238, 242), "C": (232, 214, 70), "O": (230, 64, 60), "N": (70, 120, 235), "Cl": (90, 220, 120),
          "F": (140, 230, 140), "Br": (170, 60, 40), "I": (150, 60, 190), "S": (240, 200, 60), "P": (240, 140, 40),
          "Na": (170, 120, 240), "K": (150, 90, 220), "Ca": (120, 200, 120), "Mg": (120, 220, 90), "Fe": (220, 120, 60),
          "Cu": (220, 140, 80), "Zn": (150, 160, 190), "Ag": (200, 200, 210)}
RADII = {"H": 0.55}

# chemical naming rules (no topic list): "<x>ic acid", "<metal or ammonium> <x>ide/ate/ite", organic name endings, written formulas.
# A candidate becomes a molecule only when PubChem knows it.
_ACID = re.compile(r"\b([a-z]+ic)\s+acid\b", re.I)
_SALT = re.compile(r"\b(sodium|potassium|calcium|magnesium|ammonium|hydrogen|silver|copper|zinc|iron|barium|lithium|aluminium|aluminum)"
                   r"\s+([a-z]+(?:ide|ate|ite))\b", re.I)
_ORGANIC = re.compile(r"\b([a-z]+(?:anol|enol|ol|ane|ene|yne|one|aldehyde|amine|oform|ose|benzene|phenol|ether|ester))\b", re.I)
_FORMULA = re.compile(r"\b((?:[A-Z][a-z]?\d*){2,}(?:\([A-Z][a-z]?\d*\))?(?:[A-Z][a-z]?\d*)*)\b")
_NOT_COMPOUNDS = {"control", "alone", "done", "someone", "none", "phone", "zone", "tone", "bone", "stone", "one", "gone", "scene",
                  "general", "online", "everyone", "anyone", "symbol", "school", "tool", "pool", "cool", "wool", "protocol", "patrol",
                  "throne", "ozone", "sane", "plane", "lane", "crane", "membrane", "mundane", "insane", "airplane", "humane",
                  "gene", "serene", "intervene", "convene", "obscene", "purpose", "close", "those", "whose", "chose", "dose", "nose",
                  "loose", "suppose", "propose", "compose", "expose", "rose", "pose", "choose", "goose", "verbose", "diagnose",
                  "either", "neither", "whether", "together", "rather", "other", "another", "weather", "father", "mother", "further"}


def candidates(text: str) -> list[str]:
    text = text or ""
    found = [f"{m.group(1)} acid" for m in _ACID.finditer(text)]
    found += [f"{m.group(1)} {m.group(2)}" for m in _SALT.finditer(text)]
    found += [m.group(1) for m in _ORGANIC.finditer(text) if m.group(1).lower() not in _NOT_COMPOUNDS and len(m.group(1)) >= 5]
    found += [m.group(1) for m in _FORMULA.finditer(text) if re.search(r"\d", m.group(1)) or len(re.findall(r"[A-Z]", m.group(1))) >= 2]
    seen, out = set(), []
    for name in found:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name.lower() if not re.search(r"[A-Z].*[A-Z0-9]", name) else name)
    return out


def _load_cache() -> dict:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def smiles_for(name: str, timeout: float = 8.0) -> str | None:
    """PubChem name/formula -> SMILES ("" cached for names PubChem does not know)."""
    key = name.strip().lower()
    with _LOCK:
        cache = _load_cache()
    if key in cache:
        return cache[key] or None
    try:
        import requests
        from urllib.parse import quote
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/property/IsomericSMILES/TXT"
        response = requests.get(url, timeout=timeout)
        smiles = response.text.strip().splitlines()[0] if response.ok and response.text.strip() else ""
    except Exception:
        return None                                   # network problem: not cached, tried again next time
    with _LOCK:
        cache = _load_cache()
        cache[key] = smiles
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CACHE)
    return smiles or None


def first_molecule(text: str, max_atoms: int = 60) -> dict | None:
    """The first compound named in the text that has a structure: {'name', 'smiles'}; small ions/single atoms are skipped."""
    found = []
    for name in candidates(text):
        smiles = smiles_for(name)
        if not smiles:
            continue
        model = model_for(smiles)
        if model is not None and 3 <= len(model[0]) <= max_atoms:
            found.append({"name": name if not name.islower() else name.title(), "smiles": smiles})
    # one connected molecule (phenol, acetic acid) explains more than an ionic salt drawn as loose ions (NaOH, NaCl)
    whole = [m for m in found if "." not in m["smiles"]]
    return whole[0] if whole else None          # an ionic salt drawn as loose ions (NaCl, CH3COO- Na+) explains nothing: no picture


@lru_cache(maxsize=64)
def model_for(smiles: str):
    """(elements, xyz centred, bonds) of a 3D conformer, or None."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol, randomSeed=11) != 0:
            return None
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
        conf = mol.GetConformer()
        xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())])
        xyz -= xyz.mean(axis=0)
        elements = tuple(atom.GetSymbol() for atom in mol.GetAtoms())
        bonds = tuple((b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in mol.GetBonds())
        return elements, xyz, bonds
    except Exception:
        return None


@lru_cache(maxsize=128)
def _sphere(color: tuple, radius: int) -> Image.Image:
    """A shaded sphere sprite (highlight up-left, darker rim)."""
    size = radius * 2 + 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx = cy = size / 2
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius
    light = np.sqrt(np.clip(1 - dist ** 2, 0, 1))
    spec = np.clip(1 - np.sqrt((xx - cx + radius * 0.35) ** 2 + (yy - cy + radius * 0.35) ** 2) / (radius * 0.55), 0, 1) ** 2
    rgb = np.stack([np.clip(c * (0.35 + 0.65 * light) + 255 * 0.55 * spec, 0, 255) for c in color], axis=-1)
    alpha = np.clip((1 - dist) * radius * 1.5, 0, 1) * 255
    return Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA")


def render(smiles: str, size: tuple[int, int], angle: float, labels: bool = True) -> Image.Image | None:
    """RGBA picture of the molecule rotated by `angle` radians about the vertical axis (plus a fixed tilt)."""
    model = model_for(smiles)
    if model is None:
        return None
    elements, xyz, bonds = model
    tilt = 0.35
    rot_y = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    rot_x = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
    pts = xyz @ rot_y.T @ rot_x.T
    width, height = size
    span = max(1e-6, np.abs(pts[:, :2]).max())
    scale = min(width, height) * 0.40 / span
    to2d = lambda p: (width / 2 + p[0] * scale * (1 + p[2] * 0.04), height / 2 - p[1] * scale * (1 + p[2] * 0.04))
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    base_r = max(6, round(scale * 0.42))
    order = np.argsort(pts[:, 2])                       # far atoms first
    drawn_bonds = set()
    label_font = font(max(12, round(base_r * 0.9)), True)
    radius_of = lambda i: base_r * RADII.get(elements[i], 1.0) * (1 + pts[i, 2] * 0.04)

    def rod(a, b):                                    # from sphere edge to sphere edge, so element letters stay readable
        (xa, ya), (xb, yb) = to2d(pts[a]), to2d(pts[b])
        length = max(1e-6, math.hypot(xb - xa, yb - ya))
        ux, uy = (xb - xa) / length, (yb - ya) / length
        ra, rb = radius_of(a) * 0.85, radius_of(b) * 0.85
        if length > ra + rb:
            draw.line(((xa + ux * ra, ya + uy * ra), (xb - ux * rb, yb - uy * rb)), fill=(205, 210, 220, 255), width=max(3, round(base_r * 0.30)))

    for index in order:
        for a, b, kind in bonds:                          # rods behind the nearer of their two atoms
            if index in (a, b) and (a, b) not in drawn_bonds:
                other = b if index == a else a
                if pts[other, 2] < pts[index, 2]:
                    rod(a, b)
                    drawn_bonds.add((a, b))
        element = elements[index]
        radius = round(base_r * RADII.get(element, 1.0) * (1 + pts[index, 2] * 0.04))
        x, y = to2d(pts[index])
        sprite = _sphere(COLORS.get(element, (200, 200, 210)), max(4, radius))
        image.alpha_composite(sprite, (round(x - sprite.width / 2), round(y - sprite.height / 2)))
        if labels and element != "H" and radius >= 12:
            box = draw.textbbox((0, 0), element, font=label_font)
            draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]), element, font=label_font, fill=(25, 25, 30, 255))
    for a, b, kind in bonds:                              # any rod not yet drawn (both atoms at equal depth)
        if (a, b) not in drawn_bonds:
            rod(a, b)
    return image
