"""Deterministic circuit schematics (schemdraw, open source) for series/parallel layouts.

Contract rows (``columns``): ``"<Caption> | <series|parallel>: battery 12 V; resistor R1; resistor R2"``. Components are drawn
exactly as listed; nothing is inferred from the narration here.
"""
from __future__ import annotations

import io
import re

from PIL import Image

COMPONENTS = {"battery", "cell", "resistor", "lamp", "bulb", "capacitor", "inductor", "switch", "ammeter", "voltmeter", "diode", "led"}
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


class CircuitError(ValueError):
    pass


def parse_row(row: str) -> tuple[str, str, list[tuple[str, str]]]:
    """-> (caption, topology, [(component, label), ...])"""
    caption, _, spec = row.partition("|")
    match = re.match(r"(?i)^\s*(series|parallel|electrolysis)\s*:\s*(.+)$", spec)
    if not match:
        raise CircuitError(f"circuit row needs 'series:', 'parallel:' or 'electrolysis:' followed by components: {row!r}")
    topology = match.group(1).lower()
    if topology == "electrolysis":
        return caption.strip() or "Electrolysis cell", topology, _electrolysis_parts(match.group(2))
    parts = []
    for item in re.split(r"[;,]", match.group(2)):
        item = item.strip()
        if not item:
            continue
        kind, _, label = item.partition(" ")
        kind = kind.lower()
        if kind not in COMPONENTS:
            raise CircuitError(f"unsupported circuit component {kind!r} (supported: {sorted(COMPONENTS)})")
        parts.append((kind, label.strip()))
    if not any(k in {"battery", "cell"} for k, _ in parts):
        raise CircuitError("circuit needs a battery or cell")
    if not any(k not in {"battery", "cell"} for k, _ in parts):
        raise CircuitError("circuit needs at least one load component")
    return caption.strip() or topology.title() + " circuit", topology, parts


ELECTROLYSIS_PARTS = ("battery", "anode", "cathode", "electrolyte")


def _electrolysis_parts(spec: str) -> list[tuple[str, str]]:
    parts = []
    for item in re.split(r"[;,]", spec):
        item = item.strip()
        if not item:
            continue
        kind, _, label = item.partition(" ")
        kind = kind.lower()
        if kind not in ELECTROLYSIS_PARTS:
            raise CircuitError(f"unsupported electrolysis part {kind!r} (supported: {list(ELECTROLYSIS_PARTS)})")
        parts.append((kind, label.strip()))
    missing = [name for name in ELECTROLYSIS_PARTS if name not in {k for k, _ in parts}]
    if missing:
        raise CircuitError(f"electrolysis cell needs {missing}")
    return parts


def pretty(label: str) -> str:
    """R1 -> R with a subscript digit; keeps units and Greek capital omega as plain text."""
    label = label.strip().replace("ohm", "Ω")
    return re.sub(r"^([A-Za-z])(\d+)$", lambda m: m.group(1) + m.group(2).translate(_SUB), label)


def _component(elm, kind: str, label: str, loc: str = "top"):
    mapping = {"resistor": elm.Resistor, "lamp": elm.Lamp, "bulb": elm.Lamp, "capacitor": elm.Capacitor, "inductor": elm.Inductor,
               "switch": elm.Switch, "diode": elm.Diode, "led": elm.LED}
    if kind in mapping:
        return mapping[kind]().label(pretty(label), loc=loc)
    return (elm.MeterA() if kind == "ammeter" else elm.MeterV()).label(pretty(label) or ("A" if kind == "ammeter" else "V"), loc=loc)


def _draw_electrolysis(parts: list[tuple[str, str]]) -> Image.Image:
    """Battery wired to two electrodes dipped in an electrolyte: the standard electroplating / electrolysis diagram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    label = {kind: text for kind, text in parts}
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    ink, wire = "#1f2933", "#2b2b2b"
    # beaker and liquid
    ax.add_patch(Rectangle((0.7, 0.5), 8.6, 2.6, facecolor="#cfe8f7", edgecolor="none", zorder=1))
    ax.plot([0.7, 0.7, 9.3, 9.3], [3.5, 0.5, 0.5, 3.5], color=ink, lw=3, zorder=3)
    ax.text(5.0, 0.18, label.get("electrolyte", "") and f"Electrolyte: {label['electrolyte']}", ha="center", va="center", fontsize=13, color=ink)
    # electrodes: anode (+) left, cathode (-) right
    for x, kind, sign, colour in ((3.6, "anode", "+", "#c0392b"), (6.4, "cathode", "−", "#1f5fa8")):
        ax.add_patch(Rectangle((x - 0.18, 0.9), 0.36, 3.3, facecolor="#9aa5b1", edgecolor=ink, lw=2, zorder=4))
        ax.text(x, 4.45, f"{sign}", ha="center", va="center", fontsize=22, fontweight="bold", color=colour)
        name = label.get(kind, "")
        ax.text(x + (-0.4 if kind == "anode" else 0.4), 1.55, kind.title() + ("\n" + name if name else ""),
                ha="right" if kind == "anode" else "left", va="center", fontsize=13, color=ink, zorder=5)
        ax.plot([x, x], [4.2, 5.0], color=wire, lw=2.6, zorder=2)
    # wires and battery
    ax.plot([3.6, 3.6, 4.4], [5.0, 5.0, 5.0], color=wire, lw=2.6)
    ax.plot([5.6, 6.4], [5.0, 5.0], color=wire, lw=2.6)
    ax.plot([4.4, 4.4], [4.6, 5.4], color=ink, lw=4.5)          # long plate = positive
    ax.plot([5.0, 5.0], [4.8, 5.2], color=ink, lw=4.5)
    ax.plot([5.0, 5.6], [5.0, 5.0], color=wire, lw=2.6)
    ax.text(4.2, 5.5, "+", ha="center", va="center", fontsize=14, color="#c0392b", fontweight="bold")
    ax.text(5.25, 5.5, "−", ha="center", va="center", fontsize=14, color="#1f5fa8", fontweight="bold")
    if label.get("battery"):
        ax.text(4.8, 4.55, label["battery"], ha="center", va="center", fontsize=12, color=ink)
    fig.tight_layout(pad=0.4)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def draw(row: str) -> tuple[str, Image.Image]:
    caption, topology, parts = parse_row(row)
    if topology == "electrolysis":
        return caption, _draw_electrolysis(parts)
    import schemdraw
    import schemdraw.elements as elm
    schemdraw.use("matplotlib")
    drawing = schemdraw.Drawing(show=False)
    drawing.config(unit=3, fontsize=20, lw=2.6)
    _kind, battery_label = next(p for p in parts if p[0] in {"battery", "cell"})
    loads = [p for p in parts if p[0] not in {"battery", "cell"}]
    battery = elm.Battery().up().reverse().label(pretty(battery_label), loc="left")
    drawing += battery
    if topology == "series":
        drawing += elm.Line().right(1.0)
        for kind, label in loads:
            drawing += _component(elm, kind, label).right()
            drawing += elm.Line().right(1.0)
        drawing += elm.Line().down().toy(battery.start)
        drawing += elm.Line().tox(battery.start)
    else:
        drawing += elm.Line().right(1.0)
        for index, (kind, label) in enumerate(loads):
            drawing += elm.Dot()
            drawing.push()
            drawing += _component(elm, kind, label, "right").down().length(3.0)
            drawing += elm.Dot()
            drawing.pop()
            drawing += elm.Line().right(2.6 if index < len(loads) - 1 else 1.0)
        drawing += elm.Line().down().toy(battery.start)
        drawing += elm.Line().tox(battery.start)
    buffer = io.BytesIO()
    drawing.save(buffer, dpi=200, transparent=False)
    buffer.seek(0)
    return caption, Image.open(buffer).convert("RGB")
