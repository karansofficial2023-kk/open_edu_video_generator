"""Deterministic Mermaid diagram renderer for science/engineering flowcharts and cycles.

Follows OpenMontage's diagram_gen approach:
1. Attempts to use local Mermaid CLI (`mmdc`) if installed.
2. Provides a robust built-in PIL fallback parser and renderer when Mermaid CLI
   is not present on the host, ensuring the pipeline never fails or halts.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw

from .typography import font as _font_loader


def is_cli_available() -> bool:
    """Whether the mmdc (Mermaid CLI) executable exists on PATH."""
    return bool(shutil.which("mmdc") or shutil.which("mmdc.cmd"))


# ---------------------------------------------------------------------------
# Fallback Parser & Renderer
# ---------------------------------------------------------------------------

_NODE_PATTERN = re.compile(r"([A-Za-z0-9_]+)\s*(?:\[([^\]]+)\]|\(([^\)]+)\)|\{([^\}]+)\})")
_EDGE_PATTERN = re.compile(
    r"([A-Za-z0-9_]+)(?:\s*(?:\[[^\]]*\]|\([^\)]*\)|\{[^\}]*\}))?\s*--+>\s*(?:\|([^\|]+)\|\s*)?([A-Za-z0-9_]+)(?:\s*(?:\[([^\]]+)\]|\(([^\)]+)\)|\{([^\}]+)\}))?"
)


def parse_mermaid(text: str) -> tuple[dict[str, str], list[tuple[str, str, str]], str]:
    """Extract nodes, directed edges (src, dst, label), and orientation (TD or LR)."""
    nodes: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    orientation = "TD"

    for raw_line in text.strip().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%%"):
            continue

        # Check orientation
        if "graph" in line or "flowchart" in line:
            if "LR" in line:
                orientation = "LR"
            elif "RL" in line:
                orientation = "RL"
            elif "BT" in line:
                orientation = "BT"
            else:
                orientation = "TD"
            continue

        # Extract node labels on this line first
        for n_match in _NODE_PATTERN.finditer(line):
            nid = n_match.group(1)
            lbl = n_match.group(2) or n_match.group(3) or n_match.group(4) or nid
            if nid not in nodes or (lbl != nid and nodes[nid] == nid):
                nodes[nid] = lbl.strip()

        # Match directed edges on this line
        for m in _EDGE_PATTERN.finditer(line):
            src = m.group(1)
            edge_lbl = (m.group(2) or "").strip()
            dst = m.group(3)
            dst_lbl = m.group(4) or m.group(5) or m.group(6)
            if dst_lbl and (dst not in nodes or nodes[dst] == dst):
                nodes[dst] = dst_lbl.strip()
            edges.append((src, dst, edge_lbl))

    # Ensure all edge targets exist as nodes
    for src, dst, _ in edges:
        if src not in nodes:
            nodes[src] = src
        if dst not in nodes:
            nodes[dst] = dst

    return nodes, edges, orientation


def _render_fallback(
    mermaid_code: str,
    output_path: Path,
    size: tuple[int, int] = (1920, 1080),
    heading: str = "",
) -> Path:
    """High-quality dark teacher board diagram drawn using PIL."""
    width, height = size
    nodes, edges, orientation = parse_mermaid(mermaid_code)

    if not nodes:
        # Default fallback if parsing fails completely
        nodes = {"A": "Process Start", "B": "Transformation", "C": "Completion"}
        edges = [("A", "B", ""), ("B", "C", "")]

    img = Image.new("RGBA", size, (14, 20, 32, 255))
    draw = ImageDraw.Draw(img)

    # Soft glowing background grid / particles
    for gx in range(0, width, 60):
        draw.line((gx, 0, gx, height), fill=(24, 34, 52, 100), width=1)
    for gy in range(0, height, 60):
        draw.line((0, gy, width, gy), fill=(24, 34, 52, 100), width=1)

    # Optional heading
    top_offset = 120
    if heading:
        hfnt = _font_loader(44, bold=True)
        draw.text((80, 50), heading, font=hfnt, fill=(240, 246, 255))
        draw.rounded_rectangle((80, 105, 260, 110), radius=3, fill=(56, 189, 248))
        top_offset = 150

    # Assign topological ranks (levels)
    ranks: dict[str, int] = {n: 0 for n in nodes}
    for _ in range(len(nodes)):
        for src, dst, _ in edges:
            if src in ranks and dst in ranks:
                ranks[dst] = max(ranks[dst], ranks[src] + 1)

    max_rank = max(ranks.values()) if ranks else 0
    rank_groups: dict[int, list[str]] = {}
    for n, r in ranks.items():
        rank_groups.setdefault(r, []).append(n)

    # Compute bounding boxes for each node
    node_boxes: dict[str, tuple[int, int, int, int]] = {}
    box_w, box_h = 280, 90

    if orientation == "LR":
        # Horizontal flow: ranks left to right
        num_ranks = max_rank + 1
        x_step = (width - 320) // max(1, num_ranks)
        for r, nlist in rank_groups.items():
            cx = 160 + r * x_step + box_w // 2
            y_step = (height - top_offset - 100) // max(1, len(nlist))
            for i, nid in enumerate(nlist):
                cy = top_offset + 50 + i * y_step + y_step // 2
                node_boxes[nid] = (cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2)
    else:
        # Vertical flow (TD): ranks top to bottom
        num_ranks = max_rank + 1
        y_step = (height - top_offset - 120) // max(1, num_ranks)
        for r, nlist in rank_groups.items():
            cy = top_offset + 60 + r * y_step + box_h // 2
            x_step = (width - 240) // max(1, len(nlist))
            for i, nid in enumerate(nlist):
                cx = 120 + i * x_step + x_step // 2
                node_boxes[nid] = (cx - box_w // 2, cy - box_h // 2, cx + box_w // 2, cy + box_h // 2)

    # Draw Edges (Arrows)
    fnt_edge = _font_loader(20, bold=False)
    for src, dst, lbl in edges:
        if src not in node_boxes or dst not in node_boxes:
            continue
        sb = node_boxes[src]
        db = node_boxes[dst]

        # Connect centres clipped to boundaries
        if orientation == "LR":
            p1 = (sb[2], (sb[1] + sb[3]) // 2)
            p2 = (db[0], (db[1] + db[3]) // 2)
        else:
            p1 = ((sb[0] + sb[2]) // 2, sb[3])
            p2 = ((db[0] + db[2]) // 2, db[1])

        # Draw arrow line
        arrow_col = (56, 189, 248, 220)
        draw.line([p1, p2], fill=arrow_col, width=4)

        # Arrowhead
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        if length > 1e-3:
            ux, uy = dx / length, dy / length
            head_len = 16
            head_w = 9
            bx, by = p2[0] - ux * head_len, p2[1] - uy * head_len
            poly = [
                p2,
                (bx - uy * head_w, by + ux * head_w),
                (bx + uy * head_w, by - ux * head_w),
            ]
            draw.polygon(poly, fill=(56, 189, 248, 255))

        # Edge label
        if lbl:
            mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
            draw.text((mx + 8, my - 16), lbl, font=fnt_edge, fill=(203, 213, 225, 230))

    # Draw Nodes (Cards)
    fnt_node = _font_loader(24, bold=True)
    for nid, lbl in nodes.items():
        if nid not in node_boxes:
            continue
        bx0, by0, bx1, by1 = node_boxes[nid]

        # Outer glow and fill
        draw.rounded_rectangle((bx0 - 3, by0 - 3, bx1 + 3, by1 + 3), radius=16, fill=(56, 189, 248, 30))
        draw.rounded_rectangle((bx0, by0, bx1, by1), radius=14, fill=(22, 32, 50, 240), outline=(56, 189, 248, 255), width=3)

        # Node label (wrapped if needed)
        words = lbl.split()
        if len(words) > 3:
            line1 = " ".join(words[: len(words) // 2])
            line2 = " ".join(words[len(words) // 2 :])
            lines = [line1, line2]
        else:
            lines = [lbl]

        th = len(lines) * 30
        ty = (by0 + by1 - th) // 2
        for line in lines:
            try:
                left, top, right, bottom = fnt_node.getbbox(line)
                lw = right - left
            except Exception:
                lw = len(line) * 14
            tx = (bx0 + bx1 - lw) // 2
            draw.text((tx, ty), line, font=fnt_node, fill=(255, 255, 255, 255))
            ty += 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_mermaid_diagram(
    mermaid_code: str,
    output_path: str | Path,
    dark_theme: bool = True,
    size: tuple[int, int] = (1920, 1080),
    heading: str = "",
) -> Path:
    """Render Mermaid diagram code to a PNG image.

    Uses `mmdc` if installed; otherwise falls back cleanly to the built-in PIL engine.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if is_cli_available():
        with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False, encoding="utf-8") as f:
            f.write(mermaid_code)
            tmp_mmd = f.name
        try:
            cmd = [
                shutil.which("mmdc") or shutil.which("mmdc.cmd") or "mmdc",
                "-i", tmp_mmd,
                "-o", str(out),
                "-w", str(size[0]),
                "-H", str(size[1]),
                "-b", "transparent",
                "-t", "dark" if dark_theme else "default",
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=25, check=False)
            if res.returncode == 0 and out.is_file() and out.stat().st_size > 0:
                return out
        except Exception:
            pass
        finally:
            try:
                os.remove(tmp_mmd)
            except OSError:
                pass

    # Built-in fallback
    return _render_fallback(mermaid_code, out, size=size, heading=heading)
