# Pro Style Storyboard Creation Guide

Target style: polished educational video like the uploaded Kakumanu reference, using open-source tools and local ComfyUI/LTX where useful.

## Core Rule

Do not ask AI video/image models to create exact text, labels, arrows, formulas, legends, or scientific names inside the image. The storyboard should describe those items as separate overlay elements so the renderer can draw them precisely.

Good split:
- AI/asset layer: clean background image or short natural motion.
- Overlay layer: title, labels, arrows, legends, formulas, highlights, subtitles.
- Motion layer: zoom, pan, fade, slide-in labels, reveal one item at a time.

## Recommended Storyboard Segment Fields

Each segment should ideally include these fields:

```yaml
scene_number: 1
segment_number: 1
template: title_card | labeled_image | comparison | process | formula | split_screen | video_broll
heading: "Clear screen title"
narration: "Voice-over line for this segment."
visual_subject: "What the background image/video should show."
asset_path: "optional/path/to/reviewed-image-or-video.png"
labels:
  - text: "Anther"
    target: "yellow pollen-bearing tip"
    position: "upper left"
  - text: "Stigma"
    target: "central sticky tip"
    position: "upper right"
arrows:
  - from_label: "Anther"
    to_target: "pollen-bearing tip"
highlights:
  - target: "pollen grains"
    color: "yellow"
motion:
  camera: "slow_zoom_in | slow_pan_left | slow_pan_right | static"
  label_reveal: "one_by_one"
  transition_in: "fade"
  transition_out: "crossfade"
subtitle_style: "reference_band"
```

## Template List

### 1. title_card

Use for opening scene and major section starts.

```yaml
template: title_card
heading: "Types of Pollination"
subheading: "Self-pollination and cross-pollination"
visual_subject: "sharp macro flower background with pollinators, cinematic daylight"
motion:
  camera: "slow_zoom_in"
  title_animation: "fade_up"
```

Quality notes:
- Title should be drawn by renderer, not generated inside image.
- Use large bold title, slight shadow, background darkening/blur behind title if needed.
- Good for Adobe-like title card using FFmpeg/Pillow/OpenCV only.

### 2. labeled_image

Use for anther, stigma, cathode, anode, DNA parts, graph parts, apparatus parts.

```yaml
template: labeled_image
heading: "Pollen Transfer: Anther to Stigma"
visual_subject: "sharp close-up of hibiscus flower reproductive parts, no text"
labels:
  - text: "Anther"
    target: "pollen-bearing tip"
    position: "left"
  - text: "Stigma"
    target: "central sticky tip"
    position: "right"
arrows:
  - from_label: "Anther"
    to_target: "pollen-bearing tip"
  - from_label: "Stigma"
    to_target: "central sticky tip"
motion:
  camera: "slow_zoom_in"
  label_reveal: "one_by_one"
```

Quality notes:
- Use reviewed asset if scientific accuracy matters.
- Labels and arrows must be overlayed by code.
- Use consistent label pill, line thickness, arrowhead, and highlight color.

### 3. process

Use for one-by-one naming or steps.

```yaml
template: process
heading: "Self-Pollination"
steps:
  - "Pollen forms in the anther"
  - "Pollen reaches stigma of the same flower"
  - "Fertilization can follow"
motion:
  step_reveal: "one_by_one"
  camera: "static"
```

Quality notes:
- Best for precise educational explanation.
- Use simple icon/diagram/photo in background, steps overlaid cleanly.

### 4. comparison

Use for self vs cross, strong vs weak electrolyte, homologous vs analogous.

```yaml
template: comparison
heading: "Self-Pollination vs Cross-Pollination"
columns:
  - title: "Self-Pollination"
    points:
      - "Same flower or same plant"
      - "Less genetic variation"
  - title: "Cross-Pollination"
    points:
      - "Different plants of same species"
      - "More genetic variation"
motion:
  column_reveal: "left_then_right"
```

### 5. split_screen

Use like the reference: two related visuals side by side.

```yaml
template: split_screen
heading: "Habitats Show Diversity"
left:
  visual_subject: "forest ecosystem with animals"
  label: "Land"
right:
  visual_subject: "coral reef ecosystem"
  label: "Water"
motion:
  camera: "subtle_independent_zoom"
```

### 6. formula

Use for chemistry/physics/math derivations. Do not use AI photo for equations.

```yaml
template: formula
heading: "Kohlrausch Law"
formula_lines:
  - "Lambda0 = lambda0+ + lambda0-"
  - "Lambda0(CH3COOH) = Lambda0(CH3COONa) + Lambda0(HCl) - Lambda0(NaCl)"
explain_steps:
  - "Use strong electrolytes"
  - "Cancel common ions"
  - "Obtain weak electrolyte value"
motion:
  formula_reveal: "line_by_line"
```

Best tool: Manim or renderer text layer.

### 7. video_broll

Use LTXV only for simple motion where exact anatomy/text is not critical.

```yaml
template: video_broll
heading: "Pollinator Movement"
visual_subject: "realistic bee slowly approaching one flower, no text, no labels"
duration: 4
ltx: true
motion:
  camera: "stable"
```

Avoid LTXV for:
- exact anther/stigma labeling
- formulas
- graphs with readable numbers
- legends
- small insect anatomy closeups if precision is required

## Subtitle Style

Use reference style:

```yaml
subtitle_style:
  placement: "bottom_band"
  band_color: "black"
  band_opacity: 0.38
  text_color: "white"
  font_size: 44
  max_lines: 2
  align: "center"
```

Avoid white rectangle subtitles because they cover visuals and feel less polished.

## Scene Design Rules

For a 1-2 minute polished lesson:
- 8 to 14 segments total.
- Each segment 5 to 9 seconds.
- One concept per segment.
- Every 2-3 segments, use a different visual rhythm: title card, labeled image, comparison, process, b-roll.
- Prefer strong visual headings over long narration.
- Keep narration short enough that the viewer can read labels.

## Asset Quality Rules

Use these priorities:

1. Reviewed real/curriculum asset with `asset_path`.
2. Generated high-resolution still image with no text.
3. Deterministic diagram rendered by code/Manim.
4. LTXV short b-roll only.

For 12GB RTX 3060:
- Keep LTXV at 768x512 or similar, then upscale/compose into 1080p.
- Generate fewer video clips, 1-3 per short lesson.
- Use still images plus motion overlays for most segments.
- Load/unload ComfyUI models between image/video batches if VRAM becomes tight.

For Linux server/Colab:
- Same storyboard format can be reused.
- Increase LTX resolution/frames/steps only when VRAM allows.
- Final export should remain 1080p at 8-10 Mbps.

## Open-Source Tool Mapping

- Title cards: Pillow/OpenCV + FFmpeg, optionally Blender for advanced 3D titles.
- Labels/arrows/legends: Pillow/OpenCV overlay layer.
- Formulas/derivations: Manim, then composite into final video.
- Slow zoom/pan/fade: FFmpeg zoompan or OpenCV frame rendering.
- Split-screen: FFmpeg/Pillow composition.
- Subtitles: FFmpeg drawtext/ASS subtitles or Pillow-rendered subtitle band.
- Background image generation: ComfyUI SDXL/Flux/Qwen image model.
- Short motion clips: LTXV through ComfyUI.
- Upscaling: Real-ESRGAN/SUPIR/ComfyUI upscale nodes.

## Qwen/Model Notes

Qwen can help with storyboard writing, scene planning, labels, and prompts. It should not be trusted to create final scientific images with embedded text. For 12GB VRAM, use quantized or smaller local Qwen text models for planning if available; image/video generation should remain batched through ComfyUI so models can be loaded and unloaded.
