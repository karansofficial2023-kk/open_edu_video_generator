# Educational animation workflow

The implementation uses Python, Pillow and FFmpeg for deterministic 2D animation.
Your voice is unchanged: Edge TTS `en-IN-NeerjaNeural`, rate `-20%`.
Edge TTS requires internet access. There are no paid APIs in this pipeline.

## Run on this computer

The copied `.venv` points to an unavailable Python installation. `run_video.ps1`
tries it first, then uses the existing ComfyUI portable Python with project-local
dependencies in `.runtime-deps`. It does not install anything into ComfyUI.

```powershell
cd D:\Python\open_edu_video_generator
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input samples/pollination_animated.json --output outputs/my_animation_preview --preview
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input samples/pollination_animated.json --output outputs/my_animation_voice
```

The first command creates a silent MP4 with approximate timing. The second uses
your Edge voice and real word timestamps. Neither command needs Ollama or ComfyUI:
the sample uses only built-in animation templates. Open `final.mp4` in the chosen
output directory. Use separate directories for previews and final renders.

For this pollination topic, use the teacher-approved Word storyboard or a
coverage-passing JSON storyboard. The LLM-generated storyboard can be useful as a
draft, but the current checked copy misses several required topic areas.

Run the teacher storyboard like this:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input "C:\Users\User\Downloads\1788516588515_5913833 - Types of Pollination.docx" --output outputs/pollination_quality_animation
```

## Standard installation on another computer

Install Python 3.12 and FFmpeg/FFprobe. In the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m app.main --config config.12gb.yaml --input samples/pollination_animated.json --output outputs/demo --preview
```

Set `ffmpeg_path` and `ffprobe_path` in your config to the installed executables.
Keep the resolution 16:9, e.g. 1920x1080 or 1280x720, with even dimensions.

## Turn your own content into animation

1. Start Ollama if planning from text. Keep your existing `llama3.1:8b` model.
2. Run planning only:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input samples/pollination.txt --output outputs/my_plan --plan-only
   ```

3. Edit `outputs/my_plan/storyboard.json`. Check narration, terminology, sources,
   and the `shot` for each segment. `review.json` is a small lint report, not proof
   of scientific accuracy. Known bad terms and missing animation cues stop rendering.
4. Preview that JSON using `--preview --limit-segments 3` in a new output directory.
5. Render the same JSON without `--preview` for the voice version.

Existing DOCX/JSON inputs remain readable. Legacy Word storyboard segments are
cleaned for known spelling problems and assigned structured animation shots from
their narration. For best quality, review and refine those `shot` objects in
JSON. JSON is the authoritative editable format: the legacy DOCX export does not
preserve the new shot fields.

For files whose name contains `Types of Pollination`, the review also writes
`coverage.json` and warns if the storyboard misses too many core
topics: definition, self-pollination, cross-pollination, wind, water, insects,
birds, bats, animals, and advantages/disadvantages.
For final lessons, add `--strict-coverage` to block incomplete storyboards.

## Templates and fields

| Template | Behavior | Required fields |
|---|---|---|
| `pollination` | Pollen travels between two schematic flowers | None |
| `protandry` | Highlights anthers, then stigma | None |
| `protogyny` | Highlights stigma, then anthers | None |
| `process` | 2-4 ordered cards, highlighted in sequence | `steps` |
| `comparison` | 2-4 comparison cards, highlighted in sequence | `steps` |
| `photo` | Contained artwork with a short heading and timed captions | `asset_path`, reviewed asset, or ComfyUI |
| `video` | Imported clip, normalized to output size and fps | `asset_path` |

All templates support `heading` and `learning_objective`. Paths are relative to
the project working directory, or absolute. Example:

```json
"shot": {
  "template": "process",
  "heading": "Three stages",
  "learning_objective": "Explain the order of the process",
  "steps": ["First stage", "Second stage", "Third stage"],
  "cues": ["First", "Next", "Finally"]
}
```

Each cue must appear in narration in order. Supply one per card, or two for a
flower template. Cue timing comes from Edge word boundaries; preview/old audio
uses approximate timing. If cues are omitted, `stage_fractions` can specify
positions in speech duration, e.g. `[0, 0.5]`; otherwise equal stages are used.
Source text never executes Python or shell code.

Flower templates are simplified teaching diagrams, not botanically precise
species models. Use reviewed custom artwork/clips when species anatomy matters.
The process and comparison templates support other educational subjects.

## Inkscape, Blender, and Manim

These are optional authoring tools, not required installations for the sample.

* Inkscape: draw and review a diagram; export a PNG at 1920 pixels wide or larger.
  Save it under `assets/reviewed/` and use a `photo` shot with its `asset_path`.
  PNG/JPEG are supported; direct SVG import and per-part SVG animation are not yet implemented.
* Blender: author the model and animation, export a silent MP4, and use
  `{"template": "video", "asset_path": "assets/reviewed/my_animation.mp4"}`.
* Manim: author/render an animation externally and import its MP4 in the same way.
  No automatic Blender/Manim scene generation is implemented.

Imported video audio is discarded; Edge narration is used. Long clips are trimmed;
short clips hold their final frame. By default, subtitles are external/soft: the
pipeline writes `subtitles.srt` and embeds a selectable MP4 subtitle track.
Set `render.burn_captions: true` or pass `--burn-captions` only when the captions
must always be visible in the video pixels.

## ComfyUI and models

`visual_mode: hybrid` routes structured diagrams directly to the template renderer.
ComfyUI is only used for photo shots without supplied assets. Your SDXL checkpoint
and Llama settings have not changed. The earlier Llama/SDXL links were license
references; Qwen3-8B was optional and is not required or downloaded by this change.
You can set `comfyui.enabled: false` and `comfyui.rewrite_prompts: false` if you use
only reviewed assets and templates.

For optional generated photos, start ComfyUI at the configured URL and keep your
existing workflow/checkpoint. Increase image quality only after the scene explains
the intended concept. Photographs alone do not execute animation directions.

For generated video, set:

```yaml
comfyui:
  video_enabled: true
  video_workflow_path: "workflows/your_wan_or_ltx_api_workflow.json"
  video_width: 832
  video_height: 480
  video_frames: 81
  video_fps: 16
```

Then use a storyboard segment with:

```json
"shot": {
  "template": "video",
  "heading": "Bee visiting a flower"
}
```

The Python code queues the ComfyUI workflow, patches common prompt, seed, size,
frame-count, and save nodes, downloads the generated MP4/WebP output, normalizes
it to the final 16:9 video, and muxes it with Edge narration and subtitles.

On an RTX 3060 12GB system, prefer a practical local ComfyUI video workflow such
as Wan2.2 5B first. Use 480p/832x480-ish short clips and upscale/contain them in
the final 1080p edit. LTX is useful, but current local LTX Desktop guidance points
to 16GB VRAM for local generation; on 12GB it may require API mode or a lighter
ComfyUI setup. Wan/LTX model weights can have their own licenses, so accept and
record those separately from this project code.

Recommended hybrid rule:

| Scene need | Preferred output |
|---|---|
| Definitions, classifications, comparisons, labels | local animation template |
| Exact scientific anatomy | reviewed teacher diagram or Manim/Blender clip |
| Natural b-roll, flower visits, cinematic transitions | ComfyUI generated video |
| Important final lesson | teacher-reviewed storyboard JSON before rendering |

Tell the transcribe/storyboard maker to output structured `shot` fields directly.
It should choose `video` only for cinematic b-roll where small hallucinations are
acceptable after review. It should choose deterministic templates for facts,
labels, taxonomies, and comparisons.

## Files and checks

Outputs include storyboard JSON, legacy storyboard DOCX, review JSON, SRT captions,
per-segment clips, narration WAV, and final MP4. Diagrams also get an initial PNG.
`--skip-tts` builds assets without producing a final MP4; use `--preview` to inspect motion.

Run tests with a configured Python environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

No silent planning fallback is used by default. `planner_fallback: true` opts into
draft text slides if Ollama fails. The renderer does not automatically certify facts,
invent species anatomy, or turn arbitrary instructions into a designer-quality film.
