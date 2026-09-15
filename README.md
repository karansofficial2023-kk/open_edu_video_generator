# Open Edu Video Generator

Educational video generator for Windows. It turns text, DOCX, or storyboard JSON
into narrated MP4 lessons with timed captions and 2D educational animation.

The current quality path is the `hybrid` renderer: structured biology/process
shots are animated locally with Python, Pillow, and FFmpeg. ComfyUI is optional
and is used only for photo-style shots that need generated images.

Your voice setting is unchanged:

```yaml
edge_tts:
  voice: "en-IN-NeerjaNeural"
  rate: "-20%"
```

Edge TTS needs internet access. No paid API key is required.

## Quick Run

On this computer, use the launcher. It tries the project `.venv` first, then uses
the existing ComfyUI portable Python with project-local dependencies from
`.runtime-deps`.

```powershell
cd D:\Python\open_edu_video_generator
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input samples/pollination_animated.json --output outputs/my_animation_preview --preview
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input samples/pollination_animated.json --output outputs/my_animation_voice
```

The preview command makes a silent MP4 quickly. The voice command uses Edge TTS
word timings and writes `final.mp4`.

For this pollination topic, use the teacher-approved Word storyboard or a
coverage-passing JSON storyboard:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input "C:\Users\User\Downloads\1788516588515_5913833 - Types of Pollination.docx" --output outputs/pollination_quality_animation
```

## Normal Setup

On another Windows machine:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy config.example.yaml config.yaml
```

Set `ffmpeg_path` and `ffprobe_path` in the config to your installed FFmpeg
executables. Keep output resolution 16:9, such as 1920x1080 or 1280x720.

## Workflow

Use `--plan-only` when creating a storyboard from raw text:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.12gb.yaml --input samples/pollination.txt --output outputs/my_plan --plan-only
```

Then edit `outputs/my_plan/storyboard.json`. The JSON is the main editable
format for animation. Existing Word storyboards are normalized into safer wording
and structured animation shots automatically. The legacy DOCX export is still
written for reading, but it does not preserve the new animation fields.

For `Types of Pollination` files, the review also writes `coverage.json` and
warns when the storyboard misses important topic areas. Use `--strict-coverage`
when you want to block incomplete final lessons.

Useful shot templates:

| Template | Use |
|---|---|
| `pollination` | Animated pollen movement between two flower diagrams |
| `protandry` | Highlights anthers first, then stigma |
| `protogyny` | Highlights stigma first, then anthers |
| `process` | 2-4 ordered teaching cards |
| `comparison` | 2-4 comparison cards |
| `photo` | Reviewed image or generated photo with captions |
| `video` | Imported silent MP4 from Blender, Manim, or another editor |

Example segment shot:

```json
"shot": {
  "template": "process",
  "heading": "Three stages",
  "learning_objective": "Explain the order of the process",
  "steps": ["Pollen lands", "Pollen tube grows", "Male gametes reach ovule"],
  "cues": ["lands", "tube grows", "reach ovule"]
}
```

Cue text must appear in the narration in order. With Edge TTS, cue timing comes
from real word boundaries. In preview mode it is estimated from narration length.

## Models And Licenses

The Llama and SDXL links are license references for the optional model stack.
This change does not download Qwen3-8B and does not require it. Keep your current
Llama/Ollama and SDXL settings unless you want to change them separately.

`visual_mode: hybrid` means structured templates are rendered locally. Ollama is
used for planning from raw text. ComfyUI/SDXL is used only when a `photo` shot has
no reviewed asset.

Optional free tools:

| Tool | How to use it here |
|---|---|
| Inkscape | Export reviewed diagrams as PNG and use them as `photo` assets |
| Blender | Export a silent MP4 and use a `video` shot |
| Manim | Render an educational animation externally and import its MP4 |

Direct SVG animation and automatic Blender/Manim scene generation are not
implemented yet.

ComfyUI video is supported through a supplied API workflow JSON. Set
`comfyui.video_enabled: true` and `comfyui.video_workflow_path`, then use
`"shot": {"template": "video"}` for segments that should be generated as short
video clips. Keep factual/label-heavy scenes on local animation templates.

## Output Files

Each run writes files such as:

- `storyboard.json`
- `storyboard.docx`
- `review.json`
- `subtitles.srt`
- `narration.wav`
- `frames/*.png`
- `clips/*.mp4`
- `final.mp4`

By default, subtitles are written as `subtitles.srt` and embedded as a soft MP4
subtitle track. They are not burned into the video picture. Use `--burn-captions`
only when you want always-visible captions.

`review.json` is deterministic lint for obvious storyboard issues. It is not a
scientific fact checker, so review subject content before publishing lessons.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On this machine, the launcher can use the portable Python fallback described in
`docs/ANIMATION_WORKFLOW.md`.

For full details, read `docs/ANIMATION_WORKFLOW.md`.
