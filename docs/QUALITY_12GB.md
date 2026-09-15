# Clear educational visuals on a 12 GB GPU

The uploaded examples contain invented lettering and incorrect anatomy. More
VRAM or a higher video bitrate cannot repair those errors. This pipeline now
rewrites image prompts, draws real text with Pillow, and exports stable 1080p
frames. It remains a narrated still-image video pipeline, not generative motion
video. It cannot guarantee the quality or scientific accuracy of a manual film.

## Install and select the model

1. Keep your working ComfyUI installation. This workflow uses only built-in nodes;
   no additional custom nodes are required.
2. Download `sd_xl_base_1.0.safetensors` from the
   [official SDXL files](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main).
3. Put it in `ComfyUI/models/checkpoints/`, then restart ComfyUI.
4. Use `config.12gb.yaml` directly. Copy your machine-specific FFmpeg paths into
   the new profile if FFmpeg/FFprobe are not on PATH.

The new starting settings are 1344 x 768, batch size 1, 30 steps, CFG 5.5,
`dpmpp_2m`, `karras`. They target a 12 GB GPU but are not benchmarked on your machine.
If memory is insufficient, reduce image size to 1152 x 640. Keep output resolution
at 1920 x 1080; scaling does not create missing detail. Avoid running other GPU jobs.

Do not use 30 steps/CFG 5.5 with the Lightning checkpoint. To keep Lightning for
fast previews, set checkpoint `sdxl_lightning_4step.safetensors`, steps 4, CFG 1,
sampler `euler`, scheduler `sgm_uniform`. See the
[Lightning model instructions](https://huggingface.co/ByteDance/SDXL-Lightning).

## Planner and voice

Your existing `llama3.1:8b` works. An alternative with Apache-2.0 model licensing is
[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):

```powershell
ollama pull qwen2.5:7b
```

Set `ollama_model: "qwen2.5:7b"` to use it. The planner rewrites prompts even when
the input DOCX already has a storyboard, and releases its model after the final
rewrite before ComfyUI generates images. Prompt rewriting needs Ollama running.
Set `comfyui.rewrite_prompts: false` only for prompts you have already prepared.

Neerja at -20% remains configured to preserve your requested voice. Edge TTS uses
Microsoft's online service: this configuration is not fully open source/offline.
For local speech, the project supports [Piper](https://github.com/OHF-Voice/piper1-gpl):

```powershell
python -m pip install piper-tts
python -m piper.download_voices en_US-lessac-medium
```

In the YAML set `tts_provider: "piper"`, `piper_path: "piper"`,
`piper_model: "en_US-lessac-medium.onnx"`, and
`piper_config: "en_US-lessac-medium.onnx.json"`. Activate the Python environment
before running. This voice differs from Neerja; review its model card/license.
SDXL itself uses Open RAIL++ with use restrictions, and Llama has its own license.
Local execution and open weights do not mean an entirely permissive open-source stack.

## Preview first

From the updated project folder on your rendering machine:

```powershell
python -m pip install -r requirements.txt
python -m app.main --input "C:\Users\admin\Downloads\5913833-Types of Pollination_storyboard.docx" --output ".\outputs\quality_preview" --config ".\config.12gb.yaml" --limit-segments 6 --skip-tts
```

Inspect `frames/` for the actual composition; `generated_images/` contains raw AI
images without software labels. `visual_plan.json` contains the rewritten prompts
and flags likely diagram/anatomy segments for review. These flags are keyword
hints, not an automated scientific accuracy check. AI can still invent text even
when prompts discourage it; reject and replace those images.

## Accurate diagrams and labels

For flower cross-sections, stigma/anther identification, ovules, and fertilization,
use a teacher-reviewed diagram. Create it with an open-source drawing tool such
as Inkscape or Blender, or use a suitably licensed educational asset. Export at
least 1600 pixels wide with clear labels and generous margins.

Place overrides in the project, for example:

```text
assets/reviewed/scene_02_segment_02.png
assets/reviewed/scene_06_segment_03.png
```

The pipeline selects matching PNG/JPG/JPEG files before calling ComfyUI. It keeps
the complete image visible instead of cropping it. Labels identifying parts must
be in the reviewed diagram: the application cannot reliably locate anatomy inside
arbitrary generated images. The separate keyword band is a topic legend, not an
anatomical annotation. Check numbering against the exported storyboard JSON.

Keep narration to one short sentence per segment. All narration is fitted to the
reserved band; very long segments raise an error instead of silently dropping text.
Set `render.font_path` to a TTF file (for example Noto Sans) for another language.

## Full render

```powershell
python -m app.main --input "C:\Users\admin\Downloads\5913833-Types of Pollination_storyboard.docx" --output ".\outputs\pollination_quality_v2" --config ".\config.12gb.yaml"
```

Use a new output folder. Existing malformed images are not repaired. The render
uses CRF 17, 1080p, 30 fps, stationary typography, and copies encoded clips into
the final MP4 without encoding their video again. Segment boundaries use cumulative
frame rounding to limit timing drift. Captions remain visible for their segment;
they are not word-highlighted karaoke subtitles.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests exercise prompt polarity, rewriting existing storyboards, text wrapping,
reviewed overrides, and an FFmpeg smoke render when FFmpeg is installed.
GPU generation and visual/scientific review must be performed on the render machine.
