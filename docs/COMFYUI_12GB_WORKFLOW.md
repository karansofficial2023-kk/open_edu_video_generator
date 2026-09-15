# ComfyUI 12GB Workflow

> The current 1080p quality profile and installation instructions are in
> [Quality on 12 GB](QUALITY_12GB.md). The Lightning settings below describe the
> earlier fast-preview workflow, not the current `config.12gb.yaml` defaults.

This project now supports `visual_mode: comfyui-image` for an RTX 3050 12 GB system.

The 12 GB workflow is:

```text
DOCX or TXT
-> Ollama llama3.1:8b storyboard planning
-> Edge TTS Neerja voice at -20%
-> ComfyUI image generation per storyboard segment
-> full-frame educational composite
-> FFmpeg pan/zoom video clips
-> final synced MP4
```

## Best Starting Model

Use this first:

```text
ByteDance SDXL-Lightning 4-step full checkpoint
File: sdxl_lightning_4step.safetensors
Folder: ComfyUI/models/checkpoints
```

Recommended settings:

```yaml
checkpoint: "sdxl_lightning_4step.safetensors"
width: 1024
height: 576
steps: 4
cfg: 1.0
sampler_name: "euler"
scheduler: "sgm_uniform"
```

This is fast enough for iteration and good enough to replace the low-quality placeholder visuals.

## Better Quality Model Option

After the first workflow works, try an SDXL realistic or documentary-style checkpoint:

```text
Juggernaut XL
RealVisXL
DreamShaper XL
```

Use:

```yaml
steps: 20
cfg: 5.0
sampler_name: "dpmpp_2m_sde"
scheduler: "karras"
```

This is slower but usually more polished than Lightning.

## Install ComfyUI Nodes

Install these with ComfyUI Manager:

```text
ComfyUI-Manager
ComfyUI-VideoHelperSuite
ComfyUI_essentials
ComfyUI-KJNodes
ComfyUI-Impact-Pack
```

For later image-to-video tests:

```text
ComfyUI-AnimateDiff-Evolved
ComfyUI-Frame-Interpolation
```

## Project Config

Use:

```powershell
copy config.12gb.yaml config.yaml
```

Important config section:

```yaml
visual_mode: "comfyui-image"

comfyui:
  api_url: "http://127.0.0.1:8188"
  enabled: true
  workflow_path: "workflows/sdxl_txt2img_api.json"
  checkpoint: "sdxl_lightning_4step.safetensors"
  width: 1024
  height: 576
  steps: 4
  cfg: 1.0
  sampler_name: "euler"
  scheduler: "sgm_uniform"
  fallback_to_slideshow: false
```

## Run Order

1. Start ComfyUI and confirm it opens:

```text
http://127.0.0.1:8188
```

2. Start Ollama:

```powershell
ollama serve
ollama pull llama3.1:8b
```

3. Run the generator:

```powershell
python -m app.main --input "C:\Users\admin\Downloads\5913833-Types of Pollination_storyboard.docx" --output ".\outputs\pollination_12gb" --config ".\config.yaml"
```

## Output

```text
outputs/pollination_12gb/
  generated_images/
  frames/
  clips/
  audio/
  final.mp4
```

## Why This Is Better Than The First Output

The first version drew simple placeholder educational graphics. This 12 GB mode creates real AI images per segment in ComfyUI, then uses those images as the video source. That is the step needed to approach your manually created benchmark video.

## Next Upgrade

Once still-image quality is good, add image-to-video:

```text
generated_images/
-> AnimateDiff or Stable Video Diffusion workflow
-> short MP4 clips
-> same audio/subtitle/final FFmpeg pipeline
```

Do this only after image generation is stable. Bad source images make bad video clips.
