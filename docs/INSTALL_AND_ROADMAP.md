# Install And Roadmap

For the updated 12 GB quality profile, model download, readable labels, reviewed
diagram overrides, and preview commands, follow [Quality on 12 GB](QUALITY_12GB.md).

## Current Target

Your current practical target is:

- Windows
- Python 3.12.10
- GTX 1650 4 GB VRAM
- 16 GB RAM
- 720p educational video
- Open-source rendering tools; the default Edge voice is an online service

The default project mode is intentionally conservative. It does not rely on heavy diffusion video models on the GTX 1650. It creates storyboard-driven educational visuals, renders motion with FFmpeg, and syncs everything to Edge TTS using the `en-IN-NeerjaNeural` voice at `-20%` speed.

## Required Tools For Windows 4 GB

Install these first:

1. Python 3.12.10
2. Git
3. Git LFS
4. FFmpeg full build
5. Ollama
6. NVIDIA driver
7. 7-Zip
8. Visual C++ Redistributable
9. Internet access for Edge TTS voice generation

Optional:

1. ComfyUI Desktop
2. ComfyUI Manager
3. ComfyUI-VideoHelperSuite

## Recommended Folder Layout

```text
C:/ai-tools/
  ffmpeg/
  comfyui/

C:/ai-output/
  edu-videos/
```

Then set `config.yaml` like this:

```yaml
ffmpeg_path: "C:/ai-tools/ffmpeg/bin/ffmpeg.exe"
ffprobe_path: "C:/ai-tools/ffmpeg/bin/ffprobe.exe"
tts_provider: "edge"
edge_tts:
  voice: "en-IN-NeerjaNeural"
  rate: "-20%"
```

## Python Environment

```powershell
cd "C:\Users\karan\OneDrive\Desktop\New folder\open_edu_video_generator"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy config.4gb.yaml config.yaml
```

If `py -3.12` does not work, use your full Python path:

```powershell
"C:\Users\admin\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv
```

## Ollama Models

For 4 GB VRAM, run LLM mostly on CPU/RAM and use a smaller model:

```powershell
ollama serve
ollama pull llama3.2:3b
```

Better quality if acceptable speed:

```powershell
ollama pull llama3.1:8b
```

Good future models:

```text
4 GB: llama3.2:3b
12 GB: llama3.1:8b or qwen2.5:7b
48 GB: qwen2.5:14b, mistral-small class models, or larger local models as available
```

## Edge TTS

Use Edge TTS for the requested Neerja voice.

The project config should include:

```yaml
tts_provider: "edge"
edge_tts:
  voice: "en-IN-NeerjaNeural"
  rate: "-20%"
```

Install the Python package:

```powershell
pip install edge-tts
```

Note: Edge TTS is not fully offline/open-source. Piper remains in the code as the optional offline/open-source provider.

## Run The Project

Plain text input:

```powershell
.\.venv\Scripts\Activate.ps1
python -m app.main --input ".\samples\pollination.txt" --output ".\outputs\pollination" --config ".\config.yaml"
```

Storyboard DOCX input:

```powershell
python -m app.main --input "C:\Users\karan\Downloads\5913833-Types of Pollination_storyboard.docx" --output ".\outputs\pollination_docx" --config ".\config.yaml"
```

Debug without audio:

```powershell
python -m app.main --input ".\samples\pollination.txt" --output ".\outputs\debug_frames" --skip-tts
```

## What The Project Produces

```text
outputs/name/
  storyboard.json
  storyboard.docx
  visual_prompts.txt
  subtitles.srt
  narration.wav
  final.mp4
  audio/
  frames/
  clips/
```

## 4 GB Production Strategy

Use:

```yaml
visual_mode: "slideshow"
ollama_model: "llama3.2:3b"
output_resolution:
  width: 1280
  height: 720
```

Keep scenes short. A strong pattern is:

```text
1 scene = 20 to 45 seconds
1 segment = 1 narration sentence
1 visual = one clean educational screen
```

This gives reliable video with:

- voiceover
- subtitles
- animated image motion
- concept labels
- scene titles
- final MP4

## 12 GB Upgrade Strategy

Install ComfyUI and generate one image per segment.

Recommended path:

```text
storyboard.json
-> visual_prompts.txt
-> ComfyUI image generation
-> use generated images instead of built-in frames
-> same FFmpeg timing/audio/subtitle pipeline
```

Use image models before video models. They are cheaper, more reliable, and enough for many educational videos.

This project now has a working `comfyui-image` integration. See `docs/COMFYUI_12GB_WORKFLOW.md`.

Suggested experiments:

```text
SD 1.5 Lightning/LCM
SDXL Turbo/Lightning
AnimateDiff short clips
Stable Video Diffusion image-to-video
```

## 48 GB Linux Server Strategy

Use the same storyboard JSON, but move visual generation to server workflows:

```text
text/docx
-> storyboard JSON
-> Edge TTS, Piper, or XTTS voice
-> ComfyUI image/video workflow
-> per-scene clips
-> FFmpeg final assembly
```

On 48 GB VRAM, experiment with:

```text
Wan-class image-to-video workflows
LTX Video workflows
higher resolution ComfyUI pipelines
longer clips per scene
```

Do not build your business logic inside ComfyUI. Keep ComfyUI as a visual generation worker and keep Python as the pipeline controller.

## Future Features To Add

1. ComfyUI API image provider
2. ComfyUI API image-to-video provider
3. DOCX storyboard round-trip editing
4. Per-segment voice emotion and speed
5. Auto glossary and term cards
6. Diagram templates for science/math topics
7. Human approval step before final render
8. Batch folder processing
9. Web UI using FastAPI
10. Reuse your existing Java project outputs as direct inputs
