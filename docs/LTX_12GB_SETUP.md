# LTX 12GB setup

This copy is the LTX experiment branch. Keep `D:\Python\open_edu_video_generator` unchanged and run this folder when testing LTX.

Use LTX only for short cinematic b-roll clips. On an RTX 3060 12GB, do not target a single uninterrupted one-minute local render. Generate 4-6 second clips and let this project assemble them with narration, captions, and deterministic educational animation.

Recommended command:

```powershell
cd D:\Python\open_edu_video_generator_LTX
powershell -ExecutionPolicy Bypass -File .\run_video.ps1 --config config.ltx.12gb.yaml --input samples/pollination_animated.json --output outputs/ltx_test --limit-segments 3
```

ComfyUI requirements:

- Install or update LTX nodes from ComfyUI Manager. Search for `LTXVideo` / `ComfyUI-LTXVideo`.
- Load an official LTX text-to-video workflow in ComfyUI first and use its missing-model downloader when possible.
- Export that workflow in API format to `workflows/ltx_video_12gb_api.json`.

12GB starting settings in `config.ltx.12gb.yaml`:

- `video_width: 768`
- `video_height: 512`
- `video_frames: 97`
- `video_fps: 24`
- `video_steps: 8`
- `video_max_segments: 1`

LTX 2.5 model locations if installing manually:

- `ltx-2.5-22b-distilled-transformer-bf16.safetensors` -> `ComfyUI/models/diffusion_models/`
- `gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` -> `ComfyUI/models/text_encoders/`
- `gemma4_e2b_it_bf16.safetensors` -> `ComfyUI/models/text_encoders/`
- `ltx-2.5-video-vae-bf16.safetensors` -> `ComfyUI/models/vae/`
- `ltx-2.5-audio-vae-bf16.safetensors` -> `ComfyUI/models/vae/`

If the 22B BF16 workflow is too heavy on 12GB, prefer the smallest/distilled/quantized LTX workflow available in your ComfyUI Manager or use the existing Wan config as fallback.

## Installed on this machine

Installed for RTX 3060 12GB:

- `D:\AI\ComfyUI_windows_portable\ComfyUI\models\checkpoints\ltxv-2b-0.9.8-distilled-fp8.safetensors`
- `D:\AI\ComfyUI_windows_portable\ComfyUI\models\text_encoders\t5xxl_fp16.safetensors`
- `D:\AI\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-LTXVideo`

The heavy LTX 2.5 / 22B workflow is not the target for this computer. Use the 2B FP8 LTXV text-to-video workflow in ComfyUI, then export it in API format to:

`D:\Python\open_edu_video_generator_LTX\workflows\ltx_video_12gb_api.json`

Recommended first LTX test values:

- width: `768`
- height: `512`
- frames: `81`
- fps: `16`
- steps: `8`

That produces about 5 seconds per generated clip. Build one-minute educational videos by assembling many short clips, not by one huge local LTX render.
