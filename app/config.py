from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class Resolution(BaseModel):
    width: int = 1280
    height: int = 720


class VoiceConfig(BaseModel):
    sentence_gap_seconds: float = 0.25
    scene_gap_seconds: float = 0.7


class EdgeTTSConfig(BaseModel):
    voice: str = "en-IN-NeerjaNeural"
    rate: str = "-20%"


class RenderConfig(BaseModel):
    background: str = "#F7F8F3"
    ink: str = "#182026"
    accent: str = "#2E7D63"
    second_accent: str = "#D18B32"
    subtitle_bg: str = "#101820"
    subtitle_fg: str = "#FFFFFF"
    seconds_per_segment_without_audio: float = 5.0
    video_crf: int = 18
    video_preset: str = "slow"
    font_path: str = ""
    reviewed_assets_dir: str = ""
    layout: str = "legacy"
    burn_captions: bool = False
    embed_subtitles: bool = True


class ComfyUIConfig(BaseModel):
    api_url: str = "http://127.0.0.1:8188"
    enabled: bool = False
    workflow_path: str = "workflows/sdxl_txt2img_api.json"
    checkpoint: str = "sdxl_lightning_4step.safetensors"
    width: int = 1024
    height: int = 576
    steps: int = 4
    cfg: float = 1.0
    sampler_name: str = "euler"
    scheduler: str = "sgm_uniform"
    seed: int = -1
    timeout_seconds: int = 900
    fallback_to_slideshow: bool = True
    rewrite_prompts: bool = False
    prompt_prefix: str = "high quality educational documentary still, clean scientific visual, cinematic lighting, detailed, accurate, natural colors,"
    prompt_suffix: str = "no watermark, no logo, no extra text, no distorted anatomy, no blurry image"
    negative_prompt: str = "low quality, blurry, watermark, logo, extra text, misspelled text, deformed, distorted, noisy, oversaturated, cartoonish"
    video_enabled: bool = False
    video_workflow_path: str = ""
    video_width: int = 832
    video_height: int = 480
    video_frames: int = 81
    video_fps: int = 16
    video_steps: int = 20
    video_cfg: float = 5.0
    video_sampler_name: str = "uni_pc"
    video_scheduler: str = "simple"
    video_prompt_prefix: str = "high quality educational cinematic video, smooth motion,"
    video_prompt_suffix: str = "no text, no subtitles, no watermark, coherent motion"


class SubtitleStyle(BaseModel):
    font_px: int = 42
    band_opacity: float = 0.55
    band_color: str = "#111417"
    text_color: str = "#FFFFFF"
    max_lines: int = 2
    margin_x: float = 0.06       # fraction of width kept clear left/right
    margin_bottom: float = 0.04  # fraction of height kept clear under the band
    padding_px: int = 16


class OpenTtsConfig(BaseModel):
    """Open-source Indic voices (AI4Bharat Indic-TTS) run in their own Python environment; see docs/OPEN_VOICES.md."""
    python: str = "D:/AI/venvs/indictts/Scripts/python.exe"
    models_dir: str = "D:/AI/models/indic-tts"
    speaker: str = "female"            # each language ships a female and a male voice


class ProductionConfig(BaseModel):
    """Production pipeline (contract-driven storyboards). All tools are open source and local."""
    vision_model: str = "qwen3-vl:8b-instruct"
    vision_fallback_model: str = "qwen2.5vl:7b"
    prompt_director: bool = True           # repair generic image prompts from approved narration
    director_model: str = "qwen3:8b"
    max_attempts: int = 3
    min_semantic_score: float = 0.65
    generate_title_backgrounds: bool = True    # False: title cards use the built-in gradient, so lessons without photos never load the image model
    ocr_text_check: bool = True            # reject stills with readable lettering found by OCR (needs rapidocr-onnxruntime)
    gallery_thumbnails: bool = True        # board "uses" lists get one reviewed picture per item (physics/chemistry boards only)
    min_sharpness: float = 45.0            # Laplacian variance floor at 1024px
    image_workflow_path: str = "workflows/flux_schnell_txt2img_api.json"
    image_checkpoint: str = "flux1-schnell-fp8.safetensors"
    image_width: int = 1344
    image_height: int = 768
    image_steps: int = 4
    label_min_confidence: float = 0.6
    label_font_px: int = 32                # >= 28 px at 1080p per spec
    label_hold_seconds: float = 3.0
    label_stagger_seconds: float = 0.9
    crossfade_seconds: float = 0.4
    logo_path: str = ""
    logo_height_px: int = 64
    watermark_text: str = ""
    retrieval_enabled: bool = False        # Wikimedia Commons first, license-checked
    retrieval_min_width: int = 1600
    semantic_qa: bool = True
    tts_provider: str = "edge"             # edge = cloud voices; open = open-source voices only (fails if none is installed); auto = open when installed
    open_tts: OpenTtsConfig = OpenTtsConfig()
    pronunciations: dict[str, dict[str, str]] = {}   # per language code: {"CO2": "spoken form"}, applied to narration before synthesis
    indic_rate: str = "-10%"               # speaking rate for non-English lessons (Edge TTS)
    indic_voices: dict[str, str] = {
        "hi": "hi-IN-SwaraNeural", "mr": "mr-IN-AarohiNeural", "ta": "ta-IN-PallaviNeural", "te": "te-IN-ShrutiNeural",
        "ml": "ml-IN-SobhanaNeural", "kn": "kn-IN-SapnaNeural", "bn": "bn-IN-TanishaaNeural", "gu": "gu-IN-DhwaniNeural",
        "ur": "ur-IN-GulNeural", "ne": "ne-NP-HemkalaNeural", "si": "si-LK-ThiliniNeural"}
    music_path: str = ""                  # optional royalty-free track; mixed under the narration and ducked
    music_volume_db: float = -22.0
    motion_enabled: bool = False           # animate approved stills with LTX image-to-video (about 1.5 min per 1080p clip on a 12 GB card)
    motion_workflow_path: str = "workflows/ltx_i2v_1080_api.json"   # ComfyUI API workflow: LoadImage -> LTXVImgToVideo -> sampler -> save
    motion_width: int = 1920               # multiples of 32; the 1088-high result is cover-cropped to 1080
    motion_height: int = 1088
    motion_frames: int = 97                # 8n+1 frames; 97 at 24 fps = 4 s
    motion_fps: int = 24
    motion_max_clips: int = 6              # per lesson: the contract's `animate` shots first, by order, then none
    motion_attempts: int = 2               # new seed when a clip fails the checks
    motion_start_match: float = 0.90       # first frame must still look like the approved picture (luminance correlation, 0..1)
    motion_end_match: float = 0.55         # and the last frame must not have drifted into a different scene
    retry_rejected: bool = False           # rerun images that already failed every attempt
    auto_labels: bool = False              # propose labels (parts named in the narration) for photographs the contract left unlabelled
    subtitle: SubtitleStyle = SubtitleStyle()


class AppConfig(BaseModel):
    project_name: str = "Open Edu Video Generator"
    output_resolution: Resolution = Resolution()
    fps: int = 30
    visual_mode: str = "slideshow"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    use_ollama: bool = True
    planner_fallback: bool = False
    tts_provider: str = "edge"
    edge_tts: EdgeTTSConfig = EdgeTTSConfig()
    piper_path: str = "piper"
    piper_model: str = ""
    piper_config: str = ""
    voice: VoiceConfig = VoiceConfig()
    render: RenderConfig = RenderConfig()
    comfyui: ComfyUIConfig = ComfyUIConfig()
    production: ProductionConfig = ProductionConfig()
    # Optional, topic-agnostic quality hooks (all empty by default):
    # text_replacements: {"misspelt": "correct"} applied to imported storyboards.
    # coverage_topics: {"topic": ["marker", ...]} checklist merged with storyboard.required_topics.
    text_replacements: dict[str, str] = {}
    coverage_topics: dict[str, list[str]] = {}
    coverage_min_ratio: float = 0.75


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
