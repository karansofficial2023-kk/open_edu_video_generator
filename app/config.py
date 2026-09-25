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
    video_bitrate: str = ""
    video_maxrate: str = ""
    video_bufsize: str = ""
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
    real_image_auto_promote: bool = True
    video_auto_promote: bool = True
    video_max_segments: int = 1
    video_prompt_prefix: str = "high quality educational cinematic video, smooth motion,"
    video_prompt_suffix: str = "no text, no subtitles, no watermark, coherent motion"


class VisionQAConfig(BaseModel):
    enabled: bool = False
    model: str = "qwen2.5vl:7b"
    max_attempts: int = 3
    minimum_relevance: float = 0.78
    minimum_subject_match: float = 0.72
    block_on_failure: bool = True
    timeout_seconds: int = 600


class AssetRetrievalConfig(BaseModel):
    enabled: bool = False
    provider: str = "wikimedia_commons"
    max_candidates: int = 8
    timeout_seconds: int = 45
    thumbnail_width: int = 1600
    user_agent: str = "OpenEduVideoGenerator/0.1 (local educational video application)"
    allowed_licenses: list[str] = [
        "CC0",
        "Public domain",
        "CC BY 4.0",
        "CC BY-SA 4.0",
        "CC BY 3.0",
        "CC BY-SA 3.0",
        "CC BY 2.0",
        "CC BY-SA 2.0",
    ]


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
    vision_qa: VisionQAConfig = VisionQAConfig()
    asset_retrieval: AssetRetrievalConfig = AssetRetrievalConfig()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    data: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)
