from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", r"D:\AI\ComfyUI_windows_portable\ComfyUI"))
CACHE_DIR = Path(os.environ.get("WAN_HF_CACHE", r"D:\AI\ComfyUI_windows_portable\hf_cache"))

FILES = [
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors",
        COMFY_ROOT / "models" / "diffusion_models" / "wan2.2_ti2v_5B_fp16.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "split_files/vae/wan2.2_vae.safetensors",
        COMFY_ROOT / "models" / "vae" / "wan2.2_vae.safetensors",
    ),
    (
        "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        COMFY_ROOT / "models" / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    ),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN") or None
    print(f"ComfyUI root: {COMFY_ROOT}")
    print(f"Cache: {CACHE_DIR}")
    for repo_id, filename, target in FILES:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 1024 * 1024:
            print(f"Already installed: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
            continue
        print(f"Downloading {repo_id}/{filename}")
        downloaded = Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=CACHE_DIR,
            token=token,
        ))
        print(f"Copying to {target}")
        shutil.copy2(downloaded, target)
        print(f"Installed: {target} ({target.stat().st_size / 1024**3:.2f} GB)")
    print("Wan2.2 5B model files are installed.")


if __name__ == "__main__":
    main()
