from __future__ import annotations

import json
import random
import time
import uuid
from collections.abc import MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from .config import AppConfig
from .schema import Segment


class ComfyUIClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.base_url = config.comfyui.api_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def generate_image(self, segment: Segment, output_path: str | Path, filename_prefix: str) -> Path:
        workflow = self._load_workflow()
        prompt = self._build_prompt(segment)
        workflow = self._patch_workflow(workflow, prompt, filename_prefix)
        prompt_id = self._queue_prompt(workflow)
        image_info = self._wait_for_image(prompt_id)
        return self._download_image(image_info, output_path)

    def generate_video(self, segment: Segment, output_path: str | Path, filename_prefix: str) -> Path:
        workflow = self._load_video_workflow()
        prompt = self._build_video_prompt(segment)
        workflow = self._patch_workflow(workflow, prompt, filename_prefix, video=True)
        prompt_id = self._queue_prompt(workflow)
        video_info = self._wait_for_media(prompt_id, ("videos", "gifs", "animated", "images"))
        return self._download_media(video_info, output_path)

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def _load_workflow(self) -> dict[str, Any]:
        workflow_path = Path(self.config.comfyui.workflow_path)
        if not workflow_path.is_absolute():
            workflow_path = Path.cwd() / workflow_path
        return json.loads(workflow_path.read_text(encoding="utf-8"))

    def _load_video_workflow(self) -> dict[str, Any]:
        if not self.config.comfyui.video_workflow_path.strip():
            raise ValueError("Set comfyui.video_workflow_path to a ComfyUI API workflow JSON.")
        workflow_path = Path(self.config.comfyui.video_workflow_path)
        if not workflow_path.is_absolute():
            workflow_path = Path.cwd() / workflow_path
        return json.loads(workflow_path.read_text(encoding="utf-8"))

    def _build_prompt(self, segment: Segment) -> str:
        parts = [
            "Single coherent image, text-free, unlabeled, no typography.",
            self.config.comfyui.prompt_prefix,
            segment.image_prompt or segment.visual or segment.narration,
            self.config.comfyui.prompt_suffix,
        ]
        return " ".join(part for part in parts if part).strip()

    def _build_video_prompt(self, segment: Segment) -> str:
        parts = [
            self.config.comfyui.video_prompt_prefix,
            segment.image_prompt or segment.visual or segment.narration,
            self.config.comfyui.video_prompt_suffix,
        ]
        return " ".join(part for part in parts if part).strip()

    def _patch_workflow(self, workflow: dict[str, Any], prompt: str, filename_prefix: str, video: bool = False) -> dict[str, Any]:
        patched = deepcopy(workflow)
        for node in patched.values():
            class_type = node.get("class_type")
            inputs = node.setdefault("inputs", {})
            if class_type == "CheckpointLoaderSimple" and not video:
                inputs["ckpt_name"] = self.config.comfyui.checkpoint
            elif class_type == "EmptyLatentImage":
                inputs["width"] = self.config.comfyui.video_width if video else self.config.comfyui.width
                inputs["height"] = self.config.comfyui.video_height if video else self.config.comfyui.height
                inputs["batch_size"] = 1
            elif class_type in {"EmptyHunyuanLatentVideo", "WanImageToVideoLatent", "Wan22ImageToVideoLatent", "EmptyLatentVideo", "EmptyLTXVLatentVideo", "LTXVEmptyLatentVideo", "LTXVLatentVideo", "LTXVideoLatent"}:
                if "width" in inputs:
                    inputs["width"] = self.config.comfyui.video_width
                if "height" in inputs:
                    inputs["height"] = self.config.comfyui.video_height
                for key in ("length", "num_frames", "frames"):
                    if key in inputs:
                        inputs[key] = self.config.comfyui.video_frames
            elif class_type == "KSampler":
                inputs["seed"] = self._seed()
                inputs["steps"] = self.config.comfyui.video_steps if video else self.config.comfyui.steps
                inputs["cfg"] = self.config.comfyui.video_cfg if video else self.config.comfyui.cfg
                inputs["sampler_name"] = self.config.comfyui.video_sampler_name if video else self.config.comfyui.sampler_name
                inputs["scheduler"] = self.config.comfyui.video_scheduler if video else self.config.comfyui.scheduler
            elif class_type in {"SaveImage", "SaveAnimatedWEBP", "VHS_VideoCombine", "SaveVideo"}:
                inputs["filename_prefix"] = filename_prefix
                if "frame_rate" in inputs:
                    inputs["frame_rate"] = self.config.comfyui.video_fps
                if "fps" in inputs:
                    inputs["fps"] = self.config.comfyui.video_fps
            elif video and class_type in {"CreateVideo", "LTXVCreateVideo"}:
                if "fps" in inputs:
                    inputs["fps"] = self.config.comfyui.video_fps
                if "frame_rate" in inputs:
                    inputs["frame_rate"] = self.config.comfyui.video_fps

        self._patch_generic_video_inputs(patched, prompt, filename_prefix, video)

        # Conditioning connections, not JSON key order, identify prompt polarity.
        for node in patched.values():
            if node.get("class_type") != "KSampler":
                continue
            for polarity, text in [("positive", prompt), ("negative", self.config.comfyui.negative_prompt)]:
                link = node["inputs"].get(polarity)
                if not isinstance(link, list):
                    continue
                encoder = patched[str(link[0])]
                if encoder.get("class_type") != "CLIPTextEncode":
                    continue
                encoder["inputs"]["text"] = text

        return patched

    def _patch_generic_video_inputs(
        self,
        workflow: dict[str, Any],
        prompt: str,
        filename_prefix: str,
        video: bool,
    ) -> None:
        if not video:
            return
        prompt_keys = {"prompt", "positive_prompt", "positive", "text_prompt"}
        negative_keys = {"negative_prompt", "negative"}
        width_keys = {"width", "video_width"}
        height_keys = {"height", "video_height"}
        frame_keys = {"length", "num_frames", "frames", "frame_count", "video_frames"}
        fps_keys = {"fps", "frame_rate", "video_fps"}
        step_keys = {"steps", "num_steps", "video_steps"}
        cfg_keys = {"cfg", "cfg_scale", "guidance", "guidance_scale"}
        seed_keys = {"seed", "noise_seed"}
        output_keys = {"filename_prefix", "prefix"}

        for node in workflow.values():
            inputs = node.get("inputs")
            if not isinstance(inputs, MutableMapping):
                continue
            for key in list(inputs):
                normalized = key.lower().replace(" ", "_").replace("-", "_")
                if normalized in prompt_keys and isinstance(inputs[key], str):
                    inputs[key] = prompt
                elif normalized in negative_keys and isinstance(inputs[key], str):
                    inputs[key] = self.config.comfyui.negative_prompt
                elif normalized in width_keys and isinstance(inputs[key], int):
                    inputs[key] = self.config.comfyui.video_width
                elif normalized in height_keys and isinstance(inputs[key], int):
                    inputs[key] = self.config.comfyui.video_height
                elif normalized in frame_keys and isinstance(inputs[key], int):
                    inputs[key] = self.config.comfyui.video_frames
                elif normalized in fps_keys and isinstance(inputs[key], (int, float)):
                    inputs[key] = self.config.comfyui.video_fps
                elif normalized in step_keys and isinstance(inputs[key], int):
                    inputs[key] = self.config.comfyui.video_steps
                elif normalized in cfg_keys and isinstance(inputs[key], (int, float)):
                    inputs[key] = self.config.comfyui.video_cfg
                elif normalized in seed_keys and isinstance(inputs[key], int):
                    inputs[key] = self._seed()
                elif normalized in output_keys and isinstance(inputs[key], str):
                    inputs[key] = filename_prefix

    def _seed(self) -> int:
        if self.config.comfyui.seed >= 0:
            return self.config.comfyui.seed
        return random.randint(1, 2**31 - 1)

    def _queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        if not response.ok:
            try:
                details = json.dumps(response.json(), indent=2, ensure_ascii=False)
            except ValueError:
                details = response.text
            raise RuntimeError(
                f"ComfyUI rejected the workflow (HTTP {response.status_code}).\n"
                f"Server: {self.base_url}\n"
                f"Configured checkpoint: {self.config.comfyui.checkpoint}\n"
                f"Sampler: {self.config.comfyui.sampler_name}; "
                f"scheduler: {self.config.comfyui.scheduler}\n"
                "ComfyUI response:\n" + details + "\n\n"
                "If ckpt_name is not in the available list, install the checkpoint in "
                "ComfyUI/models/checkpoints, restart ComfyUI, and set comfyui.checkpoint "
                "in the YAML passed with --config to the exact dropdown name "
                "(including its subfolder, if any).\n"
                "For other errors, use the node_errors details above to identify the rejected input."
            )
        return response.json()["prompt_id"]

    def _wait_for_image(self, prompt_id: str) -> dict[str, str]:
        return self._wait_for_media(prompt_id, ("images",))

    def _wait_for_media(self, prompt_id: str, media_keys: tuple[str, ...]) -> dict[str, str]:
        deadline = time.time() + self.config.comfyui.timeout_seconds
        while time.time() < deadline:
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            history = response.json()
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI generation failed: {status.get('messages', [])}")
                outputs = history[prompt_id].get("outputs", {})
                for output in outputs.values():
                    for media_key in media_keys:
                        item = self._first_media_file(output.get(media_key, []))
                        if item:
                            return item
            time.sleep(1.5)
        raise TimeoutError(f"ComfyUI timed out waiting for prompt {prompt_id}")

    def _first_media_file(self, items: Any) -> dict[str, str] | None:
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            return None
        for item in items:
            if isinstance(item, dict) and item.get("filename"):
                return item
        return None

    def _download_image(self, image_info: dict[str, str], output_path: str | Path) -> Path:
        return self._download_media(image_info, output_path)

    def _download_media(self, image_info: dict[str, str], output_path: str | Path) -> Path:
        if not isinstance(image_info, dict) or not image_info.get("filename"):
            raise RuntimeError(f"ComfyUI did not return a downloadable media file: {image_info!r}")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        query = urlencode(
            {
                "filename": image_info["filename"],
                "subfolder": image_info.get("subfolder", ""),
                "type": image_info.get("type", "output"),
            }
        )
        response = requests.get(f"{self.base_url}/view?{query}", timeout=120)
        response.raise_for_status()
        output.write_bytes(response.content)
        return output


