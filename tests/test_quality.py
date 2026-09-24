from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

from app.comfyui_client import ComfyUIClient
from app.config import AppConfig, load_config
from app.input_reader import _shot_from_media_type
from app.renderer import _subtitle_filter, render_video
from app.schema import Scene, Segment, Shot, Storyboard
from app.visual_planner import prepare_visual_prompts
from app.storyboard_cleanup import normalize_storyboard, promote_real_image_shots, promote_video_shots
from app.visuals import _font, _pixel_wrap, render_ai_image_frame, render_segment_frames
from app.vision_qa import _normalize_proposals


ROOT = Path(__file__).resolve().parents[1]


def storyboard():
    segment = Segment(segment_number=1, narration="Pollen is transferred to the stigma.",
                      visual="Label the anther and stigma on a flower diagram.",
                      image_prompt="A flower with labels", keywords=["Pollen", "Stigma"], end=0.6)
    return Storyboard(title="Pollination", scenes=[Scene(scene_number=1, title="Pollen transfer",
                     narration=segment.narration, segments=[segment])])


class QualityTests(unittest.TestCase):
    def test_queue_error_preserves_server_validation_details(self):
        response = Mock(ok=False, status_code=400)
        response.json.return_value = {"node_errors": {"4": {"errors": [
            {"details": "ckpt_name: missing.safetensors not in []"}]}}}
        with patch("app.comfyui_client.requests.post", return_value=response):
            with self.assertRaises(RuntimeError) as caught:
                ComfyUIClient(AppConfig())._queue_prompt({})
        self.assertIn("HTTP 400", str(caught.exception))
        self.assertIn("missing.safetensors not in []", str(caught.exception))

    def test_queue_error_preserves_non_json_response(self):
        response = Mock(ok=False, status_code=500, text="Worker unavailable")
        response.json.side_effect = ValueError("Not JSON")
        with patch("app.comfyui_client.requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "Worker unavailable"):
                ComfyUIClient(AppConfig())._queue_prompt({})

    def test_queue_success_returns_prompt_id(self):
        response = Mock(ok=True)
        response.json.return_value = {"prompt_id": "queued-id"}
        with patch("app.comfyui_client.requests.post", return_value=response):
            self.assertEqual(ComfyUIClient(AppConfig())._queue_prompt({}), "queued-id")

    def test_wait_for_media_skips_non_file_video_entries(self):
        history_response = Mock()
        history_response.raise_for_status.return_value = None
        history_response.json.return_value = {
            "prompt-id": {
                "status": {"status_str": "success"},
                "outputs": {
                    "11": {"videos": [True, {"filename": "clip.mp4", "subfolder": "video", "type": "output"}]}
                },
            }
        }
        with patch("app.comfyui_client.requests.get", return_value=history_response):
            media = ComfyUIClient(AppConfig())._wait_for_media("prompt-id", ("videos",))
        self.assertEqual(media["filename"], "clip.mp4")

    def test_generate_video_accepts_mp4_returned_under_images_key(self):
        history_response = Mock()
        history_response.raise_for_status.return_value = None
        history_response.json.return_value = {
            "prompt-id": {
                "status": {"status_str": "success"},
                "outputs": {
                    "58": {
                        "images": [{"filename": "clip.mp4", "subfolder": "", "type": "output"}],
                        "animated": [True],
                    }
                },
            }
        }
        with patch("app.comfyui_client.requests.get", return_value=history_response):
            media = ComfyUIClient(AppConfig())._wait_for_media("prompt-id", ("videos", "gifs", "animated", "images"))
        self.assertEqual(media["filename"], "clip.mp4")

    def test_prompt_polarity_does_not_depend_on_json_order(self):
        workflow = json.loads((ROOT / "workflows/sdxl_txt2img_api.json").read_text())
        workflow = dict(reversed(list(workflow.items())))
        config = AppConfig()
        result = ComfyUIClient(config)._patch_workflow(workflow, "A flower", "test")
        self.assertEqual(result["6"]["inputs"]["text"], "A flower")
        self.assertEqual(result["7"]["inputs"]["text"], config.comfyui.negative_prompt)
        self.assertNotEqual(workflow["6"]["inputs"]["text"], "A flower")

    def test_flux_workflow_patches_sd3_latent_dimensions(self):
        workflow = json.loads((ROOT / "workflows/flux_schnell_fp8_api.json").read_text())
        config = AppConfig()
        config.comfyui.checkpoint = "flux1-schnell-fp8.safetensors"
        config.comfyui.width = 960
        config.comfyui.height = 544

        result = ComfyUIClient(config)._patch_workflow(workflow, "A sunflower", "flux_test")

        self.assertEqual(result["4"]["inputs"]["ckpt_name"], config.comfyui.checkpoint)
        self.assertEqual(result["5"]["inputs"]["width"], 960)
        self.assertEqual(result["5"]["inputs"]["height"], 544)
        self.assertEqual(result["6"]["inputs"]["text"], "A sunflower")
        self.assertEqual(result["9"]["inputs"]["filename_prefix"], "flux_test")

    def test_vision_coordinate_proposals_are_normalized_but_not_approved(self):
        proposals = _normalize_proposals({
            "Stigma": {"x": 512, "y": 288, "confidence": 0.9},
            "Unknown": None,
        }, 1024, 576)
        self.assertEqual(proposals["Stigma"]["x"], 0.5)
        self.assertEqual(proposals["Stigma"]["y"], 0.5)
        self.assertEqual(proposals["Unknown"], None)

    def test_subtitle_filter_uses_one_readable_bottom_band(self):
        with tempfile.TemporaryDirectory() as directory:
            subtitles = Path(directory) / "lesson.srt"
            subtitles.write_text(
                "1\n00:00:05,250 --> 00:00:06,500\nPollen transfer\n",
                encoding="utf-8",
            )
            result = _subtitle_filter(subtitles, AppConfig())
            self.assertIn("drawbox=x=0:y=ih*0.72", result)
            self.assertIn("color=black@0.55", result)
            self.assertIn("between(t,5.250,6.500)", result)
            self.assertIn("Fontsize=26", result)

    def test_raw_label_instructions_are_not_appended(self):
        segment = storyboard().all_segments()[0][1]
        segment.image_prompt = "One flower outdoors"
        prompt = ComfyUIClient(AppConfig())._build_prompt(segment)
        self.assertIn("One flower outdoors", prompt)
        self.assertNotIn(segment.visual, prompt)

    def test_video_workflow_patches_common_video_nodes(self):
        config = AppConfig()
        config.comfyui.video_width = 832
        config.comfyui.video_height = 480
        config.comfyui.video_frames = 81
        config.comfyui.video_fps = 16
        config.comfyui.video_steps = 20
        workflow = {
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "3": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "negative": ["2", 0], "seed": 0,
                                                        "steps": 1, "cfg": 1, "sampler_name": "", "scheduler": ""}},
            "4": {"class_type": "Wan22ImageToVideoLatent", "inputs": {"width": 1, "height": 1, "length": 1}},
            "5": {"class_type": "VHS_VideoCombine", "inputs": {"filename_prefix": "", "frame_rate": 1}},
            "9": {"class_type": "CreateVideo", "inputs": {"fps": 24}},
        }
        patched = ComfyUIClient(config)._patch_workflow(workflow, "A moving flower", "lesson_video", video=True)
        self.assertEqual(patched["4"]["inputs"]["width"], 832)
        self.assertEqual(patched["4"]["inputs"]["height"], 480)
        self.assertEqual(patched["4"]["inputs"]["length"], 81)
        self.assertEqual(patched["5"]["inputs"]["filename_prefix"], "lesson_video")
        self.assertEqual(patched["5"]["inputs"]["frame_rate"], 16)
        self.assertEqual(patched["9"]["inputs"]["fps"], 16)
        self.assertEqual(patched["3"]["inputs"]["steps"], 20)
        self.assertEqual(patched["3"]["inputs"]["cfg"], config.comfyui.video_cfg)
        self.assertEqual(patched["3"]["inputs"]["sampler_name"], config.comfyui.video_sampler_name)
        self.assertEqual(patched["3"]["inputs"]["scheduler"], config.comfyui.video_scheduler)
        self.assertEqual(patched["1"]["inputs"]["text"], "A moving flower")

    def test_existing_storyboard_prompts_are_rewritten_and_audited(self):
        config = load_config(ROOT / "config.12gb.yaml")
        board = storyboard()
        response = Mock()
        response.json.return_value = {"response": json.dumps({"image_prompt": "A single flower outdoors."})}
        with tempfile.TemporaryDirectory() as directory, patch("app.visual_planner.requests.post", return_value=response) as post:
            prepare_visual_prompts(board, Path(directory), config)
            self.assertEqual(board.all_segments()[0][1].image_prompt, "A single flower outdoors.")
            audit = json.loads((Path(directory) / "visual_plan.json").read_text())
            self.assertTrue(audit[0]["scientific_review_recommended"])
            self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)

    def test_pixel_wrap_preserves_long_words_without_overflow(self):
        draw = ImageDraw.Draw(Image.new("RGB", (500, 500)))
        font = _font(30)
        text = "WWWWWWWWWWWWWWWWWWWW pollen transfer"
        lines = _pixel_wrap(draw, text, font, 150)
        self.assertEqual("".join(lines).replace(" ", ""), text.replace(" ", ""))
        self.assertTrue(all(draw.textlength(line, font=font) <= 150 for line in lines))

    def test_reviewed_asset_skips_comfy_and_preserves_image_edges(self):
        config = load_config(ROOT / "config.12gb.yaml")
        config.render.layout = "legacy"  # Original layout remains supported explicitly.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.render.reviewed_assets_dir = directory
            source = root / "scene_01_segment_01.png"
            Image.new("RGB", (1000, 600), "red").save(source)
            with patch.object(ComfyUIClient, "generate_image") as generate, patch.object(ComfyUIClient, "is_available") as available:
                frames = render_segment_frames(storyboard(), root, config)
            generate.assert_not_called()
            available.assert_not_called()
            with Image.open(frames[0]) as frame:
                self.assertEqual(frame.size, (1920, 1080))
                # 1000x600 art fits inside the 691-pixel-high art band.
                self.assertEqual(frame.getpixel((960, 125)), (255, 0, 0))
                self.assertEqual(frame.getpixel((960, 805)), (255, 0, 0))

    def test_modern_reviewed_asset_is_returned_for_animation_renderer(self):
        config = load_config(ROOT / "config.12gb.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.render.reviewed_assets_dir = directory
            source = root / "scene_01_segment_01.png"
            Image.new("RGB", (1000, 600), "red").save(source)
            with patch.object(ComfyUIClient, "generate_image") as generate, patch.object(ComfyUIClient, "is_available") as available:
                frames = render_segment_frames(storyboard(), root, config)
            generate.assert_not_called()
            available.assert_not_called()
            self.assertEqual(frames, [source])

    def test_generated_video_segment_uses_comfy_video_client(self):
        config = load_config(ROOT / "config.12gb.yaml")
        config.comfyui.video_enabled = True
        board = storyboard()
        segment = board.all_segments()[0][1]
        segment.shot = Shot(template="video", heading="Moving flower")
        with tempfile.TemporaryDirectory() as directory, patch.object(ComfyUIClient, "is_available", return_value=True), \
             patch.object(ComfyUIClient, "generate_video") as generate_video:
            generated = Path(directory) / "generated_videos" / "scene_01_segment_01.mp4"
            generated.parent.mkdir(parents=True)
            generated.write_bytes(b"video")
            generate_video.return_value = generated
            frames = render_segment_frames(board, Path(directory), config)
        self.assertEqual(frames, [generated])
        generate_video.assert_called_once()




    def test_docx_media_metadata_is_not_used_as_teaching_text(self):
        shot = _shot_from_media_type(
            "Animation with labels",
            "Local animation",
            "Step-by-step animation of the electroplating process, showing the setup, current flow, and metal deposition",
            "Educational animation of electroplating process, electron flow, cathode, anode, metal deposition",
            ["Electron Flow", "Cathode", "Anode"],
        )
        self.assertEqual(shot.template, "process")
        teaching_text = " ".join([shot.heading, *shot.steps]).lower()
        self.assertNotIn("animation with labels", teaching_text)
        self.assertNotIn("local animation", teaching_text)
        self.assertIn("electron flow", teaching_text)

    def test_chemistry_storyboard_does_not_auto_promote_generic_wan(self):
        config = load_config(ROOT / "config.12gb.yaml")
        config.comfyui.enabled = True
        config.comfyui.video_enabled = True
        config.comfyui.video_max_segments = 1
        segment = Segment(
            segment_number=1,
            narration="Kohlrausch law determines limiting molar conductivity of weak electrolytes.",
            visual="Diagram | Static image | Formula with ion contributions",
            image_prompt="Educational diagram of Kohlrausch law formula with ion contributions",
            keywords=["Kohlrausch law", "limiting molar conductivity"],
            shot=Shot(template="photo", heading="Kohlrausch law"),
        )
        board = Storyboard(title="Applications of Kohlrausch Law", source=segment.narration,
                           scenes=[Scene(scene_number=1, title="Conductivity", narration=segment.narration, segments=[segment])])
        promote_video_shots(board, config)
        self.assertEqual(board.all_segments()[0][1].shot.template, "photo")

    def test_electroplating_terms_do_not_trigger_bat_video_keyword(self):
        config = load_config(ROOT / "config.12gb.yaml")
        config.comfyui.enabled = True
        config.comfyui.video_enabled = True
        config.comfyui.video_max_segments = 1
        segment = Segment(
            segment_number=1,
            narration="The red wire is connected to the battery positive terminal.",
            visual="Animation with labels | A diagram showing the red and blue wires connected to the battery terminals.",
            image_prompt="Electroplating setup with battery, wires, switch, anode and cathode.",
            keywords=["battery", "red wire", "blue wire"],
            shot=Shot(template="photo", heading="Battery wiring"),
        )
        board = Storyboard(title="Electroplating", scenes=[Scene(scene_number=1, title="Setup", narration=segment.narration, segments=[segment])])
        promote_video_shots(board, config)
        self.assertEqual(board.all_segments()[0][1].shot.template, "photo")

    def test_word_storyboard_prefers_hd_photos_and_limits_wan_video(self):
        config = load_config(ROOT / "config.12gb.yaml")
        config.comfyui.enabled = True
        config.comfyui.video_enabled = True
        config.comfyui.real_image_auto_promote = True
        config.comfyui.video_auto_promote = True
        config.comfyui.video_max_segments = 1
        config.comfyui.video_workflow_path = "workflows/wan2_2_5b_video_api.json"
        bee = Segment(segment_number=1, narration="Entomophily uses bees to carry pollen between flowers.", visual="Bee visiting a flower", image_prompt="Bee carrying pollen between flowers")
        hydro = Segment(segment_number=2, narration="Hydrophily carries floating pollen across water to a receptive flower.", visual="Floating pollen on water", image_prompt="Floating pollen moving across water to aquatic flowers")
        proto = Segment(segment_number=3, narration="In protandry, anthers release pollen earlier and stigma becomes receptive later.", visual="Two step timing diagram", image_prompt="Protandry timing diagram")
        board = Storyboard(title="Pollination", scenes=[Scene(scene_number=1, title="Types", narration="Pollination types", segments=[bee, hydro, proto])])
        normalize_storyboard(board)
        promote_real_image_shots(board, config)
        promote_video_shots(board, config)
        templates = [segment.shot.template for _, segment in board.all_segments()]
        self.assertEqual(templates.count("video"), 1)
        self.assertEqual(templates.count("photo"), 1)
        self.assertEqual(board.all_segments()[2][1].shot.template, "protandry")
        self.assertEqual(board.all_segments()[1][1].shot.template, "photo")
        self.assertIn("high resolution", board.all_segments()[1][1].image_prompt.lower())
        self.assertIn("bee slowly visiting", board.all_segments()[0][1].image_prompt.lower())

    def test_impossible_narration_fails_instead_of_disappearing(self):
        config = AppConfig()
        segment = storyboard().all_segments()[0][1]
        segment.narration = "Narration " * 5000
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            Image.new("RGB", (100, 100), "green").save(source)
            with self.assertRaisesRegex(ValueError, "Split this narration"):
                render_ai_image_frame("Title", "Scene", segment, source,
                                      Path(directory) / "frame.png", config)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe not installed")
    def test_burned_subtitles_render_without_subtitle_stream(self):
        config = AppConfig()
        config.output_resolution.width = 640
        config.output_resolution.height = 360
        config.render.video_preset = "ultrafast"
        config.render.burn_captions = True
        config.render.embed_subtitles = True
        config.render.subtitle_fg = "#000000"
        config.render.subtitle_bg = "#FFFFFF"
        board = storyboard()
        first = board.all_segments()[0][1]
        first.shot = Shot(template="photo", heading="Real flower")
        first.end = 1.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(bytes(24000 * 2))
            frame = root / "frame.png"
            Image.new("RGB", (640, 360), "white").save(frame)
            subtitles = root / "subtitles.srt"
            subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\nPollen transfer\n", encoding="utf-8")
            output = render_video(board, [frame], audio, root, config, subtitles)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                                   capture_output=True, text=True, check=True)
            streams = json.loads(probe.stdout)["streams"]
            self.assertTrue(any(stream["codec_type"] == "video" for stream in streams))
            self.assertFalse(any(stream["codec_type"] == "subtitle" for stream in streams))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe not installed")
    def test_photo_shot_renders_full_video_size(self):
        config = AppConfig()
        config.output_resolution.width = 640
        config.output_resolution.height = 360
        config.render.video_preset = "ultrafast"
        board = storyboard()
        first = board.all_segments()[0][1]
        first.shot = Shot(template="photo", heading="Real flower")
        first.end = 1.0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(bytes(24000 * 2))
            frame = root / "portrait.png"
            Image.new("RGB", (300, 600), "blue").save(frame)
            output = render_video(board, [frame], audio, root, config, None)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                                   capture_output=True, text=True, check=True)
            video = next(stream for stream in json.loads(probe.stdout)["streams"] if stream["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"]), (640, 360))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe not installed")
    def test_ffmpeg_mux_with_spaces_and_fractional_segment_timing(self):
        config = AppConfig()
        config.output_resolution.width = 640
        config.output_resolution.height = 360
        config.render.video_preset = "ultrafast"
        board = storyboard()
        first = board.all_segments()[0][1]
        first.end = 0.517
        second = first.model_copy(update={"segment_number": 2, "start": 0.517, "end": 1.034})
        board.scenes[0].segments.append(second)
        with tempfile.TemporaryDirectory(prefix="edu quality ") as directory:
            root = Path(directory)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(24000)
                handle.writeframes(b"\0\0" * 24816)
            frame = root / "frame.png"
            Image.new("RGB", (640, 360), "green").save(frame)
            subtitles = root / "subtitles.srt"
            subtitles.write_text("1\n00:00:00,000 --> 00:00:01,000\nPollen transfer\n", encoding="utf-8")
            output = render_video(board, [frame, frame], audio, root, config, subtitles)
            probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output)],
                                   capture_output=True, text=True, check=True)
            streams = json.loads(probe.stdout)["streams"]
            video = next(stream for stream in streams if stream["codec_type"] == "video")
            self.assertEqual((video["width"], video["height"]), (640, 360))
            self.assertEqual(int(video["nb_frames"]), round(1.034 * 30))
            self.assertTrue(any(stream["codec_type"] == "audio" for stream in streams))
            self.assertTrue(any(stream["codec_type"] == "subtitle" for stream in streams))


if __name__ == "__main__":
    unittest.main()
