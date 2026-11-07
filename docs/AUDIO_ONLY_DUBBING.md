# Audio-only dubbing

This mode takes a translated Transcribe contract and the original video. It creates localized TTS, fits each narration segment into the contract's shot-duration slot, and muxes the new audio while stream-copying the original video. It does not generate or alter video frames and does not create subtitles. The original audio track is replaced entirely.

## Prepare the translated contract

In Transcribe, set `storyboard.target.languages=hi` in `application.properties`, then run the normal production transcription for the source video. For another supported target, replace `hi` with its language code. Transcribe writes a localized file named `<source>_hi_contract.json` beside the storyboard outputs.

## Run audio replacement

From PowerShell:

```powershell
Set-Location -LiteralPath "D:\Python\claude version\open_edu_video_generator"
& ".\run_video.ps1" --config ".\config.realistic.12gb.yaml" --input "D:\path\to\<source>_hi_contract.json" --output ".\outputs\dub_hi" --replace-audio --source-video "D:\path\to\original_video.mp4"
```

The result is `<original-name>_dub_hi.mp4` in the output folder. `audio_replacement_manifest.json` records the target language, codecs, duration, and any per-shot speech speed changes. The video codec is stream-copied; the original soundtrack is not retained. If a translated line would need to be sped up beyond 1.35x to fit its slot, the job stops with a message rather than producing clipped or unnaturally fast speech. This aligns narration to the storyboard's shot timing and the full video duration; it does not perform speaker lip-sync.

The workflow needs FFmpeg/FFprobe and a configured TTS voice for the target language. ComfyUI and image/video generation are not used.