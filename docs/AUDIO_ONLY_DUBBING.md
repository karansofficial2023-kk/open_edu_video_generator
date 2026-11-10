# Audio-only dubbing

This mode takes a translation-only Transcribe contract and the original video. Transcribe translates each recognized ASR segment without paraphrasing or planning visuals, and carries the segment's source timestamps into the contract. The generator creates localized TTS, fits speech to source timing groups, and muxes the audio while stream-copying the original video. When a short cue cannot fit its translation, it combines neighboring utterances into a larger group and aligns the next group to its source cue. It does not generate or alter video frames and does not create subtitles. Default mode replaces the original audio; `--keep-original-audio` keeps English as the default and adds Hindi as a selectable second track.

## Prepare the translated contract

Run Transcribe with `-AudioTranslationOnly -TargetLanguages "hi"`. For another supported target, replace `hi` with its language code. Transcribe writes a localized file named `<source>_hi_contract.json` in the output folder.

## Run audio replacement

From PowerShell:

```powershell
Set-Location -LiteralPath "D:\Python\claude version\open_edu_video_generator"
& ".\run_video.ps1" --config ".\config.realistic.12gb.yaml" --input "D:\path\to\<source>_hi_contract.json" --output ".\outputs\dub_hi" --replace-audio --source-video "D:\path\to\original_video.mp4"
```

The result is `<original-name>_dub_hi.mp4` in the output folder. `audio_replacement_manifest.json` records the target language, codecs, duration, timing groups, and speech speed changes. The video codec is stream-copied; the original soundtrack is not retained. If even a combined timing group would need more than 1.50x speed, the job stops rather than dropping speech or shifting all later cues. Re-running after a fit-limit failure reuses completed TTS audio from the local cache. This aligns narration to ASR segment start times at group boundaries; it does not perform speaker lip-sync.

To retain English and add Hindi as a selectable second audio stream, add `--keep-original-audio` to `run_video.ps1`. English stays selected by default; Hindi is labeled Hindi in players with audio-track selection. The output is `<original-name>_dub_hi_dual.mp4`. For the folder queue, add `-KeepOriginalAudio` to `run_audio_dub_queue.ps1`.

## Restarting an interrupted folder queue

Run the same queue command again **without** `-RerunTranscribe` or `-RerunDubbing`. Valid translation contracts are reused, completed dubs are skipped when their audio mode matches, and videos without a valid contract or completed dub are retried. If switching from replacement mode to dual-audio mode, add `-KeepOriginalAudio`; completed replacement-mode dubs are regenerated in dual-audio mode. Recovery is per video, not per sentence: if power fails during a video's TTS/mux step, that video's dubbing starts again, while already completed videos are skipped. The current Piper path does not checkpoint individual synthesized sentences. The queue exits nonzero if any video fails so a failed run is visible to a calling shell or scheduler.

The queue uses the local translation model, so internet loss does not normally stop Hindi translation. Ollama-dependent optional punctuation may be skipped while Ollama is unreachable. Keep the output folder intact between restarts; contracts, manifests, and completed MP4s there are the recovery checkpoints.

Before TTS, the generator checks every translated line against its stored source narration for chemical formulas and numeric
values. It corrects source formula identifiers that were spelled phonetically in Hindi (for example, `B2` written as “बी 2”),
and restores formula and number tokens when source and translation have matching token counts and order. If a complete spoken
arithmetic equation has a number-count mismatch, the generator rebuilds that line from a Hindi equation template only when the
source equation is mathematically consistent. It then runs the checks again. Corrections and remaining issues are recorded in
`translation_qa_report.json`; corrected narration is passed to TTS.

If the source equation itself is inconsistent, a source line is missing, or a mismatch cannot be corrected safely, dubbing still
continues to honor the queue's retry-and-review workflow. The report is marked `needs_teacher_review`, and the queue records
`completed_needs_teacher_review`. Review that report and the corresponding source video before approving the dub. These checks
catch protected formula and number errors; they cannot prove that every translated sentence preserves its full scientific
meaning or infer the correct value from video frames.

Speaker profiling uses Silero voice activity detection on audio after the opening 20 seconds, so intro music does not set the
voice pitch estimate. The detected speech onset is also used to preserve the original soundtrack through the intro. If acoustic
classification is still uncertain, add a source-adjacent `<video-name>.voice.json` containing `{"gender":"male"}` or
`{"gender":"female"}`; the queue will use that reviewed override and record it in the manifest.

Transcribe also supports a Hindi terminology glossary. Starter biology terms include `bacteria` → `जीवाणु` and `virus` →
`विषाणु`, and `red cabbage` → `लाल गोभी`; chemistry includes `vinegar` → `सिरका`. Common wrong variants are normalized only
when the English source line contains the corresponding term. Hindi TTS also speaks `NaOH` as `सोडियम हाइड्रॉक्साइड`.
Add reviewed terms under
`transcribe/terminology/hi/` or make per-lesson overrides in the Transcribe storyboard materials folder; see
`transcribe/terminology/README.md`.

The workflow needs FFmpeg/FFprobe and a configured TTS voice for the target language. ComfyUI and image/video generation are not used.
