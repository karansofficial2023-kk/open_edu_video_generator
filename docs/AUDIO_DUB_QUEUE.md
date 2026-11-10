# Audio detection, Hindi translation, and dubbing queue

The queue bridge is `run_audio_dub_queue.ps1`. It recursively inventories supported video files, writes videos with no audio stream to a separate CSV, and excludes them from translation. Videos with an audio stream are passed together to one Transcribe queue process. The generator then replaces the narration using timed Hindi contracts, preserves any original audio before the first speech cue, and copies the original video stream.

For visible Windows launchers with persistent per-video IDs, periodic totals, logs, resume behavior, and teacher reruns, see [`windows_queue/README.md`](../windows_queue/README.md). Copy the complete `windows_queue` folder to `D:\Python\video_dubbing` to use its batch files there.

## Run a folder queue

Put the source videos in one input folder, then run:

```powershell
Set-Location -LiteralPath "D:\Python\claude version\open_edu_video_generator"

& ".\run_audio_dub_queue.ps1" `
  -InputPath "D:\Videos\ToDub" `
  -OutputRoot ".\outputs\hindi_audio_queue" `
  -TargetLanguage "hi" `
  -GradeLevel "10th standard" `
  -AcademicLevel "secondary"
```

Use the same output root to resume a queue. Completed contracts and dubs are reused. Pass `-RerunTranscribe` to recreate contracts for all audio-bearing inputs. After manually correcting one translated contract, use `-RerunDubbing` for the queue or the ID-based teacher launcher described in `windows_queue/README.md`. Duplicate filenames with the same stem are listed in `reports\duplicate_names.csv` and skipped because Transcribe writes contracts by base filename.

## Reports and output

- `reports\inventory.csv`: one row per detected video.
- `reports\no_audio.csv`: videos that have no audio stream.
- `reports\probe_failures.csv`: files FFprobe could not inspect.
- `reports\duplicate_names.csv`: basename collisions excluded from processing.
- `reports\queue_results.csv`: queued, transcribing, dubbing, completed, teacher-review-needed, and failed item status.
- `transcribe\`: translation contracts and translation review reports.
- `dubbed\<video-name>\`: Hindi dubbed video and synchronization/voice manifest.

Audio presence means a media audio stream exists; Transcribe can still report that a video has no usable speech (for example, music-only audio). Translation review JSON files are retained for teacher review. Before TTS, the generator compares formula and numeric tokens to each source line, safely repairs unambiguous mismatches, and rechecks the corrected text. It records any unresolved issue in `translation_qa_report.json` as `needs_teacher_review` and continues dubbing so the queue is not blocked. Items with translation or voice-review flags finish as `completed_needs_teacher_review`; they are not marked release-ready. This is mechanical QA and cannot guarantee the full semantic accuracy of every translation.

## Voice matching

The dubbing code estimates the perceived voice category from the source audio's pitch. It does not identify a speaker's gender identity or inspect a person's appearance in the video. The Hindi male and female Piper voices are selected when the estimate is clear. If pitch is mixed or ambiguous, the queue uses the configured female default. The manifest records the estimate, confidence, and model path.

For a manual per-video override, place a sidecar beside the source video using the same base name and `.voice.json` extension. For `Lesson.mp4`, create `Lesson.voice.json` with:

```json
{"gender":"male"}
```

Use `"female"` for the female voice. A sidecar override takes precedence over pitch estimation.
