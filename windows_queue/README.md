# Video dubbing queue

Put source videos under `D:\Python\video_dubbing` (subfolders are scanned too). Open a Command Prompt in that folder and run one launcher:

| Command | What it does |
| --- | --- |
| `RUN_FIX_AND_RERUN.bat` | Resume the replacement-audio queue. Reuse existing translated contracts, run QA corrections before needed TTS, skip completed replacement dubs, and retry unfinished videos. Missing contracts are translated first. |
| `RUN_SECOND_AUDIO_TRACK.bat` | Resume in dual-track mode. English remains the default and Hindi is selectable; completed dual-track dubs are skipped and unfinished videos are retried. Existing replacement-mode videos are converted to dual-track outputs. |
| `TEACHER_RERUN_BY_ID.bat V000001` | Rerun just one video selected by its persistent ID. It reuses that video's translation contract so a teacher's contract edit is not overwritten. |

First run creates `video_ids.csv`. Each video receives a persistent ID such as `V000001`; use that ID for teacher reruns. IDs identify videos in logs and commands; source files are not renamed. Each run prints live Transcribe/generator output plus a status line every ten seconds with total, QA-passed completed, teacher-review, failed, pending, and no-audio counts. No-audio items are listed separately and are not counted as failures.

To assign/list IDs without starting production, run `py -3 video_dubbing_queue.py --mode list`.

Outputs and logs:

- Videos: `D:\Python\video_dubbing_output\dubbed\<video name>\`
- Live run logs: `D:\Python\video_dubbing_output\logs\`
- Queue CSV: `D:\Python\video_dubbing_output\reports\queue_results.csv`
- Teacher rerun result: `D:\Python\video_dubbing_output\reports\teacher_<ID>_last_result.csv`
- No-audio list: `D:\Python\video_dubbing_output\reports\no_audio.csv`
- Input audio inventory: `D:\Python\video_dubbing_output\reports\inventory.csv`

`completed_QA_ok` means dubbing finished and there are no translation QA flags. A video with unresolved translation or voice-review flags is counted under `needs_teacher_review`, not as a passed completion. Actual processing errors are counted under failed.

## Teacher correction workflow

1. Find the video ID and translation contract path in `video_ids.csv` and `D:\Python\video_dubbing_output\transcribe\`.
2. Edit the Hindi contract narration for that video, saving the corrected file in place.
3. Run `TEACHER_RERUN_BY_ID.bat V000001` with that video's ID. The generator runs its formula/number QA before TTS and regenerates only that video's dub.
4. Review `translation_qa_report.json` and the new output. If QA still reports an issue, correct the contract and rerun the same ID.

The teacher-only command preserves the last whole-folder `queue_results.csv` and saves its one-video result separately under the teacher rerun filename above.

The automated QA can safely restore formulas and numbers when the source makes the intended correction unambiguous. It reports uncertain semantic issues for teacher review; it does not independently prove the full scientific meaning of a translation.

To run from PowerShell instead of opening a batch file:

```powershell
Set-Location -LiteralPath 'D:\Python\video_dubbing'
& '.\RUN_FIX_AND_RERUN.bat'
& '.\RUN_SECOND_AUDIO_TRACK.bat'
& '.\TEACHER_RERUN_BY_ID.bat' V000001
```


## Restart after interruption

After network or power interruption, run the same mode launcher again. Completed videos with a matching audio mode are skipped; valid translation contracts are reused; failed or unfinished videos are retried. If power fails while one video is in TTS/mux, that current video starts again from its dubbing step because per-sentence Piper work is not checkpointed. Switching from replacement audio to dual-track mode intentionally rebuilds videos that do not yet have both tracks. The teacher-ID launcher intentionally reruns that one video even if a completed dub exists.
