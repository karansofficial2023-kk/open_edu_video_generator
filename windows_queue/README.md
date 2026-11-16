# Video dubbing queue (V2 deep dive)

Every launcher in this folder now runs the **v2 deep-dive pipeline**: formulas, equations, bonds and labels are spoken by an English
voice; the translated contract is checked and repaired before dubbing; ASR uses the STEM vocabulary prompt; formulas are masked during
translation. Each launcher first runs `_v2_preflight.bat` (English voices present + tests pass) and stops if that fails.
Output now goes to `D:\Python\video_dubbing_output_v2` (set `DUB_OUTPUT_ROOT` to change it).

| Launcher | Use it for |
| --- | --- |
| `RUN_FIX_AND_RERUN.bat` | **New videos / resume.** Videos in `D:\Python\video_dubbing` that have no finished v2 dub are processed fresh; finished v2 dubs are skipped. |
| `RUN_SECOND_AUDIO_TRACK.bat` | Same, but English stays the default track and Hindi is added as a selectable second track. |
| `TEACHER_RERUN_BY_ID.bat V000001` | Apply reviewed Hindi corrections for one video, then re-dub with all v2 checks. |
| `RUN_HINDI_DUB_V2_DEEP_DIVE.bat` | Direct runner with modes: none = 3 test videos fresh, `NEW`, `ALL`, `QUICK` (reuse old contracts), `KEEP`. |

**v2.1 (permanent fixes from the two-video review).** Both `RUN_FIX_AND_RERUN.bat` and `RUN_SECOND_AUDIO_TRACK.bat` use the same code and differ only in the audio-track mode. Whisper decodes without conditioning on previous text (no phantom words, punctuation kept), rescues the closing words, and contracts record `asr_pipeline: v2.1`; the queue rebuilds contracts from an older recipe and dubs below QA version 21. Ambiguous speaker pitch (185-190 Hz) chooses the nearer voice at confidence 0.5 and is listed as `voice_selection_unverified`; a `<video>.voice.json` sidecar always wins.

The text below is the original queue documentation; paths that mention `video_dubbing_output` now refer to `video_dubbing_output_v2`.


Put source videos under `D:\Python\video_dubbing` (subfolders are scanned too). Open a Command Prompt in that folder and run one launcher:

| Command | What it does |
| --- | --- |
| `RUN_FIX_AND_RERUN.bat` | Resume the replacement-audio queue. Reuse existing translated contracts, run QA corrections before needed TTS, skip completed replacement dubs, and retry unfinished videos. Missing contracts are translated first. |
| `RUN_SECOND_AUDIO_TRACK.bat` | Resume in dual-track mode. English remains the default and Hindi is selectable; completed dual-track dubs are skipped and unfinished videos are retried. Existing replacement-mode videos are converted to dual-track outputs. |
| `TEACHER_RERUN_BY_ID.bat V000001` | Apply that video's reviewed Hindi corrections by shot ID, then rerun dubbing and QA. It preserves the English source and timing. |

First run creates `video_ids.csv`. Each video receives a persistent ID such as `V000001`; use that ID for teacher reruns. IDs identify videos in logs and commands. After a queue run, generated MP4 filenames are prefixed with the ID (for example, V000001_AC Motor_dub_hi_dual.mp4); older existing outputs are prefixed the next time a launcher scans them. Source videos are not renamed. Each run prints live Transcribe/generator output plus a status line every ten seconds with total, QA-passed completed, teacher-review, failed, pending, and no-audio counts. No-audio items are listed separately and are not counted as failures.

To assign/list IDs without starting production, run `py -3 video_dubbing_queue.py --mode list`.

Outputs and logs:

- Videos: `D:\Python\video_dubbing_output\dubbed\<video name>\`
- Live run logs: `D:\Python\video_dubbing_output\logs\`
- Queue CSV: `D:\Python\video_dubbing_output\reports\queue_results.csv`
- Teacher rerun result: `D:\Python\video_dubbing_output\reports\teacher_<ID>_last_result.csv`
- No-audio list: `D:\Python\video_dubbing_output\reports\no_audio.csv`
- Input audio inventory: `D:\Python\video_dubbing_output\reports\inventory.csv`

`completed_QA_ok` means dubbing finished and there are no translation QA flags. A video with unresolved translation or voice-review flags is counted under `needs_teacher_review`, not as a passed completion. Actual processing errors are counted under failed.

The audio contract keeps English source text, Hindi narration, and source timing for each grouped sentence. Transcribe checks uncertain ASR words, runs MADLAD followed by a separate Qwen candidate/audit for flagged lines, and audits all translated sentence groups. Semantic failures receive one more Hindi candidate and a separate audit pass. The generator checks numbers, formulas, operators, known terminology, and Transcribe review findings before TTS; after TTS it transcribes the generated Hindi speech and compares it with approved text. The final QA report includes `qa_version`, `contract_sha256`, and `post_tts_speech_check`. A resume run rebuilds a dub when its QA version or contract hash is outdated. Review copies continue through dubbing, but unresolved findings never count as `completed_QA_ok`. Hindi `एक` can be a grammatical article, so numeric QA caps its count at the source count; scientific identity and meaning still require the bilingual audit and teacher review when flagged.

## Teacher correction workflow

1. Find the video ID in `video_ids.csv`, then find the shot IDs and timing in `D:\Python\video_dubbing_output\transcribe\<video name>_hi_contract.json`.
2. Save reviewed Hindi lines to `D:\Python\video_dubbing\teacher_corrections\V000001.json`:

```json
{"corrections":[{"shot_id":"19.1","narration":"Corrected Hindi sentence here."}]}
```

If the report flags `source_asr`, first listen to the original audio at that shot's `source_start`/`source_end`. Include the verified English text and its matching Hindi correction together:

```json
{"corrections":[{"shot_id":"19.1","source_narration":"Verified English sentence.","narration":"Matching corrected Hindi sentence."}]}
```

Changing only Hindi does not clear a `source_asr` finding. The teacher tool will not mark an uncertain source transcription as reviewed until a verified English correction is supplied.

3. Run `TEACHER_RERUN_BY_ID.bat V000001`. It validates all correction IDs before changing the contract, then runs source-aligned QA before TTS and regenerates only that dub.
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


The ID is a filename prefix for identification; it is not burned into the visible video frames.
