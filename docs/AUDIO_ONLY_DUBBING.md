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
when the English source line contains the corresponding term. Add reviewed terms under
`transcribe/terminology/hi/` or make per-lesson overrides in the Transcribe storyboard materials folder; see
`transcribe/terminology/README.md`.

The workflow needs FFmpeg/FFprobe and a configured TTS voice for the target language. ComfyUI and image/video generation are not used.

## Dubbing v2 (deep dive)

Launcher: `windows_queue\RUN_HINDI_DUB_V2_DEEP_DIVE.bat` (`ALL` = every video, `KEEP` = keep the English track as a second audio track).

**Strict formula rule.** Chemical formulas, chemical equations, atom/point labels (`AB`, `B1`) and acronyms (`AC`) are never read
in Hindi. `app/spoken_forms.py` splits each narration into Hindi and English spans: `H2SO4` is spoken by an English Piper voice as
"H two S O four", `2NaOH + H2SO4 -> Na2SO4 + 2H2O` as one English equation, and everything else (digits, percentages, units)
by the Hindi voice with numbers expanded into Hindi words. Configure the English voices with `piper_english_model` /
`piper_english_gender_models` (or `PIPER_EN_MODEL`); without one the dub fails loudly instead of silently reading a formula in Hindi.
Hindi pronunciation maps can no longer replace a formula. The QA report lists every English span under `strict_formula_rule`.

**Contract precheck.** Before any TTS, `app/contract_precheck.py` (also `python scripts/precheck_contract.py CONTRACT.json`)
removes foreign-script ASR debris, repairs known science mistranslations from the source term (`app/data/hi_dub_glossary.json`),
restores Devanagari-spelled formulas to Latin, speaks bare numbering cues ("2.") as Hindi ordinals, and repairs cue timing. Repairs are
written to `<name>_contract.precheck.json`; the original is kept as `<name>_contract.json.v1.bak`. Low-risk review notes (ASR
doubt only on function words, unverified language-model concerns) become `advisory_notes` in the QA report instead of teacher-review
flags; anything touching a formula, number or term stays a finding.

**QA.** Post-TTS checking uses Hindi-only speech coverage per cue (threshold 0.55) and over the dubbed span (0.70), excluding the
untouched English intro. QA version is 20; the queue rebuilds any older output.


**v2 additions from the STEM prompt review.** Bonds, groups and suffixes are spoken as chemistry (`C=O` = "C double bond O",
`C≡N` = "C triple bond N", `-COOH` = "dash C O O H", `-one` = "suffix own"; physics formulas such as `F=BIL` stay equations).
Transcribe's MADLAD worker masks formulas, bonds and diagram labels as `[1]`, `[2]` before translating and restores them (a line that
loses a placeholder is translated again unmasked). Whisper gets a STEM vocabulary prompt (`WHISPER_STEM_VOCAB=0` disables it). The Qwen
repair prompt carries the NCERT Hindi glossary. QA flags a `-one` suffix that was replaced by `C=O`
(`iupac_suffix_replaced_by_group`), and an audit claiming the source should say `C=O` is recorded as advisory.
Launcher modes: default = fresh ASR + translation, `QUICK` = reuse old contracts, `ALL`, `KEEP`.

**Two-voice mismatch (v2.1).** The English voice for formulas is a second speaker, so it is matched to the Hindi narrator: the male
English voice is `en_US-hfc_male-medium` (measured closest to the Hindi `pratham` voice: spectral distance 16 against 27 for the old
`en_GB-alan`, pitch gap 3 Hz against 18 Hz); the female one stays `en_US-lessac-medium` (the other candidates do not run on the
installed Piper build). Each English span is also pitch-matched (cached per voice pair), level-matched to the Hindi speech of the same
line, slowed slightly (`piper_english_length_scale: 1.08`) and faded at the joins. `piper_english_match_voice: false` turns the
matching off.

**Narrator gender.** The detected gender picks BOTH the Hindi voice and the English formula voice. It is decided by the original
approach: the source speech after the music intro (voice-activity filtered) is listened to and its pitch gives male or female
(male at or below 185 Hz, female at or above 190 Hz; the narrow band between is resolved to the nearer side at confidence 0.5 and
reported as `voice_selection_unverified`). This was right on every lesson tested (Acid-base, Bleaching Powder, Allotropes,
Carbonyl). Overrides win: `<video>.voice.json`, then `default.voice.json` in the video's folder (one narrator for a series).
A missing male/female voice in the config is reported as `voice_gender_mismatch`.
