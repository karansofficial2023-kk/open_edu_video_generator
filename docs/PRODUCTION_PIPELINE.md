# Production pipeline (contract-driven storyboards)

Implements `Open_Edu_Video_Generator_Production_Specification.pdf`. All tooling is open source and local:
FFmpeg, Pillow, OpenCV, matplotlib (mathtext), ComfyUI + FLUX.1 Schnell FP8, Ollama (Qwen text + Qwen-VL), Edge TTS.

```powershell
# Transcribe writes <name>_contract.json (preferred) and <name>_storyboard.docx; both carry the same fields.
python -m app.main --config config.12gb.yaml --input "path\to\lesson_contract.json" --output outputs\lesson
python -m app.main --config config.12gb.yaml --input "path\to\folder" --output outputs\batch      # queue, isolated per lesson
python -m app.main --config config.12gb.yaml --input lesson_contract.json --output outputs\x --preflight-only
python -m app.main --config config.12gb.yaml --input lesson_contract.json --output outputs\x --preview    # silent draft
```

## Stages and gates

| Stage | Module | Gate / output |
|---|---|---|
| Preflight | `preflight.py` | `preflight.json` - `{shot_id, field, severity, renderer, reason, repair}`; ComfyUI/Ollama/FFmpeg/font health before any voice or GPU work |
| Route | `router.py` | renderer chosen from `visual_type`, `formula_lines`, `steps`, `columns` only (no topic keywords) |
| Stills | `assets.py`, `commons.py` | Commons retrieval (license allow-list) then FLUX candidates; technical QA (blank/blur/size) + Qwen-VL semantic QA; corrected-prompt retry rounds; `assets/shot_*.json` keeps every attempt, seed and score; `licenses.json` |
| Labels | `labels.py` | Qwen-VL proposes a box, an independent crop check verifies it, low confidence is omitted and reported (never an approximate arrow); collision-free layout (>= 28 px text); leader line grows first, then the label fades in |
| Deterministic art | `formulas.py`, `charts.py`, `animation_renderer.py` | formulas (mathtext, validated), bar charts (`columns` = `Label: value`, needs `Source:` in `review_notes`), 2-3 panel comparisons (`columns` = `Caption | image prompt`), process cards (only for real ordered steps) |
| Compose | `production_renderer.py`, `typography.py`, `captions.py` | Ken Burns (disabled when labels need fixed anchors), title cards, crossfade, subtitle band (<= 2 lines, measured wrap, safe margins), optional logo |
| Voice | `tts_edge.py` | cached by narration + voice + rate; retries with backoff |
| Assemble | `production.py` | reading holds for labels/formulas, loudness-normalised AAC, `final.mp4` only when nothing is unresolved (else `preview.mp4`) |
| Final QA | `qa.py` | `final_qa.json`: container, A/V drift, blank/repeated frames, subtitle bounds, clipping, metadata leak, Qwen-VL frame-vs-narration review |

`production_issues.json` lists unresolved shots; approved assets and clips are cached, so a rerun only redoes what failed.

## Memory policy (RTX 3060 12 GB)
Only one heavy model is resident: image generation and vision review run in separate phases, Ollama models are unloaded
(`keep_alive: 0`) and ComfyUI is asked to free memory between them.

## Configuration
See the `production:` block in `config.12gb.yaml` (`logo_path`, `retrieval_enabled`, `subtitle.*`, `label_font_px`, ...).
`render.font_path` overrides fonts; Indic/Arabic-script lines automatically use Nirmala UI / Noto.

## Subject coverage (what each discipline renders deterministically)
| Subject | Visual language |
|---|---|
| Biology | FLUX/Commons photographs with QA, verified labels (leader line then label), split-screen comparisons, concept cards for processes |
| Chemistry | typeset reactions with upright species, ions, charges and states (`Cu^{2+}(aq) + 2e^- -> Cu(s)`), data bar charts, labelled apparatus photographs |
| Physics | typeset laws and worked substitutions, series/parallel **circuit schematics** (schemdraw), bar charts and function plots |
| Mathematics | derivations one transformation per shot (`mathparse` -> LaTeX), **symbolic step check** (`derivation.py`), animated function plots with computed roots |
| Indic languages | Qt/HarfBuzz shaping (Pillow has no Raqm here), script-aware wrapping, Edge Indic voices from the language package, native-script titles/cards/labels |

## Gates added on top of the stages above
- `tools/storyboard_scorecard.py` / `tools/run_golden_suite.py`: storyboard score (renderer readiness, completeness, source coverage/support, subject routing,
  language integrity, derivation consistency, variety, pacing) and release gate over a folder of lessons; `--render` adds final-QA results.
- `derivation.py`: consecutive equations must be algebraic rewrites of each other; a wrong intermediate step is reported once as a warning for SME review.
- Concept-card fallback: a photograph that cannot be verified after all attempts becomes deterministic key-idea cards (warning), a title card falls back
  to a gradient background; nothing unverified reaches a `final.mp4`.
- `release/review.html`, `manifest.json` (SHA-256), `sign_off.md`: subject-matter review sheet with thumbnails, flags and provenance.
- `app/gpu_lease.py`: one heavy GPU job at a time across Transcribe and the generator.
- Optional: `production.motion_enabled` (LTX clips with flicker/semantic QA, still fallback), `production.music_path` (ducked background music),
  `production.retrieval_enabled` (Wikimedia Commons first, license allow-list).
