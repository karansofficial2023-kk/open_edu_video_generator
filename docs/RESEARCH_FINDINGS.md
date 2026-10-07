# Research findings (Oct 2026): ideas adopted from open-source projects

Sources were read by research agents from READMEs, papers and model cards; source code and issues were mostly not read. Treat as leads to validate, not as measured results.

## Constraint conflict to resolve
- **Edge TTS is not open source** (free client for a Microsoft cloud service). Requirement is open-source only.
  Candidates (Apache-2.0): Indic Parler-TTS (21 Indic languages, 0.9B), Kokoro (82M; English + Hindi). Audition on real lessons before switching.

## Ranked ideas
| # | Idea | Source | Status |
|---|------|--------|--------|
| 1 | Reviewer checklist: per-component yes/no (DSG/TIFA style), reject on a missed required item | DSG, VQAScore | planned |
| 2 | Structured scenes (circuits, apparatus, flows, cycles) always use deterministic drawers, never FLUX | schemdraw, Graphviz, RDKit | partly done (circuit/graph/formula/cards) |
| 3 | Stray-text rejection with OCR on every still | PaddleOCR (Apache-2.0) | planned |
| 4 | Object presence + label anchors from open-set detection | GroundingDINO (Apache-2.0) | planned |
| 5 | Retrieval first with licence + attribution (Commons done; add Openverse) | Openverse API | partly done |
| 6 | Molecules from validated SMILES | RDKit (BSD) | planned |
| 7 | Glossary hotwords for ASR before paraphrase | faster-whisper `hotwords` | **done** |
| 8 | Quiz-based regression: questions from source, answered by VLM from frames | TeachQuiz / PresentQuiz | planned |
| 9 | Named layout grid + occupancy check for deterministic renderers | Code2Video (6x6 anchors) | planned |
| 10 | Escalating repair: field, then shot, then section | Code2Video ScopeRefine | planned |
| 11 | Plan-only dry run + cheap text gates before any GPU work | TheoremExplainAgent `--only_plan` | partly (preflight) |
| 12 | Short reasoning field before structured fields | LVD, "Let Me Speak Freely?" | planned (storyboard) |
| 13 | Pronunciation overrides for formulas/terms in the contract | deck2video | planned |

## Speed / memory (RTX 3060 12 GB, 31 GB RAM)
- Installed ComfyUI 0.35.0 / PyTorch 2.13 cu130, no restrictive flags: Dynamic VRAM should already be on; the 17 GB all-in-one FLUX checkpoint is still the RAM hog (ComfyUI issue #4239).
- **Measured fix: launch ComfyUI with `--disable-smart-memory`.** Peak RAM while generating stays about 13.5 GB (was ~18 GB and never released); it drops back to ~2 GB after each image, and ~20 GB of system RAM stays free between images. Generation time is unchanged (15-20 s per 1344x768 still).
- Splitting the all-in-one checkpoint (done locally with `workflows/flux_schnell_split_txt2img_api.json`, no download) gave the same 13.5 GB peak as the original workflow, so the split is not needed; the flag is what helps.
- `POST /free {"unload_models":true,"free_memory":true}` works (verified here: 18.6 GB -> 2 GB). The generator calls it after each image round and when a run ends.
- Fallback flags if RAM is still high: `--disable-smart-memory --cache-lru 2 --reserve-vram 1.5`.
- Ollama: `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_MAX_LOADED_MODELS=1`.

## Not suitable
Remotion (source-available licence), Wan2.2 5B (24 GB), OmniSVG (17-26 GB, non-commercial dataset), FLUX Kontext dev (non-commercial), Paper2Video talking head (48 GB), stock-footage tools (MoneyPrinterTurbo, ShortGPT) for educational accuracy.
