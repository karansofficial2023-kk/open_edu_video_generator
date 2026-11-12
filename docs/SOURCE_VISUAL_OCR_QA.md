# Source video formula and arithmetic review

The queue checks original video slides after Transcribe writes the timed English/Hindi contract and before TTS. It samples three frames within each timed science or arithmetic cue. Cues without science or math words receive one screening frame; if that frame contains a printed formula or calculation, it is reused as the middle of a three-frame check. RapidOCR reads printed text. Exact formula tokens and simple numeric arithmetic are checked with deterministic code. Qwen3-VL is consulted only when OCR suggests a conflict. The text-only Qwen translation review in Transcribe continues to run separately.

The contract remains unchanged. One OCR evidence folder is stored beside it as `<lesson>_<language>_source_qa/visual_source_qa/`. The generator validates the source file, contract, review, OCR version, and model against that report and reuses it for dubbing. A changed input causes a new scan at the same path. If Qwen or OCR was temporarily unavailable, the current dubbing run keeps that finding without scanning twice; the next queue launch retries it. The dubbing QA report contains the evidence path and summary, not a second copy of OCR lines or frames. Interrupted scans do not create a valid cache entry.

The visual stage **does not rewrite science facts from a frame**. Animated labels can change, OCR can confuse similar symbols, and a slide can contain a related equation that is not being spoken. A conflict is reported only with repeated OCR evidence and, for source-to-slide disagreements, a vision-model crosscheck. Review the saved frames and source audio before correcting the English contract or Hindi translation. `translation_qa_report.json` links to `visual_source_qa/report.json`; affected frames live beside that report.

## Installation on this Windows setup

The currently used portable Python has RapidOCR installed in `.runtime-deps`. For a fresh installation, from the generator directory in PowerShell:

```powershell
& 'D:\AI\ComfyUI_windows_portable\python_embeded\python.exe' -m pip install --target '.runtime-deps' 'rapidocr-onnxruntime==1.2.3'
ollama list
```

The OCR dependency is also pinned in `requirements.txt` for a working project virtual environment. Keep the local `qwen3-vl:8b-instruct` model available in Ollama. The queue invokes this stage automatically when `production.source_visual_ocr` is true (the default).

## Run an OCR-only check without TTS

```powershell
& 'D:\AI\ComfyUI_windows_portable\python_embeded\python.exe' 'D:\Python\claude version\open_edu_video_generator\scripts\audit_source_video.py' `
  --video 'D:\Python\video_dubbing\AC Motor.mp4' `
  --contract 'D:\Python\video_dubbing_output_qwen_improved\transcribe\AC Motor_hi_contract.json' `
  --config 'D:\Python\claude version\open_edu_video_generator\config.realistic.12gb.yaml'
```

Current arithmetic coverage is numeric `+`, `-`, `×`, and `÷` with an equals sign, including parentheses around the left expression. It does not parse arbitrary algebra, chemical reaction balancing, handwritten derivations, or tables whose numbers are split across OCR boxes. For those cases, preserve source-frame evidence and use a formula-specific recognizer such as PaddleOCR PP-FormulaNet after benchmarking it on this curriculum and video resolution.

The one-frame screening pass can miss a formula shown only briefly between sample times. Review the source video when a teacher reports a mismatch; OCR is evidence, not a guarantee that every visible symbol was read.
