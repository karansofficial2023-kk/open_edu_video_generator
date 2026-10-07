# Open-source narration voices (Indian languages)

Edge TTS is a free client for a Microsoft cloud service: it is not open source. For fully open pipelines the generator can
narrate with **AI4Bharat Indic-TTS** (FastPitch + HiFi-GAN, MIT licence, no login): Tamil, Telugu, Kannada, Malayalam, Hindi,
Marathi, Bengali, Gujarati, Odia, Punjabi, Assamese, Bodo, Manipuri and Rajasthani - each with a female and a male voice.
English stays on Edge unless you add an open English voice.

## Install (once)
```
# 1. voice engine in its own environment (PyTorch 2.8 CPU + Coqui TTS; do not mix with other environments)
python -m venv D:/AI/venvs/indictts
D:/AI/venvs/indictts/Scripts/python.exe -m pip install coqui-tts "transformers>=4.50,<5" "torch==2.8.0" "torchaudio==2.8.0" --extra-index-url https://download.pytorch.org/whl/cpu
# 2. voices (about 1.5 GB per language)
python tools/install_open_voice.py ta te
```
The versions matter: Coqui needs `torchcodec` with PyTorch 2.9+, and transformers 5.x removed a function Coqui imports.

## Use
In the config (`production:` block):
```
tts_provider: auto        # edge = cloud voices (default) | open = open voices only, error if missing | auto = open when installed
open_tts: {python: "D:/AI/venvs/indictts/Scripts/python.exe", models_dir: "D:/AI/models/indic-tts", speaker: female}
pronunciations:
  ta: {"CO2": "கார்பன் டை ஆக்சைடு"}
```
The voice reads native script only. Therefore:
* **digits** are spoken as words (Telugu/Kannada/Bengali via `num2words`; Tamil has a table for 0-20 and the tens, other numbers are read digit by digit - have a Tamil speaker review numeric lessons);
* **Latin terms** (formulas, symbols, English words) are silent unless listed under `pronunciations`;
* **caption timings** are derived from the audio (these voices give no word timings): words share the speaking time by length, with pauses after punctuation.

## Measured
Tamil, three lesson sentences, synthesized on the CPU and transcribed back by Whisper large-v3: character error rate 6.0%
(the Edge Tamil voice scored 6.8% with the same test). It speaks about 5x faster than real time on the CPU, so the GPU stays free.
This measures intelligibility, not naturalness: have a native speaker listen before publishing.

## Other options checked
| Option | Why not default |
|--------|-----------------|
| Indic Parler-TTS (Apache-2.0) | needs a Hugging Face account that accepted its terms |
| Meta MMS-TTS (Tamil, Telugu) | CC-BY-NC: non-commercial |
| Piper | no Tamil/Kannada/Gujarati voices |
| Kokoro | Hindi and English only |
