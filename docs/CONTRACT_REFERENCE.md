# Storyboard contract reference (Transcribe -> generator)

One shot = one record. The DOCX (vertical field tables) and `<lesson>_contract.json` (schema 2.0) carry the same field names.
`review_notes` is never shown on screen. Subject-independent: renderers are chosen by `visual_type` and the structured fields.

| `visual_type` | Renderer | Required fields | Notes |
|---|---|---|---|
| `title_card` | background still + Pillow title | `heading`, `image_prompt` | exactly one, first |
| `realistic_image` | still (Commons -> FLUX, QA'd) + Ken Burns | `image_prompt` (+ `image_requirement`) | text-free prompt |
| `realistic_labeled_image` | still + verified labels | `labels`, `label_placement` (`label \| morphology \| COORDINATES_PENDING_APPROVED_IMAGE`) | unverifiable labels are omitted and reported |
| `process_steps` | concept/step cards, or **formula** when `formula_lines` exist | `steps` (2-4) or `formula_lines` + `explain_steps` | steps that merely cut the narration into pieces fall back to a photo |
| `split_screen` | 2-3 panels | `columns`: `Caption \| single-subject prompt` | each panel is retrieved/generated and QA'd |
| `graph` (bars) | animated bar chart | `columns`: `Label: value[unit]`, `explain_steps[0]` axis note, `Source:` in `review_notes` | values must come from the narration |
| `graph` (function) | animated plot with computed roots | `columns`: `y = <expression in x>`, `x range: a to b` | safe expression evaluator |
| `circuit` | schemdraw schematics side by side | `columns`: `Caption \| series: battery 12 V; resistor R1; resistor R2` (or `parallel:`) | components: battery, resistor, lamp, capacitor, inductor, switch, ammeter, voltmeter, diode, led |
| `short_motion_clip` | LTX/Wan clip | `ltx_video_prompt`/`wan_video_prompt` | no labels on motion |

## Formulas
`formula_lines` hold **plain math only** (no prose): `x = (-b ± sqrt(b^2 - 4ac))/(2a)`, `1/R = 1/R1 + 1/R2`,
`Cu^{2+}(aq) + 2e^- -> Cu(s)`. `app/mathparse.py` turns fractions, roots, powers, subscripts and Greek into typeset LaTeX; chemistry
(arrows, state symbols, charges, bare formulas) keeps upright species. One line per equation/derivation step; `explain_steps` gives
one short sentence per line. Numbers in a formula must be stated in the narration (checked by Transcribe).

## Language package
`language: {bcp47, script, direction, voice_preference, glossary}`. The generator picks the Edge voice from it
(`production.indic_voices`), renders non-Latin text through Qt/HarfBuzz (`app/shaping.py`), wraps subtitles by measured, shaped width
and refuses to render complex scripts when shaping is unavailable. English technical terms inside native-script narration are kept as spoken.
