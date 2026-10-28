# Safe visual style for body / reproduction / hormone lessons

Source: the seven reference videos in `sample_bio_videos` (Female Reproductive Anatomy, Male Reproductive System in Humans,
Male Sex Hormones, Puberty/Menstruation/Menopause, Role of Hormones, Contraceptive Methods, AIDS). 214-397 s each, 1280x720.
Frames were sampled every 6-12 s into contact sheets and viewed.

## What the references do
- **Never a photograph of a body.** Every sensitive topic is a stylised, false-colour 3D animation: ONE organ / gland / cell cluster
  glowing (orange, teal, blue, violet) on a dark background. Ovary sections, follicles, sperm and egg, testis, epididymis, prostate,
  Leydig cells, hormone molecules, virus particles.
- **Whole-body shots are abstract:** a muscle-and-skeleton ecorche (opening of every lesson), or a translucent blue/green hologram
  with only the organs lit, or a simple clothed cartoon avatar. No skin photographs.
- **Everyday context by objects and small insets:** baby socks, a calendar-style "Years" timeline bar, a medicine pack, hospital
  scene as a cartoon, a chart of hormone levels, a flowchart (peptide vs steroid hormones), a chemical structure.
- **Fixed frame:** short branded title card, presenter (clothed) beside the visual, lower-third label pills, dotted leader lines to
  parts, then a "Summary" slide whose bullets build up one by one over a faded copy of the lesson visual, then the logo.
- Shots last about 6-12 s; transitions dissolve (few hard cuts).

## Where we stay stricter than the references
The references show a bare-chested mannequin avatar, hologram silhouettes with a visible body outline and realistic external
male anatomy. Our rule (school students, no adult exposure) does not allow these: no body outline or silhouette, no bare skin on any
figure, no external-genital renders (internal organs, ducts and glands as isolated models only).

## How the generator applies it
- `assets.MEDICAL_STYLE` replaces the "photorealistic documentary photograph" suffix for every shot matching `SENSITIVE_TOPIC`:
  the old mixed prompt (organ 3D render + natural colour photograph) produced flesh-coloured organs inside a skin-toned body, which the
  safety check rightly rejected.
- `safety.py`: organs drawn inside a body outline are unsafe on their own; every picture and motion frame is checked twice.
- Final-frame check (`qa.py`) stays the last gate; a video with an unsafe frame cannot pass.
- Transcribe `VisualDirector` prompt carries the same rule and the list of safe visual ideas.
