from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, model_validator


class Caption(BaseModel):
    text: str
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end <= self.start:
            raise ValueError("Caption end must be after start")
        return self


class Shot(BaseModel):
    """Declarative templates only: source documents never supply executable code."""
    template: Literal["photo", "video", "process", "comparison", "title_card", "formula", "split_screen", "graph", "circuit"]
    heading: str = ""
    learning_objective: str = ""
    asset_path: str | None = None
    motion_asset: str | None = None      # accepted short generated clip (LTX); the still in asset_path stays as fallback
    steps: list[str] = Field(default_factory=list, max_length=4)
    # Cue phrases are matched against speech boundaries. Fractions are a fallback.
    cues: list[str] = Field(default_factory=list, max_length=4)
    stage_fractions: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_template(self):
        step_templates = {"process", "comparison"}
        count = len(self.steps) if self.template in step_templates else 2
        if self.template == "split_screen" and self.stage_fractions:
            count = 2
        if self.template in step_templates and len(self.steps) < 2:
            raise ValueError("Process/comparison templates need 2 to 4 steps")
        if self.cues and len(self.cues) != count:
            raise ValueError("Provide one cue per stage")
        if self.stage_fractions:
            if len(self.stage_fractions) != count or self.stage_fractions[0] != 0:
                raise ValueError("Provide one stage fraction per stage, beginning at 0")
            if any(not 0 <= x < 1 for x in self.stage_fractions) or any(
                a >= b for a, b in zip(self.stage_fractions, self.stage_fractions[1:])
            ):
                raise ValueError("Stage fractions must increase in [0, 1)")
        return self


class Segment(BaseModel):
    segment_number: int = Field(ge=1)
    narration: str
    visual: str = ""
    image_prompt: str = ""
    keywords: List[str] = Field(default_factory=list)
    audio_path: str | None = None
    start: float = 0.0
    end: float = 0.0
    shot: Shot | None = None
    captions: list[Caption] = Field(default_factory=list)
    speech_duration: float = 0.0
    source_references: list[str] = Field(default_factory=list)
    # ---- renderer-ready contract fields (Transcribe storyboard contract v2) ----
    shot_id: str = ""
    visual_type: str = ""
    media_type: str = ""
    image_requirement: str = ""
    labels: list[str] = Field(default_factory=list)
    label_placement: list[str] = Field(default_factory=list)
    label_style: str = ""
    labels_auto: bool = False             # labels were proposed by the generator (auto_labels), not given by the contract
    continuation: bool = False            # the shot re-uses the previous picture (it only names or closes what that picture showed)
    motion: str = ""
    subtitle: str = ""
    subtitle_style: str = ""
    formula_lines: list[str] = Field(default_factory=list)
    explain_steps: list[str] = Field(default_factory=list)
    duration_hint: float = 0.0
    motion_prompt: str = ""
    animate: bool = False                 # the storyboard asks for the approved picture to be brought to life (LTX image-to-video)
    layout: str = ""                      # board layout from the storyboard: definition, bullets, table, derivation, diagram, gallery, summary
    table_rows: list[str] = Field(default_factory=list)       # "left | right", first row = column headers
    gallery_items: list[str] = Field(default_factory=list)
    review_notes: str = ""
    heading: str = ""
    extra_hold: float = 0.0
    panel_assets: list[str] = Field(default_factory=list)
    asset_path: str = ""
    steps: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)


class Scene(BaseModel):
    scene_number: int = Field(ge=1)
    title: str
    narration: str
    segments: List[Segment] = Field(default_factory=list)


class Storyboard(BaseModel):
    title: str
    source: str = ""
    # Optional coverage checklist: {"topic name": ["marker phrase", ...]}. A topic counts as
    # covered when any marker appears in the narration. Empty means no coverage check.
    required_topics: dict[str, list[str]] = Field(default_factory=dict)
    # Language package: {"bcp47": "en-IN", "script": "Latn", "direction": "ltr", "voice": "", "glossary": []}
    language: dict = Field(default_factory=dict)
    subject: str = ""                     # as named by the storyboard ("Chemistry"); see subject.py for how it is used
    scenes: List[Scene] = Field(default_factory=list)

    def all_segments(self) -> list[tuple[Scene, Segment]]:
        items: list[tuple[Scene, Segment]] = []
        for scene in self.scenes:
            for segment in scene.segments:
                items.append((scene, segment))
        return items
