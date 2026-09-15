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
    template: Literal[
        "photo", "video", "process", "comparison", "pollination", "protandry", "protogyny",
        "classification", "agents", "wind", "water", "insect", "life_cycle", "pros_cons",
    ]
    heading: str = ""
    learning_objective: str = ""
    asset_path: str | None = None
    steps: list[str] = Field(default_factory=list, max_length=4)
    # Cue phrases are matched against speech boundaries. Fractions are a fallback.
    cues: list[str] = Field(default_factory=list, max_length=4)
    stage_fractions: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_template(self):
        step_templates = {"process", "comparison", "classification", "agents", "pros_cons"}
        count = len(self.steps) if self.template in step_templates else 2
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
    visual: str
    image_prompt: str
    keywords: List[str] = Field(default_factory=list)
    audio_path: str | None = None
    start: float = 0.0
    end: float = 0.0
    shot: Shot | None = None
    captions: list[Caption] = Field(default_factory=list)
    speech_duration: float = 0.0
    source_references: list[str] = Field(default_factory=list)


class Scene(BaseModel):
    scene_number: int = Field(ge=1)
    title: str
    narration: str
    segments: List[Segment] = Field(default_factory=list)


class Storyboard(BaseModel):
    title: str
    source: str = ""
    scenes: List[Scene] = Field(default_factory=list)

    def all_segments(self) -> list[tuple[Scene, Segment]]:
        items: list[tuple[Scene, Segment]] = []
        for scene in self.scenes:
            for segment in scene.segments:
                items.append((scene, segment))
        return items
