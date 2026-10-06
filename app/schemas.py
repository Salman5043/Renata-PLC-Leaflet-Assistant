from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ============================================================
# Citation
# ============================================================

class Citation(BaseModel):
    label: str
    source: str | None = None
    page: int | str | None = None
    section: str | None = None
    product: str | None = None
    active_ingredient: str | None = None
    score: float | None = None


# ============================================================
# Single question
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask the Renata leaflet assistant.",
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)


# ============================================================
# Batch questions
# ============================================================

class BatchQuestion(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
    )


class AskBatchRequest(BaseModel):
    questions: list[str] = Field(
        ...,
        min_length=1,
        description="List of questions to process.",
    )


class AskBatchResponse(BaseModel):
    results: list[AskResponse] = Field(
        default_factory=list,
    )


# ============================================================
# Health
# ============================================================

class HealthResponse(BaseModel):
    status: str
    service: str | None = None


# ============================================================
# Optional generic response models
# ============================================================

class ErrorResponse(BaseModel):
    detail: str


class AskResult(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)