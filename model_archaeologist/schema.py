"""Pydantic models defining the structured report schema for Model Archaeologist.

Defines the full hierarchy of data models used to represent architectural
hypotheses, training paradigm analysis, and capability source identification.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DesignHypothesis(BaseModel):
    """A single design decision hypothesis with supporting evidence.

    Attributes:
        hypothesis: The hypothesized design decision in plain English.
        confidence: Confidence score between 0.0 (no evidence) and 1.0 (certain).
        evidence_quotes: Direct quotes from source material supporting this hypothesis.
        open_questions: Unanswered questions about this design aspect.
    """

    hypothesis: str = Field(
        ...,
        description="The hypothesized design decision.",
        min_length=1,
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0.0 and 1.0.",
        ge=0.0,
        le=1.0,
    )
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Direct quotes from source material.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unanswered questions about this design aspect.",
    )

    @field_validator("hypothesis")
    @classmethod
    def hypothesis_not_empty(cls, v: str) -> str:
        """Ensure hypothesis is not just whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("hypothesis cannot be empty or whitespace")
        return stripped

    @property
    def confidence_label(self) -> str:
        """Return a human-readable confidence label.

        Returns:
            'high', 'medium', or 'low' based on confidence score.
        """
        if self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.4:
            return "medium"
        else:
            return "low"


class ArchitectureDetails(BaseModel):
    """Hypotheses about the core architectural design of a model.

    Attributes:
        attention_mechanism: Hypothesis about the attention type used.
        model_size: Hypothesis about model scale and structure.
        positional_encoding: Hypothesis about positional encoding strategy.
        normalization: Hypothesis about normalization technique and placement.
        moe_usage: Hypothesis about Mixture-of-Experts usage.
    """

    attention_mechanism: DesignHypothesis = Field(
        ...,
        description="Hypothesis about attention type (MHA, GQA, MQA, sparse, etc.).",
    )
    model_size: DesignHypothesis = Field(
        ...,
        description="Hypothesis about model scale (parameters, layers, heads).",
    )
    positional_encoding: DesignHypothesis = Field(
        ...,
        description="Hypothesis about positional encoding (RoPE, ALiBi, absolute, etc.).",
    )
    normalization: DesignHypothesis = Field(
        ...,
        description="Hypothesis about normalization strategy (LayerNorm, RMSNorm, placement).",
    )
    moe_usage: DesignHypothesis = Field(
        ...,
        description="Hypothesis about Mixture-of-Experts architecture usage.",
    )


class TrainingParadigm(BaseModel):
    """Hypotheses about training methodology and data strategies.

    Attributes:
        data_curation: Hypothesis about training data sourcing and filtering.
        alignment_technique: Hypothesis about alignment method (RLHF, DPO, etc.).
        scaling_strategy: Hypothesis about compute and data scaling approach.
    """

    data_curation: DesignHypothesis = Field(
        ...,
        description="Hypothesis about training data sourcing and curation.",
    )
    alignment_technique: DesignHypothesis = Field(
        ...,
        description="Hypothesis about alignment method (RLHF, DPO, Constitutional AI, etc.).",
    )
    scaling_strategy: DesignHypothesis = Field(
        ...,
        description="Hypothesis about scaling laws and compute strategy.",
    )


class CapabilitySource(BaseModel):
    """Hypotheses about the sources of a model's capabilities.

    Attributes:
        emergent_behaviors: Hypothesis about emergent capabilities and when they appear.
        fine_tuning_approach: Hypothesis about instruction tuning and fine-tuning strategy.
        efficiency_optimizations: Hypothesis about inference and serving optimizations.
    """

    emergent_behaviors: DesignHypothesis = Field(
        ...,
        description="Hypothesis about emergent behaviors and capability sources.",
    )
    fine_tuning_approach: DesignHypothesis = Field(
        ...,
        description="Hypothesis about instruction tuning and fine-tuning methodology.",
    )
    efficiency_optimizations: DesignHypothesis = Field(
        ...,
        description="Hypothesis about inference efficiency techniques.",
    )


class ArchitectureReport(BaseModel):
    """Complete architectural dossier for an AI model.

    This is the top-level report object produced by the analysis pipeline.
    It aggregates all hypotheses into a structured, human-readable format.

    Attributes:
        model_name: The AI model that was analyzed.
        generated_at: UTC timestamp when the report was generated.
        executive_summary: High-level narrative summary of key findings.
        architecture: Hypotheses about core architectural design.
        training: Hypotheses about training methodology.
        capabilities: Hypotheses about capability sources.
        overall_confidence: Average confidence across all hypotheses.
        sources_analyzed: List of source URLs/files that were analyzed.
        additional_notes: Free-form notes not fitting other categories.
    """

    model_name: str = Field(
        ...,
        description="The AI model that was analyzed.",
        min_length=1,
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this report was generated.",
    )
    executive_summary: str = Field(
        ...,
        description="High-level narrative summary of key architectural findings.",
    )
    architecture: ArchitectureDetails = Field(
        ...,
        description="Hypotheses about core architectural design decisions.",
    )
    training: TrainingParadigm = Field(
        ...,
        description="Hypotheses about training methodology and data strategies.",
    )
    capabilities: CapabilitySource = Field(
        ...,
        description="Hypotheses about the sources of model capabilities.",
    )
    overall_confidence: float = Field(
        default=0.0,
        description="Average confidence across all hypotheses (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )
    sources_analyzed: list[str] = Field(
        default_factory=list,
        description="Source URLs and file paths that were analyzed.",
    )
    additional_notes: str = Field(
        default="",
        description="Free-form notes not fitting other categories.",
    )

    @model_validator(mode="after")
    def compute_overall_confidence(self) -> ArchitectureReport:
        """Compute overall_confidence as the mean of all hypothesis confidences.

        Only recomputes if overall_confidence is 0.0 (not explicitly set).
        """
        if self.overall_confidence == 0.0:
            confidences = [
                self.architecture.attention_mechanism.confidence,
                self.architecture.model_size.confidence,
                self.architecture.positional_encoding.confidence,
                self.architecture.normalization.confidence,
                self.architecture.moe_usage.confidence,
                self.training.data_curation.confidence,
                self.training.alignment_technique.confidence,
                self.training.scaling_strategy.confidence,
                self.capabilities.emergent_behaviors.confidence,
                self.capabilities.fine_tuning_approach.confidence,
                self.capabilities.efficiency_optimizations.confidence,
            ]
            if confidences:
                self.overall_confidence = round(sum(confidences) / len(confidences), 3)
        return self

    @property
    def all_hypotheses(self) -> list[tuple[str, str, DesignHypothesis]]:
        """Return all hypotheses as (category, aspect, hypothesis) tuples.

        Returns:
            List of (category, aspect_name, DesignHypothesis) tuples.
        """
        return [
            ("Architecture", "Attention Mechanism", self.architecture.attention_mechanism),
            ("Architecture", "Model Size & Structure", self.architecture.model_size),
            ("Architecture", "Positional Encoding", self.architecture.positional_encoding),
            ("Architecture", "Normalization Strategy", self.architecture.normalization),
            ("Architecture", "Mixture-of-Experts", self.architecture.moe_usage),
            ("Training", "Data Curation", self.training.data_curation),
            ("Training", "Alignment Technique", self.training.alignment_technique),
            ("Training", "Scaling Strategy", self.training.scaling_strategy),
            ("Capabilities", "Emergent Behaviors", self.capabilities.emergent_behaviors),
            ("Capabilities", "Fine-tuning Approach", self.capabilities.fine_tuning_approach),
            ("Capabilities", "Efficiency Optimizations", self.capabilities.efficiency_optimizations),
        ]

    @field_validator("model_name")
    @classmethod
    def model_name_not_empty(cls, v: str) -> str:
        """Ensure model_name is not just whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_name cannot be empty or whitespace")
        return stripped


class IngestionMetadata(BaseModel):
    """Metadata about an ingested document source.

    Attributes:
        source: URL or file path of the source.
        content_type: MIME type of the source (e.g. 'text/html', 'application/pdf').
        title: Optional title extracted from the document.
        char_count: Number of characters in the extracted text.
        chunk_count: Number of chunks produced from this document.
    """

    source: str = Field(..., description="URL or file path of the source.")
    content_type: str = Field(default="unknown", description="MIME type of the source.")
    title: Optional[str] = Field(default=None, description="Document title if available.")
    char_count: int = Field(default=0, ge=0, description="Character count of extracted text.")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks produced.")
