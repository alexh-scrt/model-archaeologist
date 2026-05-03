"""Pydantic models defining the structured report schema for Model Archaeologist.

Defines the full hierarchy of data models used to represent architectural
hypotheses, training paradigm analysis, and capability source identification.
These models are the central data contract between the analyzer, renderer,
and CLI layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class DesignHypothesis(BaseModel):
    """A single design decision hypothesis with supporting evidence.

    Represents one facet of an architectural or training decision,
    combining a human-readable hypothesis statement with a confidence
    score, direct evidence quotes from source material, and open
    questions that remain unresolved.

    Attributes:
        hypothesis: The hypothesized design decision in plain English.
        confidence: Confidence score between 0.0 (no evidence) and 1.0 (certain).
        evidence_quotes: Direct quotes from source material supporting this hypothesis.
        open_questions: Unanswered questions about this design aspect.
    """

    hypothesis: str = Field(
        ...,
        description="The hypothesized design decision in plain English.",
        min_length=1,
    )
    confidence: float = Field(
        ...,
        description="Confidence score between 0.0 (no evidence) and 1.0 (certain).",
        ge=0.0,
        le=1.0,
    )
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Direct quotes from source material supporting this hypothesis.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        description="Unanswered questions about this design aspect.",
    )

    model_config = {"str_strip_whitespace": True}

    @field_validator("hypothesis")
    @classmethod
    def hypothesis_not_empty(cls, v: str) -> str:
        """Ensure hypothesis is not just whitespace.

        Args:
            v: The raw hypothesis string.

        Returns:
            The stripped hypothesis string.

        Raises:
            ValueError: If the hypothesis is empty or whitespace-only.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("hypothesis cannot be empty or whitespace-only")
        return stripped

    @field_validator("confidence")
    @classmethod
    def confidence_rounded(cls, v: float) -> float:
        """Round confidence to three decimal places for consistency.

        Args:
            v: The raw confidence value.

        Returns:
            Confidence rounded to 3 decimal places.
        """
        return round(v, 3)

    @field_validator("evidence_quotes", "open_questions", mode="before")
    @classmethod
    def filter_empty_strings(cls, v: object) -> list[str]:
        """Remove empty or whitespace-only strings from list fields.

        Args:
            v: The raw list value.

        Returns:
            Filtered list with empty strings removed.
        """
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if str(item).strip()]

    @property
    def confidence_label(self) -> str:
        """Return a human-readable confidence label.

        Returns:
            'high' if confidence >= 0.7, 'medium' if >= 0.4, else 'low'.
        """
        if self.confidence >= 0.7:
            return "high"
        elif self.confidence >= 0.4:
            return "medium"
        else:
            return "low"

    @property
    def has_evidence(self) -> bool:
        """Return True if at least one evidence quote is present.

        Returns:
            True when evidence_quotes is non-empty.
        """
        return len(self.evidence_quotes) > 0


class ArchitectureDetails(BaseModel):
    """Hypotheses about the core architectural design of a model.

    Aggregates individual DesignHypothesis instances covering the major
    structural decisions that define a neural network architecture.

    Attributes:
        attention_mechanism: Hypothesis about the attention type used.
        model_size: Hypothesis about model scale and structure.
        positional_encoding: Hypothesis about positional encoding strategy.
        normalization: Hypothesis about normalization technique and placement.
        moe_usage: Hypothesis about Mixture-of-Experts usage.
    """

    attention_mechanism: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about attention type used "
            "(MHA, GQA, MQA, sparse attention, etc.)."
        ),
    )
    model_size: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about model scale: parameter count, layer depth, "
            "hidden dimensions, head counts."
        ),
    )
    positional_encoding: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about positional encoding strategy "
            "(RoPE, ALiBi, absolute sinusoidal, relative, NoPE, etc.)."
        ),
    )
    normalization: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about normalization technique and placement "
            "(Pre-LN, Post-LN, LayerNorm, RMSNorm, etc.)."
        ),
    )
    moe_usage: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about Mixture-of-Experts architecture: "
            "sparse vs dense, number of experts, routing strategy."
        ),
    )

    @property
    def mean_confidence(self) -> float:
        """Return the mean confidence across all architecture hypotheses.

        Returns:
            Float mean rounded to 3 decimal places.
        """
        values = [
            self.attention_mechanism.confidence,
            self.model_size.confidence,
            self.positional_encoding.confidence,
            self.normalization.confidence,
            self.moe_usage.confidence,
        ]
        return round(sum(values) / len(values), 3)

    def as_dict(self) -> dict[str, DesignHypothesis]:
        """Return a mapping of aspect name to DesignHypothesis.

        Returns:
            Dict with human-readable keys mapping to hypothesis objects.
        """
        return {
            "Attention Mechanism": self.attention_mechanism,
            "Model Size & Structure": self.model_size,
            "Positional Encoding": self.positional_encoding,
            "Normalization Strategy": self.normalization,
            "Mixture-of-Experts": self.moe_usage,
        }


class TrainingParadigm(BaseModel):
    """Hypotheses about training methodology and data strategies.

    Covers the three principal axes of modern LLM training: how data
    was sourced and filtered, how the model was aligned to human
    preferences, and what scaling strategy was applied.

    Attributes:
        data_curation: Hypothesis about training data sourcing and filtering.
        alignment_technique: Hypothesis about alignment method (RLHF, DPO, etc.).
        scaling_strategy: Hypothesis about compute and data scaling approach.
    """

    data_curation: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about training data sourcing, filtering, and curation "
            "methodology (web crawl quality filters, deduplication, domains, etc.)."
        ),
    )
    alignment_technique: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about alignment method used to steer model behavior "
            "(RLHF with PPO, DPO, Constitutional AI, SFT-only, etc.)."
        ),
    )
    scaling_strategy: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about scaling laws applied: compute-optimal training, "
            "Chinchilla ratios, over-training on tokens, etc."
        ),
    )

    @property
    def mean_confidence(self) -> float:
        """Return the mean confidence across all training hypotheses.

        Returns:
            Float mean rounded to 3 decimal places.
        """
        values = [
            self.data_curation.confidence,
            self.alignment_technique.confidence,
            self.scaling_strategy.confidence,
        ]
        return round(sum(values) / len(values), 3)

    def as_dict(self) -> dict[str, DesignHypothesis]:
        """Return a mapping of aspect name to DesignHypothesis.

        Returns:
            Dict with human-readable keys mapping to hypothesis objects.
        """
        return {
            "Data Curation": self.data_curation,
            "Alignment Technique": self.alignment_technique,
            "Scaling Strategy": self.scaling_strategy,
        }


class CapabilitySource(BaseModel):
    """Hypotheses about the sources of a model's capabilities.

    Addresses how the model gained its reported abilities: through scale
    and emergent phenomena, through fine-tuning and instruction following,
    or through inference-time efficiency optimizations.

    Attributes:
        emergent_behaviors: Hypothesis about emergent capabilities and when they appear.
        fine_tuning_approach: Hypothesis about instruction tuning and fine-tuning strategy.
        efficiency_optimizations: Hypothesis about inference and serving optimizations.
    """

    emergent_behaviors: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about emergent capabilities: what behaviors appeared at scale, "
            "benchmark inflection points, chain-of-thought reasoning, etc."
        ),
    )
    fine_tuning_approach: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about instruction tuning and fine-tuning methodology: "
            "SFT dataset size, RLHF stages, LoRA vs full fine-tune, etc."
        ),
    )
    efficiency_optimizations: DesignHypothesis = Field(
        ...,
        description=(
            "Hypothesis about inference efficiency techniques: Flash Attention, "
            "speculative decoding, quantization, KV-cache optimizations, etc."
        ),
    )

    @property
    def mean_confidence(self) -> float:
        """Return the mean confidence across all capability hypotheses.

        Returns:
            Float mean rounded to 3 decimal places.
        """
        values = [
            self.emergent_behaviors.confidence,
            self.fine_tuning_approach.confidence,
            self.efficiency_optimizations.confidence,
        ]
        return round(sum(values) / len(values), 3)

    def as_dict(self) -> dict[str, DesignHypothesis]:
        """Return a mapping of aspect name to DesignHypothesis.

        Returns:
            Dict with human-readable keys mapping to hypothesis objects.
        """
        return {
            "Emergent Behaviors": self.emergent_behaviors,
            "Fine-tuning Approach": self.fine_tuning_approach,
            "Efficiency Optimizations": self.efficiency_optimizations,
        }


class ArchitectureReport(BaseModel):
    """Complete architectural dossier for an AI model.

    This is the top-level report object produced by the full analysis
    pipeline. It aggregates all hypotheses into a structured,
    human-readable format suitable for Markdown or JSON rendering.

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
        description="The AI model that was analyzed (e.g. 'GPT-4', 'LLaMA 3').",
        min_length=1,
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when this report was generated.",
    )
    executive_summary: str = Field(
        ...,
        description=(
            "High-level narrative summary of key architectural findings, "
            "2-3 paragraphs covering the most notable hypotheses."
        ),
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
        description="Hypotheses about the sources of the model's capabilities.",
    )
    overall_confidence: float = Field(
        default=0.0,
        description=(
            "Mean confidence across all 11 hypotheses (0.0-1.0). "
            "Auto-computed if left at 0.0."
        ),
        ge=0.0,
        le=1.0,
    )
    sources_analyzed: list[str] = Field(
        default_factory=list,
        description="Source URLs and file paths that were analyzed.",
    )
    additional_notes: str = Field(
        default="",
        description=(
            "Free-form notes, caveats, or observations that do not fit "
            "neatly into the structured sections above."
        ),
    )

    model_config = {"str_strip_whitespace": True}

    @field_validator("model_name")
    @classmethod
    def model_name_not_empty(cls, v: str) -> str:
        """Ensure model_name is not just whitespace.

        Args:
            v: The raw model name string.

        Returns:
            The stripped model name string.

        Raises:
            ValueError: If model_name is empty or whitespace-only.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("model_name cannot be empty or whitespace-only")
        return stripped

    @field_validator("overall_confidence")
    @classmethod
    def overall_confidence_rounded(cls, v: float) -> float:
        """Round overall_confidence to three decimal places.

        Args:
            v: The raw overall_confidence value.

        Returns:
            Confidence rounded to 3 decimal places.
        """
        return round(v, 3)

    @field_validator("sources_analyzed", mode="before")
    @classmethod
    def filter_empty_sources(cls, v: object) -> list[str]:
        """Remove empty or whitespace-only strings from sources_analyzed.

        Args:
            v: The raw sources list.

        Returns:
            Filtered list with empty strings removed.
        """
        if not isinstance(v, list):
            return []
        return [str(item).strip() for item in v if str(item).strip()]

    @model_validator(mode="after")
    def compute_overall_confidence(self) -> ArchitectureReport:
        """Auto-compute overall_confidence as the mean of all 11 hypothesis confidences.

        Only recomputes if overall_confidence is exactly 0.0 (indicating it
        was not explicitly set by the caller). This allows callers to supply
        a pre-computed value while still benefiting from auto-computation.

        Returns:
            The same ArchitectureReport instance with overall_confidence set.
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
        """Return all 11 hypotheses as (category, aspect, hypothesis) tuples.

        Provides a flat view of the entire report for iteration, rendering,
        or bulk operations on all hypotheses.

        Returns:
            List of (category_name, aspect_name, DesignHypothesis) tuples
            covering all architecture, training, and capability hypotheses.
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
            (
                "Capabilities",
                "Efficiency Optimizations",
                self.capabilities.efficiency_optimizations,
            ),
        ]

    @property
    def high_confidence_hypotheses(self) -> list[tuple[str, str, DesignHypothesis]]:
        """Return only hypotheses with confidence >= 0.7.

        Returns:
            Filtered list of (category, aspect, DesignHypothesis) tuples.
        """
        return [
            (cat, aspect, hyp)
            for cat, aspect, hyp in self.all_hypotheses
            if hyp.confidence >= 0.7
        ]

    @property
    def low_confidence_hypotheses(self) -> list[tuple[str, str, DesignHypothesis]]:
        """Return only hypotheses with confidence < 0.4.

        Returns:
            Filtered list of (category, aspect, DesignHypothesis) tuples.
        """
        return [
            (cat, aspect, hyp)
            for cat, aspect, hyp in self.all_hypotheses
            if hyp.confidence < 0.4
        ]

    def summary_table(self) -> list[dict[str, str]]:
        """Return a summary table of all hypotheses as a list of dicts.

        Useful for rendering tabular overviews in reports or CLI output.

        Returns:
            List of dicts with keys: 'category', 'aspect', 'confidence',
            'confidence_label', 'hypothesis_preview'.
        """
        rows = []
        for category, aspect, hyp in self.all_hypotheses:
            rows.append(
                {
                    "category": category,
                    "aspect": aspect,
                    "confidence": f"{hyp.confidence:.2f}",
                    "confidence_label": hyp.confidence_label,
                    "hypothesis_preview": (
                        hyp.hypothesis[:80] + "..."
                        if len(hyp.hypothesis) > 80
                        else hyp.hypothesis
                    ),
                }
            )
        return rows


class IngestionMetadata(BaseModel):
    """Metadata about an ingested document source.

    Tracks provenance and processing statistics for a single document
    that was fed into the analysis pipeline.

    Attributes:
        source: URL or file path of the source.
        content_type: MIME type of the source (e.g. 'text/html', 'application/pdf').
        title: Optional title extracted from the document.
        char_count: Number of characters in the extracted text.
        chunk_count: Number of chunks produced from this document.
    """

    source: str = Field(
        ...,
        description="URL or file path of the source document.",
        min_length=1,
    )
    content_type: str = Field(
        default="unknown",
        description="MIME type of the source (e.g. 'text/html', 'application/pdf').",
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title if available (extracted from HTML <title> tag or filename).",
    )
    char_count: int = Field(
        default=0,
        ge=0,
        description="Number of characters in the extracted text.",
    )
    chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of token-aware chunks produced from this document.",
    )

    @field_validator("source")
    @classmethod
    def source_not_empty(cls, v: str) -> str:
        """Ensure source is not just whitespace.

        Args:
            v: The raw source string.

        Returns:
            The stripped source string.

        Raises:
            ValueError: If source is empty or whitespace-only.
        """
        stripped = v.strip()
        if not stripped:
            raise ValueError("source cannot be empty or whitespace-only")
        return stripped

    @property
    def is_pdf(self) -> bool:
        """Return True if this source is a PDF document.

        Returns:
            True when content_type is 'application/pdf'.
        """
        return self.content_type.lower() == "application/pdf"

    @property
    def display_name(self) -> str:
        """Return a human-friendly display name for this source.

        Returns the title if available, otherwise the source path/URL.

        Returns:
            Title string or source string.
        """
        return self.title if self.title else self.source
