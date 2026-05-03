"""Unit tests for Pydantic schema models in model_archaeologist/schema.py.

Verifies that all models parse correctly, enforce validation rules,
reject invalid data with clear errors, and expose correct computed properties.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from model_archaeologist.schema import (
    ArchitectureDetails,
    ArchitectureReport,
    CapabilitySource,
    DesignHypothesis,
    IngestionMetadata,
    TrainingParadigm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_hypothesis(
    hypothesis: str = "Uses Multi-Head Attention.",
    confidence: float = 0.8,
    evidence_quotes: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> DesignHypothesis:
    """Build a DesignHypothesis with sensible defaults."""
    return DesignHypothesis(
        hypothesis=hypothesis,
        confidence=confidence,
        evidence_quotes=evidence_quotes or ["The model employs multi-head attention."],
        open_questions=open_questions or ["How many heads?"],
    )


def make_architecture(
    confidence: float = 0.7,
) -> ArchitectureDetails:
    """Build an ArchitectureDetails object with default hypothesis values."""
    h = make_hypothesis(confidence=confidence)
    return ArchitectureDetails(
        attention_mechanism=h,
        model_size=h,
        positional_encoding=h,
        normalization=h,
        moe_usage=h,
    )


def make_training(confidence: float = 0.6) -> TrainingParadigm:
    """Build a TrainingParadigm object with default hypothesis values."""
    h = make_hypothesis(confidence=confidence)
    return TrainingParadigm(
        data_curation=h,
        alignment_technique=h,
        scaling_strategy=h,
    )


def make_capabilities(confidence: float = 0.5) -> CapabilitySource:
    """Build a CapabilitySource object with default hypothesis values."""
    h = make_hypothesis(confidence=confidence)
    return CapabilitySource(
        emergent_behaviors=h,
        fine_tuning_approach=h,
        efficiency_optimizations=h,
    )


def make_report(
    model_name: str = "TestModel",
    arch_confidence: float = 0.7,
    training_confidence: float = 0.6,
    cap_confidence: float = 0.5,
    overall_confidence: float = 0.0,
    sources: list[str] | None = None,
) -> ArchitectureReport:
    """Build a complete ArchitectureReport for testing."""
    return ArchitectureReport(
        model_name=model_name,
        executive_summary="This is a test executive summary covering key findings.",
        architecture=make_architecture(arch_confidence),
        training=make_training(training_confidence),
        capabilities=make_capabilities(cap_confidence),
        overall_confidence=overall_confidence,
        sources_analyzed=sources or ["https://example.com/paper"],
    )


# ---------------------------------------------------------------------------
# DesignHypothesis tests
# ---------------------------------------------------------------------------


class TestDesignHypothesis:
    """Tests for the DesignHypothesis model."""

    def test_valid_creation(self) -> None:
        """A valid DesignHypothesis is created without error."""
        h = make_hypothesis()
        assert h.hypothesis == "Uses Multi-Head Attention."
        assert h.confidence == 0.8
        assert len(h.evidence_quotes) == 1
        assert len(h.open_questions) == 1

    def test_empty_lists_allowed(self) -> None:
        """Evidence quotes and open questions can be empty."""
        h = DesignHypothesis(
            hypothesis="Some hypothesis",
            confidence=0.5,
            evidence_quotes=[],
            open_questions=[],
        )
        assert h.evidence_quotes == []
        assert h.open_questions == []

    def test_confidence_boundary_zero(self) -> None:
        """Confidence of exactly 0.0 is valid."""
        h = DesignHypothesis(hypothesis="Unknown", confidence=0.0)
        assert h.confidence == 0.0

    def test_confidence_boundary_one(self) -> None:
        """Confidence of exactly 1.0 is valid."""
        h = DesignHypothesis(hypothesis="Certain", confidence=1.0)
        assert h.confidence == 1.0

    def test_confidence_below_zero_rejected(self) -> None:
        """Confidence below 0.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DesignHypothesis(hypothesis="Bad", confidence=-0.1)
        assert "confidence" in str(exc_info.value).lower() or "ge" in str(exc_info.value)

    def test_confidence_above_one_rejected(self) -> None:
        """Confidence above 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            DesignHypothesis(hypothesis="Bad", confidence=1.01)

    def test_empty_hypothesis_rejected(self) -> None:
        """An empty hypothesis string raises ValidationError."""
        with pytest.raises(ValidationError):
            DesignHypothesis(hypothesis="", confidence=0.5)

    def test_whitespace_hypothesis_rejected(self) -> None:
        """A whitespace-only hypothesis raises ValidationError."""
        with pytest.raises(ValidationError):
            DesignHypothesis(hypothesis="   ", confidence=0.5)

    def test_hypothesis_stripped(self) -> None:
        """Leading/trailing whitespace in hypothesis is stripped."""
        h = DesignHypothesis(hypothesis="  some hypothesis  ", confidence=0.5)
        assert h.hypothesis == "some hypothesis"

    def test_confidence_label_high(self) -> None:
        """Confidence >= 0.7 returns 'high' label."""
        h = DesignHypothesis(hypothesis="High", confidence=0.7)
        assert h.confidence_label == "high"
        h2 = DesignHypothesis(hypothesis="High", confidence=1.0)
        assert h2.confidence_label == "high"

    def test_confidence_label_medium(self) -> None:
        """Confidence in [0.4, 0.7) returns 'medium' label."""
        h = DesignHypothesis(hypothesis="Medium", confidence=0.4)
        assert h.confidence_label == "medium"
        h2 = DesignHypothesis(hypothesis="Medium", confidence=0.69)
        assert h2.confidence_label == "medium"

    def test_confidence_label_low(self) -> None:
        """Confidence < 0.4 returns 'low' label."""
        h = DesignHypothesis(hypothesis="Low", confidence=0.0)
        assert h.confidence_label == "low"
        h2 = DesignHypothesis(hypothesis="Low", confidence=0.39)
        assert h2.confidence_label == "low"

    def test_has_evidence_true(self) -> None:
        """has_evidence is True when evidence_quotes is non-empty."""
        h = make_hypothesis(evidence_quotes=["some quote"])
        assert h.has_evidence is True

    def test_has_evidence_false(self) -> None:
        """has_evidence is False when evidence_quotes is empty."""
        h = DesignHypothesis(hypothesis="No evidence", confidence=0.3)
        assert h.has_evidence is False

    def test_confidence_rounded_to_three_decimals(self) -> None:
        """Confidence is rounded to 3 decimal places."""
        h = DesignHypothesis(hypothesis="Test", confidence=0.123456789)
        assert h.confidence == 0.123

    def test_empty_strings_filtered_from_lists(self) -> None:
        """Empty strings are removed from evidence_quotes and open_questions."""
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.5,
            evidence_quotes=["", "  ", "valid quote"],
            open_questions=["valid question", ""],
        )
        assert h.evidence_quotes == ["valid quote"]
        assert h.open_questions == ["valid question"]

    def test_missing_confidence_raises_error(self) -> None:
        """Missing confidence field raises ValidationError."""
        with pytest.raises(ValidationError):
            DesignHypothesis(hypothesis="test")  # type: ignore[call-arg]

    def test_missing_hypothesis_raises_error(self) -> None:
        """Missing hypothesis field raises ValidationError."""
        with pytest.raises(ValidationError):
            DesignHypothesis(confidence=0.5)  # type: ignore[call-arg]

    def test_default_empty_lists(self) -> None:
        """evidence_quotes and open_questions default to empty lists."""
        h = DesignHypothesis(hypothesis="Test", confidence=0.5)
        assert h.evidence_quotes == []
        assert h.open_questions == []


# ---------------------------------------------------------------------------
# ArchitectureDetails tests
# ---------------------------------------------------------------------------


class TestArchitectureDetails:
    """Tests for the ArchitectureDetails model."""

    def test_valid_creation(self) -> None:
        """ArchitectureDetails is created with all required fields."""
        arch = make_architecture()
        assert arch.attention_mechanism is not None
        assert arch.model_size is not None
        assert arch.positional_encoding is not None
        assert arch.normalization is not None
        assert arch.moe_usage is not None

    def test_mean_confidence(self) -> None:
        """mean_confidence computes the mean of all 5 hypothesis confidences."""
        arch = make_architecture(confidence=0.8)
        assert arch.mean_confidence == pytest.approx(0.8, abs=1e-3)

    def test_mean_confidence_mixed(self) -> None:
        """mean_confidence with mixed confidences computes correctly."""
        h_high = make_hypothesis(confidence=1.0)
        h_low = make_hypothesis(confidence=0.0)
        arch = ArchitectureDetails(
            attention_mechanism=h_high,
            model_size=h_low,
            positional_encoding=h_high,
            normalization=h_low,
            moe_usage=h_high,
        )
        # (1.0 + 0.0 + 1.0 + 0.0 + 1.0) / 5 = 0.6
        assert arch.mean_confidence == pytest.approx(0.6, abs=1e-3)

    def test_as_dict_returns_five_keys(self) -> None:
        """as_dict returns a dict with exactly 5 keys."""
        arch = make_architecture()
        d = arch.as_dict()
        assert len(d) == 5
        assert "Attention Mechanism" in d
        assert "Model Size & Structure" in d
        assert "Positional Encoding" in d
        assert "Normalization Strategy" in d
        assert "Mixture-of-Experts" in d

    def test_missing_field_raises_error(self) -> None:
        """Missing any required field raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            ArchitectureDetails(
                attention_mechanism=h,
                model_size=h,
                positional_encoding=h,
                normalization=h,
                # moe_usage missing
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TrainingParadigm tests
# ---------------------------------------------------------------------------


class TestTrainingParadigm:
    """Tests for the TrainingParadigm model."""

    def test_valid_creation(self) -> None:
        """TrainingParadigm is created with all required fields."""
        t = make_training()
        assert t.data_curation is not None
        assert t.alignment_technique is not None
        assert t.scaling_strategy is not None

    def test_mean_confidence(self) -> None:
        """mean_confidence computes the mean of all 3 hypothesis confidences."""
        t = make_training(confidence=0.6)
        assert t.mean_confidence == pytest.approx(0.6, abs=1e-3)

    def test_mean_confidence_mixed(self) -> None:
        """mean_confidence with mixed confidences computes correctly."""
        h_a = make_hypothesis(confidence=0.9)
        h_b = make_hypothesis(confidence=0.3)
        h_c = make_hypothesis(confidence=0.6)
        t = TrainingParadigm(
            data_curation=h_a,
            alignment_technique=h_b,
            scaling_strategy=h_c,
        )
        # (0.9 + 0.3 + 0.6) / 3 = 0.6
        assert t.mean_confidence == pytest.approx(0.6, abs=1e-3)

    def test_as_dict_returns_three_keys(self) -> None:
        """as_dict returns a dict with exactly 3 keys."""
        t = make_training()
        d = t.as_dict()
        assert len(d) == 3
        assert "Data Curation" in d
        assert "Alignment Technique" in d
        assert "Scaling Strategy" in d

    def test_missing_field_raises_error(self) -> None:
        """Missing any required field raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            TrainingParadigm(
                data_curation=h,
                alignment_technique=h,
                # scaling_strategy missing
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# CapabilitySource tests
# ---------------------------------------------------------------------------


class TestCapabilitySource:
    """Tests for the CapabilitySource model."""

    def test_valid_creation(self) -> None:
        """CapabilitySource is created with all required fields."""
        c = make_capabilities()
        assert c.emergent_behaviors is not None
        assert c.fine_tuning_approach is not None
        assert c.efficiency_optimizations is not None

    def test_mean_confidence(self) -> None:
        """mean_confidence computes the mean of all 3 hypothesis confidences."""
        c = make_capabilities(confidence=0.5)
        assert c.mean_confidence == pytest.approx(0.5, abs=1e-3)

    def test_as_dict_returns_three_keys(self) -> None:
        """as_dict returns a dict with exactly 3 keys."""
        c = make_capabilities()
        d = c.as_dict()
        assert len(d) == 3
        assert "Emergent Behaviors" in d
        assert "Fine-tuning Approach" in d
        assert "Efficiency Optimizations" in d

    def test_missing_field_raises_error(self) -> None:
        """Missing any required field raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            CapabilitySource(
                emergent_behaviors=h,
                fine_tuning_approach=h,
                # efficiency_optimizations missing
            )  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ArchitectureReport tests
# ---------------------------------------------------------------------------


class TestArchitectureReport:
    """Tests for the top-level ArchitectureReport model."""

    def test_valid_creation(self) -> None:
        """ArchitectureReport is created with all required fields."""
        report = make_report()
        assert report.model_name == "TestModel"
        assert isinstance(report.generated_at, datetime)
        assert report.generated_at.tzinfo is not None

    def test_generated_at_defaults_to_utc_now(self) -> None:
        """generated_at defaults to the current UTC time."""
        before = datetime.now(timezone.utc)
        report = make_report()
        after = datetime.now(timezone.utc)
        assert before <= report.generated_at <= after

    def test_empty_model_name_rejected(self) -> None:
        """Empty model_name raises ValidationError."""
        with pytest.raises(ValidationError):
            make_report(model_name="")

    def test_whitespace_model_name_rejected(self) -> None:
        """Whitespace-only model_name raises ValidationError."""
        with pytest.raises(ValidationError):
            make_report(model_name="   ")

    def test_model_name_stripped(self) -> None:
        """Leading/trailing whitespace in model_name is stripped."""
        report = ArchitectureReport(
            model_name="  GPT-4  ",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
        )
        assert report.model_name == "GPT-4"

    def test_overall_confidence_auto_computed(self) -> None:
        """overall_confidence is auto-computed when left at 0.0."""
        report = make_report(
            arch_confidence=0.9,
            training_confidence=0.6,
            cap_confidence=0.3,
            overall_confidence=0.0,
        )
        # 5 * 0.9 + 3 * 0.6 + 3 * 0.3 = 4.5 + 1.8 + 0.9 = 7.2
        # 7.2 / 11 ≈ 0.655
        expected = round((5 * 0.9 + 3 * 0.6 + 3 * 0.3) / 11, 3)
        assert report.overall_confidence == pytest.approx(expected, abs=1e-3)

    def test_overall_confidence_explicit_not_overwritten(self) -> None:
        """Explicit overall_confidence is preserved and not overwritten."""
        # overall_confidence only auto-computes when exactly 0.0
        # Providing a non-zero value prevents auto-computation
        report = make_report(
            arch_confidence=0.9,
            training_confidence=0.6,
            cap_confidence=0.3,
            overall_confidence=0.75,
        )
        assert report.overall_confidence == pytest.approx(0.75, abs=1e-3)

    def test_all_hypotheses_returns_eleven(self) -> None:
        """all_hypotheses returns exactly 11 tuples."""
        report = make_report()
        hyps = report.all_hypotheses
        assert len(hyps) == 11

    def test_all_hypotheses_structure(self) -> None:
        """Each entry in all_hypotheses is a (str, str, DesignHypothesis) tuple."""
        report = make_report()
        for cat, aspect, hyp in report.all_hypotheses:
            assert isinstance(cat, str)
            assert isinstance(aspect, str)
            assert isinstance(hyp, DesignHypothesis)

    def test_all_hypotheses_categories(self) -> None:
        """all_hypotheses covers Architecture, Training, and Capabilities categories."""
        report = make_report()
        categories = {cat for cat, _, _ in report.all_hypotheses}
        assert categories == {"Architecture", "Training", "Capabilities"}

    def test_high_confidence_hypotheses_filter(self) -> None:
        """high_confidence_hypotheses returns only hypotheses with confidence >= 0.7."""
        report = make_report(arch_confidence=0.8, training_confidence=0.5, cap_confidence=0.3)
        high = report.high_confidence_hypotheses
        for _, _, hyp in high:
            assert hyp.confidence >= 0.7
        # Architecture has 5 hypotheses all at 0.8
        assert len(high) == 5

    def test_low_confidence_hypotheses_filter(self) -> None:
        """low_confidence_hypotheses returns only hypotheses with confidence < 0.4."""
        report = make_report(arch_confidence=0.8, training_confidence=0.5, cap_confidence=0.2)
        low = report.low_confidence_hypotheses
        for _, _, hyp in low:
            assert hyp.confidence < 0.4
        # Capabilities has 3 hypotheses all at 0.2
        assert len(low) == 3

    def test_summary_table_returns_eleven_rows(self) -> None:
        """summary_table returns exactly 11 rows."""
        report = make_report()
        table = report.summary_table()
        assert len(table) == 11

    def test_summary_table_row_keys(self) -> None:
        """Each summary_table row has the expected keys."""
        report = make_report()
        for row in report.summary_table():
            assert "category" in row
            assert "aspect" in row
            assert "confidence" in row
            assert "confidence_label" in row
            assert "hypothesis_preview" in row

    def test_summary_table_hypothesis_preview_truncated(self) -> None:
        """Hypothesis longer than 80 chars is truncated in summary_table."""
        long_text = "A" * 100
        report = ArchitectureReport(
            model_name="TestModel",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=CapabilitySource(
                emergent_behaviors=make_hypothesis(hypothesis=long_text),
                fine_tuning_approach=make_hypothesis(),
                efficiency_optimizations=make_hypothesis(),
            ),
        )
        table = report.summary_table()
        emergent_row = next(
            r for r in table if r["aspect"] == "Emergent Behaviors"
        )
        assert emergent_row["hypothesis_preview"].endswith("...")

    def test_sources_empty_strings_filtered(self) -> None:
        """Empty strings are removed from sources_analyzed."""
        report = ArchitectureReport(
            model_name="Test",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
            sources_analyzed=["", "  ", "https://example.com"],
        )
        assert report.sources_analyzed == ["https://example.com"]

    def test_additional_notes_default_empty(self) -> None:
        """additional_notes defaults to empty string."""
        report = make_report()
        assert report.additional_notes == ""

    def test_model_dump_is_json_serializable(self) -> None:
        """model_dump(mode='json') returns a JSON-serializable dict."""
        import json

        report = make_report()
        data = report.model_dump(mode="json")
        # Should not raise
        json_str = json.dumps(data)
        assert "TestModel" in json_str
        assert "executive_summary" in json_str

    def test_report_round_trips_through_dict(self) -> None:
        """An ArchitectureReport can be round-tripped through dict serialization."""
        original = make_report()
        data = original.model_dump()
        restored = ArchitectureReport.model_validate(data)
        assert restored.model_name == original.model_name
        assert restored.overall_confidence == original.overall_confidence
        assert (
            restored.architecture.attention_mechanism.hypothesis
            == original.architecture.attention_mechanism.hypothesis
        )

    def test_overall_confidence_ge_zero_le_one(self) -> None:
        """overall_confidence value must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            make_report(overall_confidence=1.5)
        with pytest.raises(ValidationError):
            make_report(overall_confidence=-0.1)


# ---------------------------------------------------------------------------
# IngestionMetadata tests
# ---------------------------------------------------------------------------


class TestIngestionMetadata:
    """Tests for the IngestionMetadata model."""

    def test_valid_creation(self) -> None:
        """IngestionMetadata is created with all fields."""
        meta = IngestionMetadata(
            source="https://example.com/paper",
            content_type="text/html",
            title="Example Paper",
            char_count=5000,
            chunk_count=3,
        )
        assert meta.source == "https://example.com/paper"
        assert meta.content_type == "text/html"
        assert meta.title == "Example Paper"
        assert meta.char_count == 5000
        assert meta.chunk_count == 3

    def test_default_values(self) -> None:
        """IngestionMetadata uses correct defaults."""
        meta = IngestionMetadata(source="/path/to/file.txt")
        assert meta.content_type == "unknown"
        assert meta.title is None
        assert meta.char_count == 0
        assert meta.chunk_count == 0

    def test_empty_source_rejected(self) -> None:
        """Empty source raises ValidationError."""
        with pytest.raises(ValidationError):
            IngestionMetadata(source="")

    def test_whitespace_source_rejected(self) -> None:
        """Whitespace-only source raises ValidationError."""
        with pytest.raises(ValidationError):
            IngestionMetadata(source="   ")

    def test_negative_char_count_rejected(self) -> None:
        """Negative char_count raises ValidationError."""
        with pytest.raises(ValidationError):
            IngestionMetadata(source="test", char_count=-1)

    def test_negative_chunk_count_rejected(self) -> None:
        """Negative chunk_count raises ValidationError."""
        with pytest.raises(ValidationError):
            IngestionMetadata(source="test", chunk_count=-1)

    def test_is_pdf_true(self) -> None:
        """is_pdf returns True for application/pdf content type."""
        meta = IngestionMetadata(
            source="paper.pdf",
            content_type="application/pdf",
        )
        assert meta.is_pdf is True

    def test_is_pdf_false(self) -> None:
        """is_pdf returns False for non-PDF content types."""
        meta = IngestionMetadata(
            source="https://example.com",
            content_type="text/html",
        )
        assert meta.is_pdf is False

    def test_display_name_uses_title_when_present(self) -> None:
        """display_name returns title when available."""
        meta = IngestionMetadata(
            source="https://example.com/long-url",
            title="Short Title",
        )
        assert meta.display_name == "Short Title"

    def test_display_name_falls_back_to_source(self) -> None:
        """display_name falls back to source when title is None."""
        meta = IngestionMetadata(source="https://example.com/long-url")
        assert meta.display_name == "https://example.com/long-url"
