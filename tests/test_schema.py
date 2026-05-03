"""Unit tests for Pydantic schema models in model_archaeologist/schema.py.

Verifies that all models parse correctly, enforce validation rules,
reject invalid data with clear errors, and expose correct computed properties.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
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
# Fixtures / helpers
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
        evidence_quotes=evidence_quotes if evidence_quotes is not None else ["The model employs multi-head attention."],
        open_questions=open_questions if open_questions is not None else ["How many heads?"],
    )


def make_architecture(confidence: float = 0.7) -> ArchitectureDetails:
    """Build an ArchitectureDetails object with uniform hypothesis confidence."""
    h = make_hypothesis(confidence=confidence)
    return ArchitectureDetails(
        attention_mechanism=h,
        model_size=h,
        positional_encoding=h,
        normalization=h,
        moe_usage=h,
    )


def make_training(confidence: float = 0.6) -> TrainingParadigm:
    """Build a TrainingParadigm object with uniform hypothesis confidence."""
    h = make_hypothesis(confidence=confidence)
    return TrainingParadigm(
        data_curation=h,
        alignment_technique=h,
        scaling_strategy=h,
    )


def make_capabilities(confidence: float = 0.5) -> CapabilitySource:
    """Build a CapabilitySource object with uniform hypothesis confidence."""
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
        sources_analyzed=sources if sources is not None else ["https://example.com/paper"],
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
        with pytest.raises(ValidationError):
            DesignHypothesis(hypothesis="Bad", confidence=-0.1)

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

    def test_empty_strings_filtered_from_evidence_quotes(self) -> None:
        """Empty strings are removed from evidence_quotes."""
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.5,
            evidence_quotes=["", "  ", "valid quote"],
        )
        assert h.evidence_quotes == ["valid quote"]

    def test_empty_strings_filtered_from_open_questions(self) -> None:
        """Empty strings are removed from open_questions."""
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.5,
            open_questions=["valid question", ""],
        )
        assert h.open_questions == ["valid question"]

    def test_empty_strings_filtered_from_both_lists(self) -> None:
        """Empty strings are removed from both list fields simultaneously."""
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

    def test_confidence_at_0_4_is_medium(self) -> None:
        """Confidence exactly at 0.4 is 'medium', not 'low'."""
        h = DesignHypothesis(hypothesis="Boundary", confidence=0.4)
        assert h.confidence_label == "medium"

    def test_confidence_at_0_7_is_high(self) -> None:
        """Confidence exactly at 0.7 is 'high', not 'medium'."""
        h = DesignHypothesis(hypothesis="Boundary", confidence=0.7)
        assert h.confidence_label == "high"

    def test_non_list_evidence_quotes_becomes_empty_list(self) -> None:
        """Non-list value for evidence_quotes is coerced to empty list."""
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.5,
            evidence_quotes=None,  # type: ignore[arg-type]
        )
        assert h.evidence_quotes == []

    def test_non_list_open_questions_becomes_empty_list(self) -> None:
        """Non-list value for open_questions is coerced to empty list."""
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.5,
            open_questions=None,  # type: ignore[arg-type]
        )
        assert h.open_questions == []

    def test_confidence_0_999_rounds_to_three_decimals(self) -> None:
        """Confidence 0.9999 rounds to 1.0 after 3 decimal rounding."""
        h = DesignHypothesis(hypothesis="Test", confidence=0.9994)
        assert h.confidence == 0.999

    def test_hypothesis_with_special_characters(self) -> None:
        """Hypothesis containing special characters is stored correctly."""
        text = "Uses RoPE (Rotary Position Embedding) with \u03b8=10000."
        h = DesignHypothesis(hypothesis=text, confidence=0.8)
        assert h.hypothesis == text

    def test_multiple_evidence_quotes_preserved(self) -> None:
        """Multiple valid evidence quotes are all preserved."""
        quotes = ["Quote one.", "Quote two.", "Quote three."]
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.6,
            evidence_quotes=quotes,
        )
        assert len(h.evidence_quotes) == 3
        for q in quotes:
            assert q in h.evidence_quotes

    def test_multiple_open_questions_preserved(self) -> None:
        """Multiple valid open questions are all preserved."""
        questions = ["Question A?", "Question B?", "Question C?"]
        h = DesignHypothesis(
            hypothesis="Test",
            confidence=0.6,
            open_questions=questions,
        )
        assert len(h.open_questions) == 3


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

    def test_mean_confidence_uniform(self) -> None:
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

    def test_as_dict_key_names(self) -> None:
        """as_dict contains the expected human-readable keys."""
        arch = make_architecture()
        d = arch.as_dict()
        assert "Attention Mechanism" in d
        assert "Model Size & Structure" in d
        assert "Positional Encoding" in d
        assert "Normalization Strategy" in d
        assert "Mixture-of-Experts" in d

    def test_as_dict_values_are_design_hypothesis(self) -> None:
        """as_dict values are all DesignHypothesis instances."""
        arch = make_architecture()
        for key, val in arch.as_dict().items():
            assert isinstance(val, DesignHypothesis), f"Expected DesignHypothesis for key '{key}'"

    def test_missing_attention_mechanism_raises_error(self) -> None:
        """Missing attention_mechanism raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            ArchitectureDetails(
                model_size=h,
                positional_encoding=h,
                normalization=h,
                moe_usage=h,
            )  # type: ignore[call-arg]

    def test_missing_model_size_raises_error(self) -> None:
        """Missing model_size raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            ArchitectureDetails(
                attention_mechanism=h,
                positional_encoding=h,
                normalization=h,
                moe_usage=h,
            )  # type: ignore[call-arg]

    def test_missing_positional_encoding_raises_error(self) -> None:
        """Missing positional_encoding raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            ArchitectureDetails(
                attention_mechanism=h,
                model_size=h,
                normalization=h,
                moe_usage=h,
            )  # type: ignore[call-arg]

    def test_missing_normalization_raises_error(self) -> None:
        """Missing normalization raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            ArchitectureDetails(
                attention_mechanism=h,
                model_size=h,
                positional_encoding=h,
                moe_usage=h,
            )  # type: ignore[call-arg]

    def test_missing_moe_usage_raises_error(self) -> None:
        """Missing moe_usage raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            ArchitectureDetails(
                attention_mechanism=h,
                model_size=h,
                positional_encoding=h,
                normalization=h,
            )  # type: ignore[call-arg]

    def test_mean_confidence_rounded_to_three_decimals(self) -> None:
        """mean_confidence is rounded to 3 decimal places."""
        arch = make_architecture(confidence=0.333333)
        # 0.333 rounded to 3 places
        assert isinstance(arch.mean_confidence, float)
        assert len(str(arch.mean_confidence).split(".")[1]) <= 3 or arch.mean_confidence == round(arch.mean_confidence, 3)


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

    def test_mean_confidence_uniform(self) -> None:
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

    def test_as_dict_key_names(self) -> None:
        """as_dict contains the expected human-readable keys."""
        t = make_training()
        d = t.as_dict()
        assert "Data Curation" in d
        assert "Alignment Technique" in d
        assert "Scaling Strategy" in d

    def test_as_dict_values_are_design_hypothesis(self) -> None:
        """as_dict values are all DesignHypothesis instances."""
        t = make_training()
        for key, val in t.as_dict().items():
            assert isinstance(val, DesignHypothesis), f"Expected DesignHypothesis for key '{key}'"

    def test_missing_data_curation_raises_error(self) -> None:
        """Missing data_curation raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            TrainingParadigm(
                alignment_technique=h,
                scaling_strategy=h,
            )  # type: ignore[call-arg]

    def test_missing_alignment_technique_raises_error(self) -> None:
        """Missing alignment_technique raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            TrainingParadigm(
                data_curation=h,
                scaling_strategy=h,
            )  # type: ignore[call-arg]

    def test_missing_scaling_strategy_raises_error(self) -> None:
        """Missing scaling_strategy raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            TrainingParadigm(
                data_curation=h,
                alignment_technique=h,
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

    def test_mean_confidence_uniform(self) -> None:
        """mean_confidence computes the mean of all 3 hypothesis confidences."""
        c = make_capabilities(confidence=0.5)
        assert c.mean_confidence == pytest.approx(0.5, abs=1e-3)

    def test_mean_confidence_mixed(self) -> None:
        """mean_confidence with mixed confidences computes correctly."""
        h_a = make_hypothesis(confidence=0.8)
        h_b = make_hypothesis(confidence=0.2)
        h_c = make_hypothesis(confidence=0.5)
        c = CapabilitySource(
            emergent_behaviors=h_a,
            fine_tuning_approach=h_b,
            efficiency_optimizations=h_c,
        )
        # (0.8 + 0.2 + 0.5) / 3 = 0.5
        assert c.mean_confidence == pytest.approx(0.5, abs=1e-3)

    def test_as_dict_returns_three_keys(self) -> None:
        """as_dict returns a dict with exactly 3 keys."""
        c = make_capabilities()
        d = c.as_dict()
        assert len(d) == 3

    def test_as_dict_key_names(self) -> None:
        """as_dict contains the expected human-readable keys."""
        c = make_capabilities()
        d = c.as_dict()
        assert "Emergent Behaviors" in d
        assert "Fine-tuning Approach" in d
        assert "Efficiency Optimizations" in d

    def test_as_dict_values_are_design_hypothesis(self) -> None:
        """as_dict values are all DesignHypothesis instances."""
        c = make_capabilities()
        for key, val in c.as_dict().items():
            assert isinstance(val, DesignHypothesis), f"Expected DesignHypothesis for key '{key}'"

    def test_missing_emergent_behaviors_raises_error(self) -> None:
        """Missing emergent_behaviors raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            CapabilitySource(
                fine_tuning_approach=h,
                efficiency_optimizations=h,
            )  # type: ignore[call-arg]

    def test_missing_fine_tuning_approach_raises_error(self) -> None:
        """Missing fine_tuning_approach raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            CapabilitySource(
                emergent_behaviors=h,
                efficiency_optimizations=h,
            )  # type: ignore[call-arg]

    def test_missing_efficiency_optimizations_raises_error(self) -> None:
        """Missing efficiency_optimizations raises ValidationError."""
        h = make_hypothesis()
        with pytest.raises(ValidationError):
            CapabilitySource(
                emergent_behaviors=h,
                fine_tuning_approach=h,
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
        """Explicit non-zero overall_confidence is preserved and not overwritten."""
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

    def test_all_hypotheses_architecture_count(self) -> None:
        """Exactly 5 hypotheses belong to the Architecture category."""
        report = make_report()
        arch_hyps = [h for cat, _, h in report.all_hypotheses if cat == "Architecture"]
        assert len(arch_hyps) == 5

    def test_all_hypotheses_training_count(self) -> None:
        """Exactly 3 hypotheses belong to the Training category."""
        report = make_report()
        train_hyps = [h for cat, _, h in report.all_hypotheses if cat == "Training"]
        assert len(train_hyps) == 3

    def test_all_hypotheses_capabilities_count(self) -> None:
        """Exactly 3 hypotheses belong to the Capabilities category."""
        report = make_report()
        cap_hyps = [h for cat, _, h in report.all_hypotheses if cat == "Capabilities"]
        assert len(cap_hyps) == 3

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

    def test_high_confidence_hypotheses_empty_when_all_low(self) -> None:
        """high_confidence_hypotheses is empty when all confidences are below 0.7."""
        report = make_report(arch_confidence=0.3, training_confidence=0.3, cap_confidence=0.3)
        assert len(report.high_confidence_hypotheses) == 0

    def test_low_confidence_hypotheses_empty_when_all_high(self) -> None:
        """low_confidence_hypotheses is empty when all confidences are >= 0.4."""
        report = make_report(arch_confidence=0.9, training_confidence=0.8, cap_confidence=0.7)
        assert len(report.low_confidence_hypotheses) == 0

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
        """Hypothesis longer than 80 chars is truncated with '...' in summary_table."""
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
        assert len(emergent_row["hypothesis_preview"]) <= 83  # 80 + len("...")

    def test_summary_table_hypothesis_preview_not_truncated_for_short(self) -> None:
        """Hypothesis <= 80 chars is not truncated in summary_table."""
        short_text = "Short hypothesis."
        report = ArchitectureReport(
            model_name="TestModel",
            executive_summary="Summary",
            architecture=ArchitectureDetails(
                attention_mechanism=make_hypothesis(hypothesis=short_text),
                model_size=make_hypothesis(),
                positional_encoding=make_hypothesis(),
                normalization=make_hypothesis(),
                moe_usage=make_hypothesis(),
            ),
            training=make_training(),
            capabilities=make_capabilities(),
        )
        table = report.summary_table()
        attn_row = next(r for r in table if r["aspect"] == "Attention Mechanism")
        assert not attn_row["hypothesis_preview"].endswith("...")
        assert attn_row["hypothesis_preview"] == short_text

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

    def test_sources_analyzed_defaults_to_empty_list(self) -> None:
        """sources_analyzed defaults to an empty list when not provided."""
        report = ArchitectureReport(
            model_name="Test",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
        )
        assert report.sources_analyzed == []

    def test_additional_notes_default_empty(self) -> None:
        """additional_notes defaults to empty string."""
        report = make_report()
        assert report.additional_notes == ""

    def test_additional_notes_can_be_set(self) -> None:
        """additional_notes can be explicitly set."""
        report = ArchitectureReport(
            model_name="Test",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
            additional_notes="Some additional commentary.",
        )
        assert report.additional_notes == "Some additional commentary."

    def test_model_dump_is_json_serializable(self) -> None:
        """model_dump(mode='json') returns a JSON-serializable dict."""
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

    def test_report_round_trips_through_json(self) -> None:
        """An ArchitectureReport can be round-tripped through JSON serialization."""
        original = make_report()
        json_data = json.loads(json.dumps(original.model_dump(mode="json")))
        restored = ArchitectureReport.model_validate(json_data)
        assert restored.model_name == original.model_name

    def test_overall_confidence_ge_zero_le_one_above(self) -> None:
        """overall_confidence value above 1.0 is rejected."""
        with pytest.raises(ValidationError):
            make_report(overall_confidence=1.5)

    def test_overall_confidence_ge_zero_le_one_below(self) -> None:
        """overall_confidence value below 0.0 is rejected."""
        with pytest.raises(ValidationError):
            make_report(overall_confidence=-0.1)

    def test_overall_confidence_rounded(self) -> None:
        """overall_confidence is rounded to 3 decimal places."""
        report = ArchitectureReport(
            model_name="Test",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
            overall_confidence=0.123456,
        )
        assert report.overall_confidence == 0.123

    def test_sources_analyzed_non_list_becomes_empty(self) -> None:
        """Non-list value for sources_analyzed is coerced to empty list."""
        report = ArchitectureReport(
            model_name="Test",
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
            sources_analyzed=None,  # type: ignore[arg-type]
        )
        assert report.sources_analyzed == []

    def test_missing_executive_summary_raises_error(self) -> None:
        """Missing executive_summary raises ValidationError."""
        with pytest.raises(ValidationError):
            ArchitectureReport(
                model_name="Test",
                architecture=make_architecture(),
                training=make_training(),
                capabilities=make_capabilities(),
            )  # type: ignore[call-arg]

    def test_missing_architecture_raises_error(self) -> None:
        """Missing architecture raises ValidationError."""
        with pytest.raises(ValidationError):
            ArchitectureReport(
                model_name="Test",
                executive_summary="Summary",
                training=make_training(),
                capabilities=make_capabilities(),
            )  # type: ignore[call-arg]

    def test_missing_training_raises_error(self) -> None:
        """Missing training raises ValidationError."""
        with pytest.raises(ValidationError):
            ArchitectureReport(
                model_name="Test",
                executive_summary="Summary",
                architecture=make_architecture(),
                capabilities=make_capabilities(),
            )  # type: ignore[call-arg]

    def test_missing_capabilities_raises_error(self) -> None:
        """Missing capabilities raises ValidationError."""
        with pytest.raises(ValidationError):
            ArchitectureReport(
                model_name="Test",
                executive_summary="Summary",
                architecture=make_architecture(),
                training=make_training(),
            )  # type: ignore[call-arg]

    def test_all_hypotheses_aspect_names(self) -> None:
        """all_hypotheses contains the expected aspect names."""
        report = make_report()
        aspects = {aspect for _, aspect, _ in report.all_hypotheses}
        expected_aspects = {
            "Attention Mechanism",
            "Model Size & Structure",
            "Positional Encoding",
            "Normalization Strategy",
            "Mixture-of-Experts",
            "Data Curation",
            "Alignment Technique",
            "Scaling Strategy",
            "Emergent Behaviors",
            "Fine-tuning Approach",
            "Efficiency Optimizations",
        }
        assert aspects == expected_aspects

    def test_generated_at_has_timezone(self) -> None:
        """generated_at always has timezone info (UTC)."""
        report = make_report()
        assert report.generated_at.tzinfo is not None

    def test_explicit_generated_at_preserved(self) -> None:
        """An explicitly provided generated_at is preserved."""
        fixed_time = datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        report = ArchitectureReport(
            model_name="Test",
            generated_at=fixed_time,
            executive_summary="Summary",
            architecture=make_architecture(),
            training=make_training(),
            capabilities=make_capabilities(),
        )
        assert report.generated_at == fixed_time

    def test_summary_table_confidence_format(self) -> None:
        """summary_table confidence values are formatted as '0.XX' strings."""
        report = make_report(arch_confidence=0.85)
        table = report.summary_table()
        for row in table:
            assert isinstance(row["confidence"], str)
            # Should contain a dot (decimal point)
            assert "." in row["confidence"]

    def test_summary_table_confidence_label_values(self) -> None:
        """summary_table confidence_label is one of 'high', 'medium', 'low'."""
        report = make_report()
        for row in report.summary_table():
            assert row["confidence_label"] in ("high", "medium", "low")

    def test_overall_confidence_auto_compute_uses_all_11_fields(self) -> None:
        """Auto-computed overall_confidence uses all 11 hypothesis fields."""
        # Create report with all different confidence values
        h_arch = make_hypothesis(confidence=1.0)
        h_train = make_hypothesis(confidence=0.0)
        h_cap = make_hypothesis(confidence=0.5)

        report = ArchitectureReport(
            model_name="Test",
            executive_summary="Summary",
            architecture=ArchitectureDetails(
                attention_mechanism=h_arch,
                model_size=h_arch,
                positional_encoding=h_arch,
                normalization=h_arch,
                moe_usage=h_arch,
            ),
            training=TrainingParadigm(
                data_curation=h_train,
                alignment_technique=h_train,
                scaling_strategy=h_train,
            ),
            capabilities=CapabilitySource(
                emergent_behaviors=h_cap,
                fine_tuning_approach=h_cap,
                efficiency_optimizations=h_cap,
            ),
        )
        # 5 * 1.0 + 3 * 0.0 + 3 * 0.5 = 5.0 + 0.0 + 1.5 = 6.5
        # 6.5 / 11 ≈ 0.5909...
        expected = round(6.5 / 11, 3)
        assert report.overall_confidence == pytest.approx(expected, abs=1e-3)


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

    def test_zero_char_count_allowed(self) -> None:
        """char_count of 0 is valid."""
        meta = IngestionMetadata(source="test", char_count=0)
        assert meta.char_count == 0

    def test_zero_chunk_count_allowed(self) -> None:
        """chunk_count of 0 is valid."""
        meta = IngestionMetadata(source="test", chunk_count=0)
        assert meta.chunk_count == 0

    def test_is_pdf_true(self) -> None:
        """is_pdf returns True for application/pdf content type."""
        meta = IngestionMetadata(
            source="paper.pdf",
            content_type="application/pdf",
        )
        assert meta.is_pdf is True

    def test_is_pdf_false_html(self) -> None:
        """is_pdf returns False for text/html content type."""
        meta = IngestionMetadata(
            source="https://example.com",
            content_type="text/html",
        )
        assert meta.is_pdf is False

    def test_is_pdf_false_plain_text(self) -> None:
        """is_pdf returns False for text/plain content type."""
        meta = IngestionMetadata(
            source="file.txt",
            content_type="text/plain",
        )
        assert meta.is_pdf is False

    def test_is_pdf_false_unknown(self) -> None:
        """is_pdf returns False for unknown content type."""
        meta = IngestionMetadata(
            source="file",
            content_type="unknown",
        )
        assert meta.is_pdf is False

    def test_is_pdf_case_insensitive(self) -> None:
        """is_pdf uses case-insensitive comparison."""
        meta = IngestionMetadata(
            source="paper.pdf",
            content_type="APPLICATION/PDF",
        )
        assert meta.is_pdf is True

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
        assert meta.title is None

    def test_source_stripped_of_whitespace(self) -> None:
        """Source field has leading/trailing whitespace stripped."""
        meta = IngestionMetadata(source="  https://example.com  ")
        assert meta.source == "https://example.com"

    def test_large_char_count_allowed(self) -> None:
        """Large char_count values are allowed."""
        meta = IngestionMetadata(source="large.txt", char_count=10_000_000)
        assert meta.char_count == 10_000_000

    def test_large_chunk_count_allowed(self) -> None:
        """Large chunk_count values are allowed."""
        meta = IngestionMetadata(source="large.txt", chunk_count=500)
        assert meta.chunk_count == 500

    def test_title_none_by_default(self) -> None:
        """title is None by default."""
        meta = IngestionMetadata(source="file.txt")
        assert meta.title is None

    def test_display_name_returns_string(self) -> None:
        """display_name always returns a string."""
        meta = IngestionMetadata(source="test")
        assert isinstance(meta.display_name, str)

    def test_model_dump_serializable(self) -> None:
        """IngestionMetadata can be serialized to a JSON-compatible dict."""
        meta = IngestionMetadata(
            source="https://example.com",
            content_type="text/html",
            title="Test",
            char_count=100,
            chunk_count=2,
        )
        data = meta.model_dump()
        json_str = json.dumps(data)
        assert "example.com" in json_str

    def test_round_trip_through_dict(self) -> None:
        """IngestionMetadata round-trips through dict serialization."""
        original = IngestionMetadata(
            source="https://example.com/paper.pdf",
            content_type="application/pdf",
            title="Great Paper",
            char_count=12345,
            chunk_count=7,
        )
        data = original.model_dump()
        restored = IngestionMetadata.model_validate(data)
        assert restored.source == original.source
        assert restored.content_type == original.content_type
        assert restored.title == original.title
        assert restored.char_count == original.char_count
        assert restored.chunk_count == original.chunk_count
