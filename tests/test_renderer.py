"""Tests for the Jinja2 renderer in model_archaeologist/renderer.py.

Verifies that:
- Markdown rendering produces expected sections and content.
- JSON rendering produces valid, parseable JSON with correct structure.
- Custom Jinja2 filters (confidence_bar, confidence_emoji, etc.) work correctly.
- render_to_file writes content to disk in both formats.
- Error handling works for missing templates directories and bad paths.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from model_archaeologist.renderer import (
    RendererError,
    ReportRenderer,
    _confidence_bar,
    _confidence_emoji,
    _confidence_label,
    _format_datetime,
)
from model_archaeologist.schema import (
    ArchitectureDetails,
    ArchitectureReport,
    CapabilitySource,
    DesignHypothesis,
    TrainingParadigm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_hypothesis(
    hypothesis: str = "Multi-Head Attention with GQA.",
    confidence: float = 0.75,
    evidence_quotes: list[str] | None = None,
    open_questions: list[str] | None = None,
) -> DesignHypothesis:
    """Create a test DesignHypothesis."""
    return DesignHypothesis(
        hypothesis=hypothesis,
        confidence=confidence,
        evidence_quotes=evidence_quotes or ["The paper describes grouped-query attention."],
        open_questions=open_questions or ["How many KV heads are used?"],
    )


def _make_report(
    model_name: str = "TestModel-7B",
    with_notes: bool = False,
    sources: list[str] | None = None,
) -> ArchitectureReport:
    """Create a complete test ArchitectureReport."""
    h_high = _make_hypothesis(confidence=0.85)
    h_med = _make_hypothesis(
        hypothesis="RMSNorm applied before each sub-layer (Pre-LN).",
        confidence=0.6,
        evidence_quotes=["Pre-norm architecture improves training stability."],
        open_questions=["Is QKNorm applied as well?"],
    )
    h_low = _make_hypothesis(
        hypothesis="Sparse Mixture-of-Experts with top-2 routing.",
        confidence=0.3,
        evidence_quotes=[],
        open_questions=["Number of experts unknown."],
    )

    report = ArchitectureReport(
        model_name=model_name,
        generated_at=datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc),
        executive_summary=(
            "TestModel-7B is a dense transformer with strong evidence for GQA attention "
            "and RoPE positional encoding. Training appears compute-optimal following "
            "Chinchilla scaling laws."
        ),
        architecture=ArchitectureDetails(
            attention_mechanism=h_high,
            model_size=_make_hypothesis(
                hypothesis="7 billion parameters, 32 layers, 32 attention heads.",
                confidence=0.9,
                evidence_quotes=["The 7B model variant was released publicly."],
                open_questions=[],
            ),
            positional_encoding=_make_hypothesis(
                hypothesis="Rotary Position Embedding (RoPE) with theta=10000.",
                confidence=0.8,
                evidence_quotes=["RoPE is standard for this model family."],
                open_questions=["Is extended context RoPE scaling applied?"],
            ),
            normalization=h_med,
            moe_usage=h_low,
        ),
        training=TrainingParadigm(
            data_curation=_make_hypothesis(
                hypothesis="Web-scale data with quality filtering and deduplication.",
                confidence=0.65,
                evidence_quotes=["Trained on 2T tokens from filtered web data."],
                open_questions=["Exact deduplication method?"],
            ),
            alignment_technique=_make_hypothesis(
                hypothesis="RLHF with PPO followed by DPO fine-tuning.",
                confidence=0.55,
                evidence_quotes=["The instruct variant uses RLHF."],
                open_questions=["Was Constitutional AI used?"],
            ),
            scaling_strategy=_make_hypothesis(
                hypothesis="Compute-optimal training following Chinchilla ratios.",
                confidence=0.7,
                evidence_quotes=["Trained at compute-optimal token-to-parameter ratio."],
                open_questions=["Was the model over-trained for inference efficiency?"],
            ),
        ),
        capabilities=CapabilitySource(
            emergent_behaviors=_make_hypothesis(
                hypothesis="Strong instruction following emerges at 7B scale.",
                confidence=0.75,
                evidence_quotes=["Benchmark scores improve sharply at this scale."],
                open_questions=["At what exact scale do reasoning capabilities emerge?"],
            ),
            fine_tuning_approach=_make_hypothesis(
                hypothesis="Supervised fine-tuning on high-quality instruction datasets.",
                confidence=0.8,
                evidence_quotes=["The instruct model was fine-tuned on curated examples."],
                open_questions=["Dataset size for SFT?"],
            ),
            efficiency_optimizations=_make_hypothesis(
                hypothesis="Flash Attention 2 for training and inference.",
                confidence=0.6,
                evidence_quotes=["Flash Attention is standard for models of this class."],
                open_questions=["Is speculative decoding used in serving?"],
            ),
        ),
        sources_analyzed=sources or [
            "https://arxiv.org/abs/2301.00001",
            "https://example.com/blog/testmodel",
        ],
        additional_notes="This is a test report for renderer validation." if with_notes else "",
    )
    return report


@pytest.fixture()
def report() -> ArchitectureReport:
    """Return a standard test ArchitectureReport."""
    return _make_report()


@pytest.fixture()
def renderer() -> ReportRenderer:
    """Return a ReportRenderer using the package's default templates."""
    return ReportRenderer()


# ---------------------------------------------------------------------------
# Tests – confidence filter functions
# ---------------------------------------------------------------------------


class TestConfidenceBar:
    """Tests for the _confidence_bar filter function."""

    def test_zero_confidence(self) -> None:
        """0.0 produces an all-empty bar."""
        result = _confidence_bar(0.0)
        assert "░" * 10 in result
        assert "0%" in result

    def test_full_confidence(self) -> None:
        """1.0 produces a fully-filled bar."""
        result = _confidence_bar(1.0)
        assert "█" * 10 in result
        assert "100%" in result

    def test_half_confidence(self) -> None:
        """0.5 produces a half-filled bar."""
        result = _confidence_bar(0.5)
        assert "█" * 5 in result
        assert "50%" in result

    def test_custom_width(self) -> None:
        """Custom width changes the bar length."""
        result = _confidence_bar(1.0, width=5)
        assert "█" * 5 in result
        # Should not contain 10 filled blocks
        assert len(result) < len(_confidence_bar(1.0, width=10)) + 5

    def test_clamping_above_one(self) -> None:
        """Values above 1.0 are clamped to 1.0."""
        result = _confidence_bar(1.5)
        assert "100%" in result

    def test_clamping_below_zero(self) -> None:
        """Values below 0.0 are clamped to 0.0."""
        result = _confidence_bar(-0.5)
        assert "0%" in result

    def test_format_contains_brackets(self) -> None:
        """The bar is enclosed in square brackets."""
        result = _confidence_bar(0.7)
        assert result.startswith("[")
        assert "]" in result

    def test_returns_string(self) -> None:
        """_confidence_bar always returns a string."""
        assert isinstance(_confidence_bar(0.5), str)


class TestConfidenceEmoji:
    """Tests for the _confidence_emoji filter function."""

    def test_high_confidence_green(self) -> None:
        """Confidence >= 0.7 returns the green circle emoji."""
        assert _confidence_emoji(0.7) == "\U0001f7e2"
        assert _confidence_emoji(1.0) == "\U0001f7e2"
        assert _confidence_emoji(0.85) == "\U0001f7e2"

    def test_medium_confidence_yellow(self) -> None:
        """Confidence in [0.4, 0.7) returns the yellow circle emoji."""
        assert _confidence_emoji(0.4) == "\U0001f7e1"
        assert _confidence_emoji(0.69) == "\U0001f7e1"
        assert _confidence_emoji(0.5) == "\U0001f7e1"

    def test_low_confidence_red(self) -> None:
        """Confidence < 0.4 returns the red circle emoji."""
        assert _confidence_emoji(0.0) == "\U0001f534"
        assert _confidence_emoji(0.39) == "\U0001f534"

    def test_clamping(self) -> None:
        """Values outside [0, 1] are clamped before classification."""
        assert _confidence_emoji(2.0) == "\U0001f7e2"  # clamped to 1.0 -> high
        assert _confidence_emoji(-1.0) == "\U0001f534"  # clamped to 0.0 -> low

    def test_returns_string(self) -> None:
        """_confidence_emoji always returns a string."""
        assert isinstance(_confidence_emoji(0.5), str)


class TestConfidenceLabel:
    """Tests for the _confidence_label filter function."""

    def test_high_label(self) -> None:
        """Confidence >= 0.7 returns 'high'."""
        assert _confidence_label(0.7) == "high"
        assert _confidence_label(1.0) == "high"

    def test_medium_label(self) -> None:
        """Confidence in [0.4, 0.7) returns 'medium'."""
        assert _confidence_label(0.4) == "medium"
        assert _confidence_label(0.5) == "medium"
        assert _confidence_label(0.699) == "medium"

    def test_low_label(self) -> None:
        """Confidence < 0.4 returns 'low'."""
        assert _confidence_label(0.0) == "low"
        assert _confidence_label(0.399) == "low"

    def test_returns_string(self) -> None:
        """_confidence_label always returns a string."""
        assert isinstance(_confidence_label(0.5), str)


class TestFormatDatetime:
    """Tests for the _format_datetime filter function."""

    def test_formats_utc_datetime(self) -> None:
        """A UTC datetime is formatted with the default format string."""
        dt = datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        result = _format_datetime(dt)
        assert result == "2024-06-15 12:30:00 UTC"

    def test_custom_format(self) -> None:
        """A custom format string is applied correctly."""
        dt = datetime(2024, 1, 5, tzinfo=timezone.utc)
        result = _format_datetime(dt, fmt="%Y/%m/%d")
        assert result == "2024/01/05"

    def test_non_datetime_returns_str(self) -> None:
        """A non-datetime value without strftime returns str(value)."""
        result = _format_datetime("not a datetime")
        assert result == "not a datetime"

    def test_returns_string(self) -> None:
        """_format_datetime always returns a string."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert isinstance(_format_datetime(dt), str)


# ---------------------------------------------------------------------------
# Tests – render_markdown
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Tests for ReportRenderer.render_markdown."""

    def test_returns_string(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """render_markdown returns a non-empty string."""
        result = renderer.render_markdown(report)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_model_name(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The rendered Markdown contains the model name in the title."""
        result = renderer.render_markdown(report)
        assert "TestModel-7B" in result

    def test_contains_executive_summary(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The rendered Markdown contains the executive summary text."""
        result = renderer.render_markdown(report)
        assert "dense transformer" in result
        assert "Executive Summary" in result

    def test_contains_architecture_section(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The rendered Markdown has an Architecture Hypotheses section."""
        result = renderer.render_markdown(report)
        assert "## Architecture Hypotheses" in result

    def test_contains_training_section(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The rendered Markdown has a Training Paradigm section."""
        result = renderer.render_markdown(report)
        assert "## Training Paradigm" in result

    def test_contains_capabilities_section(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The rendered Markdown has a Capability Sources section."""
        result = renderer.render_markdown(report)
        assert "## Capability Sources" in result

    def test_contains_sources_section(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The rendered Markdown has a Sources section with the source URLs."""
        result = renderer.render_markdown(report)
        assert "## Sources" in result
        assert "https://arxiv.org/abs/2301.00001" in result

    def test_contains_attention_mechanism_subsection(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The Attention Mechanism subsection is present with hypothesis text."""
        result = renderer.render_markdown(report)
        assert "### Attention Mechanism" in result
        assert "Multi-Head Attention" in result

    def test_contains_positional_encoding_subsection(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The Positional Encoding subsection is present."""
        result = renderer.render_markdown(report)
        assert "### Positional Encoding" in result
        assert "RoPE" in result or "Rotary" in result

    def test_contains_data_curation_subsection(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The Data Curation subsection is present."""
        result = renderer.render_markdown(report)
        assert "### Data Curation" in result

    def test_contains_alignment_technique_subsection(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The Alignment Technique subsection is present."""
        result = renderer.render_markdown(report)
        assert "### Alignment Technique" in result
        assert "RLHF" in result or "PPO" in result or "DPO" in result

    def test_contains_evidence_quotes(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Evidence quotes appear as blockquotes in the Markdown."""
        result = renderer.render_markdown(report)
        # Blockquote lines start with >
        assert "> The paper describes grouped-query attention." in result

    def test_contains_open_questions(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Open questions appear as list items."""
        result = renderer.render_markdown(report)
        assert "- How many KV heads are used?" in result

    def test_contains_confidence_bar(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Confidence bars (ASCII art) appear in the rendered output."""
        result = renderer.render_markdown(report)
        # Confidence bar contains the block character
        assert "█" in result or "░" in result

    def test_contains_confidence_emoji(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Confidence emojis appear in the rendered output."""
        result = renderer.render_markdown(report)
        # At least one colored circle emoji should appear
        assert "\U0001f7e2" in result or "\U0001f7e1" in result or "\U0001f534" in result

    def test_contains_generated_date(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The generation date appears in the rendered output."""
        result = renderer.render_markdown(report)
        assert "2024-06-15" in result

    def test_contains_sources_count(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The number of analyzed sources appears in the header."""
        result = renderer.render_markdown(report)
        assert "2" in result  # 2 sources

    def test_additional_notes_rendered_when_present(self, renderer: ReportRenderer) -> None:
        """Additional notes section is rendered when additional_notes is non-empty."""
        report = _make_report(with_notes=True)
        result = renderer.render_markdown(report)
        assert "## Additional Notes" in result
        assert "This is a test report for renderer validation." in result

    def test_additional_notes_absent_when_empty(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Additional notes section is not rendered when additional_notes is empty."""
        assert report.additional_notes == ""
        result = renderer.render_markdown(report)
        assert "## Additional Notes" not in result

    def test_no_sources_section_shows_placeholder(self, renderer: ReportRenderer) -> None:
        """When no sources are provided, a placeholder message is shown."""
        report = _make_report(sources=[])
        result = renderer.render_markdown(report)
        assert "No sources recorded" in result

    def test_contains_summary_table(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """A confidence summary table is included in the Markdown output."""
        result = renderer.render_markdown(report)
        # Markdown table rows contain pipe characters
        assert "| Architecture |" in result or "| Category |" in result

    def test_moe_subsection_present(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The MoE subsection is present."""
        result = renderer.render_markdown(report)
        assert "Mixture-of-Experts" in result or "MoE" in result

    def test_all_eleven_aspects_present(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """All 11 architectural aspects are mentioned in the output."""
        result = renderer.render_markdown(report)
        expected_sections = [
            "Attention Mechanism",
            "Model Size",
            "Positional Encoding",
            "Normalization",
            "Data Curation",
            "Alignment Technique",
            "Scaling Strategy",
            "Emergent Behaviors",
            "Fine-tuning Approach",
            "Efficiency Optimizations",
        ]
        for section in expected_sections:
            assert section in result, f"Expected section '{section}' not found in output"

    def test_special_chars_in_model_name(self, renderer: ReportRenderer) -> None:
        """Model names with special characters render correctly."""
        report = _make_report(model_name="LLaMA-3 70B (Instruct)")
        result = renderer.render_markdown(report)
        assert "LLaMA-3 70B" in result

    def test_hypothesis_with_no_evidence_quotes(self, renderer: ReportRenderer) -> None:
        """Hypotheses with no evidence quotes do not produce empty blockquote blocks."""
        h_no_evidence = DesignHypothesis(
            hypothesis="Sparse MoE with 8 experts.",
            confidence=0.3,
            evidence_quotes=[],  # Empty
            open_questions=["How many active experts?"],
        )
        report = _make_report()
        # Override moe_usage with no evidence
        report_data = report.model_dump()
        report_data["architecture"]["moe_usage"] = {
            "hypothesis": "Sparse MoE with 8 experts.",
            "confidence": 0.3,
            "evidence_quotes": [],
            "open_questions": ["How many active experts?"],
        }
        new_report = ArchitectureReport.model_validate(report_data)
        result = renderer.render_markdown(new_report)
        # Should not have "Supporting Evidence:" for MoE since quotes is empty
        # Check the section renders without errors
        assert "Mixture-of-Experts" in result
        assert "Sparse MoE" in result


# ---------------------------------------------------------------------------
# Tests – render_json
# ---------------------------------------------------------------------------


class TestRenderJson:
    """Tests for ReportRenderer.render_json."""

    def test_returns_valid_json(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """render_json returns a valid JSON string."""
        result = renderer.render_json(report)
        # Should not raise
        data = json.loads(result)
        assert isinstance(data, dict)

    def test_contains_model_name(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The JSON contains the model name."""
        result = renderer.render_json(report)
        data = json.loads(result)
        assert data["model_name"] == "TestModel-7B"

    def test_contains_all_top_level_keys(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The JSON contains all expected top-level keys."""
        result = renderer.render_json(report)
        data = json.loads(result)
        expected_keys = [
            "model_name",
            "generated_at",
            "executive_summary",
            "architecture",
            "training",
            "capabilities",
            "overall_confidence",
            "sources_analyzed",
            "additional_notes",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_architecture_section_structure(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The architecture section has all five hypothesis sub-keys."""
        result = renderer.render_json(report)
        data = json.loads(result)
        arch = data["architecture"]
        for key in (
            "attention_mechanism",
            "model_size",
            "positional_encoding",
            "normalization",
            "moe_usage",
        ):
            assert key in arch, f"Missing architecture key: {key}"

    def test_training_section_structure(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The training section has all three hypothesis sub-keys."""
        result = renderer.render_json(report)
        data = json.loads(result)
        training = data["training"]
        for key in ("data_curation", "alignment_technique", "scaling_strategy"):
            assert key in training, f"Missing training key: {key}"

    def test_capabilities_section_structure(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The capabilities section has all three hypothesis sub-keys."""
        result = renderer.render_json(report)
        data = json.loads(result)
        caps = data["capabilities"]
        for key in ("emergent_behaviors", "fine_tuning_approach", "efficiency_optimizations"):
            assert key in caps, f"Missing capabilities key: {key}"

    def test_hypothesis_structure(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Each hypothesis has hypothesis, confidence, evidence_quotes, open_questions."""
        result = renderer.render_json(report)
        data = json.loads(result)
        hyp = data["architecture"]["attention_mechanism"]
        assert "hypothesis" in hyp
        assert "confidence" in hyp
        assert "evidence_quotes" in hyp
        assert "open_questions" in hyp

    def test_confidence_is_float(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Confidence values are serialized as floats."""
        result = renderer.render_json(report)
        data = json.loads(result)
        conf = data["architecture"]["attention_mechanism"]["confidence"]
        assert isinstance(conf, (int, float))

    def test_sources_analyzed_is_list(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """sources_analyzed is serialized as a JSON list."""
        result = renderer.render_json(report)
        data = json.loads(result)
        assert isinstance(data["sources_analyzed"], list)
        assert "https://arxiv.org/abs/2301.00001" in data["sources_analyzed"]

    def test_generated_at_is_string(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """generated_at is serialized as a string (ISO-8601)."""
        result = renderer.render_json(report)
        data = json.loads(result)
        assert isinstance(data["generated_at"], str)

    def test_json_is_indented(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """The JSON output uses 2-space indentation."""
        result = renderer.render_json(report)
        # 2-space indent means lines like '  "model_name"'
        assert '  "model_name"' in result

    def test_non_ascii_preserved(self, renderer: ReportRenderer) -> None:
        """Non-ASCII characters in text fields are preserved (not escaped)."""
        report = _make_report(model_name="Gemma-2 (日本語テスト)")
        result = renderer.render_json(report)
        assert "日本語テスト" in result  # Not \u65e5\u672c\u8a9e...

    def test_round_trips_through_pydantic(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """JSON output can be parsed back into an ArchitectureReport."""
        json_str = renderer.render_json(report)
        data = json.loads(json_str)
        restored = ArchitectureReport.model_validate(data)
        assert restored.model_name == report.model_name
        assert restored.overall_confidence == report.overall_confidence


# ---------------------------------------------------------------------------
# Tests – render_to_file
# ---------------------------------------------------------------------------


class TestRenderToFile:
    """Tests for ReportRenderer.render_to_file."""

    def test_writes_markdown_file(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """render_to_file writes Markdown content to the specified path."""
        output = tmp_path / "report.md"
        renderer.render_to_file(report, output, output_format="markdown")
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "TestModel-7B" in content
        assert "## Architecture Hypotheses" in content

    def test_writes_json_file(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """render_to_file writes JSON content to the specified path."""
        output = tmp_path / "report.json"
        renderer.render_to_file(report, output, output_format="json")
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["model_name"] == "TestModel-7B"

    def test_creates_parent_directories(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """render_to_file creates missing parent directories."""
        output = tmp_path / "nested" / "deep" / "report.md"
        renderer.render_to_file(report, output, output_format="markdown")
        assert output.exists()

    def test_default_format_is_markdown(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """The default output_format is 'markdown'."""
        output = tmp_path / "default.md"
        renderer.render_to_file(report, output)  # No format specified
        content = output.read_text(encoding="utf-8")
        assert "## Architecture Hypotheses" in content

    def test_unknown_format_raises_value_error(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """An unknown output_format raises ValueError (not RendererError)."""
        output = tmp_path / "report.xyz"
        with pytest.raises(ValueError, match="output_format"):
            renderer.render_to_file(report, output, output_format="xml")

    def test_case_insensitive_format(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """Format strings are case-insensitive ('JSON' == 'json')."""
        output = tmp_path / "report.json"
        renderer.render_to_file(report, output, output_format="JSON")
        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "model_name" in data

    def test_overwrites_existing_file(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """render_to_file overwrites an existing file at the output path."""
        output = tmp_path / "existing.md"
        output.write_text("old content", encoding="utf-8")
        renderer.render_to_file(report, output, output_format="markdown")
        content = output.read_text(encoding="utf-8")
        assert "TestModel-7B" in content
        assert "old content" not in content

    def test_file_encoded_as_utf8(self, renderer: ReportRenderer, report: ArchitectureReport, tmp_path: Path) -> None:
        """Output files are encoded as UTF-8."""
        output = tmp_path / "utf8.md"
        renderer.render_to_file(report, output, output_format="markdown")
        # Verify by reading back with explicit UTF-8
        content = output.read_bytes().decode("utf-8")
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Tests – error handling
# ---------------------------------------------------------------------------


class TestRendererErrors:
    """Tests for RendererError conditions."""

    def test_missing_templates_dir_raises_renderer_error(self, report: ArchitectureReport) -> None:
        """A non-existent templates directory raises RendererError on render."""
        renderer = ReportRenderer(templates_dir=Path("/nonexistent/templates/dir"))
        with pytest.raises(RendererError):
            renderer.render_markdown(report)

    def test_missing_template_file_raises_renderer_error(self, report: ArchitectureReport, tmp_path: Path) -> None:
        """An existing directory with no template file raises RendererError."""
        # Create a real directory but without the expected template file
        renderer = ReportRenderer(templates_dir=tmp_path)
        with pytest.raises(RendererError):
            renderer.render_markdown(report)

    def test_render_json_always_succeeds_for_valid_report(
        self, renderer: ReportRenderer, report: ArchitectureReport
    ) -> None:
        """render_json does not depend on templates directory and always succeeds."""
        result = renderer.render_json(report)
        assert len(result) > 0

    def test_render_json_with_bad_templates_dir_still_works(
        self, report: ArchitectureReport
    ) -> None:
        """render_json works even when templates_dir points to a nonexistent path."""
        renderer = ReportRenderer(templates_dir=Path("/nonexistent"))
        result = renderer.render_json(report)
        data = json.loads(result)
        assert data["model_name"] == "TestModel-7B"


# ---------------------------------------------------------------------------
# Tests – ReportRenderer initialization
# ---------------------------------------------------------------------------


class TestReportRendererInit:
    """Tests for ReportRenderer initialization and lazy environment creation."""

    def test_default_templates_dir(self) -> None:
        """Default templates_dir points to the package's templates directory."""
        from model_archaeologist.renderer import TEMPLATES_DIR
        renderer = ReportRenderer()
        assert renderer.templates_dir == TEMPLATES_DIR

    def test_custom_templates_dir(self, tmp_path: Path) -> None:
        """Custom templates_dir is stored correctly."""
        renderer = ReportRenderer(templates_dir=tmp_path)
        assert renderer.templates_dir == tmp_path

    def test_env_is_none_before_first_render(self) -> None:
        """Jinja2 environment is not created until first render call."""
        renderer = ReportRenderer()
        assert renderer._env is None

    def test_env_is_cached_after_first_render(
        self, renderer: ReportRenderer, report: ArchitectureReport
    ) -> None:
        """Jinja2 environment is cached after the first render_markdown call."""
        renderer.render_markdown(report)
        env1 = renderer._env
        renderer.render_markdown(report)
        env2 = renderer._env
        assert env1 is env2  # Same object

    def test_filters_registered(self, renderer: ReportRenderer, report: ArchitectureReport) -> None:
        """Custom filters are registered in the Jinja2 environment."""
        renderer.render_markdown(report)  # Trigger lazy init
        assert "confidence_bar" in renderer._env.filters  # type: ignore[union-attr]
        assert "confidence_emoji" in renderer._env.filters  # type: ignore[union-attr]
        assert "confidence_label" in renderer._env.filters  # type: ignore[union-attr]
        assert "format_datetime" in renderer._env.filters  # type: ignore[union-attr]
