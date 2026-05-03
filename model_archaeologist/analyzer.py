"""LLM-powered architectural hypothesis engine for Model Archaeologist.

Sends chunked source text to an OpenAI-compatible LLM and synthesizes
structured hypotheses about model architecture, training paradigms,
and capability sources into a fully-populated ArchitectureReport.

The analysis proceeds in two phases:
1. Per-chunk observation extraction: each chunk is independently analyzed
   to extract relevant architectural clues with supporting evidence.
2. Synthesis: all observations are merged by the LLM into a coherent,
   deduplicated final report conforming to the ArchitectureReport schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from openai import AsyncOpenAI

from model_archaeologist.schema import (
    ArchitectureDetails,
    ArchitectureReport,
    CapabilitySource,
    DesignHypothesis,
    TrainingParadigm,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI researcher and architect specializing in reverse-engineering
the design decisions behind large language models and other AI systems.

Your task is to analyze publicly available information about AI models and produce
evidence-backed hypotheses about their architecture, training paradigms, and
capability sources.

For each hypothesis:
1. State the most likely design decision based on the available evidence.
2. Assign a numeric confidence score: use 0.8-1.0 for high confidence (direct
   evidence), 0.4-0.7 for medium confidence (indirect evidence or common patterns),
   and 0.0-0.3 for low confidence (speculation or insufficient data).
3. Quote specific passages from the source material as evidence (verbatim where
   possible, paraphrased only when necessary).
4. Identify open questions that remain unresolved.

Be specific, technical, and cite evidence. If information is insufficient,
use domain knowledge to make educated guesses but flag them as speculative
with a low confidence score.

Always respond with valid JSON only — no markdown fences, no commentary outside
the JSON structure.
"""

# Template for per-chunk observation extraction
_ANALYSIS_PROMPT = """\
Analyze the following source material about the AI model "{model_name}" and
extract architectural insights.

## Source Material (chunk {chunk_index} of {total_chunks})

{text}

---

Based solely on the content above, identify any clues about:

1. Attention mechanisms (MHA, GQA, MQA, sparse attention, flash attention, etc.)
2. Model size and structure (parameter count, layers, heads, hidden dimensions, MoE)
3. Positional encoding strategy (RoPE, ALiBi, absolute sinusoidal, NoPE, etc.)
4. Normalization strategy and placement (Pre-LN vs Post-LN, LayerNorm, RMSNorm)
5. Training data sourcing and curation
6. Alignment techniques (RLHF, PPO, DPO, Constitutional AI, SFT, etc.)
7. Scaling strategies and laws applied (Chinchilla, compute-optimal, etc.)
8. Notable capabilities or emergent behaviors
9. Inference and efficiency optimizations (quantization, speculative decoding, etc.)

Respond with JSON in exactly this format:
{{
  "observations": [
    {{
      "category": "architecture|training|capabilities|efficiency",
      "aspect": "brief aspect name",
      "hypothesis": "detailed hypothesis text",
      "confidence": 0.0,
      "evidence_quotes": ["direct quote from source"],
      "open_questions": ["unanswered question"]
    }}
  ]
}}

If this chunk contains no relevant architectural information, return:
{{"observations": []}}
"""

# Template for final synthesis from all observations
_SYNTHESIS_PROMPT = """\
You have analyzed {num_chunks} chunk(s) of source material about the AI model
"{model_name}" and collected the following observations:

{observations_json}

---

Now synthesize a comprehensive, deduplicated architectural dossier.

Instructions:
- Merge related observations into single, well-supported hypotheses.
- When observations contradict each other, select the best-supported one and
  explain the choice in the hypothesis text.
- Set confidence scores based on the weight of evidence across all observations.
- Collect all unique evidence quotes for each aspect.
- Collect all unique open questions for each aspect.
- Write the executive_summary as 2-3 paragraphs covering the most significant
  architectural findings and overall assessment.

Respond with JSON matching EXACTLY this schema (all fields are required):
{{
  "model_name": "{model_name}",
  "executive_summary": "2-3 paragraph summary",
  "architecture": {{
    "attention_mechanism": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "model_size": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "positional_encoding": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "normalization": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "moe_usage": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }}
  }},
  "training": {{
    "data_curation": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "alignment_technique": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "scaling_strategy": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }}
  }},
  "capabilities": {{
    "emergent_behaviors": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "fine_tuning_approach": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "efficiency_optimizations": {{
      "hypothesis": "string",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }}
  }},
  "overall_confidence": 0.0,
  "sources_analyzed": [],
  "additional_notes": ""
}}
"""

# Truncation limit for individual chunk text sent to the LLM (characters)
_CHUNK_TEXT_CHAR_LIMIT = 12000

# Truncation limit for observations JSON sent to synthesis prompt (characters)
_OBSERVATIONS_CHAR_LIMIT = 24000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AnalyzerError(Exception):
    """Raised when LLM analysis encounters an unrecoverable error.

    Wraps lower-level OpenAI SDK exceptions and JSON parsing errors with
    a consistent error type and descriptive message.
    """


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ModelAnalyzer:
    """Uses an OpenAI-compatible LLM to analyze source material about an AI model.

    The analyzer operates in two phases:

    1. **Chunk analysis**: Each text chunk is sent to the LLM with a structured
       prompt requesting JSON-formatted observations about the model's potential
       architectural decisions, training choices, and capabilities.

    2. **Synthesis**: All collected observations are merged in a second LLM call
       that produces a single, deduplicated :class:`~model_archaeologist.schema.ArchitectureReport`.

    If chunk analysis produces no observations (e.g. all chunks are off-topic),
    a minimal placeholder report is returned rather than raising an error.

    Example usage::

        analyzer = ModelAnalyzer(model="gpt-4o", verbose=True)
        report = await analyzer.analyze(
            model_name="GPT-4",
            chunks=["...paper text chunk 1...", "...chunk 2..."],
            sources=["https://arxiv.org/abs/2303.08774"],
        )

    Args:
        model: OpenAI model identifier to use for LLM calls.
            Defaults to ``'gpt-4o'``.
        base_url: Optional base URL for an OpenAI-compatible API endpoint
            (e.g. ``'http://localhost:11434/v1'`` for Ollama).
        api_key: Optional API key.  If ``None``, the ``OPENAI_API_KEY``
            environment variable is used automatically by the OpenAI SDK.
        verbose: If ``True``, emit progress messages via :mod:`logging` and
            ``print``.
        temperature: Sampling temperature for LLM calls.  Lower values
            (e.g. 0.2) produce more deterministic, factual output.
            Defaults to ``0.2``.
        max_retries: Number of times to retry a failed LLM call before
            giving up.  Defaults to ``2``.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> None:
        """Initialise the ModelAnalyzer and create the AsyncOpenAI client."""
        self.model = model
        self.verbose = verbose
        self.temperature = temperature
        self.max_retries = max_retries

        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key
        # Pass max_retries to the SDK for automatic retry on transient errors
        client_kwargs["max_retries"] = max_retries

        self._client = AsyncOpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        model_name: str,
        chunks: list[str],
        sources: Optional[list[str]] = None,
    ) -> ArchitectureReport:
        """Analyze document chunks and produce a structured ArchitectureReport.

        Orchestrates the two-phase analysis pipeline:

        1. Send each chunk to :meth:`_analyze_chunk` to extract observations.
        2. Pass all observations to :meth:`_synthesize_report` for final
           report generation.

        If no observations are extracted (e.g. all chunks are irrelevant), a
        minimal placeholder report is returned via :meth:`_build_empty_report`.

        Args:
            model_name: Human-readable name of the AI model being analyzed
                (e.g. ``'GPT-4'``, ``'LLaMA 3 70B'``).
            chunks: Non-empty list of text chunks to analyze.  Each chunk
                should contain at most ``chunk_size`` tokens as enforced by
                :class:`~model_archaeologist.chunker.TextChunker`.
            sources: Optional list of source identifiers (URLs or file paths)
                to embed in the final report's ``sources_analyzed`` field.

        Returns:
            A fully populated :class:`~model_archaeologist.schema.ArchitectureReport`
            with all 11 hypothesis fields, executive summary, and metadata.

        Raises:
            AnalyzerError: If ``chunks`` is empty, or if the synthesis LLM call
                fails or returns output that cannot be parsed into an
                :class:`~model_archaeologist.schema.ArchitectureReport`.
        """
        if not chunks:
            raise AnalyzerError("No chunks provided for analysis.")

        sources = sources or []
        all_observations: list[dict[str, Any]] = []
        total = len(chunks)

        # Phase 1: Extract observations from each chunk individually
        for i, chunk in enumerate(chunks, start=1):
            self._log(f"Analyzing chunk {i}/{total} ({len(chunk)} chars)...")
            try:
                observations = await self._analyze_chunk(
                    model_name=model_name,
                    text=chunk,
                    chunk_index=i,
                    total_chunks=total,
                )
                if observations:
                    self._log(f"  -> {len(observations)} observation(s) from chunk {i}")
                all_observations.extend(observations)
            except AnalyzerError as exc:
                # Log and skip failed chunks rather than aborting the whole run
                logger.warning("Chunk %d/%d analysis failed: %s", i, total, exc)
                self._log(f"  Warning: chunk {i} analysis failed: {exc}")

        if not all_observations:
            self._log("No observations collected; returning empty report.")
            return self._build_empty_report(model_name, sources)

        self._log(
            f"Collected {len(all_observations)} observation(s) total; "
            "synthesizing final report..."
        )

        # Phase 2: Synthesize all observations into the final structured report
        try:
            report_data = await self._synthesize_report(
                model_name=model_name,
                observations=all_observations,
                num_chunks=total,
            )
        except AnalyzerError as exc:
            raise AnalyzerError(f"Report synthesis failed: {exc}") from exc

        # Inject metadata that must come from outside the LLM
        report_data["model_name"] = model_name
        report_data["sources_analyzed"] = sources

        # Ensure every required nested field has at least a placeholder value
        report_data = self._fill_missing_fields(report_data, model_name)

        try:
            report = ArchitectureReport.model_validate(report_data)
        except Exception as exc:
            raise AnalyzerError(
                f"Failed to parse LLM output into ArchitectureReport: {exc}\n"
                f"Raw data keys: {list(report_data.keys())}"
            ) from exc

        self._log("Analysis complete.")
        return report

    # ------------------------------------------------------------------
    # Private: LLM calls
    # ------------------------------------------------------------------

    async def _analyze_chunk(
        self,
        model_name: str,
        text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> list[dict[str, Any]]:
        """Send a single text chunk to the LLM for observation extraction.

        Constructs the per-chunk analysis prompt, calls the LLM with
        ``response_format={"type": "json_object"}`` to enforce JSON output,
        and parses the ``observations`` list from the response.

        Args:
            model_name: Name of the AI model being analyzed.
            text: Raw text content of this chunk.  Long texts are truncated
                to :data:`_CHUNK_TEXT_CHAR_LIMIT` characters before sending.
            chunk_index: 1-based index of this chunk (used in the prompt).
            total_chunks: Total number of chunks being analyzed.

        Returns:
            A list of observation dicts, each with keys ``category``,
            ``aspect``, ``hypothesis``, ``confidence``, ``evidence_quotes``,
            and ``open_questions``.  Returns an empty list if the LLM finds
            no relevant information in this chunk.

        Raises:
            AnalyzerError: If the API call fails or the response contains
                invalid JSON.
        """
        # Safety truncation to avoid exceeding context limits
        truncated_text = text[:_CHUNK_TEXT_CHAR_LIMIT]

        prompt = _ANALYSIS_PROMPT.format(
            model_name=model_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            text=truncated_text,
        )

        raw = await self._chat_complete(prompt, context=f"chunk {chunk_index}/{total_chunks}")

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                f"LLM returned invalid JSON for chunk {chunk_index}: {exc}\n"
                f"Raw (first 500 chars): {raw[:500]}"
            ) from exc

        observations = data.get("observations", [])
        if not isinstance(observations, list):
            return []

        # Basic validation: keep only dicts with required keys
        valid: list[dict[str, Any]] = []
        for obs in observations:
            if isinstance(obs, dict) and "hypothesis" in obs and "confidence" in obs:
                valid.append(obs)
        return valid

    async def _synthesize_report(
        self,
        model_name: str,
        observations: list[dict[str, Any]],
        num_chunks: int,
    ) -> dict[str, Any]:
        """Synthesize all per-chunk observations into a final report dict.

        Serializes all observations to JSON (truncating if necessary to avoid
        context overflow), sends them to the LLM via the synthesis prompt, and
        parses the response into a raw dict suitable for Pydantic validation.

        Args:
            model_name: Name of the AI model being analyzed.
            observations: All observations collected across all chunks.
            num_chunks: Total number of chunks that were analyzed.

        Returns:
            A dict whose structure mirrors :class:`~model_archaeologist.schema.ArchitectureReport`.

        Raises:
            AnalyzerError: If the API call fails or returns non-JSON output.
        """
        observations_json = json.dumps(observations, indent=2, ensure_ascii=False)

        # Truncate if observations are very large to stay within context limits
        if len(observations_json) > _OBSERVATIONS_CHAR_LIMIT:
            observations_json = (
                observations_json[:_OBSERVATIONS_CHAR_LIMIT]
                + "\n  ... (truncated due to length)"
            )

        prompt = _SYNTHESIS_PROMPT.format(
            model_name=model_name,
            num_chunks=num_chunks,
            observations_json=observations_json,
        )

        raw = await self._chat_complete(prompt, context="synthesis")

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                f"LLM returned invalid JSON during synthesis: {exc}\n"
                f"Raw (first 500 chars): {raw[:500]}"
            ) from exc

    async def _chat_complete(self, user_prompt: str, context: str = "") -> str:
        """Execute a chat completion request and return the response content.

        Wraps the OpenAI SDK call with consistent error handling, converting
        all SDK exceptions into :class:`AnalyzerError`.

        Args:
            user_prompt: The user-role message content to send.
            context: A short description of this call for error messages
                (e.g. ``'chunk 3/10'`` or ``'synthesis'``).

        Returns:
            The raw string content of the first response choice.

        Raises:
            AnalyzerError: If the API call fails for any reason.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            label = f" ({context})" if context else ""
            raise AnalyzerError(
                f"LLM API call failed{label}: {type(exc).__name__}: {exc}"
            ) from exc

        content = response.choices[0].message.content
        if not content:
            label = f" ({context})" if context else ""
            raise AnalyzerError(f"LLM returned empty response{label}")
        return content

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        """Emit a debug message when verbose mode is enabled.

        Args:
            message: The message to print / log.
        """
        if self.verbose:
            print(f"[analyzer] {message}")
        logger.debug(message)

    @staticmethod
    def _placeholder_hypothesis(note: str = "Insufficient source material.") -> dict[str, Any]:
        """Return a minimal hypothesis dict suitable for missing LLM output fields.

        Args:
            note: Human-readable note explaining why this is a placeholder.

        Returns:
            A dict with all required DesignHypothesis fields set to safe defaults.
        """
        return {
            "hypothesis": note,
            "confidence": 0.0,
            "evidence_quotes": [],
            "open_questions": ["More source material is needed to form a hypothesis."],
        }

    def _fill_missing_fields(
        self,
        data: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any]:
        """Ensure all required fields are present in the raw report dict.

        Fills in placeholder values for any fields that the LLM omitted,
        preventing Pydantic validation errors for genuinely missing data.

        Args:
            data: Raw dict from the LLM synthesis response.
            model_name: Model name to use as fallback.

        Returns:
            The (mutated) dict with all required fields present.
        """
        ph = self._placeholder_hypothesis()

        # Top-level fields
        data.setdefault("model_name", model_name)
        data.setdefault(
            "executive_summary",
            "Insufficient source material was available to produce a detailed summary.",
        )
        data.setdefault("overall_confidence", 0.0)
        data.setdefault("sources_analyzed", [])
        data.setdefault("additional_notes", "")

        # Architecture section
        arch = data.setdefault("architecture", {})
        for field in (
            "attention_mechanism",
            "model_size",
            "positional_encoding",
            "normalization",
            "moe_usage",
        ):
            arch.setdefault(field, ph.copy())
            self._fill_hypothesis_fields(arch[field])

        # Training section
        training = data.setdefault("training", {})
        for field in ("data_curation", "alignment_technique", "scaling_strategy"):
            training.setdefault(field, ph.copy())
            self._fill_hypothesis_fields(training[field])

        # Capabilities section
        caps = data.setdefault("capabilities", {})
        for field in ("emergent_behaviors", "fine_tuning_approach", "efficiency_optimizations"):
            caps.setdefault(field, ph.copy())
            self._fill_hypothesis_fields(caps[field])

        return data

    @staticmethod
    def _fill_hypothesis_fields(hyp: dict[str, Any]) -> None:
        """Ensure a hypothesis dict has all required DesignHypothesis fields.

        Mutates the dict in place, adding missing keys with safe defaults.

        Args:
            hyp: A hypothesis dict that may be missing optional keys.
        """
        hyp.setdefault("hypothesis", "No hypothesis could be formed from available data.")
        hyp.setdefault("confidence", 0.0)
        hyp.setdefault("evidence_quotes", [])
        hyp.setdefault("open_questions", [])

        # Clamp confidence to [0.0, 1.0]
        try:
            conf = float(hyp["confidence"])
            hyp["confidence"] = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            hyp["confidence"] = 0.0

        # Ensure list fields are actually lists
        for list_field in ("evidence_quotes", "open_questions"):
            if not isinstance(hyp[list_field], list):
                hyp[list_field] = []

    def _build_empty_report(
        self,
        model_name: str,
        sources: list[str],
    ) -> ArchitectureReport:
        """Build a minimal placeholder ArchitectureReport when no data is available.

        Used when all chunk analyses fail to produce any observations, ensuring
        that the caller always receives a valid (if uninformative) report object.

        Args:
            model_name: Name of the model that was attempted.
            sources: List of source identifiers that were analyzed.

        Returns:
            An :class:`~model_archaeologist.schema.ArchitectureReport` with
            all hypotheses set to zero confidence and placeholder text.
        """
        placeholder = DesignHypothesis(
            hypothesis="Insufficient source material to form a hypothesis.",
            confidence=0.0,
            evidence_quotes=[],
            open_questions=["More source material is needed."],
        )

        return ArchitectureReport(
            model_name=model_name,
            executive_summary=(
                f"No usable observations could be extracted from the provided source "
                f"material for {model_name}. The sources may be inaccessible, off-topic, "
                f"or too sparse to support architectural inference. Please provide "
                f"additional URLs or files such as the technical paper, official blog "
                f"posts, or benchmark reports."
            ),
            architecture=ArchitectureDetails(
                attention_mechanism=placeholder,
                model_size=placeholder,
                positional_encoding=placeholder,
                normalization=placeholder,
                moe_usage=placeholder,
            ),
            training=TrainingParadigm(
                data_curation=placeholder,
                alignment_technique=placeholder,
                scaling_strategy=placeholder,
            ),
            capabilities=CapabilitySource(
                emergent_behaviors=placeholder,
                fine_tuning_approach=placeholder,
                efficiency_optimizations=placeholder,
            ),
            overall_confidence=0.0,
            sources_analyzed=sources,
            additional_notes=(
                "Analysis produced no observations from the provided sources. "
                "Consider adding more authoritative sources."
            ),
        )
