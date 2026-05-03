"""LLM-powered architectural hypothesis engine for Model Archaeologist.

Sends chunked source text to an OpenAI-compatible LLM and synthesizes
structured hypotheses about model architecture, training paradigms,
and capability sources.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from openai import AsyncOpenAI

from model_archaeologist.schema import ArchitectureReport

# System prompt for the LLM analysis
SYSTEM_PROMPT = """\
You are an expert AI researcher and architect who specializes in reverse-engineering
the design decisions behind large language models and other AI systems.

Your task is to analyze publicly available information about AI models and produce
evidence-backed hypotheses about their architecture, training paradigms, and
capability sources.

For each hypothesis:
1. State the most likely design decision based on evidence
2. Assign a confidence level: "high" (0.7-1.0), "medium" (0.4-0.7), or "low" (0.0-0.4)
3. Quote specific passages from the source material as evidence
4. Identify open questions that remain unanswered

Be specific, technical, and cite evidence. If information is insufficient,
use domain knowledge to make educated guesses but flag them as speculative.
"""

ANALYSIS_PROMPT_TEMPLATE = """\
Analyze the following source material about the AI model "{model_name}" and extract
architectural insights.

## Source Material ({chunk_index}/{total_chunks})

{text}

---

Based on this material, identify any clues about:
1. Attention mechanisms (MHA, GQA, MQA, sparse attention, etc.)
2. Model size and structure (layers, heads, hidden dimensions, MoE)
3. Positional encoding (RoPE, ALiBi, absolute, relative, etc.)
4. Normalization strategies (LayerNorm, RMSNorm, placement)
5. Training data and curation approach
6. Alignment techniques (RLHF, DPO, Constitutional AI, etc.)
7. Scaling strategies and laws applied
8. Notable capabilities or emergent behaviors
9. Inference and efficiency optimizations

Respond in JSON format matching this schema:
{{
  "observations": [
    {{
      "category": "architecture|training|capabilities|efficiency",
      "aspect": "brief aspect name",
      "hypothesis": "detailed hypothesis text",
      "confidence": 0.0-1.0,
      "evidence_quotes": ["direct quote from source"],
      "open_questions": ["question 1"]
    }}
  ]
}}

Return only valid JSON. If the source material contains no relevant information, return
{{"observations": []}}.
"""

SYNTHESIS_PROMPT_TEMPLATE = """\
You have analyzed {num_chunks} chunks of source material about the AI model "{model_name}".
Here are all the observations collected:

{observations_json}

Now synthesize a comprehensive architectural dossier in JSON format. Merge related
observations, resolve contradictions by choosing the best-supported hypothesis,
and produce the final structured report.

Respond with JSON matching this schema exactly:
{{
  "model_name": "{model_name}",
  "executive_summary": "2-3 paragraph summary of key architectural findings",
  "architecture": {{
    "attention_mechanism": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "model_size": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "positional_encoding": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "normalization": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "moe_usage": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }}
  }},
  "training": {{
    "data_curation": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "alignment_technique": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "scaling_strategy": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }}
  }},
  "capabilities": {{
    "emergent_behaviors": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "fine_tuning_approach": {{
      "hypothesis": "...",
      "confidence": 0.0,
      "evidence_quotes": [],
      "open_questions": []
    }},
    "efficiency_optimizations": {{
      "hypothesis": "...",
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


class AnalyzerError(Exception):
    """Raised when LLM analysis fails."""


class ModelAnalyzer:
    """Uses an OpenAI-compatible LLM to analyze model source material.

    Sends document chunks to an LLM with structured prompts and synthesizes
    observations into a complete ArchitectureReport.

    Args:
        model: OpenAI model identifier (e.g. 'gpt-4o').
        base_url: Optional custom API base URL for OpenAI-compatible endpoints.
        api_key: Optional API key (reads from OPENAI_API_KEY env var if not provided).
        verbose: Enable verbose logging.
        temperature: LLM sampling temperature.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        verbose: bool = False,
        temperature: float = 0.3,
    ) -> None:
        """Initialize the ModelAnalyzer."""
        self.model = model
        self.verbose = verbose
        self.temperature = temperature

        client_kwargs: dict[str, Any] = {}
        if base_url:
            client_kwargs["base_url"] = base_url
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = AsyncOpenAI(**client_kwargs)

    async def analyze(
        self,
        model_name: str,
        chunks: list[str],
        sources: Optional[list[str]] = None,
    ) -> ArchitectureReport:
        """Analyze document chunks and produce an ArchitectureReport.

        Processes each chunk individually to extract observations, then
        synthesizes all observations into a final structured report.

        Args:
            model_name: The AI model being analyzed.
            chunks: List of text chunks to analyze.
            sources: Optional list of source identifiers.

        Returns:
            A fully populated ArchitectureReport.

        Raises:
            AnalyzerError: If LLM calls fail or produce unparseable output.
        """
        if not chunks:
            raise AnalyzerError("No chunks provided for analysis.")

        sources = sources or []
        all_observations: list[dict[str, Any]] = []

        # Phase 1: Extract observations from each chunk
        for i, chunk in enumerate(chunks, start=1):
            if self.verbose:
                print(f"  Analyzing chunk {i}/{len(chunks)}...")
            try:
                observations = await self._analyze_chunk(
                    model_name=model_name,
                    text=chunk,
                    chunk_index=i,
                    total_chunks=len(chunks),
                )
                all_observations.extend(observations)
            except AnalyzerError as exc:
                if self.verbose:
                    print(f"  Warning: chunk {i} analysis failed: {exc}")

        if not all_observations:
            # Return a minimal report if no observations were extracted
            return self._build_empty_report(model_name, sources)

        # Phase 2: Synthesize observations into final report
        try:
            report_data = await self._synthesize_report(
                model_name=model_name,
                observations=all_observations,
                num_chunks=len(chunks),
            )
        except AnalyzerError as exc:
            raise AnalyzerError(f"Failed to synthesize report: {exc}") from exc

        # Inject metadata
        report_data["model_name"] = model_name
        report_data["sources_analyzed"] = sources

        try:
            return ArchitectureReport.model_validate(report_data)
        except Exception as exc:
            raise AnalyzerError(
                f"Failed to parse LLM output into ArchitectureReport: {exc}\n"
                f"Raw data: {json.dumps(report_data, indent=2)}"
            ) from exc

    async def _analyze_chunk(
        self,
        model_name: str,
        text: str,
        chunk_index: int,
        total_chunks: int,
    ) -> list[dict[str, Any]]:
        """Send a single chunk to the LLM for observation extraction.

        Args:
            model_name: The AI model being analyzed.
            text: Text content of the chunk.
            chunk_index: 1-based index of this chunk.
            total_chunks: Total number of chunks.

        Returns:
            List of observation dicts from the LLM.

        Raises:
            AnalyzerError: If the LLM call fails or returns invalid JSON.
        """
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            model_name=model_name,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            text=text[:8000],  # Safety truncation
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise AnalyzerError(f"LLM API call failed for chunk {chunk_index}: {exc}") from exc

        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            return data.get("observations", [])
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                f"LLM returned invalid JSON for chunk {chunk_index}: {exc}\nRaw: {raw[:500]}"
            ) from exc

    async def _synthesize_report(
        self,
        model_name: str,
        observations: list[dict[str, Any]],
        num_chunks: int,
    ) -> dict[str, Any]:
        """Synthesize all observations into a structured report via the LLM.

        Args:
            model_name: The AI model being analyzed.
            observations: All extracted observations from chunk analysis.
            num_chunks: Total number of chunks analyzed.

        Returns:
            A dict matching the ArchitectureReport schema.

        Raises:
            AnalyzerError: If the LLM call fails or returns invalid JSON.
        """
        observations_json = json.dumps(observations, indent=2)
        # Truncate if too large
        if len(observations_json) > 20000:
            observations_json = observations_json[:20000] + "\n... (truncated)"

        prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
            model_name=model_name,
            num_chunks=num_chunks,
            observations_json=observations_json,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise AnalyzerError(f"LLM synthesis API call failed: {exc}") from exc

        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AnalyzerError(
                f"LLM returned invalid JSON for synthesis: {exc}\nRaw: {raw[:500]}"
            ) from exc

    def _build_empty_report(self, model_name: str, sources: list[str]) -> ArchitectureReport:
        """Build a minimal ArchitectureReport when no observations are available.

        Args:
            model_name: The AI model being analyzed.
            sources: List of source identifiers.

        Returns:
            An ArchitectureReport with placeholder values.
        """
        from model_archaeologist.schema import (
            ArchitectureDetails,
            DesignHypothesis,
            TrainingParadigm,
            CapabilitySource,
        )

        placeholder = DesignHypothesis(
            hypothesis="Insufficient source material to form a hypothesis.",
            confidence=0.0,
            evidence_quotes=[],
            open_questions=["More source material needed."],
        )

        return ArchitectureReport(
            model_name=model_name,
            executive_summary="Insufficient source material was available to analyze this model.",
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
            additional_notes="Analysis produced no observations from the provided sources.",
        )
