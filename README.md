# Model Archaeologist 🔍

> Reverse-engineer any public AI model into a detective-style architectural dossier.

Model Archaeologist ingests published papers, blog posts, and benchmark results for any AI model, then uses an LLM to hypothesize the architectural and training decisions behind it. Feed it URLs or local files; get back a structured Markdown report with evidence-backed guesses on attention mechanisms, training paradigms, alignment techniques, and more.

---

## Quick Start

```bash
# Install
pip install model-archaeologist

# Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# Analyze a model from arXiv and a blog post
model-archaeologist analyze "Mistral 7B" \
  --source https://arxiv.org/abs/2310.06825 \
  --source https://mistral.ai/news/announcing-mistral-7b/

# Analyze using local PDFs
model-archaeologist analyze "Llama 3" \
  --source llama3_paper.pdf \
  --source llama3_blog.txt \
  --output llama3_dossier.md
```

That's it. Your architectural dossier is ready.

---

## Features

- **Multi-source ingestion** — Accepts arXiv/blog URLs, local PDFs, and plain text files as evidence sources for a single model analysis.
- **LLM-powered hypothesis engine** — Uses structured chain-of-thought prompting to produce evidence-backed hypotheses covering attention type, MoE usage, context length strategies, data curation, and alignment techniques.
- **Token-aware chunking** — Automatically splits large corpora and merges partial analyses into a coherent final report without exceeding context limits.
- **Structured Markdown dossier** — Outputs a clean, section-rich report with confidence levels, supporting evidence quotes, and open questions per architectural decision.
- **Pluggable LLM backend** — Defaults to GPT-4o but accepts any OpenAI-compatible endpoint via `--base-url`, including local models via Ollama.

---

## Usage Examples

### Analyze from multiple URLs

```bash
model-archaeologist analyze "GPT-4" \
  --source https://arxiv.org/abs/2303.08774 \
  --source https://openai.com/research/gpt-4 \
  --source https://openai.com/research/gpt-4-technical-report \
  --output gpt4_dossier.md
```

### Mix URLs and local files

```bash
model-archaeologist analyze "Gemma 2" \
  --source https://arxiv.org/abs/2408.00118 \
  --source ./notes/gemma2_benchmarks.txt \
  --output gemma2_dossier.md
```

### Use a local model via Ollama

```bash
model-archaeologist analyze "Phi-3" \
  --source https://arxiv.org/abs/2404.14219 \
  --base-url http://localhost:11434/v1 \
  --model llama3.1:70b \
  --output phi3_dossier.md
```

### Output as JSON

```bash
model-archaeologist analyze "Falcon 180B" \
  --source https://arxiv.org/abs/2311.16867 \
  --format json \
  --output falcon_dossier.json
```

### Sample report output

```markdown
# Architectural Dossier: Mistral 7B

**Generated:** 2024-11-01 14:32:10 UTC
**Sources Analyzed:** 2
**Overall Confidence:** 🟡 ████████░░ (medium)

## Executive Summary

Mistral 7B appears to employ a decoder-only transformer architecture with
several efficiency-focused modifications distinguishing it from vanilla
Llama-style models...

## Architecture Hypotheses

### Attention Mechanism
**Hypothesis:** Grouped-query attention (GQA) with sliding window attention (SWA)
**Confidence:** 🟢 ██████████ (high)
**Evidence:** "We use Grouped-Query Attention (GQA) [...] and Sliding Window
Attention (SWA)" — Mistral 7B paper, §3.1

### Positional Encoding
**Hypothesis:** Rotary positional embeddings (RoPE)
**Confidence:** 🟢 █████████░ (high)
...
```

---

## Project Structure

```
model_archaeologist/
├── pyproject.toml                  # Project metadata, dependencies, CLI entry point
├── README.md                       # This file
├── model_archaeologist/
│   ├── __init__.py                 # Package init, version string
│   ├── cli.py                      # Click CLI entry point
│   ├── ingestion.py                # URL/PDF/text fetching and extraction
│   ├── chunker.py                  # Token-aware document splitting
│   ├── analyzer.py                 # LLM hypothesis engine
│   ├── schema.py                   # Pydantic report schema models
│   ├── renderer.py                 # Markdown/JSON report renderer
│   └── templates/
│       └── report.md.j2            # Jinja2 Markdown dossier template
└── tests/
    ├── test_chunker.py             # Chunker unit tests
    ├── test_ingestion.py           # Ingestion layer tests (mocked HTTP)
    ├── test_schema.py              # Pydantic schema validation tests
    └── test_renderer.py            # Renderer output tests
```

---

## Configuration

All options can be passed as CLI flags. There is no config file required.

| Flag | Default | Description |
|---|---|---|
| `--source` / `-s` | *(required)* | URL or local file path. Repeatable. |
| `--output` / `-o` | stdout | Output file path for the report. |
| `--format` | `markdown` | Output format: `markdown` or `json`. |
| `--model` | `gpt-4o` | LLM model name to use for analysis. |
| `--base-url` | OpenAI API | Base URL for an OpenAI-compatible endpoint. |
| `--chunk-size` | `4000` | Max tokens per analysis chunk. |
| `--chunk-overlap` | `200` | Token overlap between consecutive chunks. |
| `--verbose` / `-v` | off | Enable verbose logging output. |

**Environment variables:**

```bash
OPENAI_API_KEY=sk-...       # Required for OpenAI backend
OPENAI_BASE_URL=...         # Optional: overrides --base-url
```

---

## Running Tests

```bash
pip install -e '.[dev]'
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built with [Jitter](https://github.com/jitter-ai) — an AI agent that ships code daily.*
