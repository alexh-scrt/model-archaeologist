# Model Archaeologist

**Model Archaeologist** is a CLI tool that reverse-engineers public AI models by ingesting their
published papers, blog posts, and benchmark results, then using an LLM to hypothesize architectural
design decisions.

Given a model name and a set of URLs or local files, it produces a structured Markdown report
covering:

- **Architecture patterns**: attention mechanisms, normalization strategies, positional encodings, MoE usage
- **Training paradigms**: data curation, RLHF/DPO signals, scaling laws
- **Capability sources**: emergent behaviors, fine-tuning approach, benchmark analysis

Think of it as an automated research analyst that reads everything public about a model and outputs
a detective-style architectural dossier.

---

## Installation

```bash
# From PyPI (once published)
pip install model-archaeologist

# From source
git clone https://github.com/example/model-archaeologist
cd model-archaeologist
pip install -e .[dev]
```

## Prerequisites

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

### Basic: Analyze a model by name with URLs

```bash
model-archaeologist analyze "GPT-4" \
  --url https://arxiv.org/abs/2303.08774 \
  --url https://openai.com/research/gpt-4 \
  --output report.md
```

### Provide local PDF or text files

```bash
model-archaeologist analyze "LLaMA 3" \
  --file llama3_paper.pdf \
  --file llama3_blog.txt \
  --output llama3_report.md
```

### Mix URLs and files

```bash
model-archaeologist analyze "Mistral 7B" \
  --url https://arxiv.org/abs/2310.06825 \
  --file mistral_notes.txt \
  --output mistral_report.md \
  --format markdown
```

### Use a local model via Ollama

```bash
model-archaeologist analyze "Gemma 2" \
  --url https://arxiv.org/abs/2408.00118 \
  --base-url http://localhost:11434/v1 \
  --model llama3.1 \
  --output gemma2_report.md
```

### Output as JSON

```bash
model-archaeologist analyze "Claude 3" \
  --url https://www.anthropic.com/news/claude-3-family \
  --format json \
  --output claude3_report.json
```

---

## CLI Reference

```
Usage: model-archaeologist analyze [OPTIONS] MODEL_NAME

  Analyze a public AI model and produce an architectural dossier.

Arguments:
  MODEL_NAME  The name of the model to analyze (e.g. 'GPT-4', 'LLaMA 3').

Options:
  -u, --url TEXT          URL to fetch as evidence (can be repeated).
  -f, --file PATH         Local PDF or text file as evidence (can be repeated).
  -o, --output PATH       Output file path. Defaults to stdout.
  --format [markdown|json] Output format. [default: markdown]
  --model TEXT            OpenAI model to use for analysis. [default: gpt-4o]
  --base-url TEXT         Custom OpenAI-compatible base URL (e.g. Ollama).
  --chunk-size INTEGER    Token chunk size for splitting documents. [default: 3000]
  --chunk-overlap INTEGER Token overlap between chunks. [default: 200]
  --max-chunks INTEGER    Max chunks to analyze per source. [default: 10]
  --verbose               Enable verbose logging.
  --help                  Show this message and exit.
```

---

## Sample Output

```markdown
# Architectural Dossier: GPT-4

**Generated:** 2024-01-15 10:30:00 UTC  
**Sources analyzed:** 3 documents, 47 chunks  
**Analysis model:** gpt-4o

---

## Executive Summary

GPT-4 appears to be a large-scale Transformer-based language model with strong evidence of
Mixture-of-Experts (MoE) architecture, likely employing multi-head attention with grouped-query
variants for efficiency. Training incorporates RLHF with PPO and potentially Constitutional AI
influences.

---

## Architecture Hypotheses

### Attention Mechanism
**Hypothesis:** Multi-Head Attention with Grouped-Query Attention (GQA)  
**Confidence:** High (0.82)  
**Evidence:**
> "The model demonstrates efficient inference characteristics consistent with GQA implementations..."

**Open Questions:**
- Exact number of attention heads and key-value heads unknown
- Flash Attention version unclear

...
```

---

## Architecture

```
model_archaeologist/
├── __init__.py       # Package version
├── cli.py            # Click CLI entry point
├── ingestion.py      # URL/PDF/file text extraction
├── chunker.py        # Token-aware document splitting
├── analyzer.py       # LLM hypothesis engine
├── schema.py         # Pydantic report models
├── renderer.py       # Jinja2 Markdown/JSON renderer
└── templates/
    └── report.md.j2  # Markdown report template
```

---

## Development

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest

# Run with verbose output
pytest -v
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
