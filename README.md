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
pip install -e .

# With dev/test dependencies
pip install -e ".[dev]"
```

### Requirements

- Python 3.11 or later
- An OpenAI API key (set `OPENAI_API_KEY` environment variable), **or** any OpenAI-compatible
  local endpoint such as [Ollama](https://ollama.com) via `--base-url`

---

## Quick Start

```bash
export OPENAI_API_KEY="sk-..."

# Analyze GPT-4 from its technical report
model-archaeologist analyze "GPT-4" \
    --url https://arxiv.org/abs/2303.08774 \
    --output gpt4_dossier.md

# Open the dossier
cat gpt4_dossier.md
```

---

## Usage

```
Usage: model-archaeologist analyze [OPTIONS] MODEL_NAME

  Analyze a public AI model and produce an architectural dossier.

  MODEL_NAME is the name of the model to analyze (e.g. 'GPT-4', 'LLaMA 3').

  At least one --url or --file source must be provided as evidence.

Options:
  -u, --url TEXT                  URL to fetch as evidence (can be repeated).
  -f, --file PATH                 Local PDF or text file (can be repeated).
  -o, --output PATH               Output file path. Defaults to stdout.
  --format [markdown|json]        Output format.  [default: markdown]
  --model TEXT                    OpenAI model to use.  [default: gpt-4o]
  --base-url TEXT                 Custom OpenAI-compatible base URL.
  --api-key TEXT                  OpenAI API key (or set OPENAI_API_KEY).
  --chunk-size INTEGER RANGE      Token chunk size.  [default: 3000]
  --chunk-overlap INTEGER RANGE   Token overlap between chunks.  [default: 200]
  --max-chunks INTEGER RANGE      Max chunks per source document.  [default: 10]
  --temperature FLOAT RANGE       LLM sampling temperature.  [default: 0.2]
  --verbose                       Enable verbose progress output.
  --help                          Show this message and exit.
```

---

## Example Commands

### Analyze LLaMA 3 from multiple sources

```bash
model-archaeologist analyze "LLaMA 3 70B" \
    --url https://arxiv.org/abs/2407.21783 \
    --url https://ai.meta.com/blog/meta-llama-3 \
    --output llama3_dossier.md \
    --verbose
```

### Analyze Mistral 7B and export as JSON

```bash
model-archaeologist analyze "Mistral 7B" \
    --url https://arxiv.org/abs/2310.06825 \
    --format json \
    --output mistral7b.json
```

### Use a local PDF paper as evidence

```bash
model-archaeologist analyze "Gemma 2" \
    --file gemma2_paper.pdf \
    --url https://storage.googleapis.com/deepmind-media/gemma/gemma2-report.pdf \
    --output gemma2_dossier.md
```

### Use a local Ollama model instead of OpenAI

```bash
model-archaeologist analyze "Phi-3" \
    --url https://arxiv.org/abs/2404.14219 \
    --base-url http://localhost:11434/v1 \
    --model llama3.1 \
    --output phi3_dossier.md
```

### Combine URL and local file sources

```bash
model-archaeologist analyze "Claude 3" \
    --url https://www.anthropic.com/news/claude-3-family \
    --file claude3_model_card.pdf \
    --file benchmark_results.txt \
    --output claude3_dossier.md \
    --verbose
```

---

## Sample Output

```markdown
# Architectural Dossier: LLaMA 3 70B

**Generated:** 2024-06-15 14:23:11 UTC
**Sources Analyzed:** 2
**Overall Confidence:** 🟢 [███████░░░] 72% (high)
**Analysis Engine:** Model Archaeologist

---

## Executive Summary

LLaMA 3 70B is a dense transformer-based large language model from Meta AI.
Based on the available technical report and blog post, the architecture shows
strong evidence for Grouped Query Attention (GQA) replacing standard Multi-Head
Attention, enabling a favorable quality-to-inference-cost tradeoff at 70B scale.

The positional encoding strategy is almost certainly Rotary Position Embedding
(RoPE), consistent with the broader LLaMA model family lineage. The normalization
strategy uses Pre-RMSNorm (Pre-LayerNorm with RMSNorm), a common stabilization
choice for large models. No Mixture-of-Experts usage is reported; LLaMA 3 70B
appears to be a dense model.

Training appears to follow a compute-optimal approach on over 15 trillion tokens,
with explicit quality filtering and deduplication of the pre-training corpus.
Alignment leverages supervised fine-tuning followed by RLHF with iterative rounds
of human preference data collection.

---

## Architecture Hypotheses

### Attention Mechanism

**Confidence:** 🟢 [████████░░] 85%

**Hypothesis:** Grouped Query Attention (GQA) with 8 KV heads and 64 query heads,
reducing KV-cache memory footprint for efficient serving at 70B scale.

**Supporting Evidence:**

> "We use grouped query attention (GQA) with a group size of 8."

**Open Questions:**

- Does the 8B variant use MQA (1 KV head) instead of GQA?

---

### Positional Encoding

**Confidence:** 🟢 [█████████░] 90%

**Hypothesis:** Rotary Position Embeddings (RoPE) with a base frequency of
theta=500,000 (significantly higher than LLaMA 2's theta=10,000), supporting
a native context length of 8,192 tokens.

**Supporting Evidence:**

> "We increase the RoPE base frequency hyperparameter to 500,000."

**Open Questions:**

- Is YaRN or dynamic NTK scaling applied for contexts beyond 8K tokens?
```

---

## Architecture

The tool processes sources through a four-stage pipeline:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌────────────────┐
│   Ingestion     │────▶│   Chunking       │────▶│   LLM Analysis  │────▶│   Rendering    │
│                 │     │                  │     │                 │     │                │
│ • URLs (HTML)   │     │ tiktoken-based   │     │ Per-chunk obs.  │     │ Jinja2 MD      │
│ • URLs (PDF)    │     │ token-aware      │     │ extraction +    │     │ or JSON via    │
│ • Local PDFs    │     │ sliding window   │     │ final synthesis │     │ Pydantic       │
│ • Local text    │     │ with overlap     │     │ into structured │     │ serialization  │
└─────────────────┘     └──────────────────┘     │ ArchReport      │     └────────────────┘
                                                  └─────────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| `cli.py` | Click-based entry point wiring all stages together |
| `ingestion.py` | Async HTTP fetching, HTML scraping, PDF/text extraction |
| `chunker.py` | Token-aware text splitter using tiktoken |
| `analyzer.py` | Two-phase LLM analysis (per-chunk + synthesis) |
| `schema.py` | Pydantic models for the structured report |
| `renderer.py` | Jinja2 Markdown and JSON output rendering |

---

## Report Structure

Every generated dossier covers 11 structured hypotheses across three categories:

### 🏗️ Architecture

| Aspect | Description |
|--------|-------------|
| Attention Mechanism | MHA, GQA, MQA, sparse attention, Flash Attention |
| Model Size & Structure | Parameter count, layers, heads, hidden dimensions |
| Positional Encoding | RoPE, ALiBi, absolute sinusoidal, NoPE |
| Normalization Strategy | Pre-LN / Post-LN, LayerNorm, RMSNorm |
| Mixture-of-Experts | Sparse/dense MoE, expert count, routing strategy |

### 🎓 Training

| Aspect | Description |
|--------|-------------|
| Data Curation | Web-scale sources, quality filtering, deduplication |
| Alignment Technique | RLHF/PPO, DPO, Constitutional AI, SFT-only |
| Scaling Strategy | Chinchilla-optimal, over-training, compute budget |

### ⚡ Capabilities

| Aspect | Description |
|--------|-------------|
| Emergent Behaviors | Scale-dependent abilities, benchmark inflections |
| Fine-tuning Approach | Instruction tuning, LoRA vs full fine-tune |
| Efficiency Optimizations | Flash Attention, speculative decoding, quantization |

Each hypothesis includes:
- **Confidence score** (0–100%) with visual progress bar and color emoji
- **Supporting evidence** as direct blockquote citations from source material
- **Open questions** highlighting remaining uncertainties

---

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required unless using `--base-url` with a key-free local model) |

### Chunk Size Tuning

The default chunk size of 3,000 tokens with 200-token overlap works well for most use cases.
For very long papers, consider increasing `--max-chunks` to analyze more of the document:

```bash
# Analyze up to 20 chunks per source (covers ~60K tokens per document)
model-archaeologist analyze "GPT-4" \
    --url https://arxiv.org/abs/2303.08774 \
    --chunk-size 4000 \
    --chunk-overlap 300 \
    --max-chunks 20 \
    --output gpt4_detailed.md
```

### Using Local Models (Ollama)

Model Archaeologist supports any OpenAI-compatible API endpoint. With Ollama:

```bash
# Start Ollama with a capable model
ollama serve
ollama pull llama3.1:70b

# Run analysis against local Ollama
model-archaeologist analyze "Qwen2" \
    --url https://arxiv.org/abs/2407.10671 \
    --base-url http://localhost:11434/v1 \
    --model llama3.1:70b \
    --api-key ollama \
    --output qwen2_dossier.md
```

> **Note:** Local models produce better results when they have strong instruction-following
> and JSON output capabilities. Models with at least 13B parameters are recommended.

---

## Development

### Setup

```bash
git clone https://github.com/example/model-archaeologist
cd model-archaeologist
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test module
pytest tests/test_chunker.py -v
pytest tests/test_ingestion.py -v
pytest tests/test_schema.py -v
pytest tests/test_renderer.py -v

# Run with coverage
pytest --cov=model_archaeologist --cov-report=term-missing
```

### Project Structure

```
model-archaeologist/
├── pyproject.toml
├── README.md
├── model_archaeologist/
│   ├── __init__.py          # Package version
│   ├── cli.py               # Click CLI entry point
│   ├── ingestion.py         # URL/file ingestion layer
│   ├── chunker.py           # Token-aware text chunker
│   ├── analyzer.py          # LLM hypothesis engine
│   ├── schema.py            # Pydantic report models
│   ├── renderer.py          # Jinja2/JSON report renderer
│   └── templates/
│       └── report.md.j2     # Markdown report template
└── tests/
    ├── __init__.py
    ├── test_chunker.py
    ├── test_ingestion.py
    ├── test_schema.py
    └── test_renderer.py
```

---

## Limitations

- **Paywalled content**: The tool cannot access papers or posts behind paywalls. Use local PDF
  files for paywalled sources.
- **Image-heavy PDFs**: PDFs that are scans (image-only) will produce no extractable text.
  Use papers with embedded text layers.
- **Closed models**: For models like GPT-4 with minimal public technical disclosure, confidence
  scores will be low and hypotheses will rely heavily on indirect inference.
- **Token limits**: Very long papers may be truncated to `--max-chunks` chunks per source.
  Increase `--max-chunks` to analyze more content at the cost of higher API usage.
- **Hallucination risk**: The LLM may confabulate details not present in the source material.
  Always verify cited evidence quotes against the original sources.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes and add tests
4. Run the test suite (`pytest`)
5. Open a pull request
