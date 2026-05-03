"""Click-based CLI entry point for Model Archaeologist.

This module wires together document ingestion, token-aware chunking,
LLM-powered analysis, and report rendering into a cohesive CLI experience.

The main entry point is the :func:`main` Click group, which exposes a single
``analyze`` subcommand accepting a model name, one or more evidence sources
(URLs or local files), and various configuration flags for the LLM backend,
chunking parameters, and output format.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from model_archaeologist import __version__

# Stderr console for status/progress messages (keeps stdout clean for report output)
console = Console(stderr=True)


@click.group()
@click.version_option(
    version=__version__,
    prog_name="model-archaeologist",
    message="%(prog)s %(version)s",
)
def main() -> None:
    """Model Archaeologist: Reverse-engineer public AI model architectures.

    Ingests papers, blog posts, and benchmark results to produce a
    detective-style architectural dossier using LLM reasoning.

    Set OPENAI_API_KEY in your environment before running, or use
    --base-url to point at a local OpenAI-compatible endpoint.
    """


@main.command("analyze")
@click.argument("model_name")
@click.option(
    "-u",
    "--url",
    "urls",
    multiple=True,
    metavar="TEXT",
    help="URL to fetch as evidence (can be repeated).",
)
@click.option(
    "-f",
    "--file",
    "files",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    metavar="PATH",
    help="Local PDF or text file as evidence (can be repeated).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    metavar="PATH",
    help="Output file path. Defaults to stdout.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"], case_sensitive=False),
    default="markdown",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--model",
    "llm_model",
    default="gpt-4o",
    show_default=True,
    metavar="TEXT",
    help="OpenAI model to use for analysis.",
)
@click.option(
    "--base-url",
    "base_url",
    default=None,
    metavar="TEXT",
    help="Custom OpenAI-compatible base URL (e.g. Ollama: http://localhost:11434/v1).",
)
@click.option(
    "--api-key",
    "api_key",
    default=None,
    metavar="TEXT",
    envvar="OPENAI_API_KEY",
    help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable.",
)
@click.option(
    "--chunk-size",
    "chunk_size",
    default=3000,
    show_default=True,
    type=click.IntRange(min=100),
    help="Token chunk size for splitting documents.",
)
@click.option(
    "--chunk-overlap",
    "chunk_overlap",
    default=200,
    show_default=True,
    type=click.IntRange(min=0),
    help="Token overlap between consecutive chunks.",
)
@click.option(
    "--max-chunks",
    "max_chunks",
    default=10,
    show_default=True,
    type=click.IntRange(min=1),
    help="Maximum number of chunks to analyze per source document.",
)
@click.option(
    "--temperature",
    "temperature",
    default=0.2,
    show_default=True,
    type=click.FloatRange(min=0.0, max=2.0),
    help="LLM sampling temperature (lower = more deterministic).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose progress and debug logging.",
)
def analyze(
    model_name: str,
    urls: tuple[str, ...],
    files: tuple[Path, ...],
    output_path: Optional[Path],
    output_format: str,
    llm_model: str,
    base_url: Optional[str],
    api_key: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    temperature: float,
    verbose: bool,
) -> None:
    """Analyze a public AI model and produce an architectural dossier.

    MODEL_NAME is the name of the model to analyze (e.g. 'GPT-4', 'LLaMA 3').

    At least one --url or --file source must be provided as evidence for the
    analysis. Multiple sources can be combined freely.

    Examples:

    \b
        model-archaeologist analyze "GPT-4" \\
            --url https://arxiv.org/abs/2303.08774 \\
            --output report.md

    \b
        model-archaeologist analyze "LLaMA 3" \\
            --file llama3_paper.pdf \\
            --url https://ai.meta.com/blog/meta-llama-3 \\
            --format json \\
            --output llama3.json

    \b
        model-archaeologist analyze "Mistral 7B" \\
            --url https://arxiv.org/abs/2310.06825 \\
            --base-url http://localhost:11434/v1 \\
            --model llama3.1 \\
            --verbose
    """
    # ------------------------------------------------------------------ #
    # Validate inputs before doing any async work                         #
    # ------------------------------------------------------------------ #
    if not urls and not files:
        console.print(
            "[bold red]Error:[/bold red] At least one [cyan]--url[/cyan] or "
            "[cyan]--file[/cyan] must be provided as evidence."
        )
        console.print(
            "Run [bold]model-archaeologist analyze --help[/bold] for usage information."
        )
        sys.exit(1)

    if chunk_overlap >= chunk_size:
        console.print(
            f"[bold red]Error:[/bold red] "
            f"--chunk-overlap ({chunk_overlap}) must be less than "
            f"--chunk-size ({chunk_size})."
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Print startup banner                                                #
    # ------------------------------------------------------------------ #
    console.print(
        f"[bold cyan]Model Archaeologist[/bold cyan] [dim]v{__version__}[/dim]"
    )
    console.print(f"Analyzing: [bold green]{model_name}[/bold green]")

    if verbose:
        _print_config_table(
            model_name=model_name,
            urls=urls,
            files=files,
            llm_model=llm_model,
            base_url=base_url,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_chunks=max_chunks,
            temperature=temperature,
            output_format=output_format,
            output_path=output_path,
        )
    else:
        console.print(
            f"Sources: [bold]{len(urls)}[/bold] URL(s), "
            f"[bold]{len(files)}[/bold] file(s)"
        )
        console.print(
            f"LLM: [bold]{llm_model}[/bold]"
            + (f" via [dim]{base_url}[/dim]" if base_url else "")
        )

    # ------------------------------------------------------------------ #
    # Lazy imports – keep startup fast and avoid circular deps            #
    # ------------------------------------------------------------------ #
    from model_archaeologist.ingestion import DocumentIngester, IngestionError
    from model_archaeologist.chunker import TextChunker, ChunkerError
    from model_archaeologist.analyzer import ModelAnalyzer, AnalyzerError
    from model_archaeologist.renderer import ReportRenderer, RendererError

    # ------------------------------------------------------------------ #
    # Initialise pipeline components                                      #
    # ------------------------------------------------------------------ #
    ingester = DocumentIngester(verbose=verbose)

    try:
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except ChunkerError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        sys.exit(1)

    analyzer = ModelAnalyzer(
        model=llm_model,
        base_url=base_url,
        api_key=api_key or None,
        verbose=verbose,
        temperature=temperature,
    )
    renderer = ReportRenderer()

    # ------------------------------------------------------------------ #
    # Run the async analysis pipeline                                     #
    # ------------------------------------------------------------------ #
    async def _run_pipeline() -> str:
        """Execute the full ingestion -> chunking -> analysis -> rendering pipeline."""

        # -------------------------------------------------------- #
        # Phase 1: Ingest all sources                             #
        # -------------------------------------------------------- #
        all_documents: list[dict] = []
        ingestion_errors: list[str] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Fetch URLs
            for url in urls:
                short_url = url[:70] + "..." if len(url) > 70 else url
                task_id = progress.add_task(
                    f"Fetching [cyan]{short_url}[/cyan]", total=None
                )
                try:
                    doc = await ingester.fetch_url(url)
                    all_documents.append(doc)
                    char_count = len(doc.get("text", ""))
                    if verbose:
                        console.print(
                            f"  [green]✓[/green] Fetched [dim]{short_url}[/dim] "
                            f"({char_count:,} chars, "
                            f"content-type: {doc.get('content_type', '?')})"
                        )
                except IngestionError as exc:
                    msg = f"URL '{short_url}': {exc}"
                    ingestion_errors.append(msg)
                    console.print(
                        f"  [yellow]⚠[/yellow]  Could not fetch [dim]{short_url}[/dim]: "
                        f"[yellow]{exc}[/yellow]"
                    )
                finally:
                    progress.remove_task(task_id)

            # Read local files
            for file_path in files:
                task_id = progress.add_task(
                    f"Reading [cyan]{file_path.name}[/cyan]", total=None
                )
                try:
                    doc = await ingester.read_file(file_path)
                    all_documents.append(doc)
                    char_count = len(doc.get("text", ""))
                    if verbose:
                        console.print(
                            f"  [green]✓[/green] Read [dim]{file_path.name}[/dim] "
                            f"({char_count:,} chars, "
                            f"content-type: {doc.get('content_type', '?')})"
                        )
                except IngestionError as exc:
                    msg = f"File '{file_path.name}': {exc}"
                    ingestion_errors.append(msg)
                    console.print(
                        f"  [yellow]⚠[/yellow]  Could not read "
                        f"[dim]{file_path.name}[/dim]: [yellow]{exc}[/yellow]"
                    )
                finally:
                    progress.remove_task(task_id)

        if not all_documents:
            console.print(
                "\n[bold red]Error:[/bold red] No documents could be ingested from "
                "the provided sources."
            )
            if ingestion_errors:
                console.print("[dim]Errors:[/dim]")
                for err in ingestion_errors:
                    console.print(f"  • {err}")
            sys.exit(1)

        console.print(
            f"[green]✓[/green] Ingested [bold]{len(all_documents)}[/bold] document(s)"
        )

        # -------------------------------------------------------- #
        # Phase 2: Chunk all documents                            #
        # -------------------------------------------------------- #
        all_chunks: list[str] = []
        source_names: list[str] = []

        for doc in all_documents:
            text = doc.get("text", "")
            source = doc.get("source", "unknown")
            source_names.append(source)

            if not text.strip():
                if verbose:
                    console.print(
                        f"  [yellow]⚠[/yellow]  Skipping empty document: "
                        f"[dim]{source}[/dim]"
                    )
                continue

            chunks = chunker.split(text)
            total_chunks = len(chunks)
            limited_chunks = chunks[:max_chunks]
            all_chunks.extend(limited_chunks)

            if verbose:
                omitted = total_chunks - len(limited_chunks)
                msg = (
                    f"  Chunked [dim]{_short_source(source)}[/dim]: "
                    f"{total_chunks} chunk(s)"
                )
                if omitted > 0:
                    msg += f" (using {len(limited_chunks)}, omitting {omitted})"
                console.print(msg)

        if not all_chunks:
            console.print(
                "[bold red]Error:[/bold red] All ingested documents produced empty text. "
                "No content available for analysis."
            )
            sys.exit(1)

        console.print(
            f"[green]✓[/green] Prepared [bold]{len(all_chunks)}[/bold] chunk(s) for analysis"
        )

        # -------------------------------------------------------- #
        # Phase 3: LLM analysis                                  #
        # -------------------------------------------------------- #
        console.print(
            f"[dim]Sending chunks to [bold]{llm_model}[/bold] for analysis...[/dim]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(
                f"Analyzing {len(all_chunks)} chunk(s) with [bold]{llm_model}[/bold]...",
                total=None,
            )
            try:
                report = await analyzer.analyze(
                    model_name=model_name,
                    chunks=all_chunks,
                    sources=source_names,
                )
            except AnalyzerError as exc:
                progress.remove_task(task_id)
                console.print(
                    f"[bold red]Analysis error:[/bold red] {exc}"
                )
                sys.exit(1)
            finally:
                try:
                    progress.remove_task(task_id)
                except Exception:
                    pass

        console.print("[green]✓[/green] Analysis complete")

        if verbose:
            _print_confidence_summary(report)

        # -------------------------------------------------------- #
        # Phase 4: Render report                                  #
        # -------------------------------------------------------- #
        try:
            if output_format.lower() == "json":
                output_text = renderer.render_json(report)
            else:
                output_text = renderer.render_markdown(report)
        except RendererError as exc:
            console.print(
                f"[bold red]Rendering error:[/bold red] {exc}"
            )
            sys.exit(1)

        return output_text

    # Run the async pipeline in a new event loop
    try:
        output_text = asyncio.run(_run_pipeline())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(130)
    except SystemExit:
        raise
    except Exception as exc:
        console.print(
            f"[bold red]Unexpected error:[/bold red] "
            f"{type(exc).__name__}: {exc}"
        )
        if verbose:
            import traceback
            console.print_exception()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Write output                                                        #
    # ------------------------------------------------------------------ #
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
            console.print(
                f"[green]✓[/green] Report written to [bold]{output_path}[/bold] "
                f"([dim]{output_format}[/dim])"
            )
        except OSError as exc:
            console.print(
                f"[bold red]Error writing output file:[/bold red] {exc}"
            )
            sys.exit(1)
    else:
        # Write to stdout — use click.echo to handle encoding correctly
        click.echo(output_text)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _short_source(source: str, max_len: int = 60) -> str:
    """Truncate a source identifier for display purposes.

    Args:
        source: URL or file path string.
        max_len: Maximum display length before truncation.

    Returns:
        Possibly-truncated string with '...' appended if truncated.
    """
    if len(source) <= max_len:
        return source
    return source[:max_len] + "..."


def _print_config_table(
    model_name: str,
    urls: tuple[str, ...],
    files: tuple[Path, ...],
    llm_model: str,
    base_url: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    temperature: float,
    output_format: str,
    output_path: Optional[Path],
) -> None:
    """Print a Rich table summarising the analysis configuration.

    Only called when ``--verbose`` is active.  Provides a quick visual
    confirmation of all runtime parameters.

    Args:
        model_name: Name of the AI model being analyzed.
        urls: Tuple of URL strings to ingest.
        files: Tuple of local file paths to ingest.
        llm_model: LLM model identifier.
        base_url: Optional custom API base URL.
        chunk_size: Tokens per chunk.
        chunk_overlap: Token overlap between chunks.
        max_chunks: Maximum chunks per source.
        temperature: LLM sampling temperature.
        output_format: Output format string.
        output_path: Optional output file path.
    """
    table = Table(
        title="Analysis Configuration",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Parameter", style="bold", no_wrap=True)
    table.add_column("Value", style="dim")

    table.add_row("Model name", model_name)
    table.add_row("LLM model", llm_model)
    table.add_row("Base URL", base_url or "(default OpenAI)")
    table.add_row("Temperature", str(temperature))
    table.add_row("URLs", str(len(urls)))
    table.add_row("Files", str(len(files)))
    table.add_row("Chunk size", f"{chunk_size} tokens")
    table.add_row("Chunk overlap", f"{chunk_overlap} tokens")
    table.add_row("Max chunks/source", str(max_chunks))
    table.add_row("Output format", output_format)
    table.add_row("Output path", str(output_path) if output_path else "(stdout)")

    console.print(table)

    if urls:
        console.print("[bold]URLs:[/bold]")
        for url in urls:
            console.print(f"  • [link={url}]{_short_source(url)}[/link]")
    if files:
        console.print("[bold]Files:[/bold]")
        for f in files:
            console.print(f"  • {f}")
    console.print()


def _print_confidence_summary(report: object) -> None:  # ArchitectureReport
    """Print a compact confidence summary table to the console.

    Displays the confidence score and label for every hypothesis in the
    report as a Rich table.  Only invoked when ``--verbose`` is active.

    Args:
        report: A fully-populated ArchitectureReport instance.
    """
    table = Table(
        title="Confidence Summary",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Category", style="bold", no_wrap=True)
    table.add_column("Aspect", no_wrap=True)
    table.add_column("Confidence", justify="right")
    table.add_column("Level", justify="center")

    _LEVEL_STYLES = {
        "high": "bold green",
        "medium": "bold yellow",
        "low": "bold red",
    }

    for category, aspect, hyp in report.all_hypotheses:  # type: ignore[attr-defined]
        label = hyp.confidence_label
        style = _LEVEL_STYLES.get(label, "")
        table.add_row(
            category,
            aspect,
            f"{hyp.confidence:.0%}",
            f"[{style}]{label}[/{style}]",
        )

    console.print(table)
    console.print(
        f"Overall confidence: [bold]{report.overall_confidence:.0%}[/bold]"  # type: ignore[attr-defined]
    )
    console.print()


if __name__ == "__main__":
    main()
