"""Click-based CLI entry point for Model Archaeologist.

This module wires together document ingestion, token-aware chunking,
LLM-powered analysis, and report rendering into a cohesive CLI experience.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from model_archaeologist import __version__

console = Console(stderr=True)


@click.group()
@click.version_option(version=__version__, prog_name="model-archaeologist")
def main() -> None:
    """Model Archaeologist: Reverse-engineer public AI model architectures.

    Ingests papers, blog posts, and benchmark results to produce a
    detective-style architectural dossier using LLM reasoning.
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
    help="Custom OpenAI-compatible base URL (e.g., Ollama).",
)
@click.option(
    "--chunk-size",
    "chunk_size",
    default=3000,
    show_default=True,
    type=int,
    help="Token chunk size for splitting documents.",
)
@click.option(
    "--chunk-overlap",
    "chunk_overlap",
    default=200,
    show_default=True,
    type=int,
    help="Token overlap between chunks.",
)
@click.option(
    "--max-chunks",
    "max_chunks",
    default=10,
    show_default=True,
    type=int,
    help="Max chunks to analyze per source.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)
def analyze(
    model_name: str,
    urls: tuple[str, ...],
    files: tuple[Path, ...],
    output_path: Optional[Path],
    output_format: str,
    llm_model: str,
    base_url: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks: int,
    verbose: bool,
) -> None:
    """Analyze a public AI model and produce an architectural dossier.

    MODEL_NAME is the name of the model to analyze (e.g. 'GPT-4', 'LLaMA 3').
    """
    if not urls and not files:
        console.print(
            "[bold red]Error:[/bold red] At least one --url or --file must be provided."
        )
        sys.exit(1)

    console.print(f"[bold cyan]Model Archaeologist[/bold cyan] v{__version__}")
    console.print(f"Analyzing model: [bold]{model_name}[/bold]")
    console.print(
        f"Sources: {len(urls)} URL(s), {len(files)} file(s)",
    )

    if verbose:
        console.print(f"LLM model: {llm_model}")
        if base_url:
            console.print(f"Base URL: {base_url}")
        console.print(f"Chunk size: {chunk_size} tokens, overlap: {chunk_overlap} tokens")

    # Lazy imports to avoid circular dependencies and speed up startup
    from model_archaeologist.ingestion import DocumentIngester
    from model_archaeologist.chunker import TextChunker
    from model_archaeologist.analyzer import ModelAnalyzer
    from model_archaeologist.renderer import ReportRenderer

    async def run_analysis() -> str:
        """Execute the full analysis pipeline asynchronously."""
        ingester = DocumentIngester(verbose=verbose)
        chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        analyzer = ModelAnalyzer(
            model=llm_model,
            base_url=base_url,
            verbose=verbose,
        )
        renderer = ReportRenderer()

        all_documents: list[dict] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            # Ingest URLs
            for url in urls:
                task = progress.add_task(f"Fetching {url[:60]}...", total=None)
                try:
                    doc = await ingester.fetch_url(url)
                    all_documents.append(doc)
                    if verbose:
                        console.print(
                            f"  [green]✓[/green] Fetched {url[:60]} "
                            f"({len(doc.get('text', ''))} chars)"
                        )
                except Exception as exc:
                    console.print(f"  [yellow]⚠[/yellow] Failed to fetch {url}: {exc}")
                finally:
                    progress.remove_task(task)

            # Ingest files
            for file_path in files:
                task = progress.add_task(f"Reading {file_path.name}...", total=None)
                try:
                    doc = await ingester.read_file(file_path)
                    all_documents.append(doc)
                    if verbose:
                        console.print(
                            f"  [green]✓[/green] Read {file_path.name} "
                            f"({len(doc.get('text', ''))} chars)"
                        )
                except Exception as exc:
                    console.print(f"  [yellow]⚠[/yellow] Failed to read {file_path}: {exc}")
                finally:
                    progress.remove_task(task)

        if not all_documents:
            console.print("[bold red]Error:[/bold red] No documents could be ingested.")
            sys.exit(1)

        console.print(f"[green]✓[/green] Ingested {len(all_documents)} document(s)")

        # Chunk documents
        all_chunks: list[str] = []
        for doc in all_documents:
            text = doc.get("text", "")
            if text.strip():
                chunks = chunker.split(text)
                limited = chunks[:max_chunks]
                all_chunks.extend(limited)
                if verbose:
                    console.print(
                        f"  Chunked '{doc.get('source', 'unknown')}': "
                        f"{len(chunks)} chunks, using {len(limited)}"
                    )

        console.print(f"[green]✓[/green] Created {len(all_chunks)} chunk(s) for analysis")

        # Analyze with LLM
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running LLM analysis...", total=None)
            try:
                report = await analyzer.analyze(
                    model_name=model_name,
                    chunks=all_chunks,
                    sources=[doc.get("source", "unknown") for doc in all_documents],
                )
            finally:
                progress.remove_task(task)

        console.print("[green]✓[/green] Analysis complete")

        # Render output
        if output_format.lower() == "json":
            output_text = renderer.render_json(report)
        else:
            output_text = renderer.render_markdown(report)

        return output_text

    output_text = asyncio.run(run_analysis())

    if output_path:
        output_path.write_text(output_text, encoding="utf-8")
        console.print(f"[green]✓[/green] Report written to [bold]{output_path}[/bold]")
    else:
        click.echo(output_text)


if __name__ == "__main__":
    main()
