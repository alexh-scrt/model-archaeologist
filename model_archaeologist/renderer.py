"""Report renderer for Model Archaeologist.

Renders ArchitectureReport objects into Markdown or JSON output
using a Jinja2 template for Markdown and standard JSON serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from model_archaeologist.schema import ArchitectureReport

# Path to the templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATE_NAME = "report.md.j2"


class RendererError(Exception):
    """Raised when report rendering fails."""


class ReportRenderer:
    """Renders ArchitectureReport objects into Markdown or JSON.

    Uses a Jinja2 template for Markdown rendering and the Pydantic
    model's serialization for JSON output.

    Args:
        templates_dir: Path to the Jinja2 templates directory.
            Defaults to the package's built-in templates.
    """

    def __init__(self, templates_dir: Path = TEMPLATES_DIR) -> None:
        """Initialize the ReportRenderer."""
        self.templates_dir = templates_dir
        self._env: Environment | None = None

    def _get_env(self) -> Environment:
        """Lazily initialize and return the Jinja2 environment.

        Returns:
            Configured Jinja2 Environment instance.

        Raises:
            RendererError: If the templates directory does not exist.
        """
        if self._env is None:
            if not self.templates_dir.is_dir():
                raise RendererError(
                    f"Templates directory not found: '{self.templates_dir}'"
                )
            self._env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=select_autoescape([]),  # No autoescaping for Markdown
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True,
            )
            # Register custom filters
            self._env.filters["confidence_bar"] = self._confidence_bar_filter
            self._env.filters["confidence_emoji"] = self._confidence_emoji_filter
        return self._env

    def render_markdown(self, report: ArchitectureReport) -> str:
        """Render a report as a Markdown string using the Jinja2 template.

        Args:
            report: The ArchitectureReport to render.

        Returns:
            A Markdown-formatted string representing the dossier.

        Raises:
            RendererError: If template rendering fails.
        """
        try:
            env = self._get_env()
            template = env.get_template(TEMPLATE_NAME)
            return template.render(report=report)
        except RendererError:
            raise
        except Exception as exc:
            raise RendererError(f"Failed to render Markdown report: {exc}") from exc

    def render_json(self, report: ArchitectureReport) -> str:
        """Render a report as a pretty-printed JSON string.

        Uses Pydantic's model serialization to produce JSON output
        with proper datetime formatting.

        Args:
            report: The ArchitectureReport to render.

        Returns:
            A JSON-formatted string.

        Raises:
            RendererError: If JSON serialization fails.
        """
        try:
            data = report.model_dump(mode="json")
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as exc:
            raise RendererError(f"Failed to render JSON report: {exc}") from exc

    def render_to_file(
        self,
        report: ArchitectureReport,
        output_path: Path,
        output_format: str = "markdown",
    ) -> None:
        """Render a report and write it to a file.

        Args:
            report: The ArchitectureReport to render.
            output_path: Destination file path.
            output_format: 'markdown' or 'json'.

        Raises:
            RendererError: If rendering or file writing fails.
            ValueError: If output_format is not recognized.
        """
        if output_format.lower() == "json":
            content = self.render_json(report)
        elif output_format.lower() == "markdown":
            content = self.render_markdown(report)
        else:
            raise ValueError(
                f"Unknown output_format '{output_format}'. Expected 'markdown' or 'json'."
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise RendererError(
                f"Failed to write report to '{output_path}': {exc}"
            ) from exc

    @staticmethod
    def _confidence_bar_filter(value: float, width: int = 10) -> str:
        """Jinja2 filter: render a confidence score as an ASCII progress bar.

        Args:
            value: Confidence score between 0.0 and 1.0.
            width: Width of the bar in characters.

        Returns:
            ASCII progress bar string, e.g. '[████████░░]'.
        """
        filled = round(value * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}] {value:.0%}"

    @staticmethod
    def _confidence_emoji_filter(value: float) -> str:
        """Jinja2 filter: return an emoji representing confidence level.

        Args:
            value: Confidence score between 0.0 and 1.0.

        Returns:
            '🟢' for high, '🟡' for medium, '🔴' for low confidence.
        """
        if value >= 0.7:
            return "🟢"
        elif value >= 0.4:
            return "🟡"
        else:
            return "🔴"
