"""Report renderer for Model Archaeologist.

Renders :class:`~model_archaeologist.schema.ArchitectureReport` objects into
either a Markdown string (via a Jinja2 template) or a pretty-printed JSON
string (via Pydantic's model serialization).

The Markdown renderer uses the Jinja2 ``Environment`` with ``trim_blocks`` and
``lstrip_blocks`` enabled to produce clean, human-readable output without
spurious blank lines from template control tags.

Custom Jinja2 filters provided:

- ``confidence_bar(value, width=10)`` — renders an ASCII progress bar.
- ``confidence_emoji(value)`` — returns a colored circle emoji.
- ``confidence_label(value)`` — returns ``'high'``, ``'medium'``, or ``'low'``.
- ``format_datetime(dt, fmt)`` — formats a datetime with a strftime format string.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, TemplateNotFound, select_autoescape

from model_archaeologist.schema import ArchitectureReport

# Absolute path to the package's built-in templates directory.
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Name of the primary Markdown report template.
MARKDOWN_TEMPLATE_NAME = "report.md.j2"


class RendererError(Exception):
    """Raised when report rendering encounters an unrecoverable error.

    Wraps Jinja2 exceptions and JSON serialization errors with a consistent
    error type and descriptive message.
    """


class ReportRenderer:
    """Renders ArchitectureReport objects into Markdown or JSON output.

    Uses a Jinja2 ``Environment`` backed by a ``FileSystemLoader`` for
    Markdown rendering and Pydantic's built-in ``model_dump(mode='json')``
    for JSON rendering.

    The Jinja2 environment is created lazily on first use and cached for
    subsequent calls, so creating a ``ReportRenderer`` instance is cheap.

    Example usage::

        renderer = ReportRenderer()
        markdown_str = renderer.render_markdown(report)
        json_str = renderer.render_json(report)
        renderer.render_to_file(report, Path("output.md"), output_format="markdown")

    Args:
        templates_dir: Path to the directory containing Jinja2 template files.
            Defaults to the package's built-in ``templates/`` directory.
    """

    def __init__(self, templates_dir: Path = TEMPLATES_DIR) -> None:
        """Initialise the ReportRenderer.

        Args:
            templates_dir: Override the default templates directory.
        """
        self.templates_dir = templates_dir
        self._env: Environment | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, report: ArchitectureReport) -> str:
        """Render a report as a Markdown-formatted string.

        Uses the ``report.md.j2`` Jinja2 template from the configured
        templates directory.  All custom filters are available within the
        template (``confidence_bar``, ``confidence_emoji``, ``confidence_label``,
        ``format_datetime``).

        Args:
            report: The :class:`~model_archaeologist.schema.ArchitectureReport`
                to render.

        Returns:
            A Markdown string ready for writing to a ``.md`` file or stdout.

        Raises:
            RendererError: If the templates directory does not exist, the
                template file is missing, or Jinja2 raises any rendering
                exception.
        """
        try:
            env = self._get_env()
            template = env.get_template(MARKDOWN_TEMPLATE_NAME)
            return template.render(report=report)
        except RendererError:
            raise
        except TemplateNotFound as exc:
            raise RendererError(
                f"Markdown template not found: '{exc}'.  "
                f"Expected '{MARKDOWN_TEMPLATE_NAME}' in '{self.templates_dir}'."
            ) from exc
        except Exception as exc:
            raise RendererError(
                f"Failed to render Markdown report: {type(exc).__name__}: {exc}"
            ) from exc

    def render_json(self, report: ArchitectureReport) -> str:
        """Render a report as a pretty-printed JSON string.

        Uses Pydantic's ``model_dump(mode='json')`` which serializes
        :class:`~datetime.datetime` objects to ISO-8601 strings automatically.

        Args:
            report: The :class:`~model_archaeologist.schema.ArchitectureReport`
                to serialize.

        Returns:
            A JSON string with 2-space indentation, UTF-8 safe (non-ASCII
            characters are preserved, not escaped).

        Raises:
            RendererError: If Pydantic serialization or ``json.dumps`` fails.
        """
        try:
            data = report.model_dump(mode="json")
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as exc:
            raise RendererError(
                f"Failed to render JSON report: {type(exc).__name__}: {exc}"
            ) from exc

    def render_to_file(
        self,
        report: ArchitectureReport,
        output_path: Path,
        output_format: str = "markdown",
    ) -> None:
        """Render a report and write the result to a file.

        Creates parent directories if they do not exist.  The file is written
        with UTF-8 encoding.

        Args:
            report: The :class:`~model_archaeologist.schema.ArchitectureReport`
                to render.
            output_path: Destination file path.  Parent directories are
                created automatically.
            output_format: ``'markdown'`` (default) or ``'json'``.

        Raises:
            ValueError: If ``output_format`` is not ``'markdown'`` or
                ``'json'`` (case-insensitive).
            RendererError: If rendering fails or the file cannot be written.
        """
        fmt = output_format.lower()
        if fmt == "json":
            content = self.render_json(report)
        elif fmt == "markdown":
            content = self.render_markdown(report)
        else:
            raise ValueError(
                f"Unknown output_format '{output_format}'.  "
                "Expected 'markdown' or 'json'."
            )

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
        except OSError as exc:
            raise RendererError(
                f"Failed to write report to '{output_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_env(self) -> Environment:
        """Lazily create and cache the Jinja2 Environment.

        Returns:
            A configured :class:`jinja2.Environment` with custom filters
            registered and ``trim_blocks`` / ``lstrip_blocks`` enabled.

        Raises:
            RendererError: If :attr:`templates_dir` does not exist or is not
                a directory.
        """
        if self._env is not None:
            return self._env

        if not self.templates_dir.exists():
            raise RendererError(
                f"Templates directory does not exist: '{self.templates_dir}'"
            )
        if not self.templates_dir.is_dir():
            raise RendererError(
                f"Templates path is not a directory: '{self.templates_dir}'"
            )

        env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            # No HTML autoescaping — we are producing Markdown, not HTML
            autoescape=select_autoescape([]),
            # Remove the newline after block tags ({% ... %}) to avoid extra
            # blank lines in the rendered Markdown output
            trim_blocks=True,
            # Strip leading whitespace before block tags
            lstrip_blocks=True,
            # Preserve the trailing newline at the end of templates
            keep_trailing_newline=True,
            # Raise UndefinedError instead of rendering empty string for
            # missing variables — helps catch template bugs early
            undefined=_StrictUndefined,
        )

        # Register custom Jinja2 filters
        env.filters["confidence_bar"] = _confidence_bar
        env.filters["confidence_emoji"] = _confidence_emoji
        env.filters["confidence_label"] = _confidence_label
        env.filters["format_datetime"] = _format_datetime

        self._env = env
        return env


# ---------------------------------------------------------------------------
# Jinja2 filter functions (module-level for easy unit testing)
# ---------------------------------------------------------------------------


def _confidence_bar(value: float, width: int = 10) -> str:
    """Render a confidence score as an ASCII progress bar string.

    The bar consists of filled (``█``) and empty (``░``) block characters
    followed by the percentage representation of the score.

    Examples::

        _confidence_bar(0.0)   ->  "[░░░░░░░░░░]  0%"
        _confidence_bar(0.5)   ->  "[█████░░░░░] 50%"
        _confidence_bar(1.0)   ->  "[██████████] 100%"

    Args:
        value: Confidence score in the range ``[0.0, 1.0]``.
        width: Total width of the bar in block characters.  Defaults to 10.

    Returns:
        An ASCII progress bar string.
    """
    clamped = max(0.0, min(1.0, float(value)))
    filled = round(clamped * width)
    empty = width - filled
    bar = "\u2588" * filled + "\u2591" * empty
    return f"[{bar}] {clamped:.0%}"


def _confidence_emoji(value: float) -> str:
    """Return a colored circle emoji representing a confidence level.

    - 🟢 green for high confidence (>= 0.7)
    - 🟡 yellow for medium confidence (>= 0.4)
    - 🔴 red for low confidence (< 0.4)

    Args:
        value: Confidence score in the range ``[0.0, 1.0]``.

    Returns:
        A single emoji character string.
    """
    clamped = max(0.0, min(1.0, float(value)))
    if clamped >= 0.7:
        return "\U0001f7e2"  # 🟢
    elif clamped >= 0.4:
        return "\U0001f7e1"  # 🟡
    else:
        return "\U0001f534"  # 🔴


def _confidence_label(value: float) -> str:
    """Return a human-readable confidence label string.

    - ``'high'`` for confidence >= 0.7
    - ``'medium'`` for confidence >= 0.4
    - ``'low'`` for confidence < 0.4

    Args:
        value: Confidence score in the range ``[0.0, 1.0]``.

    Returns:
        One of ``'high'``, ``'medium'``, or ``'low'``.
    """
    clamped = max(0.0, min(1.0, float(value)))
    if clamped >= 0.7:
        return "high"
    elif clamped >= 0.4:
        return "medium"
    else:
        return "low"


def _format_datetime(dt: Any, fmt: str = "%Y-%m-%d %H:%M:%S UTC") -> str:
    """Format a datetime object using a strftime format string.

    Args:
        dt: A :class:`~datetime.datetime` object or any object with a
            ``strftime`` method.
        fmt: A strftime format string.  Defaults to
            ``'%Y-%m-%d %H:%M:%S UTC'``.

    Returns:
        The formatted datetime string, or ``str(dt)`` if ``dt`` lacks a
        ``strftime`` method.
    """
    if hasattr(dt, "strftime"):
        return dt.strftime(fmt)
    return str(dt)


# ---------------------------------------------------------------------------
# Custom Jinja2 Undefined class
# ---------------------------------------------------------------------------


try:
    from jinja2 import StrictUndefined as _StrictUndefined
except ImportError:
    from jinja2 import Undefined as _StrictUndefined  # type: ignore[assignment]
