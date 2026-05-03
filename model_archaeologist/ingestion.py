"""Document ingestion layer for Model Archaeologist.

Handles fetching and extracting clean text from:
- URLs (via async HTTP with httpx and HTML scraping with BeautifulSoup4)
- Local PDF files (via pdfplumber)
- Local plain text files (direct read)

All public methods return a normalized dict with 'source', 'text',
'content_type', and 'title' keys for downstream consumption by the
chunker and analyzer.
"""

from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path
from typing import Any

import httpx
import pdfplumber
from bs4 import BeautifulSoup

# Default HTTP request timeout in seconds
DEFAULT_TIMEOUT = 30.0

# User-agent string to avoid being blocked by basic bot filters
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; ModelArchaeologist/0.1; "
    "+https://github.com/example/model-archaeologist)"
)

# HTML tags whose content is considered boilerplate and should be removed
_BOILERPLATE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript"]

# Maximum number of consecutive whitespace characters to normalize
_MULTI_SPACE_RE = re.compile(r" {2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


class IngestionError(Exception):
    """Raised when a document cannot be fetched or its text cannot be extracted.

    Wraps lower-level exceptions from httpx, pdfplumber, and the stdlib
    with a consistent error type and descriptive message.
    """


class DocumentIngester:
    """Fetches and extracts clean plain text from URLs and local files.

    Provides two primary async entry points:

    - :meth:`fetch_url`: Retrieve a remote resource over HTTP/HTTPS and
      extract its text content. Handles both HTML pages and PDF responses.
    - :meth:`read_file`: Read a local file from disk and extract its text
      content. Handles ``.pdf`` files and any other extension as plain text.

    All methods return a normalized dict with the following keys:

    - ``source`` (str): The original URL or file path.
    - ``text`` (str): Extracted plain text content.
    - ``content_type`` (str): Detected MIME type.
    - ``title`` (str | None): Page title if available.

    Args:
        timeout: HTTP request timeout in seconds. Defaults to 30.
        user_agent: User-agent header string for HTTP requests.
        verbose: If True, print debug information to stdout.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        verbose: bool = False,
    ) -> None:
        """Initialize the DocumentIngester with HTTP configuration."""
        self.timeout = timeout
        self.user_agent = user_agent
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public async API
    # ------------------------------------------------------------------

    async def fetch_url(self, url: str) -> dict[str, Any]:
        """Fetch a URL and extract its plain text content.

        Determines the content type from the HTTP response headers and
        URL suffix, then dispatches to the appropriate extractor:

        - **HTML**: Scrapes meaningful text using BeautifulSoup4, stripping
          boilerplate tags and preferring ``<article>``/``<main>`` containers.
        - **PDF**: Downloads the raw bytes and extracts text page-by-page
          via pdfplumber.

        Args:
            url: The URL to fetch. Must be an http:// or https:// URL.

        Returns:
            A dict with keys:

            - ``source``: The original URL.
            - ``text``: Extracted plain text (may be empty for binary-only
              or image-heavy PDFs).
            - ``content_type``: The value of the ``Content-Type`` response
              header (lower-cased), e.g. ``'text/html; charset=utf-8'``.
            - ``title``: Page title string for HTML responses, ``None`` for
              PDFs and pages without a ``<title>`` element.

        Raises:
            IngestionError: If the network request fails, the server returns
                a non-2xx status code, or the content cannot be parsed.
        """
        if self.verbose:
            print(f"[ingestion] Fetching URL: {url}")

        response = await self._http_get(url)
        content_type = response.headers.get("content-type", "").lower()

        # Detect PDF by content-type header or URL suffix
        is_pdf = "pdf" in content_type or url.lower().rstrip("/").endswith(".pdf")

        if is_pdf:
            text = self._extract_pdf_from_bytes(response.content, source=url)
            return {
                "source": url,
                "text": text,
                "content_type": "application/pdf",
                "title": None,
            }

        # Default: treat as HTML / text
        text, title = self._extract_html_text(response.text, url=url)
        return {
            "source": url,
            "text": text,
            "content_type": content_type or "text/html",
            "title": title,
        }

    async def read_file(self, file_path: Path) -> dict[str, Any]:
        """Read a local file and extract its plain text content.

        Dispatches based on the file extension:

        - ``.pdf``: Extracts text via pdfplumber (runs in a thread pool to
          avoid blocking the async event loop).
        - Everything else: Reads as UTF-8 plain text (with ``errors='replace'``
          to handle non-UTF-8 bytes gracefully).

        Args:
            file_path: :class:`~pathlib.Path` pointing to the local file.
                The path must exist and must be a regular file.

        Returns:
            A dict with keys:

            - ``source``: The file path as a string.
            - ``text``: Extracted plain text.
            - ``content_type``: ``'application/pdf'`` or ``'text/plain'``.
            - ``title``: The file's stem (name without extension).

        Raises:
            IngestionError: If the file does not exist, is not a regular
                file, cannot be opened, or its content cannot be parsed.
        """
        if not file_path.exists():
            raise IngestionError(f"File not found: '{file_path}'")
        if not file_path.is_file():
            raise IngestionError(f"Path is not a regular file: '{file_path}'")

        if self.verbose:
            print(f"[ingestion] Reading file: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            # Run synchronous pdfplumber in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, self._extract_pdf_from_path, file_path
            )
            return {
                "source": str(file_path),
                "text": text,
                "content_type": "application/pdf",
                "title": file_path.stem,
            }

        # Plain text (or any other extension treated as text)
        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                lambda: file_path.read_text(encoding="utf-8", errors="replace"),
            )
        except OSError as exc:
            raise IngestionError(
                f"Cannot read file '{file_path}': {exc}"
            ) from exc

        return {
            "source": str(file_path),
            "text": text,
            "content_type": "text/plain",
            "title": file_path.stem,
        }

    # ------------------------------------------------------------------
    # Private helpers – HTTP
    # ------------------------------------------------------------------

    async def _http_get(self, url: str) -> httpx.Response:
        """Perform an async HTTP GET request and return the response.

        Applies the configured timeout and user-agent, follows redirects,
        and raises :class:`IngestionError` for all failure modes.

        Args:
            url: The URL to request.

        Returns:
            The successful :class:`httpx.Response` object.

        Raises:
            IngestionError: On timeout, non-2xx status, or network error.
        """
        headers = {"User-Agent": self.user_agent}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
                headers=headers,
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
                return response
        except httpx.TimeoutException as exc:
            raise IngestionError(
                f"Request timed out for URL '{url}': {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise IngestionError(
                f"HTTP {exc.response.status_code} error fetching URL '{url}'"
            ) from exc
        except httpx.RequestError as exc:
            raise IngestionError(
                f"Network error fetching URL '{url}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Private helpers – HTML extraction
    # ------------------------------------------------------------------

    def _extract_html_text(self, html: str, url: str = "") -> tuple[str, str | None]:
        """Extract readable plain text from an HTML string.

        Processing steps:

        1. Parse the HTML with ``html.parser`` (no external C dependencies).
        2. Extract the ``<title>`` element text.
        3. Remove all boilerplate tags: ``<script>``, ``<style>``, ``<nav>``,
           ``<header>``, ``<footer>``, ``<aside>``, ``<noscript>``.
        4. Prefer semantic containers: if an ``<article>`` or ``<main>``
           element exists, extract text only from that subtree; otherwise
           use the full document body.
        5. Join all non-empty text nodes with spaces and normalize whitespace.

        Args:
            html: Raw HTML source code.
            url: Source URL used only for error messages.

        Returns:
            A ``(extracted_text, page_title)`` tuple. ``page_title`` is
            ``None`` if no ``<title>`` tag was found.

        Raises:
            IngestionError: If BeautifulSoup raises an unexpected exception
                during parsing.
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            raise IngestionError(
                f"Failed to parse HTML from '{url}': {exc}"
            ) from exc

        # ---- Extract page title ----
        title_tag = soup.find("title")
        title: str | None = title_tag.get_text(strip=True) if title_tag else None

        # ---- Remove boilerplate tags in-place ----
        for tag in soup.find_all(_BOILERPLATE_TAGS):
            tag.decompose()

        # ---- Prefer semantic content containers ----
        main_content = soup.find("article") or soup.find("main") or soup.find("body")
        target = main_content if main_content else soup

        # ---- Collect all visible text nodes ----
        lines: list[str] = []
        for element in target.find_all(string=True):
            text = element.strip()
            if text:
                lines.append(text)

        # ---- Normalize whitespace ----
        raw = " ".join(lines)
        raw = _MULTI_SPACE_RE.sub(" ", raw)
        raw = _MULTI_NEWLINE_RE.sub("\n\n", raw)
        extracted = raw.strip()

        return extracted, title

    # ------------------------------------------------------------------
    # Private helpers – PDF extraction
    # ------------------------------------------------------------------

    def _extract_pdf_from_path(self, file_path: Path) -> str:
        """Extract all text from a local PDF file using pdfplumber.

        Iterates over every page and concatenates non-empty page text
        blocks separated by double newlines, preserving approximate
        document structure.

        This method is synchronous and is intended to be called from a
        thread-pool executor to avoid blocking the async event loop.

        Args:
            file_path: Absolute or relative path to a ``.pdf`` file.

        Returns:
            All extracted text as a single string. Returns an empty string
            if the PDF contains no extractable text (e.g. scanned images).

        Raises:
            IngestionError: If pdfplumber cannot open or parse the file.
        """
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                pages: list[str] = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text.strip())
                return "\n\n".join(pages)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                f"Failed to extract text from PDF '{file_path}': {exc}"
            ) from exc

    def _extract_pdf_from_bytes(self, content: bytes, source: str = "") -> str:
        """Extract all text from PDF content held in memory using pdfplumber.

        Identical to :meth:`_extract_pdf_from_path` but operates on a
        :class:`bytes` object instead of a file path. Used when a PDF is
        downloaded from a URL rather than read from disk.

        Args:
            content: Raw PDF bytes (e.g. from an HTTP response body).
            source: Source identifier (URL or description) for error messages.

        Returns:
            All extracted text as a single string. Returns an empty string
            if the PDF contains no extractable text.

        Raises:
            IngestionError: If the bytes do not represent a valid PDF or
                pdfplumber raises any exception during processing.
        """
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages: list[str] = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pages.append(page_text.strip())
                return "\n\n".join(pages)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                f"Failed to extract text from PDF bytes (source: '{source}'): {exc}"
            ) from exc
