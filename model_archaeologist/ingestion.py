"""Document ingestion layer for Model Archaeologist.

Handles fetching and extracting clean text from:
- URLs (via async HTTP with httpx and HTML scraping with BeautifulSoup4)
- Local PDF files (via pdfplumber)
- Local plain text files (direct read)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import pdfplumber
from bs4 import BeautifulSoup

# Default HTTP request timeout in seconds
DEFAULT_TIMEOUT = 30.0

# User-agent to avoid being blocked by some sites
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; ModelArchaeologist/0.1; +https://github.com/example/model-archaeologist)"
)


class IngestionError(Exception):
    """Raised when a document cannot be ingested."""


class DocumentIngester:
    """Fetches and extracts text from URLs and local files.

    This class provides async methods for ingesting documents from
    various sources and returning a normalized dictionary with the
    extracted text and metadata.

    Args:
        timeout: HTTP request timeout in seconds.
        user_agent: User-agent string for HTTP requests.
        verbose: Enable verbose logging to stdout.
    """

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        user_agent: str = DEFAULT_USER_AGENT,
        verbose: bool = False,
    ) -> None:
        """Initialize the DocumentIngester."""
        self.timeout = timeout
        self.user_agent = user_agent
        self.verbose = verbose

    async def fetch_url(self, url: str) -> dict[str, Any]:
        """Fetch a URL and extract its text content.

        Handles HTML pages (scraped with BeautifulSoup) and PDF files
        (detected via Content-Type header and extracted with pdfplumber).

        Args:
            url: The URL to fetch.

        Returns:
            A dict with keys:
                - ``source``: The original URL.
                - ``text``: Extracted plain text content.
                - ``content_type``: The detected content type.
                - ``title``: Page title if available (HTML only).

        Raises:
            IngestionError: If the URL cannot be fetched or parsed.
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
        except httpx.TimeoutException as exc:
            raise IngestionError(f"Request timed out for URL '{url}': {exc}") from exc
        except httpx.HTTPStatusError as exc:
            raise IngestionError(
                f"HTTP {exc.response.status_code} error for URL '{url}'"
            ) from exc
        except httpx.RequestError as exc:
            raise IngestionError(f"Network error fetching URL '{url}': {exc}") from exc

        content_type = response.headers.get("content-type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            text = self._extract_pdf_from_bytes(response.content, source=url)
            return {
                "source": url,
                "text": text,
                "content_type": "application/pdf",
                "title": None,
            }
        else:
            text, title = self._extract_html_text(response.text, url=url)
            return {
                "source": url,
                "text": text,
                "content_type": content_type,
                "title": title,
            }

    async def read_file(self, file_path: Path) -> dict[str, Any]:
        """Read a local file and extract its text content.

        Supports PDF files (via pdfplumber) and plain text files.
        The operation runs in a thread pool to avoid blocking the event loop.

        Args:
            file_path: Path to the local file.

        Returns:
            A dict with keys:
                - ``source``: The file path as a string.
                - ``text``: Extracted plain text content.
                - ``content_type``: 'application/pdf' or 'text/plain'.
                - ``title``: The filename stem.

        Raises:
            IngestionError: If the file cannot be read or parsed.
        """
        if not file_path.exists():
            raise IngestionError(f"File not found: '{file_path}'")
        if not file_path.is_file():
            raise IngestionError(f"Path is not a file: '{file_path}'")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text = await asyncio.get_event_loop().run_in_executor(
                None, self._extract_pdf_from_path, file_path
            )
            return {
                "source": str(file_path),
                "text": text,
                "content_type": "application/pdf",
                "title": file_path.stem,
            }
        else:
            # Treat everything else as plain text
            try:
                text = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: file_path.read_text(encoding="utf-8", errors="replace")
                )
            except OSError as exc:
                raise IngestionError(f"Cannot read file '{file_path}': {exc}") from exc
            return {
                "source": str(file_path),
                "text": text,
                "content_type": "text/plain",
                "title": file_path.stem,
            }

    def _extract_html_text(self, html: str, url: str = "") -> tuple[str, str | None]:
        """Extract readable text from an HTML string using BeautifulSoup4.

        Removes script, style, nav, header, and footer elements, then
        extracts the remaining text with normalized whitespace.

        Args:
            html: Raw HTML content as a string.
            url: Source URL for logging purposes.

        Returns:
            A tuple of (extracted_text, page_title).
        """
        try:
            soup = BeautifulSoup(html, "html.parser")
        except Exception as exc:
            raise IngestionError(f"Failed to parse HTML from '{url}': {exc}") from exc

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        # Remove boilerplate elements
        for tag in soup.find_all(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()

        # Prefer <article> or <main> content if available
        main_content = soup.find("article") or soup.find("main")
        target = main_content if main_content else soup

        # Extract lines and normalize whitespace
        lines: list[str] = []
        for element in target.find_all(string=True):
            text = element.strip()
            if text:
                lines.append(text)

        extracted = " ".join(lines)
        # Collapse multiple spaces
        import re
        extracted = re.sub(r" {2,}", " ", extracted)
        extracted = re.sub(r"\n{3,}", "\n\n", extracted)

        return extracted.strip(), title

    def _extract_pdf_from_path(self, file_path: Path) -> str:
        """Extract text from a local PDF file using pdfplumber.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Extracted text as a single string.

        Raises:
            IngestionError: If the PDF cannot be opened or parsed.
        """
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                pages: list[str] = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                return "\n\n".join(pages)
        except Exception as exc:
            raise IngestionError(f"Failed to extract PDF '{file_path}': {exc}") from exc

    def _extract_pdf_from_bytes(self, content: bytes, source: str = "") -> str:
        """Extract text from PDF content in memory using pdfplumber.

        Args:
            content: Raw PDF bytes.
            source: Source identifier for error messages.

        Returns:
            Extracted text as a single string.

        Raises:
            IngestionError: If the PDF bytes cannot be parsed.
        """
        import io

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages: list[str] = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
                return "\n\n".join(pages)
        except Exception as exc:
            raise IngestionError(
                f"Failed to extract PDF from bytes (source: '{source}'): {exc}"
            ) from exc
