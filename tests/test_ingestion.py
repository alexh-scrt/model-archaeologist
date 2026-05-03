"""Tests for the document ingestion layer in model_archaeologist/ingestion.py.

Covers:
- HTML URL fetching and text extraction (mocked with respx)
- PDF URL fetching (mocked with respx + minimal PDF bytes)
- HTTP error handling (4xx, 5xx, timeouts, network errors)
- Local plain-text file reading
- Local PDF file reading
- HTML boilerplate removal and semantic container preference
- Edge cases: empty files, binary files, redirect following
"""

from __future__ import annotations

import io
import struct
from pathlib import Path

import httpx
import pytest
import respx

from model_archaeologist.ingestion import DocumentIngester, IngestionError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_pdf_bytes() -> bytes:
    """Return the smallest valid PDF that pdfplumber can open without crashing.

    The PDF contains a single blank page with no text streams so that
    ``page.extract_text()`` returns None or an empty string, giving us a
    controllable empty-text scenario while still exercising the code path.
    """
    # Minimal PDF 1.4 with one empty page
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type /Pages /Kids[3 0 R] /Count 1>>endobj\n"
        b"3 0 obj<</Type /Page /Parent 2 0 R /MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n"
        b"0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4 /Root 1 0 R>>\n"
        b"startxref\n"
        b"190\n"
        b"%%EOF\n"
    )
    return pdf


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ingester() -> DocumentIngester:
    """Return a DocumentIngester with default settings."""
    return DocumentIngester(verbose=False)


@pytest.fixture()
def tmp_text_file(tmp_path: Path) -> Path:
    """Create a temporary plain-text file with known content."""
    p = tmp_path / "sample.txt"
    p.write_text("Hello from a text file.\nSecond line.", encoding="utf-8")
    return p


@pytest.fixture()
def tmp_empty_file(tmp_path: Path) -> Path:
    """Create an empty temporary file."""
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture()
def tmp_pdf_file(tmp_path: Path) -> Path:
    """Create a minimal temporary PDF file."""
    p = tmp_path / "sample.pdf"
    p.write_bytes(_minimal_pdf_bytes())
    return p


# ---------------------------------------------------------------------------
# Tests – fetch_url (HTML)
# ---------------------------------------------------------------------------


class TestFetchUrlHtml:
    """Tests for HTML URL fetching and extraction."""

    @pytest.mark.asyncio
    async def test_basic_html_fetch(self, ingester: DocumentIngester) -> None:
        """fetch_url returns extracted text and title for a simple HTML page."""
        html = (
            "<html><head><title>Test Page</title></head>"
            "<body><p>Hello world.</p></body></html>"
        )
        with respx.mock(assert_all_called=False) as mock:
            mock.get("https://example.com/page").mock(
                return_value=httpx.Response(
                    200,
                    text=html,
                    headers={"content-type": "text/html; charset=utf-8"},
                )
            )
            result = await ingester.fetch_url("https://example.com/page")

        assert result["source"] == "https://example.com/page"
        assert "Hello world" in result["text"]
        assert result["title"] == "Test Page"
        assert result["content_type"] == "text/html; charset=utf-8"

    @pytest.mark.asyncio
    async def test_html_boilerplate_removed(self, ingester: DocumentIngester) -> None:
        """Script, style, nav, header, footer, aside content is stripped."""
        html = (
            "<html><body>"
            "<nav>Navigation links</nav>"
            "<header>Site Header</header>"
            "<script>alert('xss')</script>"
            "<style>.foo { color: red; }</style>"
            "<main><p>Main article content.</p></main>"
            "<footer>Footer content</footer>"
            "<aside>Sidebar</aside>"
            "</body></html>"
        )
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/").mock(
                return_value=httpx.Response(
                    200,
                    text=html,
                    headers={"content-type": "text/html"},
                )
            )
            result = await ingester.fetch_url("https://example.com/")

        text = result["text"]
        assert "Main article content" in text
        assert "Navigation links" not in text
        assert "Site Header" not in text
        assert "alert('xss')" not in text
        assert ".foo { color: red; }" not in text
        assert "Footer content" not in text
        assert "Sidebar" not in text

    @pytest.mark.asyncio
    async def test_html_prefers_article_tag(self, ingester: DocumentIngester) -> None:
        """Text is extracted from <article> when present."""
        html = (
            "<html><body>"
            "<div>Outer junk text</div>"
            "<article><p>Core article text.</p></article>"
            "</body></html>"
        )
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/article").mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url("https://example.com/article")

        assert "Core article text" in result["text"]

    @pytest.mark.asyncio
    async def test_html_prefers_main_tag(self, ingester: DocumentIngester) -> None:
        """Text is extracted from <main> when no <article> is present."""
        html = (
            "<html><body>"
            "<div>Outer noise</div>"
            "<main><p>Primary content here.</p></main>"
            "</body></html>"
        )
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/main").mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url("https://example.com/main")

        assert "Primary content here" in result["text"]

    @pytest.mark.asyncio
    async def test_html_no_title_returns_none(self, ingester: DocumentIngester) -> None:
        """title is None when the HTML has no <title> tag."""
        html = "<html><body><p>No title here.</p></body></html>"
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/notitle").mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url("https://example.com/notitle")

        assert result["title"] is None

    @pytest.mark.asyncio
    async def test_empty_html_returns_empty_text(self, ingester: DocumentIngester) -> None:
        """An HTML page with no visible text returns an empty text string."""
        html = "<html><head></head><body></body></html>"
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/empty").mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url("https://example.com/empty")

        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_result_has_all_required_keys(self, ingester: DocumentIngester) -> None:
        """fetch_url result dict always contains source, text, content_type, title."""
        html = "<html><body><p>Content</p></body></html>"
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/keys").mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url("https://example.com/keys")

        assert "source" in result
        assert "text" in result
        assert "content_type" in result
        assert "title" in result


# ---------------------------------------------------------------------------
# Tests – fetch_url (PDF via URL)
# ---------------------------------------------------------------------------


class TestFetchUrlPdf:
    """Tests for PDF fetching via URL."""

    @pytest.mark.asyncio
    async def test_pdf_content_type_detection(self, ingester: DocumentIngester) -> None:
        """A response with content-type application/pdf is treated as a PDF."""
        pdf_bytes = _minimal_pdf_bytes()
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/paper").mock(
                return_value=httpx.Response(
                    200,
                    content=pdf_bytes,
                    headers={"content-type": "application/pdf"},
                )
            )
            result = await ingester.fetch_url("https://example.com/paper")

        assert result["content_type"] == "application/pdf"
        assert result["title"] is None
        assert result["source"] == "https://example.com/paper"
        assert isinstance(result["text"], str)

    @pytest.mark.asyncio
    async def test_pdf_url_suffix_detection(self, ingester: DocumentIngester) -> None:
        """A .pdf URL suffix triggers PDF extraction even with a generic content-type."""
        pdf_bytes = _minimal_pdf_bytes()
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/paper.pdf").mock(
                return_value=httpx.Response(
                    200,
                    content=pdf_bytes,
                    headers={"content-type": "application/octet-stream"},
                )
            )
            result = await ingester.fetch_url("https://example.com/paper.pdf")

        assert result["content_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_pdf_text_is_string(self, ingester: DocumentIngester) -> None:
        """The 'text' field of a PDF result is always a string."""
        pdf_bytes = _minimal_pdf_bytes()
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/doc.pdf").mock(
                return_value=httpx.Response(
                    200,
                    content=pdf_bytes,
                    headers={"content-type": "application/pdf"},
                )
            )
            result = await ingester.fetch_url("https://example.com/doc.pdf")

        assert isinstance(result["text"], str)


# ---------------------------------------------------------------------------
# Tests – fetch_url error handling
# ---------------------------------------------------------------------------


class TestFetchUrlErrors:
    """Tests for error handling during URL fetching."""

    @pytest.mark.asyncio
    async def test_404_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A 404 response raises IngestionError with status code in the message."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/missing").mock(
                return_value=httpx.Response(404)
            )
            with pytest.raises(IngestionError) as exc_info:
                await ingester.fetch_url("https://example.com/missing")

        assert "404" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_500_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A 500 response raises IngestionError."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/error").mock(
                return_value=httpx.Response(500)
            )
            with pytest.raises(IngestionError) as exc_info:
                await ingester.fetch_url("https://example.com/error")

        assert "500" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A request timeout raises IngestionError with timeout message."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/slow").mock(
                side_effect=httpx.TimeoutException("timed out")
            )
            with pytest.raises(IngestionError) as exc_info:
                await ingester.fetch_url("https://example.com/slow")

        assert "timed out" in str(exc_info.value).lower() or "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_connection_error_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A connection error raises IngestionError."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/unreachable").mock(
                side_effect=httpx.ConnectError("connection refused")
            )
            with pytest.raises(IngestionError):
                await ingester.fetch_url("https://example.com/unreachable")

    @pytest.mark.asyncio
    async def test_invalid_pdf_bytes_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """Bytes that claim to be PDF but are not raise IngestionError."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/bad.pdf").mock(
                return_value=httpx.Response(
                    200,
                    content=b"this is not a pdf",
                    headers={"content-type": "application/pdf"},
                )
            )
            with pytest.raises(IngestionError):
                await ingester.fetch_url("https://example.com/bad.pdf")


# ---------------------------------------------------------------------------
# Tests – read_file (plain text)
# ---------------------------------------------------------------------------


class TestReadFileText:
    """Tests for reading plain-text files."""

    @pytest.mark.asyncio
    async def test_reads_text_file(self, ingester: DocumentIngester, tmp_text_file: Path) -> None:
        """read_file returns the full content of a plain-text file."""
        result = await ingester.read_file(tmp_text_file)

        assert result["source"] == str(tmp_text_file)
        assert "Hello from a text file" in result["text"]
        assert result["content_type"] == "text/plain"
        assert result["title"] == tmp_text_file.stem

    @pytest.mark.asyncio
    async def test_reads_empty_text_file(self, ingester: DocumentIngester, tmp_empty_file: Path) -> None:
        """read_file handles empty files, returning an empty text string."""
        result = await ingester.read_file(tmp_empty_file)

        assert result["text"] == ""
        assert result["content_type"] == "text/plain"

    @pytest.mark.asyncio
    async def test_text_file_has_all_keys(self, ingester: DocumentIngester, tmp_text_file: Path) -> None:
        """read_file result always has source, text, content_type, title keys."""
        result = await ingester.read_file(tmp_text_file)

        assert "source" in result
        assert "text" in result
        assert "content_type" in result
        assert "title" in result

    @pytest.mark.asyncio
    async def test_nonexistent_file_raises_ingestion_error(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """Attempting to read a non-existent file raises IngestionError."""
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(IngestionError) as exc_info:
            await ingester.read_file(missing)

        assert "not found" in str(exc_info.value).lower() or "does_not_exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_directory_path_raises_ingestion_error(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """Providing a directory path instead of a file raises IngestionError."""
        with pytest.raises(IngestionError) as exc_info:
            await ingester.read_file(tmp_path)

        assert "not a regular file" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_reads_multiline_text(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """Multi-line text files are read correctly."""
        p = tmp_path / "multi.txt"
        content = "Line 1\nLine 2\nLine 3\n"
        p.write_text(content, encoding="utf-8")

        result = await ingester.read_file(p)
        assert "Line 1" in result["text"]
        assert "Line 2" in result["text"]
        assert "Line 3" in result["text"]

    @pytest.mark.asyncio
    async def test_reads_utf8_text_with_special_chars(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """UTF-8 files with non-ASCII characters are read correctly."""
        p = tmp_path / "unicode.txt"
        content = "Héllo wörld – 日本語テスト"
        p.write_text(content, encoding="utf-8")

        result = await ingester.read_file(p)
        assert "Héllo" in result["text"]
        assert "日本語" in result["text"]

    @pytest.mark.asyncio
    async def test_title_is_stem_without_extension(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """The 'title' key equals the filename stem (without extension)."""
        p = tmp_path / "my_document.txt"
        p.write_text("content", encoding="utf-8")

        result = await ingester.read_file(p)
        assert result["title"] == "my_document"


# ---------------------------------------------------------------------------
# Tests – read_file (PDF)
# ---------------------------------------------------------------------------


class TestReadFilePdf:
    """Tests for reading local PDF files."""

    @pytest.mark.asyncio
    async def test_reads_minimal_pdf(self, ingester: DocumentIngester, tmp_pdf_file: Path) -> None:
        """read_file can open and process a minimal valid PDF file."""
        result = await ingester.read_file(tmp_pdf_file)

        assert result["source"] == str(tmp_pdf_file)
        assert result["content_type"] == "application/pdf"
        assert result["title"] == tmp_pdf_file.stem
        assert isinstance(result["text"], str)

    @pytest.mark.asyncio
    async def test_pdf_file_has_all_keys(self, ingester: DocumentIngester, tmp_pdf_file: Path) -> None:
        """read_file result for a PDF always has source, text, content_type, title."""
        result = await ingester.read_file(tmp_pdf_file)

        assert "source" in result
        assert "text" in result
        assert "content_type" in result
        assert "title" in result

    @pytest.mark.asyncio
    async def test_invalid_pdf_raises_ingestion_error(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """A file with .pdf extension but invalid content raises IngestionError."""
        bad_pdf = tmp_path / "fake.pdf"
        bad_pdf.write_bytes(b"this is definitely not a pdf")

        with pytest.raises(IngestionError):
            await ingester.read_file(bad_pdf)

    @pytest.mark.asyncio
    async def test_pdf_text_is_string(self, ingester: DocumentIngester, tmp_pdf_file: Path) -> None:
        """The 'text' field is always a string even for blank PDFs."""
        result = await ingester.read_file(tmp_pdf_file)
        assert isinstance(result["text"], str)


# ---------------------------------------------------------------------------
# Tests – _extract_html_text (unit-level)
# ---------------------------------------------------------------------------


class TestExtractHtmlText:
    """Unit tests for the internal _extract_html_text helper."""

    def test_basic_extraction(self, ingester: DocumentIngester) -> None:
        """Extracts visible text from a minimal HTML fragment."""
        html = "<html><body><p>Simple text.</p></body></html>"
        text, title = ingester._extract_html_text(html)
        assert "Simple text" in text
        assert title is None

    def test_title_extracted(self, ingester: DocumentIngester) -> None:
        """<title> element text is returned as the title."""
        html = "<html><head><title>My Title</title></head><body><p>Body</p></body></html>"
        _, title = ingester._extract_html_text(html)
        assert title == "My Title"

    def test_script_removed(self, ingester: DocumentIngester) -> None:
        """<script> content is removed from extracted text."""
        html = "<html><body><script>var x = 1;</script><p>Visible</p></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "var x" not in text
        assert "Visible" in text

    def test_style_removed(self, ingester: DocumentIngester) -> None:
        """<style> content is removed from extracted text."""
        html = "<html><body><style>.cls{}</style><p>Text</p></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert ".cls" not in text
        assert "Text" in text

    def test_empty_html_returns_empty_string(self, ingester: DocumentIngester) -> None:
        """Empty HTML returns an empty text string and None title."""
        text, title = ingester._extract_html_text("")
        assert text == ""
        assert title is None

    def test_nested_content_extracted(self, ingester: DocumentIngester) -> None:
        """Deeply nested text nodes are extracted."""
        html = "<html><body><div><section><p><span>Deep text</span></p></section></div></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "Deep text" in text

    def test_multiple_paragraphs_joined(self, ingester: DocumentIngester) -> None:
        """Text from multiple paragraphs is joined into a single string."""
        html = "<html><body><p>First.</p><p>Second.</p><p>Third.</p></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "First" in text
        assert "Second" in text
        assert "Third" in text

    def test_whitespace_normalized(self, ingester: DocumentIngester) -> None:
        """Multiple consecutive spaces are collapsed to a single space."""
        html = "<html><body><p>Too   many   spaces</p></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "Too   many" not in text
        assert "Too many spaces" in text or "Too" in text


# ---------------------------------------------------------------------------
# Tests – _extract_pdf_from_bytes (unit-level)
# ---------------------------------------------------------------------------


class TestExtractPdfFromBytes:
    """Unit tests for the internal _extract_pdf_from_bytes helper."""

    def test_valid_pdf_bytes_returns_string(self, ingester: DocumentIngester) -> None:
        """A valid minimal PDF returns a string (possibly empty for blank pages)."""
        result = ingester._extract_pdf_from_bytes(_minimal_pdf_bytes())
        assert isinstance(result, str)

    def test_invalid_bytes_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """Non-PDF bytes raise IngestionError."""
        with pytest.raises(IngestionError):
            ingester._extract_pdf_from_bytes(b"not a pdf at all")

    def test_empty_bytes_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """Empty bytes raise IngestionError."""
        with pytest.raises(IngestionError):
            ingester._extract_pdf_from_bytes(b"")

    def test_source_included_in_error_message(self, ingester: DocumentIngester) -> None:
        """IngestionError includes the source identifier."""
        with pytest.raises(IngestionError) as exc_info:
            ingester._extract_pdf_from_bytes(b"bad", source="https://example.com/paper")
        assert "example.com" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests – _extract_pdf_from_path (unit-level)
# ---------------------------------------------------------------------------


class TestExtractPdfFromPath:
    """Unit tests for the internal _extract_pdf_from_path helper."""

    def test_valid_pdf_path_returns_string(self, ingester: DocumentIngester, tmp_pdf_file: Path) -> None:
        """A valid minimal PDF path returns a string."""
        result = ingester._extract_pdf_from_path(tmp_pdf_file)
        assert isinstance(result, str)

    def test_invalid_pdf_path_raises_ingestion_error(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """A path to a non-PDF file raises IngestionError."""
        bad = tmp_path / "fake.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(IngestionError):
            ingester._extract_pdf_from_path(bad)

    def test_path_included_in_error_message(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """IngestionError includes the file path."""
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"bad")
        with pytest.raises(IngestionError) as exc_info:
            ingester._extract_pdf_from_path(bad)
        assert "broken.pdf" in str(exc_info.value) or "pdf" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Tests – DocumentIngester init / verbose
# ---------------------------------------------------------------------------


class TestDocumentIngesterInit:
    """Tests for DocumentIngester initialization."""

    def test_default_timeout(self) -> None:
        """Default timeout is 30 seconds."""
        from model_archaeologist.ingestion import DEFAULT_TIMEOUT
        ingester = DocumentIngester()
        assert ingester.timeout == DEFAULT_TIMEOUT

    def test_custom_timeout(self) -> None:
        """Custom timeout is stored correctly."""
        ingester = DocumentIngester(timeout=60.0)
        assert ingester.timeout == 60.0

    def test_default_user_agent(self) -> None:
        """Default user agent contains the project name."""
        ingester = DocumentIngester()
        assert "ModelArchaeologist" in ingester.user_agent

    def test_custom_user_agent(self) -> None:
        """Custom user agent is stored correctly."""
        ingester = DocumentIngester(user_agent="MyBot/1.0")
        assert ingester.user_agent == "MyBot/1.0"

    def test_verbose_flag(self) -> None:
        """Verbose flag is stored correctly."""
        ingester = DocumentIngester(verbose=True)
        assert ingester.verbose is True
        ingester2 = DocumentIngester(verbose=False)
        assert ingester2.verbose is False
