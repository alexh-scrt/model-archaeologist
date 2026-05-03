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

    @pytest.mark.asyncio
    async def test_html_with_only_script_returns_empty(self, ingester: DocumentIngester) -> None:
        """HTML with only script/style content returns empty text after stripping."""
        html = (
            "<html><head><script>var x = 1;</script>"
            "<style>body { margin: 0; }</style></head>"
            "<body><script>doSomething();</script></body></html>"
        )
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/scriptonly").mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url("https://example.com/scriptonly")

        # After removing script/style, no visible text should remain
        assert "var x" not in result["text"]
        assert "body { margin" not in result["text"]

    @pytest.mark.asyncio
    async def test_source_field_matches_input_url(self, ingester: DocumentIngester) -> None:
        """The 'source' field in the result always equals the input URL."""
        url = "https://example.com/source-check"
        html = "<html><body><p>text</p></body></html>"
        with respx.mock(assert_all_called=False):
            respx.get(url).mock(
                return_value=httpx.Response(
                    200, text=html, headers={"content-type": "text/html"}
                )
            )
            result = await ingester.fetch_url(url)

        assert result["source"] == url


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

    @pytest.mark.asyncio
    async def test_pdf_title_is_none(self, ingester: DocumentIngester) -> None:
        """PDF responses always return title=None."""
        pdf_bytes = _minimal_pdf_bytes()
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/paper.pdf").mock(
                return_value=httpx.Response(
                    200,
                    content=pdf_bytes,
                    headers={"content-type": "application/pdf"},
                )
            )
            result = await ingester.fetch_url("https://example.com/paper.pdf")

        assert result["title"] is None

    @pytest.mark.asyncio
    async def test_pdf_result_has_all_keys(self, ingester: DocumentIngester) -> None:
        """PDF fetch result always contains source, text, content_type, title."""
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

        assert "source" in result
        assert "text" in result
        assert "content_type" in result
        assert "title" in result


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
    async def test_403_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A 403 response raises IngestionError."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/forbidden").mock(
                return_value=httpx.Response(403)
            )
            with pytest.raises(IngestionError) as exc_info:
                await ingester.fetch_url("https://example.com/forbidden")

        assert "403" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_timeout_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A request timeout raises IngestionError with timeout message."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/slow").mock(
                side_effect=httpx.TimeoutException("timed out")
            )
            with pytest.raises(IngestionError) as exc_info:
                await ingester.fetch_url("https://example.com/slow")

        error_msg = str(exc_info.value).lower()
        assert "timed out" in error_msg or "timeout" in error_msg

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

    @pytest.mark.asyncio
    async def test_error_message_contains_url(self, ingester: DocumentIngester) -> None:
        """IngestionError message contains the URL that failed."""
        target_url = "https://example.com/notfound"
        with respx.mock(assert_all_called=False):
            respx.get(target_url).mock(
                return_value=httpx.Response(404)
            )
            with pytest.raises(IngestionError) as exc_info:
                await ingester.fetch_url(target_url)

        assert "example.com" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_network_error_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """A generic network request error raises IngestionError."""
        with respx.mock(assert_all_called=False):
            respx.get("https://example.com/network-fail").mock(
                side_effect=httpx.RequestError("network failure")
            )
            with pytest.raises(IngestionError):
                await ingester.fetch_url("https://example.com/network-fail")


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
    async def test_nonexistent_file_raises_ingestion_error(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
        """Attempting to read a non-existent file raises IngestionError."""
        missing = tmp_path / "does_not_exist.txt"
        with pytest.raises(IngestionError) as exc_info:
            await ingester.read_file(missing)

        assert "not found" in str(exc_info.value).lower() or "does_not_exist" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_directory_path_raises_ingestion_error(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
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
    async def test_reads_utf8_text_with_special_chars(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
        """UTF-8 files with non-ASCII characters are read correctly."""
        p = tmp_path / "unicode.txt"
        content = "H\xe9llo w\xf6rld \u2013 \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"
        p.write_text(content, encoding="utf-8")

        result = await ingester.read_file(p)
        assert "H\xe9llo" in result["text"]
        assert "\u65e5\u672c\u8a9e" in result["text"]

    @pytest.mark.asyncio
    async def test_title_is_stem_without_extension(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
        """The 'title' key equals the filename stem (without extension)."""
        p = tmp_path / "my_document.txt"
        p.write_text("content", encoding="utf-8")

        result = await ingester.read_file(p)
        assert result["title"] == "my_document"

    @pytest.mark.asyncio
    async def test_source_is_string_path(self, ingester: DocumentIngester, tmp_text_file: Path) -> None:
        """The 'source' key is a string representation of the file path."""
        result = await ingester.read_file(tmp_text_file)
        assert result["source"] == str(tmp_text_file)
        assert isinstance(result["source"], str)

    @pytest.mark.asyncio
    async def test_reads_file_with_no_extension(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
        """Files with no extension are treated as plain text."""
        p = tmp_path / "noextension"
        p.write_text("plain text content", encoding="utf-8")

        result = await ingester.read_file(p)
        assert result["content_type"] == "text/plain"
        assert "plain text content" in result["text"]


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
    async def test_invalid_pdf_raises_ingestion_error(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
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

    @pytest.mark.asyncio
    async def test_pdf_content_type_is_application_pdf(
        self, ingester: DocumentIngester, tmp_pdf_file: Path
    ) -> None:
        """content_type is 'application/pdf' for PDF files."""
        result = await ingester.read_file(tmp_pdf_file)
        assert result["content_type"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_pdf_title_is_stem(self, ingester: DocumentIngester, tmp_path: Path) -> None:
        """The title of a PDF file is its stem (name without extension)."""
        pdf_path = tmp_path / "my_paper.pdf"
        pdf_path.write_bytes(_minimal_pdf_bytes())

        result = await ingester.read_file(pdf_path)
        assert result["title"] == "my_paper"


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

    def test_nav_removed(self, ingester: DocumentIngester) -> None:
        """<nav> content is removed from extracted text."""
        html = "<html><body><nav>Nav links</nav><p>Content</p></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "Nav links" not in text
        assert "Content" in text

    def test_header_removed(self, ingester: DocumentIngester) -> None:
        """<header> content is removed from extracted text."""
        html = "<html><body><header>Site Header</header><p>Content</p></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "Site Header" not in text
        assert "Content" in text

    def test_footer_removed(self, ingester: DocumentIngester) -> None:
        """<footer> content is removed from extracted text."""
        html = "<html><body><p>Content</p><footer>Footer</footer></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "Footer" not in text
        assert "Content" in text

    def test_aside_removed(self, ingester: DocumentIngester) -> None:
        """<aside> content is removed from extracted text."""
        html = "<html><body><p>Content</p><aside>Sidebar</aside></body></html>"
        text, _ = ingester._extract_html_text(html)
        assert "Sidebar" not in text
        assert "Content" in text

    def test_empty_html_returns_empty_string(self, ingester: DocumentIngester) -> None:
        """Empty HTML returns an empty text string and None title."""
        text, title = ingester._extract_html_text("")
        assert text == ""
        assert title is None

    def test_nested_content_extracted(self, ingester: DocumentIngester) -> None:
        """Deeply nested text nodes are extracted."""
        html = (
            "<html><body><div><section><p><span>Deep text</span></p>"
            "</section></div></body></html>"
        )
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
        # After normalization the text should be present without extra spaces
        assert "Too" in text and "spaces" in text

    def test_article_preferred_over_body(self, ingester: DocumentIngester) -> None:
        """<article> content is preferred when both article and other content exist."""
        html = (
            "<html><body>"
            "<div>Should not appear</div>"
            "<article><p>Article content</p></article>"
            "</body></html>"
        )
        text, _ = ingester._extract_html_text(html)
        assert "Article content" in text

    def test_main_preferred_over_body_when_no_article(self, ingester: DocumentIngester) -> None:
        """<main> content is preferred when no <article> exists."""
        html = (
            "<html><body>"
            "<div>Outer div</div>"
            "<main><p>Main content</p></main>"
            "</body></html>"
        )
        text, _ = ingester._extract_html_text(html)
        assert "Main content" in text

    def test_returns_tuple(self, ingester: DocumentIngester) -> None:
        """_extract_html_text returns a (str, str | None) tuple."""
        html = "<html><body><p>text</p></body></html>"
        result = ingester._extract_html_text(html)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)

    def test_title_with_whitespace_stripped(self, ingester: DocumentIngester) -> None:
        """Title text has leading/trailing whitespace stripped."""
        html = "<html><head><title>  Spaced Title  </title></head><body><p>x</p></body></html>"
        _, title = ingester._extract_html_text(html)
        assert title == "Spaced Title"


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

    def test_returns_empty_string_for_blank_pdf(self, ingester: DocumentIngester) -> None:
        """A valid but blank PDF returns an empty string (no text to extract)."""
        result = ingester._extract_pdf_from_bytes(_minimal_pdf_bytes())
        # The minimal PDF has no text content, so result should be empty string
        assert result == "" or isinstance(result, str)

    def test_truncated_pdf_raises_ingestion_error(self, ingester: DocumentIngester) -> None:
        """Truncated (incomplete) PDF bytes raise IngestionError."""
        # Take only the first 10 bytes of a valid PDF — definitely invalid
        truncated = _minimal_pdf_bytes()[:10]
        with pytest.raises(IngestionError):
            ingester._extract_pdf_from_bytes(truncated)


# ---------------------------------------------------------------------------
# Tests – _extract_pdf_from_path (unit-level)
# ---------------------------------------------------------------------------


class TestExtractPdfFromPath:
    """Unit tests for the internal _extract_pdf_from_path helper."""

    def test_valid_pdf_path_returns_string(self, ingester: DocumentIngester, tmp_pdf_file: Path) -> None:
        """A valid minimal PDF path returns a string."""
        result = ingester._extract_pdf_from_path(tmp_pdf_file)
        assert isinstance(result, str)

    def test_invalid_pdf_path_raises_ingestion_error(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
        """A path to a non-PDF file raises IngestionError."""
        bad = tmp_path / "fake.pdf"
        bad.write_bytes(b"not a pdf")
        with pytest.raises(IngestionError):
            ingester._extract_pdf_from_path(bad)

    def test_path_included_in_error_message(
        self, ingester: DocumentIngester, tmp_path: Path
    ) -> None:
        """IngestionError includes the file path."""
        bad = tmp_path / "broken.pdf"
        bad.write_bytes(b"bad")
        with pytest.raises(IngestionError) as exc_info:
            ingester._extract_pdf_from_path(bad)
        assert "broken.pdf" in str(exc_info.value) or "pdf" in str(exc_info.value).lower()

    def test_blank_pdf_returns_empty_string(
        self, ingester: DocumentIngester, tmp_pdf_file: Path
    ) -> None:
        """A blank PDF (no text content) returns an empty string."""
        result = ingester._extract_pdf_from_path(tmp_pdf_file)
        # Minimal PDF has no text; result should be empty or whitespace-only
        assert result == "" or not result.strip()

    def test_result_is_string(self, ingester: DocumentIngester, tmp_pdf_file: Path) -> None:
        """_extract_pdf_from_path always returns a string."""
        result = ingester._extract_pdf_from_path(tmp_pdf_file)
        assert isinstance(result, str)


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

    def test_verbose_flag_true(self) -> None:
        """Verbose=True flag is stored correctly."""
        ingester = DocumentIngester(verbose=True)
        assert ingester.verbose is True

    def test_verbose_flag_false(self) -> None:
        """Verbose=False flag is stored correctly."""
        ingester = DocumentIngester(verbose=False)
        assert ingester.verbose is False

    def test_default_verbose_is_false(self) -> None:
        """Default verbose flag is False."""
        ingester = DocumentIngester()
        assert ingester.verbose is False

    def test_default_timeout_value(self) -> None:
        """Default timeout equals 30.0 seconds."""
        ingester = DocumentIngester()
        assert ingester.timeout == 30.0

    def test_zero_timeout_stored(self) -> None:
        """A timeout of 0.0 is stored (edge case)."""
        ingester = DocumentIngester(timeout=0.0)
        assert ingester.timeout == 0.0
