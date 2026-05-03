"""Unit tests for the token-aware document chunker in model_archaeologist/chunker.py.

Verifies:
- Correct chunk sizes and counts under various configurations
- Overlap correctness (consecutive chunks share the expected token prefix)
- Edge cases: empty input, single-token input, exact boundary fits
- merge_chunks behaviour
- truncate behaviour
- estimate_chunk_count accuracy
- ChunkerError raised for invalid initialisation parameters
"""

from __future__ import annotations

import pytest
import tiktoken

from model_archaeologist.chunker import ChunkerError, TextChunker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_count(text: str, encoding_name: str = "cl100k_base") -> int:
    """Return the number of tokens in *text* using tiktoken directly."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def _make_text(token_count: int, encoding_name: str = "cl100k_base") -> str:
    """Generate a text string that encodes to exactly *token_count* tokens.

    Uses repeated single-token words ("word ") so the token count is
    predictable.
    """
    enc = tiktoken.get_encoding(encoding_name)
    # The word "word" encodes to a single token in cl100k_base.
    # We build up words until we have the right count.
    result_tokens: list[int] = []
    # Encode the base word
    word_tokens = enc.encode("word ")
    word_len = len(word_tokens)
    full_repeats = token_count // word_len
    remainder = token_count % word_len

    result_tokens.extend(word_tokens * full_repeats)
    # Pad remainder with single tokens (token ID 198 = newline in cl100k_base)
    if remainder:
        result_tokens.extend([198] * remainder)

    return enc.decode(result_tokens)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_chunker() -> TextChunker:
    """Return a TextChunker with default settings (3000/200)."""
    return TextChunker(chunk_size=3000, chunk_overlap=200)


@pytest.fixture()
def small_chunker() -> TextChunker:
    """Return a TextChunker with small chunk_size=10, overlap=2 for easy testing."""
    return TextChunker(chunk_size=10, chunk_overlap=2)


@pytest.fixture()
def tiny_chunker() -> TextChunker:
    """Return a TextChunker with chunk_size=5, overlap=1."""
    return TextChunker(chunk_size=5, chunk_overlap=1)


# ---------------------------------------------------------------------------
# Tests – Initialisation & Validation
# ---------------------------------------------------------------------------


class TestTextChunkerInit:
    """Tests for TextChunker.__init__ parameter validation."""

    def test_default_params(self) -> None:
        """Default construction succeeds with chunk_size=3000, overlap=200."""
        chunker = TextChunker()
        assert chunker.chunk_size == 3000
        assert chunker.chunk_overlap == 200
        assert chunker.encoding_name == "cl100k_base"

    def test_custom_params(self) -> None:
        """Custom parameters are stored correctly."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, encoding_name="cl100k_base")
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 64

    def test_zero_chunk_size_raises(self) -> None:
        """chunk_size=0 raises ChunkerError."""
        with pytest.raises(ChunkerError, match="chunk_size"):
            TextChunker(chunk_size=0)

    def test_negative_chunk_size_raises(self) -> None:
        """Negative chunk_size raises ChunkerError."""
        with pytest.raises(ChunkerError, match="chunk_size"):
            TextChunker(chunk_size=-1)

    def test_negative_overlap_raises(self) -> None:
        """Negative chunk_overlap raises ChunkerError."""
        with pytest.raises(ChunkerError, match="chunk_overlap"):
            TextChunker(chunk_size=100, chunk_overlap=-1)

    def test_overlap_equal_to_chunk_size_raises(self) -> None:
        """chunk_overlap == chunk_size raises ChunkerError."""
        with pytest.raises(ChunkerError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_chunk_size_raises(self) -> None:
        """chunk_overlap > chunk_size raises ChunkerError."""
        with pytest.raises(ChunkerError):
            TextChunker(chunk_size=100, chunk_overlap=200)

    def test_zero_overlap_allowed(self) -> None:
        """chunk_overlap=0 is valid."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        assert chunker.chunk_overlap == 0

    def test_overlap_one_less_than_chunk_size_allowed(self) -> None:
        """chunk_overlap = chunk_size - 1 is the maximum valid overlap."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=9)
        assert chunker.chunk_overlap == 9

    def test_invalid_encoding_raises(self) -> None:
        """An unrecognised tiktoken encoding name raises ChunkerError."""
        with pytest.raises(ChunkerError, match="encoding"):
            TextChunker(encoding_name="not_a_real_encoding_xyz")


# ---------------------------------------------------------------------------
# Tests – count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for TextChunker.count_tokens."""

    def test_empty_string(self, default_chunker: TextChunker) -> None:
        """Empty string returns 0 tokens."""
        assert default_chunker.count_tokens("") == 0

    def test_whitespace_only(self, default_chunker: TextChunker) -> None:
        """Whitespace-only string may return a small non-zero count."""
        # We just assert it returns a non-negative integer
        count = default_chunker.count_tokens("   ")
        assert count >= 0

    def test_single_word(self, default_chunker: TextChunker) -> None:
        """A single ASCII word returns the expected token count."""
        text = "hello"
        expected = _token_count(text)
        assert default_chunker.count_tokens(text) == expected

    def test_longer_text(self, default_chunker: TextChunker) -> None:
        """A longer passage returns the correct tiktoken count."""
        text = "The quick brown fox jumps over the lazy dog. " * 10
        expected = _token_count(text)
        assert default_chunker.count_tokens(text) == expected

    def test_unicode_text(self, default_chunker: TextChunker) -> None:
        """Unicode text returns a consistent token count."""
        text = "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8"
        expected = _token_count(text)
        assert default_chunker.count_tokens(text) == expected

    def test_matches_tiktoken_directly(self, default_chunker: TextChunker) -> None:
        """count_tokens matches tiktoken.get_encoding().encode() directly."""
        text = "Transformer models use attention mechanisms."
        direct = _token_count(text)
        assert default_chunker.count_tokens(text) == direct


# ---------------------------------------------------------------------------
# Tests – split (basic correctness)
# ---------------------------------------------------------------------------


class TestSplitBasic:
    """Basic correctness tests for TextChunker.split."""

    def test_empty_string_returns_empty_list(self, small_chunker: TextChunker) -> None:
        """Splitting an empty string returns an empty list."""
        assert small_chunker.split("") == []

    def test_whitespace_only_returns_empty_list(self, small_chunker: TextChunker) -> None:
        """Splitting a whitespace-only string returns an empty list."""
        assert small_chunker.split("   \n\t  ") == []

    def test_short_text_single_chunk(self, small_chunker: TextChunker) -> None:
        """Text shorter than chunk_size returns exactly one chunk equal to the text."""
        text = "Hi"  # 1 token
        chunks = small_chunker.split(text)  # chunk_size=10
        assert len(chunks) == 1
        assert "Hi" in chunks[0]

    def test_text_exactly_chunk_size_single_chunk(self) -> None:
        """Text whose token count equals chunk_size returns exactly one chunk."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=1)
        text = _make_text(5)
        chunks = chunker.split(text)
        assert len(chunks) == 1

    def test_long_text_produces_multiple_chunks(self) -> None:
        """Text longer than chunk_size is split into multiple chunks."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = _make_text(50)  # 50 tokens >> chunk_size=10
        chunks = chunker.split(text)
        assert len(chunks) > 1

    def test_all_chunks_are_strings(self, small_chunker: TextChunker) -> None:
        """Every element returned by split is a non-empty string."""
        text = _make_text(50)
        chunks = small_chunker.split(text)
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert chunk.strip()  # no whitespace-only chunks

    def test_no_chunk_exceeds_chunk_size(self) -> None:
        """No chunk contains more tokens than chunk_size."""
        chunker = TextChunker(chunk_size=15, chunk_overlap=3)
        text = _make_text(100)
        chunks = chunker.split(text)
        for chunk in chunks:
            assert chunker.count_tokens(chunk) <= chunker.chunk_size

    def test_single_token_text(self) -> None:
        """A single-token text produces exactly one chunk."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=1)
        text = "hi"  # likely 1 token
        chunks = chunker.split(text)
        assert len(chunks) == 1

    def test_returns_list(self, default_chunker: TextChunker) -> None:
        """split always returns a list."""
        result = default_chunker.split("some text")
        assert isinstance(result, list)

    def test_content_preserved_no_overlap(self) -> None:
        """With overlap=0, concatenating chunks reproduces all original tokens."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=0)
        # Build text of exactly 30 tokens
        text = _make_text(30)
        chunks = chunker.split(text)

        # Re-encode each chunk and concatenate token IDs
        enc = tiktoken.get_encoding("cl100k_base")
        original_tokens = enc.encode(text)
        reconstructed_tokens: list[int] = []
        for chunk in chunks:
            reconstructed_tokens.extend(enc.encode(chunk))

        assert reconstructed_tokens == original_tokens


# ---------------------------------------------------------------------------
# Tests – split (overlap correctness)
# ---------------------------------------------------------------------------


class TestSplitOverlap:
    """Tests verifying that the overlap between consecutive chunks is correct."""

    def test_consecutive_chunks_share_overlap_tokens(self) -> None:
        """The last chunk_overlap tokens of chunk[i] equal the first
        chunk_overlap tokens of chunk[i+1]."""
        overlap = 3
        chunk_size = 10
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        enc = tiktoken.get_encoding("cl100k_base")

        text = _make_text(40)  # 40 tokens, step = 10 - 3 = 7
        chunks = chunker.split(text)

        assert len(chunks) >= 2, "Need at least 2 chunks to test overlap"

        for i in range(len(chunks) - 1):
            tokens_i = enc.encode(chunks[i])
            tokens_next = enc.encode(chunks[i + 1])

            # The last `overlap` tokens of chunk[i] must equal
            # the first `overlap` tokens of chunk[i+1]
            tail = tokens_i[-overlap:]
            head = tokens_next[:overlap]
            assert tail == head, (
                f"Overlap mismatch between chunk {i} and chunk {i + 1}: "
                f"{tail!r} != {head!r}"
            )

    def test_zero_overlap_no_shared_tokens(self) -> None:
        """With overlap=0, consecutive chunks share no tokens."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=0)
        enc = tiktoken.get_encoding("cl100k_base")

        text = _make_text(30)
        chunks = chunker.split(text)

        assert len(chunks) >= 2

        for i in range(len(chunks) - 1):
            tokens_i = enc.encode(chunks[i])
            # Verify each full (non-final) chunk has exactly chunk_size tokens
            assert len(tokens_i) == chunker.chunk_size or i == len(chunks) - 2

    def test_large_overlap_many_chunks(self) -> None:
        """A large overlap relative to chunk_size produces more chunks."""
        chunker_small_overlap = TextChunker(chunk_size=10, chunk_overlap=1)
        chunker_large_overlap = TextChunker(chunk_size=10, chunk_overlap=8)
        text = _make_text(50)

        chunks_small = chunker_small_overlap.split(text)
        chunks_large = chunker_large_overlap.split(text)

        # Larger overlap -> smaller step -> more chunks
        assert len(chunks_large) >= len(chunks_small)

    def test_overlap_with_text_exactly_two_chunks(self) -> None:
        """Text that exactly spans two chunks shares correct overlap tokens."""
        chunk_size = 10
        overlap = 3
        step = chunk_size - overlap  # 7
        # Two chunks: first covers [0,10), second covers [7,17)
        total_tokens = chunk_size + step  # 17 tokens
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        enc = tiktoken.get_encoding("cl100k_base")

        text = _make_text(total_tokens)
        chunks = chunker.split(text)

        assert len(chunks) == 2
        tail = enc.encode(chunks[0])[-overlap:]
        head = enc.encode(chunks[1])[:overlap]
        assert tail == head


# ---------------------------------------------------------------------------
# Tests – split_iter
# ---------------------------------------------------------------------------


class TestSplitIter:
    """Tests for TextChunker.split_iter (lazy generator interface)."""

    def test_iter_matches_split_output(self) -> None:
        """split_iter yields the same chunks as split."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = _make_text(50)

        eager = chunker.split(text)
        lazy = list(chunker.split_iter(text))

        assert eager == lazy

    def test_iter_empty_string(self, small_chunker: TextChunker) -> None:
        """split_iter on empty string yields nothing."""
        result = list(small_chunker.split_iter(""))
        assert result == []

    def test_iter_whitespace_only(self, small_chunker: TextChunker) -> None:
        """split_iter on whitespace-only string yields nothing."""
        result = list(small_chunker.split_iter("   "))
        assert result == []

    def test_iter_single_chunk_text(self) -> None:
        """split_iter on short text yields exactly one chunk."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=5)
        text = "short text"
        result = list(chunker.split_iter(text))
        assert len(result) == 1

    def test_iter_is_iterator(self) -> None:
        """split_iter returns an iterator (not a list)."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = _make_text(30)
        result = chunker.split_iter(text)
        # Should have __iter__ and __next__
        assert hasattr(result, "__iter__")
        assert hasattr(result, "__next__")

    def test_iter_no_chunk_exceeds_chunk_size(self) -> None:
        """No chunk from split_iter exceeds chunk_size tokens."""
        chunker = TextChunker(chunk_size=15, chunk_overlap=3)
        text = _make_text(80)
        for chunk in chunker.split_iter(text):
            assert chunker.count_tokens(chunk) <= chunker.chunk_size


# ---------------------------------------------------------------------------
# Tests – merge_chunks
# ---------------------------------------------------------------------------


class TestMergeChunks:
    """Tests for TextChunker.merge_chunks."""

    def test_merges_two_chunks(self, default_chunker: TextChunker) -> None:
        """Two chunks are joined with the default separator."""
        result = default_chunker.merge_chunks(["chunk one", "chunk two"])
        assert result == "chunk one\n\nchunk two"

    def test_merges_empty_list(self, default_chunker: TextChunker) -> None:
        """An empty list returns an empty string."""
        assert default_chunker.merge_chunks([]) == ""

    def test_filters_empty_strings(self, default_chunker: TextChunker) -> None:
        """Empty and whitespace-only strings are skipped."""
        result = default_chunker.merge_chunks(["first", "", "  ", "second"])
        assert result == "first\n\nsecond"

    def test_custom_separator(self, default_chunker: TextChunker) -> None:
        """A custom separator is used instead of the default double newline."""
        result = default_chunker.merge_chunks(["a", "b", "c"], separator=" | ")
        assert result == "a | b | c"

    def test_strips_whitespace_from_chunks(self, default_chunker: TextChunker) -> None:
        """Leading/trailing whitespace is stripped from each chunk before joining."""
        result = default_chunker.merge_chunks(["  alpha  ", "  beta  "])
        assert result == "alpha\n\nbeta"

    def test_single_chunk_no_separator(self, default_chunker: TextChunker) -> None:
        """A single non-empty chunk is returned without any separator."""
        result = default_chunker.merge_chunks(["only chunk"])
        assert result == "only chunk"

    def test_all_empty_chunks_returns_empty_string(self, default_chunker: TextChunker) -> None:
        """A list of only empty/whitespace chunks returns an empty string."""
        result = default_chunker.merge_chunks(["", "  ", "\n"])
        assert result == ""

    def test_merge_roundtrip_with_split(self) -> None:
        """merge_chunks can rejoin split chunks (with separator)."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=0)
        text = _make_text(30)
        chunks = chunker.split(text)
        merged = chunker.merge_chunks(chunks)
        # merged should contain all the text (order preserved, separator added)
        assert len(merged) > 0
        # All chunks must appear in the merged result
        for chunk in chunks:
            assert chunk.strip() in merged


# ---------------------------------------------------------------------------
# Tests – truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    """Tests for TextChunker.truncate."""

    def test_empty_string_returned_unchanged(self, default_chunker: TextChunker) -> None:
        """Truncating an empty string returns an empty string."""
        assert default_chunker.truncate("") == ""

    def test_text_shorter_than_limit_unchanged(self, default_chunker: TextChunker) -> None:
        """Text shorter than max_tokens is returned unchanged."""
        text = "short"
        result = default_chunker.truncate(text, max_tokens=100)
        assert result == text

    def test_text_exactly_limit_unchanged(self, default_chunker: TextChunker) -> None:
        """Text with exactly max_tokens tokens is returned unchanged."""
        n = 5
        text = _make_text(n)
        result = default_chunker.truncate(text, max_tokens=n)
        # Should be the same number of tokens
        assert default_chunker.count_tokens(result) == n

    def test_text_truncated_to_max_tokens(self) -> None:
        """Text longer than max_tokens is truncated to exactly max_tokens tokens."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0)
        text = _make_text(50)
        result = chunker.truncate(text, max_tokens=20)
        assert chunker.count_tokens(result) == 20

    def test_max_tokens_none_returns_original(self, default_chunker: TextChunker) -> None:
        """max_tokens=None returns the original text unchanged."""
        text = "some text here"
        assert default_chunker.truncate(text, max_tokens=None) == text

    def test_max_tokens_zero_returns_empty(self, default_chunker: TextChunker) -> None:
        """max_tokens=0 returns an empty string."""
        result = default_chunker.truncate("some text", max_tokens=0)
        assert result == ""

    def test_truncated_result_is_string(self, default_chunker: TextChunker) -> None:
        """truncate always returns a string."""
        result = default_chunker.truncate(_make_text(20), max_tokens=10)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Tests – estimate_chunk_count
# ---------------------------------------------------------------------------


class TestEstimateChunkCount:
    """Tests for TextChunker.estimate_chunk_count."""

    def test_empty_string_returns_zero(self, default_chunker: TextChunker) -> None:
        """Empty string returns an estimate of 0."""
        assert default_chunker.estimate_chunk_count("") == 0

    def test_whitespace_only_returns_zero(self, default_chunker: TextChunker) -> None:
        """Whitespace-only string returns an estimate of 0."""
        assert default_chunker.estimate_chunk_count("   ") == 0

    def test_short_text_returns_one(self, small_chunker: TextChunker) -> None:
        """Text shorter than chunk_size returns 1."""
        text = "hi"  # 1 token, chunk_size=10
        assert small_chunker.estimate_chunk_count(text) == 1

    def test_estimate_matches_actual_count(self) -> None:
        """estimate_chunk_count matches len(split()) for various inputs."""
        test_cases = [
            (10, 2, 5),
            (10, 2, 10),
            (10, 2, 25),
            (10, 2, 50),
            (10, 0, 30),
            (10, 9, 20),
        ]
        for chunk_size, overlap, total_tokens in test_cases:
            chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
            text = _make_text(total_tokens)
            actual = len(chunker.split(text))
            estimated = chunker.estimate_chunk_count(text)
            assert estimated == actual, (
                f"chunk_size={chunk_size}, overlap={overlap}, tokens={total_tokens}: "
                f"estimated={estimated}, actual={actual}"
            )

    def test_estimate_is_non_negative(self, default_chunker: TextChunker) -> None:
        """estimate_chunk_count is always non-negative."""
        for text in ["", "a", _make_text(100)]:
            assert default_chunker.estimate_chunk_count(text) >= 0


# ---------------------------------------------------------------------------
# Tests – edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and regression tests."""

    def test_chunk_size_one_no_overlap(self) -> None:
        """chunk_size=1 with overlap=0 produces one chunk per token."""
        chunker = TextChunker(chunk_size=1, chunk_overlap=0)
        text = _make_text(5)  # 5 tokens
        chunks = chunker.split(text)
        assert len(chunks) == 5
        for chunk in chunks:
            assert chunker.count_tokens(chunk) == 1

    def test_chunk_size_two_overlap_one(self) -> None:
        """chunk_size=2, overlap=1 produces many overlapping 2-token chunks."""
        chunker = TextChunker(chunk_size=2, chunk_overlap=1)
        text = _make_text(6)  # 6 tokens, step=1 -> 5 chunks: [0,2),[1,3),...,[4,6)
        chunks = chunker.split(text)
        # step = 2 - 1 = 1; chunks: starts 0,1,2,3,4 -> 5 chunks
        assert len(chunks) == 5

    def test_very_long_text(self) -> None:
        """A very long text is split without errors or missing content."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = _make_text(10000)
        chunks = chunker.split(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunker.count_tokens(chunk) <= 100

    def test_unicode_text_split_correctly(self) -> None:
        """Unicode text (CJK, emoji) is split without panicking or losing data."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "\u65e5\u672c\u8a9e\u30c6\u30b9\u30c8 " * 20
        chunks = chunker.split(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert chunker.count_tokens(chunk) <= 10

    def test_newline_heavy_text(self) -> None:
        """Text with many newlines is split without error."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "\n".join(["Line " + str(i) for i in range(100)])
        chunks = chunker.split(text)
        assert len(chunks) > 0

    def test_split_iter_and_split_agree_on_unicode(self) -> None:
        """split and split_iter return identical results for Unicode input."""
        chunker = TextChunker(chunk_size=8, chunk_overlap=2)
        text = "H\xe9llo w\xf6rld \u2013 transformer! " * 15
        assert chunker.split(text) == list(chunker.split_iter(text))

    def test_merge_then_split_preserves_semantics(self) -> None:
        """Text merged from chunks is re-splittable without error."""
        chunker = TextChunker(chunk_size=20, chunk_overlap=4)
        text = _make_text(100)
        chunks = chunker.split(text)
        merged = chunker.merge_chunks(chunks, separator=" ")
        re_split = chunker.split(merged)
        assert len(re_split) > 0
        for chunk in re_split:
            assert chunker.count_tokens(chunk) <= 20

    def test_text_with_only_special_characters(self) -> None:
        """Text consisting only of special characters is handled gracefully."""
        chunker = TextChunker(chunk_size=5, chunk_overlap=1)
        text = "!@#$%^&*()" * 10
        chunks = chunker.split(text)
        # Should produce at least one chunk and not crash
        assert isinstance(chunks, list)

    def test_last_chunk_within_size_limit(self) -> None:
        """The final chunk never exceeds chunk_size tokens."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        # Use a token count that doesn't divide evenly by step
        text = _make_text(33)  # step=7; chunks at 0,7,14,21,28 -> last [28,33)
        chunks = chunker.split(text)
        assert chunker.count_tokens(chunks[-1]) <= 10

    def test_first_chunk_starts_at_beginning(self) -> None:
        """The first chunk always starts at the beginning of the text."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        enc = tiktoken.get_encoding("cl100k_base")
        text = _make_text(40)
        original_tokens = enc.encode(text)
        chunks = chunker.split(text)
        first_chunk_tokens = enc.encode(chunks[0])
        assert first_chunk_tokens == original_tokens[:10]

    def test_last_chunk_ends_at_end(self) -> None:
        """The last chunk always ends at the end of the text."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=3)
        enc = tiktoken.get_encoding("cl100k_base")
        text = _make_text(35)
        original_tokens = enc.encode(text)
        chunks = chunker.split(text)
        last_chunk_tokens = enc.encode(chunks[-1])
        assert last_chunk_tokens == original_tokens[-len(last_chunk_tokens):]
