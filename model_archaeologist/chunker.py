"""Token-aware document chunker for Model Archaeologist.

Splits large documents into LLM-friendly chunks with configurable overlap,
respecting token limits using tiktoken for accurate token counting.

The primary class :class:`TextChunker` tokenizes input text once and operates
directly on token ID arrays for efficiency, avoiding repeated encoding/decoding
roundtrips. It supports both eager (list-returning) and lazy (iterator) splitting
as well as a merge utility for reassembling analysed fragments.
"""

from __future__ import annotations

from typing import Iterator

import tiktoken

# Default tokenizer encoding:
# cl100k_base covers GPT-4, GPT-4o, GPT-3.5-turbo, and text-embedding-ada-002.
DEFAULT_ENCODING = "cl100k_base"


class ChunkerError(Exception):
    """Raised when :class:`TextChunker` encounters an unrecoverable configuration
    or runtime error.

    This is distinct from :class:`ValueError` so callers can catch chunker-specific
    problems without accidentally swallowing unrelated value errors.
    """


class TextChunker:
    """Splits plain text into token-bounded chunks with configurable overlap.

    The chunker tokenizes the input text once using :mod:`tiktoken`, then
    slides a window of ``chunk_size`` tokens across the token list, advancing
    by ``chunk_size - chunk_overlap`` tokens at each step.  This guarantees
    that:

    - Every chunk contains **at most** ``chunk_size`` tokens.
    - Consecutive chunks share **exactly** ``chunk_overlap`` tokens at their
      boundary (except for the very first and last chunks).
    - The final chunk may be shorter than ``chunk_size`` tokens.

    Example usage::

        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split(long_document_text)
        # chunks is a list[str], each ≤ 512 tokens

    Args:
        chunk_size: Maximum number of tokens per chunk.  Must be a positive
            integer.  Defaults to 3000.
        chunk_overlap: Number of tokens shared between consecutive chunks.
            Must be non-negative and strictly less than ``chunk_size``.
            Defaults to 200.
        encoding_name: Name of the :mod:`tiktoken` encoding to use for token
            counting.  Defaults to ``'cl100k_base'``.

    Raises:
        ChunkerError: If ``chunk_size <= 0``, ``chunk_overlap < 0``, or
            ``chunk_overlap >= chunk_size``, or if the tiktoken encoding
            cannot be loaded.
    """

    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
        encoding_name: str = DEFAULT_ENCODING,
    ) -> None:
        """Initialise and validate a :class:`TextChunker`.

        Args:
            chunk_size: Maximum tokens per chunk (must be > 0).
            chunk_overlap: Token overlap between consecutive chunks
                (must be >= 0 and < chunk_size).
            encoding_name: tiktoken encoding name.

        Raises:
            ChunkerError: On invalid parameter combinations or unavailable
                tiktoken encoding.
        """
        if chunk_size <= 0:
            raise ChunkerError(
                f"chunk_size must be a positive integer, got {chunk_size!r}"
            )
        if chunk_overlap < 0:
            raise ChunkerError(
                f"chunk_overlap must be non-negative, got {chunk_overlap!r}"
            )
        if chunk_overlap >= chunk_size:
            raise ChunkerError(
                f"chunk_overlap ({chunk_overlap}) must be strictly less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size: int = chunk_size
        self.chunk_overlap: int = chunk_overlap
        self.encoding_name: str = encoding_name

        try:
            self._encoding: tiktoken.Encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise ChunkerError(
                f"Failed to load tiktoken encoding '{encoding_name}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text* using the configured encoding.

        This is a convenience wrapper around the underlying tiktoken encoder and
        is useful for callers that want to pre-flight a text's token budget
        without performing a full split.

        Args:
            text: Input text string.  An empty string returns 0.

        Returns:
            Number of tokens as a non-negative integer.
        """
        if not text:
            return 0
        return len(self._encoding.encode(text))

    def split(self, text: str) -> list[str]:
        """Split *text* into a list of token-bounded chunks with overlap.

        Algorithm:

        1. Encode the entire text to a list of integer token IDs.
        2. If the token count is already within ``chunk_size``, return a
           single-element list containing the (possibly decoded) original text.
        3. Otherwise slide a window of width ``chunk_size`` over the tokens,
           stepping forward by ``chunk_size - chunk_overlap`` tokens each time.
        4. Decode each window slice back to a string and collect non-empty
           results.

        The last window is always included even if it is shorter than
        ``chunk_size``.  Empty or whitespace-only input returns an empty list.

        Args:
            text: Input text to split.  May be arbitrarily large.

        Returns:
            A (possibly empty) list of decoded text chunks.  Each chunk
            contains at most ``chunk_size`` tokens.  Consecutive chunks share
            ``chunk_overlap`` tokens at their boundary.
        """
        if not text or not text.strip():
            return []

        tokens: list[int] = self._encoding.encode(text)
        total: int = len(tokens)

        if total == 0:
            return []

        # Fast path: whole text fits in one chunk
        if total <= self.chunk_size:
            return [self._decode(tokens)]

        chunks: list[str] = []
        step: int = self.chunk_size - self.chunk_overlap
        # step is always >= 1 because chunk_overlap < chunk_size (validated in __init__)
        start: int = 0

        while start < total:
            end: int = min(start + self.chunk_size, total)
            window: list[int] = tokens[start:end]
            decoded: str = self._decode(window)
            if decoded.strip():
                chunks.append(decoded)
            if end == total:
                # We've consumed all tokens; stop to avoid an infinite loop
                # when step == 0 (guarded by validation, but be defensive)
                break
            start += step

        return chunks

    def split_iter(self, text: str) -> Iterator[str]:
        """Yield token-bounded chunks with overlap one at a time.

        A lazy, memory-efficient alternative to :meth:`split`.  Internally
        still encodes the full text once (required for accurate token
        counting), but yields decoded chunks without materialising the
        complete output list first.

        Args:
            text: Input text to split.

        Yields:
            Non-empty decoded text chunks in document order.
        """
        if not text or not text.strip():
            return

        tokens: list[int] = self._encoding.encode(text)
        total: int = len(tokens)

        if total == 0:
            return

        if total <= self.chunk_size:
            decoded = self._decode(tokens)
            if decoded.strip():
                yield decoded
            return

        step: int = self.chunk_size - self.chunk_overlap
        start: int = 0

        while start < total:
            end: int = min(start + self.chunk_size, total)
            window: list[int] = tokens[start:end]
            decoded = self._decode(window)
            if decoded.strip():
                yield decoded
            if end == total:
                break
            start += step

    def merge_chunks(self, chunks: list[str], separator: str = "\n\n") -> str:
        """Merge a list of text chunks into a single string.

        Strips leading/trailing whitespace from each chunk and joins
        non-empty results with *separator*.  Useful for reassembling partial
        analysis results from the LLM into a single context window.

        Args:
            chunks: List of text chunks to merge.  May contain empty strings
                or whitespace-only entries; these are silently skipped.
            separator: String inserted between consecutive chunks.
                Defaults to ``'\\n\\n'`` (double newline).

        Returns:
            A single merged string, or an empty string if *chunks* is empty
            or contains only whitespace-only entries.
        """
        return separator.join(chunk.strip() for chunk in chunks if chunk.strip())

    def estimate_chunk_count(self, text: str) -> int:
        """Estimate the number of chunks that :meth:`split` will produce.

        Performs a full token-count but avoids constructing the chunk list,
        making it slightly cheaper when only the count is needed.

        Args:
            text: Input text.

        Returns:
            Expected number of chunks (>= 0).  Returns 0 for empty input.
        """
        if not text or not text.strip():
            return 0

        total = self.count_tokens(text)
        if total == 0:
            return 0
        if total <= self.chunk_size:
            return 1

        step = self.chunk_size - self.chunk_overlap
        # ceiling division for the number of steps from position 0
        # First chunk covers [0, chunk_size); subsequent chunks advance by step
        return 1 + max(0, -(-( total - self.chunk_size) // step))

    def truncate(self, text: str, max_tokens: int | None = None) -> str:
        """Truncate *text* to at most *max_tokens* tokens.

        Useful for enforcing a hard token budget before sending text to an LLM
        without performing a full split.

        Args:
            text: Input text to truncate.
            max_tokens: Maximum number of tokens to keep.  If ``None`` or
                greater than or equal to the token count, the original text
                is returned unchanged (no copy is made).

        Returns:
            Truncated text as a decoded string.  Returns an empty string
            for empty input.
        """
        if not text:
            return text
        if max_tokens is None:
            return text
        if max_tokens <= 0:
            return ""

        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._decode(tokens[:max_tokens])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decode(self, tokens: list[int]) -> str:
        """Decode a sequence of token IDs back to a UTF-8 string.

        Args:
            tokens: List of integer token IDs produced by this chunker's
                encoding.

        Returns:
            Decoded text string.  May contain partial Unicode code points at
            chunk boundaries; tiktoken handles this gracefully via its
            ``decode`` method.
        """
        return self._encoding.decode(tokens)
