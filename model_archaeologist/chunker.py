"""Token-aware document chunker for Model Archaeologist.

Splits large documents into LLM-friendly chunks with configurable overlap,
respecting token limits using tiktoken for accurate token counting.
"""

from __future__ import annotations

from typing import Iterator

import tiktoken

# Default tokenizer encoding (cl100k_base covers GPT-4, GPT-3.5-turbo)
DEFAULT_ENCODING = "cl100k_base"


class ChunkerError(Exception):
    """Raised when text splitting encounters an unrecoverable error."""


class TextChunker:
    """Splits text into token-bounded chunks with overlap.

    Uses tiktoken for accurate token counting to ensure chunks stay
    within LLM context limits. Supports configurable chunk size and
    overlap between consecutive chunks.

    Args:
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of tokens to overlap between consecutive chunks.
        encoding_name: tiktoken encoding to use for token counting.
    """

    def __init__(
        self,
        chunk_size: int = 3000,
        chunk_overlap: int = 200,
        encoding_name: str = DEFAULT_ENCODING,
    ) -> None:
        """Initialize the TextChunker."""
        if chunk_size <= 0:
            raise ChunkerError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ChunkerError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ChunkerError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name

        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:
            raise ChunkerError(f"Failed to load tiktoken encoding '{encoding_name}': {exc}") from exc

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: Input text.

        Returns:
            Number of tokens according to the configured encoding.
        """
        return len(self._encoding.encode(text))

    def split(self, text: str) -> list[str]:
        """Split text into token-bounded chunks with overlap.

        The algorithm tokenizes the full text, then slides a window of
        `chunk_size` tokens advancing by `chunk_size - chunk_overlap`
        tokens each step.

        Args:
            text: Input text to split.

        Returns:
            A list of text chunks. Returns an empty list for empty input.
        """
        if not text or not text.strip():
            return []

        tokens = self._encoding.encode(text)
        total_tokens = len(tokens)

        if total_tokens == 0:
            return []

        if total_tokens <= self.chunk_size:
            return [self._decode_tokens(tokens)]

        chunks: list[str] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode_tokens(chunk_tokens)
            if chunk_text.strip():
                chunks.append(chunk_text)
            if end == total_tokens:
                break
            start += step

        return chunks

    def split_iter(self, text: str) -> Iterator[str]:
        """Iterate over token-bounded chunks with overlap.

        Memory-efficient alternative to :meth:`split` that yields
        chunks one at a time.

        Args:
            text: Input text to split.

        Yields:
            Text chunks in order.
        """
        yield from self.split(text)

    def merge_chunks(self, chunks: list[str], separator: str = "\n\n") -> str:
        """Merge a list of text chunks into a single string.

        Args:
            chunks: List of text chunks to merge.
            separator: String to insert between chunks.

        Returns:
            Merged text string.
        """
        return separator.join(chunk.strip() for chunk in chunks if chunk.strip())

    def _decode_tokens(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back to a text string.

        Args:
            tokens: List of integer token IDs.

        Returns:
            Decoded text string.
        """
        return self._encoding.decode(tokens)
