"""
Transcript Data Types

Data models for transcription results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptWord:
    """A single word with timing information."""

    word: str
    start_ms: float
    end_ms: float
    confidence: float = 1.0


@dataclass
class TranscriptChunk:
    """A chunk of transcript for streaming."""

    text: str
    is_final: bool = False
    confidence: float = 1.0
    timestamp_ms: float = 0.0


@dataclass
class Transcript:
    """Complete transcription result."""

    text: str
    confidence: float = 1.0
    language: str = "en"
    words: list[TranscriptWord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration based on word timestamps."""
        if not self.words:
            return 0.0
        return self.words[-1].end_ms - self.words[0].start_ms

    def get_text_segment(self, start_ms: float, end_ms: float) -> str:
        """Get text within a time range."""
        words_in_range = [
            w.word for w in self.words if w.start_ms >= start_ms and w.end_ms <= end_ms
        ]
        return " ".join(words_in_range)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "language": self.language,
            "words": [
                {
                    "word": w.word,
                    "start_ms": w.start_ms,
                    "end_ms": w.end_ms,
                    "confidence": w.confidence,
                }
                for w in self.words
            ],
            "metadata": self.metadata,
        }
