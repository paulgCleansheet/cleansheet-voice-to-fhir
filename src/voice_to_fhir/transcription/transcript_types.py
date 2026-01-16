"""
Transcript Data Types

Data models for transcription results.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TranscriptWord:
    """A single word with timing information."""

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptWord":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class TranscriptSegment:
    """A segment of transcript with timing information."""

    text: str
    start_time: float
    end_time: float
    confidence: float = 1.0
    speaker: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptSegment":
        """Create from dictionary."""
        words = [TranscriptWord.from_dict(w) for w in data.get("words", [])]
        return cls(
            text=data["text"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            confidence=data.get("confidence", 1.0),
            speaker=data.get("speaker"),
            words=words,
        )


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
    segments: list[TranscriptSegment] = field(default_factory=list)
    language: str = "en"
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration based on segment timestamps."""
        if not self.segments:
            return 0.0
        return self.segments[-1].end_time

    @property
    def word_count(self) -> int:
        """Count of words in the transcript text."""
        if not self.text:
            return 0
        return len(self.text.split())

    def get_text_segment(self, start_time: float, end_time: float) -> str:
        """Get text within a time range."""
        result_segments = []
        for segment in self.segments:
            if segment.start_time >= start_time and segment.end_time <= end_time:
                result_segments.append(segment.text)
        return " ".join(result_segments)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "segments": [s.to_dict() for s in self.segments],
            "language": self.language,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        """Create from dictionary."""
        segments = [TranscriptSegment.from_dict(s) for s in data.get("segments", [])]
        return cls(
            text=data["text"],
            segments=segments,
            language=data.get("language", "en"),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
        )
