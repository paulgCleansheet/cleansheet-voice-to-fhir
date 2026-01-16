"""
Tests for transcript type definitions.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import pytest

from voice_to_fhir.transcription.transcript_types import (
    Transcript,
    TranscriptSegment,
    TranscriptWord,
)


class TestTranscriptWord:
    """Tests for TranscriptWord class."""

    def test_create_word(self):
        """Test creating a transcript word."""
        word = TranscriptWord(
            text="patient",
            start_time=0.0,
            end_time=0.5,
            confidence=0.98,
        )

        assert word.text == "patient"
        assert word.start_time == 0.0
        assert word.end_time == 0.5
        assert word.confidence == 0.98

    def test_word_duration(self):
        """Test word duration calculation."""
        word = TranscriptWord(
            text="medication",
            start_time=1.0,
            end_time=1.8,
            confidence=0.95,
        )

        assert word.duration == 0.8

    def test_word_without_confidence(self):
        """Test word with default confidence."""
        word = TranscriptWord(
            text="test",
            start_time=0.0,
            end_time=0.3,
        )

        assert word.confidence == 1.0  # Default


class TestTranscriptSegment:
    """Tests for TranscriptSegment class."""

    def test_create_segment(self):
        """Test creating a transcript segment."""
        segment = TranscriptSegment(
            text="Patient reports chest pain.",
            start_time=0.0,
            end_time=2.5,
            confidence=0.95,
        )

        assert segment.text == "Patient reports chest pain."
        assert segment.start_time == 0.0
        assert segment.end_time == 2.5
        assert segment.confidence == 0.95

    def test_segment_with_words(self):
        """Test segment with word-level timestamps."""
        words = [
            TranscriptWord(text="Patient", start_time=0.0, end_time=0.5, confidence=0.98),
            TranscriptWord(text="reports", start_time=0.5, end_time=0.9, confidence=0.96),
            TranscriptWord(text="chest", start_time=0.9, end_time=1.2, confidence=0.99),
            TranscriptWord(text="pain", start_time=1.2, end_time=1.5, confidence=0.97),
        ]

        segment = TranscriptSegment(
            text="Patient reports chest pain",
            start_time=0.0,
            end_time=1.5,
            confidence=0.95,
            words=words,
        )

        assert len(segment.words) == 4
        assert segment.words[0].text == "Patient"

    def test_segment_duration(self):
        """Test segment duration calculation."""
        segment = TranscriptSegment(
            text="Test segment",
            start_time=5.0,
            end_time=8.5,
            confidence=0.9,
        )

        assert segment.duration == 3.5

    def test_segment_with_speaker(self):
        """Test segment with speaker identification."""
        segment = TranscriptSegment(
            text="The patient is doing well.",
            start_time=0.0,
            end_time=2.0,
            confidence=0.95,
            speaker="doctor",
        )

        assert segment.speaker == "doctor"


class TestTranscript:
    """Tests for Transcript class."""

    def test_create_transcript(self):
        """Test creating a transcript."""
        transcript = Transcript(
            text="Patient has chest pain and shortness of breath.",
            segments=[],
            language="en",
            confidence=0.95,
        )

        assert transcript.text == "Patient has chest pain and shortness of breath."
        assert transcript.language == "en"
        assert transcript.confidence == 0.95

    def test_transcript_with_segments(self):
        """Test transcript with multiple segments."""
        segments = [
            TranscriptSegment(
                text="First segment.",
                start_time=0.0,
                end_time=1.5,
                confidence=0.95,
            ),
            TranscriptSegment(
                text="Second segment.",
                start_time=1.5,
                end_time=3.0,
                confidence=0.92,
            ),
        ]

        transcript = Transcript(
            text="First segment. Second segment.",
            segments=segments,
            language="en",
            confidence=0.93,
        )

        assert len(transcript.segments) == 2
        assert transcript.duration == 3.0

    def test_transcript_duration(self):
        """Test transcript duration from segments."""
        segments = [
            TranscriptSegment(text="A", start_time=0.0, end_time=1.0, confidence=0.9),
            TranscriptSegment(text="B", start_time=1.0, end_time=2.5, confidence=0.9),
            TranscriptSegment(text="C", start_time=2.5, end_time=5.0, confidence=0.9),
        ]

        transcript = Transcript(
            text="A B C",
            segments=segments,
            language="en",
            confidence=0.9,
        )

        assert transcript.duration == 5.0

    def test_empty_transcript(self):
        """Test empty transcript."""
        transcript = Transcript(
            text="",
            segments=[],
            language="en",
            confidence=0.0,
        )

        assert transcript.text == ""
        assert transcript.duration == 0.0
        assert len(transcript.segments) == 0

    def test_transcript_to_dict(self):
        """Test transcript serialization to dict."""
        transcript = Transcript(
            text="Test transcript.",
            segments=[
                TranscriptSegment(
                    text="Test transcript.",
                    start_time=0.0,
                    end_time=1.0,
                    confidence=0.95,
                )
            ],
            language="en",
            confidence=0.95,
        )

        data = transcript.to_dict()

        assert data["text"] == "Test transcript."
        assert data["language"] == "en"
        assert data["confidence"] == 0.95
        assert len(data["segments"]) == 1

    def test_transcript_from_dict(self):
        """Test transcript deserialization from dict."""
        data = {
            "text": "Loaded transcript.",
            "language": "en",
            "confidence": 0.92,
            "segments": [
                {
                    "text": "Loaded transcript.",
                    "start_time": 0.0,
                    "end_time": 1.5,
                    "confidence": 0.92,
                }
            ],
        }

        transcript = Transcript.from_dict(data)

        assert transcript.text == "Loaded transcript."
        assert len(transcript.segments) == 1
        assert transcript.segments[0].end_time == 1.5

    def test_transcript_word_count(self):
        """Test word count property."""
        transcript = Transcript(
            text="This is a five word transcript",
            segments=[],
            language="en",
            confidence=0.9,
        )

        assert transcript.word_count == 6
