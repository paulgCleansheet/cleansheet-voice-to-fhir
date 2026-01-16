"""
Tests for command-line interface.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from voice_to_fhir.cli import app


runner = CliRunner()


class TestCLIProcess:
    """Tests for the process command."""

    def test_process_file_not_found(self, tmp_path: Path):
        """Test error when file doesn't exist."""
        result = runner.invoke(app, ["process", str(tmp_path / "nonexistent.wav")])

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_process_success(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test successful processing."""
        # Create test audio file
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        # Mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.process_file.return_value = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [],
        }
        mock_pipeline.to_json.return_value = '{"resourceType": "Bundle"}'
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(app, ["process", str(audio_file)])

        assert result.exit_code == 0
        mock_pipeline.process_file.assert_called_once()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_process_with_output(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test processing with output file."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        output_file = tmp_path / "output.json"

        mock_pipeline = MagicMock()
        mock_pipeline.process_file.return_value = {"resourceType": "Bundle", "type": "collection"}
        mock_pipeline.to_json.return_value = '{"resourceType": "Bundle"}'
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(
            app, ["process", str(audio_file), "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_process_with_config(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test processing with config file."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()
        config_file = tmp_path / "config.yaml"
        config_file.write_text("name: test\n")

        mock_pipeline = MagicMock()
        mock_pipeline.process_file.return_value = {"resourceType": "Bundle", "type": "collection"}
        mock_pipeline.to_json.return_value = "{}"
        mock_pipeline_class.from_config.return_value = mock_pipeline

        result = runner.invoke(
            app, ["process", str(audio_file), "--config", str(config_file)]
        )

        mock_pipeline_class.from_config.assert_called_once()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_process_with_workflow(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test processing with workflow option."""
        audio_file = tmp_path / "test.wav"
        audio_file.touch()

        mock_pipeline = MagicMock()
        mock_pipeline.process_file.return_value = {"resourceType": "Bundle", "type": "collection"}
        mock_pipeline.to_json.return_value = "{}"
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(
            app, ["process", str(audio_file), "--workflow", "emergency"]
        )

        mock_pipeline.process_file.assert_called_once()
        call_args = mock_pipeline.process_file.call_args
        assert call_args[0][1] == "emergency" or call_args[1].get("workflow") == "emergency"


class TestCLITranscript:
    """Tests for the transcript command."""

    @patch("voice_to_fhir.cli.Pipeline")
    def test_transcript_success(self, mock_pipeline_class: MagicMock):
        """Test processing transcript text."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_transcript.return_value = {
            "resourceType": "Bundle",
            "type": "collection",
        }
        mock_pipeline.to_json.return_value = '{"resourceType": "Bundle"}'
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(app, ["transcript", "Patient has chest pain"])

        assert result.exit_code == 0
        mock_pipeline.process_transcript.assert_called_once()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_transcript_with_output(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test transcript with output file."""
        output_file = tmp_path / "output.json"

        mock_pipeline = MagicMock()
        mock_pipeline.process_transcript.return_value = {"resourceType": "Bundle", "type": "collection"}
        mock_pipeline.to_json.return_value = '{"resourceType": "Bundle"}'
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(
            app, ["transcript", "Test text", "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()


class TestCLIBatch:
    """Tests for the batch command."""

    def test_batch_input_not_found(self, tmp_path: Path):
        """Test error when input directory doesn't exist."""
        result = runner.invoke(
            app,
            [
                "batch",
                str(tmp_path / "nonexistent"),
                str(tmp_path / "output"),
            ],
        )

        assert result.exit_code == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_batch_no_files(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test batch with no matching files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        result = runner.invoke(app, ["batch", str(input_dir), str(output_dir)])

        assert result.exit_code == 0
        assert "no files" in result.stdout.lower()

    @patch("voice_to_fhir.cli.Pipeline")
    def test_batch_success(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test successful batch processing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.wav").touch()
        (input_dir / "file2.wav").touch()
        output_dir = tmp_path / "output"

        mock_pipeline = MagicMock()
        mock_pipeline.process_file.return_value = {"resourceType": "Bundle", "type": "collection"}
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(app, ["batch", str(input_dir), str(output_dir)])

        assert result.exit_code == 0
        assert mock_pipeline.process_file.call_count == 2

    @patch("voice_to_fhir.cli.Pipeline")
    def test_batch_with_pattern(self, mock_pipeline_class: MagicMock, tmp_path: Path):
        """Test batch with custom file pattern."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        (input_dir / "file1.mp3").touch()
        (input_dir / "file2.wav").touch()
        output_dir = tmp_path / "output"

        mock_pipeline = MagicMock()
        mock_pipeline.process_file.return_value = {"resourceType": "Bundle", "type": "collection"}
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(
            app, ["batch", str(input_dir), str(output_dir), "--pattern", "*.mp3"]
        )

        # Should only process mp3 files
        assert mock_pipeline.process_file.call_count == 1


class TestCLIDevices:
    """Tests for the devices command."""

    @patch("voice_to_fhir.cli.AudioCapture")
    def test_devices_list(self, mock_capture_class: MagicMock):
        """Test listing audio devices."""
        mock_capture_class.list_devices.return_value = [
            {"index": 0, "name": "Default Microphone", "channels": 2, "sample_rate": 48000},
            {"index": 1, "name": "USB Microphone", "channels": 1, "sample_rate": 44100},
        ]

        result = runner.invoke(app, ["devices"])

        assert result.exit_code == 0
        assert "Default Microphone" in result.stdout
        assert "USB Microphone" in result.stdout

    @patch("voice_to_fhir.cli.AudioCapture")
    def test_devices_empty(self, mock_capture_class: MagicMock):
        """Test no devices found."""
        mock_capture_class.list_devices.return_value = []

        result = runner.invoke(app, ["devices"])

        assert result.exit_code == 0
        assert "no" in result.stdout.lower()


class TestCLIVersion:
    """Tests for the version command."""

    def test_version_output(self):
        """Test version command output."""
        result = runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "voice-to-fhir" in result.stdout.lower()


class TestCLICapture:
    """Tests for the capture command."""

    @patch("voice_to_fhir.cli.Pipeline")
    def test_capture_keyboard_interrupt(self, mock_pipeline_class: MagicMock):
        """Test capture handles keyboard interrupt."""
        mock_pipeline = MagicMock()
        mock_pipeline.capture_and_process.side_effect = KeyboardInterrupt()
        mock_pipeline_class.return_value = mock_pipeline

        result = runner.invoke(app, ["capture"])

        assert result.exit_code == 0
        assert "cancel" in result.stdout.lower()
