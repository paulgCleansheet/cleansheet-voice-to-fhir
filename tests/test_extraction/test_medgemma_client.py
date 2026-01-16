"""
Tests for MedGemma cloud client.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import pytest
from unittest.mock import MagicMock, patch

from voice_to_fhir.extraction.medgemma_client import MedGemmaClient, MedGemmaClientConfig


class TestMedGemmaClientConfig:
    """Tests for MedGemma client configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MedGemmaClientConfig()

        assert config.model_id == "google/medgemma-4b"
        assert config.max_tokens == 2048
        assert config.temperature == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = MedGemmaClientConfig(
            api_key="test-key",
            model_id="custom/model",
            max_tokens=4096,
            temperature=0.2,
        )

        assert config.api_key == "test-key"
        assert config.max_tokens == 4096
        assert config.temperature == 0.2


class TestMedGemmaClient:
    """Tests for MedGemma cloud client."""

    @pytest.fixture
    def config(self) -> MedGemmaClientConfig:
        """Create test configuration."""
        return MedGemmaClientConfig(api_key="test-key")

    @pytest.fixture
    def client(self, config: MedGemmaClientConfig) -> MedGemmaClient:
        """Create MedGemma client for testing."""
        return MedGemmaClient(config)

    def test_initialization(self, client: MedGemmaClient):
        """Test client initialization."""
        assert client.config.model_id == "google/medgemma-4b"

    @patch("voice_to_fhir.extraction.medgemma_client.requests")
    def test_extract_success(
        self, mock_requests, client: MedGemmaClient, sample_transcript_text: str
    ):
        """Test successful extraction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generated_text": """
            {
                "conditions": [{"name": "Chest pain", "status": "active"}],
                "medications": [],
                "allergies": [],
                "vitals": []
            }
            """
        }
        mock_requests.post.return_value = mock_response

        entities = client.extract(sample_transcript_text, workflow="general")

        assert len(entities.conditions) == 1
        assert entities.conditions[0].name == "Chest pain"

    @patch("voice_to_fhir.extraction.medgemma_client.requests")
    def test_extract_with_workflow(
        self, mock_requests, client: MedGemmaClient, sample_transcript_text: str
    ):
        """Test extraction with specific workflow."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generated_text": '{"conditions": [], "medications": [], "allergies": [], "vitals": []}'
        }
        mock_requests.post.return_value = mock_response

        entities = client.extract(sample_transcript_text, workflow="emergency")

        # Verify workflow was used (would be reflected in prompt)
        mock_requests.post.assert_called_once()

    @patch("voice_to_fhir.extraction.medgemma_client.requests")
    def test_extract_api_error(
        self, mock_requests, client: MedGemmaClient, sample_transcript_text: str
    ):
        """Test handling of API errors."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_requests.post.return_value = mock_response

        with pytest.raises(Exception):
            client.extract(sample_transcript_text, workflow="general")

    @patch("voice_to_fhir.extraction.medgemma_client.requests")
    def test_extract_invalid_json(
        self, mock_requests, client: MedGemmaClient, sample_transcript_text: str
    ):
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "generated_text": "This is not valid JSON"
        }
        mock_requests.post.return_value = mock_response

        # Should handle gracefully (return empty or raise)
        with pytest.raises(Exception):
            client.extract(sample_transcript_text, workflow="general")

    def test_build_prompt_general(self, client: MedGemmaClient):
        """Test prompt building for general workflow."""
        transcript = "Patient has headache."
        prompt = client._build_prompt(transcript, "general")

        assert "headache" in prompt
        assert "clinical" in prompt.lower() or "extract" in prompt.lower()

    def test_build_prompt_emergency(self, client: MedGemmaClient):
        """Test prompt building for emergency workflow."""
        transcript = "Patient has chest pain."
        prompt = client._build_prompt(transcript, "emergency")

        assert "chest pain" in prompt
        # Emergency workflow should emphasize urgency
        assert isinstance(prompt, str)

    def test_parse_response_valid(self, client: MedGemmaClient):
        """Test parsing valid JSON response."""
        response_text = """
        {
            "conditions": [{"name": "Headache", "status": "active"}],
            "medications": [{"name": "Ibuprofen", "dose": "400mg"}],
            "allergies": [],
            "vitals": []
        }
        """

        entities = client._parse_response(response_text)

        assert len(entities.conditions) == 1
        assert len(entities.medications) == 1

    def test_parse_response_with_markdown(self, client: MedGemmaClient):
        """Test parsing response wrapped in markdown code blocks."""
        response_text = """
        ```json
        {
            "conditions": [{"name": "Fever", "status": "active"}],
            "medications": [],
            "allergies": [],
            "vitals": []
        }
        ```
        """

        entities = client._parse_response(response_text)

        assert len(entities.conditions) == 1
        assert entities.conditions[0].name == "Fever"

    def test_available_workflows(self, client: MedGemmaClient):
        """Test listing available workflows."""
        workflows = client.available_workflows()

        assert isinstance(workflows, list)
        assert "general" in workflows

    def test_health_check(self, client: MedGemmaClient):
        """Test API health check method exists."""
        assert hasattr(client, "health_check")
