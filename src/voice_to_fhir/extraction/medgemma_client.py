"""
MedGemma Cloud Client

HuggingFace Inference API client for MedGemma entity extraction.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path

import requests

from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Vital,
    LabResult,
    Allergy,
    PatientDemographics,
)


@dataclass
class MedGemmaClientConfig:
    """Configuration for MedGemma cloud client."""

    api_key: str | None = None
    model_id: str = "google/medgemma-4b"
    api_url: str = "https://api-inference.huggingface.co/models"
    timeout_seconds: float = 60.0
    max_tokens: int = 2048
    temperature: float = 0.1
    prompts_dir: str | Path = "src/voice_to_fhir/extraction/prompts"


class MedGemmaClient:
    """HuggingFace Inference API client for MedGemma."""

    def __init__(self, config: MedGemmaClientConfig | None = None):
        """Initialize MedGemma client."""
        self.config = config or MedGemmaClientConfig()

        self.api_key = self.config.api_key or os.environ.get("HF_TOKEN")
        if not self.api_key:
            raise ValueError(
                "HuggingFace API key required. "
                "Set HF_TOKEN environment variable or pass api_key in config."
            )

        self._prompts_cache: dict[str, str] = {}

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @property
    def _endpoint(self) -> str:
        """API endpoint URL."""
        return f"{self.config.api_url}/{self.config.model_id}"

    def _load_prompt(self, workflow: str) -> str:
        """Load prompt template for workflow."""
        if workflow in self._prompts_cache:
            return self._prompts_cache[workflow]

        prompts_dir = Path(self.config.prompts_dir)
        prompt_file = prompts_dir / f"{workflow}.txt"

        if prompt_file.exists():
            prompt = prompt_file.read_text()
        else:
            # Use default prompt
            prompt = self._default_prompt()

        self._prompts_cache[workflow] = prompt
        return prompt

    def _default_prompt(self) -> str:
        """Default extraction prompt."""
        return """You are a medical documentation assistant. Extract structured clinical information from the following transcript.

Return a JSON object with the following structure:
{
  "patient": {
    "name": "string or null",
    "date_of_birth": "string or null",
    "gender": "string or null"
  },
  "chief_complaint": "string or null",
  "conditions": [
    {
      "name": "string",
      "severity": "mild|moderate|severe|unknown",
      "onset": "string or null",
      "icd10": "string or null"
    }
  ],
  "vitals": [
    {
      "type": "string (e.g., 'blood_pressure', 'temperature')",
      "value": "string",
      "unit": "string or null"
    }
  ],
  "lab_results": [
    {
      "name": "string",
      "value": "string",
      "unit": "string or null",
      "interpretation": "normal|high|low|critical or null"
    }
  ],
  "allergies": [
    {
      "substance": "string",
      "reaction": "string or null",
      "severity": "string or null"
    }
  ],
  "medications": [
    {
      "name": "string",
      "dose": "string or null",
      "frequency": "string or null",
      "is_new_order": false
    }
  ]
}

Only include information explicitly mentioned in the transcript.
Return valid JSON only, no additional text.

TRANSCRIPT:
"""

    def available_workflows(self) -> list[str]:
        """List available workflow types."""
        from voice_to_fhir.extraction.prompts import AVAILABLE_WORKFLOWS
        return AVAILABLE_WORKFLOWS.copy()

    def health_check(self) -> bool:
        """Check if the API is reachable."""
        try:
            response = requests.get(
                f"{self.config.api_url}/{self.config.model_id}",
                headers=self._headers,
                timeout=10.0,
            )
            return response.status_code in [200, 503]  # 503 = model loading
        except Exception:
            return False

    def _build_prompt(self, transcript: str, workflow: str) -> str:
        """Build the full prompt for extraction."""
        prompt_template = self._load_prompt(workflow)

        # Use {transcript} placeholder if present, otherwise append
        if "{transcript}" in prompt_template:
            return prompt_template.replace("{transcript}", transcript)
        else:
            return f"{prompt_template}\n{transcript}\n\nJSON:"

    def extract(self, transcript: str, workflow: str = "general") -> ClinicalEntities:
        """Extract structured clinical entities from transcript."""
        # Build prompt
        full_prompt = self._build_prompt(transcript, workflow)

        # Make API request
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "return_full_text": False,
            },
        }

        response = requests.post(
            self._endpoint,
            headers=self._headers,
            json=payload,
            timeout=self.config.timeout_seconds,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"MedGemma API error: {response.status_code} - {response.text}"
            )

        result = response.json()

        # Parse response
        if isinstance(result, list) and result:
            generated_text = result[0].get("generated_text", "")
        elif isinstance(result, dict):
            generated_text = result.get("generated_text", "")
        else:
            generated_text = str(result)

        # Parse JSON from response
        entities = self._parse_response(generated_text, transcript, workflow)
        return entities

    def _parse_response(
        self, response_text: str, transcript: str = "", workflow: str = "general"
    ) -> ClinicalEntities:
        """Parse MedGemma response into ClinicalEntities."""
        # Try to extract JSON from response
        try:
            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}

        # Build ClinicalEntities from parsed data
        entities = ClinicalEntities(
            workflow=workflow,
            raw_transcript=transcript,
            extraction_metadata={
                "model": self.config.model_id,
                "backend": "cloud",
            },
        )

        # Parse patient
        if "patient" in data and data["patient"]:
            p = data["patient"]
            entities.patient = PatientDemographics(
                name=p.get("name"),
                date_of_birth=p.get("date_of_birth"),
                gender=p.get("gender"),
                mrn=p.get("mrn"),
            )

        # Parse conditions
        for c in data.get("conditions", []):
            condition = Condition(
                name=c.get("name") or c.get("description", ""),
                severity=c.get("severity"),
                onset=c.get("onset"),
                icd10=c.get("icd10"),
                is_chief_complaint=False,
            )
            entities.conditions.append(condition)

        # Mark chief complaint
        chief = data.get("chief_complaint")
        if chief and entities.conditions:
            entities.conditions[0].is_chief_complaint = True
        elif chief:
            entities.conditions.insert(
                0, Condition(name=chief, is_chief_complaint=True)
            )

        # Parse vitals
        for v in data.get("vitals", []):
            vital = Vital(
                type=v.get("type") or v.get("name", ""),
                value=v.get("value", ""),
                unit=v.get("unit"),
            )
            entities.vitals.append(vital)

        # Parse observations (legacy format - convert to vitals)
        for o in data.get("observations", []):
            vital = Vital(
                type=o.get("name", ""),
                value=o.get("value", ""),
                unit=o.get("unit"),
            )
            entities.vitals.append(vital)

        # Parse lab results
        for lr in data.get("lab_results", []):
            lab = LabResult(
                name=lr.get("name", ""),
                value=lr.get("value", ""),
                unit=lr.get("unit"),
                interpretation=lr.get("interpretation"),
                reference_range=lr.get("reference_range"),
            )
            entities.lab_results.append(lab)

        # Parse allergies
        for a in data.get("allergies", []):
            allergy = Allergy(
                substance=a.get("substance", ""),
                reaction=a.get("reaction"),
                severity=a.get("severity"),
            )
            entities.allergies.append(allergy)

        # Parse medications
        for m in data.get("medications", []):
            med = Medication(
                name=m.get("name", ""),
                dose=m.get("dose"),
                frequency=m.get("frequency"),
                route=m.get("route"),
                rxnorm=m.get("rxnorm"),
                status="active",
                is_new_order=m.get("is_new_order", False),
            )
            entities.medications.append(med)

        # Parse current_medications (legacy format)
        for m in data.get("current_medications", []):
            med = Medication(
                name=m.get("name", ""),
                dose=m.get("dose"),
                frequency=m.get("frequency"),
                status="active",
                is_new_order=False,
            )
            entities.medications.append(med)

        # Parse new_medications (legacy format)
        for m in data.get("new_medications", []):
            med = Medication(
                name=m.get("name", ""),
                dose=m.get("dose"),
                frequency=m.get("frequency"),
                status="active",
                is_new_order=True,
            )
            entities.medications.append(med)

        return entities

    def extract_with_context(
        self, transcript: str, patient_context: dict, workflow: str = "general"
    ) -> ClinicalEntities:
        """Extract with additional patient context."""
        # Add context to transcript
        context_str = "\n".join(
            f"{k}: {v}" for k, v in patient_context.items() if v
        )
        enhanced_transcript = f"PATIENT CONTEXT:\n{context_str}\n\nTRANSCRIPT:\n{transcript}"

        return self.extract(enhanced_transcript, workflow)
