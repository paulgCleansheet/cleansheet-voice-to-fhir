"""
MedGemma Cloud Client

HuggingFace Inference API client for MedGemma entity extraction.

Supports multiple backends:
- dedicated: HuggingFace Inference Endpoints (paid, recommended)
- local: Local server running medgemma-server.py
- serverless: HF Inference API (may not work for gated models)
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
    LabOrder,
    MedicationOrder,
    ReferralOrder,
    ProcedureOrder,
    ImagingOrder,
    Allergy,
    FamilyHistory,
    SocialHistory,
    PatientDemographics,
)
from voice_to_fhir.extraction.post_processor import post_process


@dataclass
class MedGemmaClientConfig:
    """Configuration for MedGemma cloud client."""

    api_key: str | None = None
    model_id: str = "google/medgemma-4b-it"
    # Backend: "dedicated" (HF Endpoint), "local", or "serverless"
    backend: str = "dedicated"
    # For dedicated HF Endpoints (e.g., "https://xxxxx.endpoints.huggingface.cloud")
    endpoint_url: str | None = None
    # For local server (e.g., "http://localhost:3003")
    local_url: str = "http://localhost:3003"
    # Legacy serverless API
    api_url: str = "https://api-inference.huggingface.co/models"
    timeout_seconds: float = 300.0
    max_tokens: int = 8192
    temperature: float = 0.1
    prompts_dir: str | Path = "src/voice_to_fhir/extraction/prompts"

    @classmethod
    def from_env(cls) -> "MedGemmaClientConfig":
        """Create configuration from environment variables."""
        return cls(
            api_key=os.environ.get("HF_TOKEN"),
            model_id=os.environ.get("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it"),
            backend=os.environ.get("MEDGEMMA_BACKEND", "dedicated"),
            endpoint_url=os.environ.get("MEDGEMMA_ENDPOINT_URL"),
            local_url=os.environ.get("MEDGEMMA_LOCAL_URL", "http://localhost:3003"),
            timeout_seconds=float(os.environ.get("MEDGEMMA_TIMEOUT", "300.0")),
        )


class MedGemmaClient:
    """HuggingFace client for MedGemma entity extraction.

    Supports multiple backends:
    - dedicated: HuggingFace Inference Endpoints (paid, recommended)
    - local: Local server running medgemma-server.py
    - serverless: HF Inference API (may not work for gated models)
    """

    def __init__(self, config: MedGemmaClientConfig | None = None):
        """Initialize MedGemma client."""
        self.config = config or MedGemmaClientConfig()

        self.api_key = self.config.api_key or os.environ.get("HF_TOKEN")

        # Validate based on backend
        if self.config.backend in ["dedicated", "serverless"] and not self.api_key:
            raise ValueError(
                "HuggingFace API key required for cloud backends. "
                "Set HF_TOKEN environment variable or pass api_key in config."
            )

        if self.config.backend == "dedicated" and not self.config.endpoint_url:
            raise ValueError(
                "endpoint_url required for dedicated backend. "
                "Set MEDGEMMA_ENDPOINT_URL or pass endpoint_url in config."
            )

        self._prompts_cache: dict[str, str] = {}

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def _endpoint(self) -> str:
        """API endpoint URL based on backend."""
        if self.config.backend == "dedicated":
            # HF Endpoints use OpenAI-compatible format
            base = self.config.endpoint_url.rstrip("/")
            return f"{base}/v1/chat/completions"
        elif self.config.backend == "local":
            return self.config.local_url
        else:
            # Serverless
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

        backend = self.config.backend
        print(f"[MedGemma] Extracting with backend: {backend}")

        # Build payload based on backend
        if backend == "dedicated":
            # OpenAI-compatible chat completions format
            payload = {
                "model": self.config.model_id,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
        elif backend == "local":
            # Simple prompt format for local server
            payload = {
                "prompt": full_prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
        else:
            # Serverless HF Inference format
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
                f"MedGemma API error: {response.status_code} - {response.text[:500]}"
            )

        result = response.json()

        # Parse response based on backend format
        if backend == "dedicated":
            # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
            choices = result.get("choices", [])
            if choices:
                generated_text = choices[0].get("message", {}).get("content", "")
            else:
                generated_text = ""
        elif backend == "local":
            # Local server format: {"text": "..."}
            generated_text = result.get("text", "")
        else:
            # Serverless format: [{"generated_text": "..."}] or {"generated_text": "..."}
            if isinstance(result, list) and result:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = str(result)

        # Debug: Log raw MedGemma response
        print(f"[MedGemma] Raw response length: {len(generated_text)}")
        if len(generated_text) < 2000:
            print(f"[MedGemma] Raw response: {generated_text}")
        else:
            print(f"[MedGemma] Raw response preview: {generated_text[:500]}...")

        # Parse JSON from response
        entities = self._parse_response(generated_text, transcript, workflow)

        # Apply deterministic post-processing (BP normalization, ICD-10 enrichment, etc.)
        entities = post_process(entities, transcript)

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
                # Debug: Log parsed data keys and medication count
                print(f"[MedGemma Parse] Keys in response: {list(data.keys())}")
                print(f"[MedGemma Parse] Medications found: {len(data.get('medications', []))}")
                print(f"[MedGemma Parse] Allergies found: {len(data.get('allergies', []))}")
                if data.get('medications'):
                    print(f"[MedGemma Parse] Medication names: {[m.get('name') for m in data.get('medications', [])]}")
            else:
                print("[MedGemma Parse] No JSON found in response")
                data = {}
        except json.JSONDecodeError as e:
            print(f"[MedGemma Parse] JSON decode error: {e}")
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
            if isinstance(p, dict):
                entities.patient = PatientDemographics(
                    name=p.get("name"),
                    date_of_birth=p.get("date_of_birth"),
                    gender=p.get("gender"),
                    mrn=p.get("mrn"),
                )

        # Parse conditions
        for c in data.get("conditions", []):
            if not isinstance(c, dict):
                # Handle string conditions
                if isinstance(c, str) and c:
                    entities.conditions.append(Condition(name=c, is_chief_complaint=False))
                continue
            # Handle null values from JSON
            name = c.get("name") or c.get("description") or ""
            if not name:
                continue  # Skip conditions without a name
            # Extract status from MedGemma response (e.g., "resolved" for discharge diagnoses)
            condition_status = c.get("status")
            condition = Condition(
                name=name,
                status=condition_status if condition_status else "active",
                severity=c.get("severity"),
                onset=c.get("onset"),
                icd10=c.get("icd10"),
                is_chief_complaint=False,
            )
            entities.conditions.append(condition)

        # Store chief complaint text (reason for visit / presenting symptoms)
        # DO NOT add symptoms as conditions - post_processor will handle marking
        chief = data.get("chief_complaint")
        if chief:
            entities.chief_complaint_text = chief
            # Mark first condition as chief complaint if conditions exist
            if entities.conditions:
                entities.conditions[0].is_chief_complaint = True

        # Parse vitals
        for v in data.get("vitals", []):
            if not isinstance(v, dict):
                continue
            # Handle null values from JSON
            vital_type = v.get("type") or v.get("name") or ""
            vital_value = v.get("value")
            if vital_value is None or vital_value == "":
                continue  # Skip vitals without a value
            # Convert numeric values to strings (MedGemma may return numbers)
            vital_value = str(vital_value)
            # Don't skip vitals without type - post-processor will infer from unit/value
            vital = Vital(
                type=vital_type,
                value=vital_value,
                unit=v.get("unit"),
            )
            entities.vitals.append(vital)

        # Parse observations (legacy format - convert to vitals)
        for o in data.get("observations", []):
            if not isinstance(o, dict):
                continue
            obs_value = o.get("value")
            if obs_value is None or obs_value == "":
                continue  # Skip observations without a value
            # Convert numeric values to strings
            obs_value = str(obs_value)
            vital = Vital(
                type=o.get("name", ""),
                value=obs_value,
                unit=o.get("unit"),
            )
            entities.vitals.append(vital)

        # Parse lab results (completed labs with values)
        for lr in data.get("lab_results", []):
            if not isinstance(lr, dict):
                continue
            # Handle null values from JSON
            lab_name = lr.get("name") or ""
            if not lab_name:
                continue  # Skip lab results without a name
            lab = LabResult(
                name=lab_name,
                value=lr.get("value"),  # Can be None for pending labs
                unit=lr.get("unit"),
                interpretation=lr.get("interpretation"),
                reference_range=lr.get("reference_range"),
                status=lr.get("status") or "completed",
            )
            entities.lab_results.append(lab)

        # Parse lab orders (pending labs without values)
        for lo in data.get("lab_orders", []):
            if not isinstance(lo, dict):
                # Handle string lab orders
                if isinstance(lo, str) and lo:
                    entities.lab_orders.append(LabOrder(name=lo))
                continue
            # Handle null values from JSON
            lab_order_name = lo.get("name") or ""
            if not lab_order_name:
                continue  # Skip lab orders without a name
            lab_order = LabOrder(
                name=lab_order_name,
                loinc=lo.get("loinc"),
            )
            entities.lab_orders.append(lab_order)

        # Parse medication orders (new prescriptions)
        for mo in data.get("medication_orders", []):
            if not isinstance(mo, dict):
                # Handle string medication orders
                if isinstance(mo, str) and mo:
                    entities.medication_orders.append(MedicationOrder(name=mo))
                continue
            # Handle null values from JSON
            med_order_name = mo.get("name") or ""
            if not med_order_name:
                continue  # Skip medication orders without a name
            med_order = MedicationOrder(
                name=med_order_name,
                dose=mo.get("dose"),
                frequency=mo.get("frequency"),
                instructions=mo.get("instructions"),
            )
            entities.medication_orders.append(med_order)

        # Parse referral orders (consults)
        for ro in data.get("referral_orders", []):
            if not isinstance(ro, dict):
                if isinstance(ro, str) and ro:
                    entities.referral_orders.append(ReferralOrder(specialty=ro))
                continue
            # Handle null values from JSON
            specialty = ro.get("specialty") or ro.get("name") or ""
            if not specialty:
                continue  # Skip referral orders without a specialty
            ref_order = ReferralOrder(
                specialty=specialty,
                reason=ro.get("reason"),
            )
            entities.referral_orders.append(ref_order)

        # Parse procedure orders
        for po in data.get("procedure_orders", []):
            if not isinstance(po, dict):
                if isinstance(po, str) and po:
                    entities.procedure_orders.append(ProcedureOrder(name=po))
                continue
            # Handle null values from JSON
            proc_name = po.get("name") or ""
            if not proc_name:
                continue  # Skip procedure orders without a name
            proc_order = ProcedureOrder(
                name=proc_name,
            )
            entities.procedure_orders.append(proc_order)

        # Parse imaging orders
        for io in data.get("imaging_orders", []):
            if not isinstance(io, dict):
                if isinstance(io, str) and io:
                    entities.imaging_orders.append(ImagingOrder(name=io))
                continue
            # Handle null values from JSON
            img_name = io.get("name") or ""
            if not img_name:
                continue  # Skip imaging orders without a name
            img_order = ImagingOrder(
                name=img_name,
            )
            entities.imaging_orders.append(img_order)

        # Parse allergies
        for a in data.get("allergies", []):
            if not isinstance(a, dict):
                if isinstance(a, str) and a:
                    entities.allergies.append(Allergy(substance=a))
                continue
            # Handle null values from JSON
            substance = a.get("substance") or ""
            if not substance:
                continue  # Skip allergies without a substance
            allergy = Allergy(
                substance=substance,
                reaction=a.get("reaction"),
                severity=a.get("severity"),
            )
            entities.allergies.append(allergy)

        # Parse medications
        for m in data.get("medications", []):
            if not isinstance(m, dict):
                if isinstance(m, str) and m:
                    entities.medications.append(Medication(name=m, status="active"))
                continue
            # Handle null values from JSON
            med_name = m.get("name") or ""
            if not med_name:
                continue  # Skip medications without a name
            med = Medication(
                name=med_name,
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
            if not isinstance(m, dict):
                if isinstance(m, str) and m:
                    entities.medications.append(Medication(name=m, status="active"))
                continue
            # Handle null values from JSON
            med_name = m.get("name") or ""
            if not med_name:
                continue  # Skip medications without a name
            med = Medication(
                name=med_name,
                dose=m.get("dose"),
                frequency=m.get("frequency"),
                status="active",
                is_new_order=False,
            )
            entities.medications.append(med)

        # Parse new_medications (legacy format)
        for m in data.get("new_medications", []):
            if not isinstance(m, dict):
                if isinstance(m, str) and m:
                    entities.medications.append(Medication(name=m, status="active", is_new_order=True))
                continue
            # Handle null values from JSON
            med_name = m.get("name") or ""
            if not med_name:
                continue  # Skip medications without a name
            med = Medication(
                name=med_name,
                dose=m.get("dose"),
                frequency=m.get("frequency"),
                status="active",
                is_new_order=True,
            )
            entities.medications.append(med)

        # Parse family_history
        for fh in data.get("family_history", []):
            if not isinstance(fh, dict):
                if isinstance(fh, str) and fh:
                    entities.family_history.append(FamilyHistory(relationship="unknown", condition=fh))
                continue
            # Handle null values from JSON (dict.get returns None for explicit nulls)
            relationship = fh.get("relationship") or "unknown"
            condition = fh.get("condition") or ""
            if not condition:
                continue  # Skip family history entries without a condition
            family_hist = FamilyHistory(
                relationship=relationship,
                condition=condition,
                age_of_onset=fh.get("age_of_onset"),
                deceased=fh.get("deceased"),
            )
            entities.family_history.append(family_hist)

        # Parse social_history
        sh = data.get("social_history")
        if sh and isinstance(sh, dict):
            entities.social_history = SocialHistory(
                tobacco=sh.get("tobacco"),
                alcohol=sh.get("alcohol"),
                drugs=sh.get("drugs"),
                occupation=sh.get("occupation"),
                living_situation=sh.get("living_situation"),
            )

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
