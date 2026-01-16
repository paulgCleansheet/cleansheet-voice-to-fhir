"""
MedGemma Local Inference

Local inference for MedGemma on edge devices.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Observation,
    Allergy,
    PatientDemographics,
    ConditionSeverity,
    MedicationStatus,
)


@dataclass
class MedGemmaLocalConfig:
    """Configuration for local MedGemma inference."""

    model_path: str | Path = "models/medgemma-4b"
    device: str = "cuda"
    precision: str = "int8"  # fp32, fp16, int8 for edge
    use_tensorrt: bool = False
    max_tokens: int = 2048
    temperature: float = 0.1
    prompts_dir: str | Path = "src/voice_to_fhir/extraction/prompts"


class MedGemmaLocal:
    """Local MedGemma inference for edge deployment."""

    def __init__(self, config: MedGemmaLocalConfig | None = None):
        """Initialize local MedGemma model."""
        self.config = config or MedGemmaLocalConfig()
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._prompts_cache: dict[str, str] = {}

    def _ensure_initialized(self) -> None:
        """Lazy initialization of model."""
        if self._initialized:
            return

        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        """Load MedGemma model for local inference."""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'python scripts/download_models.py' first."
            )

        # Determine device and dtype
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
            if self.config.precision == "fp16":
                dtype = torch.float16
            elif self.config.precision == "int8":
                dtype = torch.float16  # Will quantize separately
            else:
                dtype = torch.float32
        else:
            device = "cpu"
            dtype = torch.float32

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model with quantization if int8
        if self.config.precision == "int8" and device == "cuda":
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device,
            )

    def _load_prompt(self, workflow: str) -> str:
        """Load prompt template for workflow."""
        if workflow in self._prompts_cache:
            return self._prompts_cache[workflow]

        prompts_dir = Path(self.config.prompts_dir)
        prompt_file = prompts_dir / f"{workflow}.txt"

        if prompt_file.exists():
            prompt = prompt_file.read_text()
        else:
            prompt = self._default_prompt()

        self._prompts_cache[workflow] = prompt
        return prompt

    def _default_prompt(self) -> str:
        """Default extraction prompt."""
        return """Extract structured clinical information from the transcript.
Return JSON with: patient, chief_complaint, conditions, observations, allergies, current_medications, new_medications.

TRANSCRIPT:
"""

    def extract(self, transcript: str, workflow: str = "general") -> ClinicalEntities:
        """Extract structured clinical entities from transcript."""
        import torch

        self._ensure_initialized()

        # Build prompt
        prompt_template = self._load_prompt(workflow)
        full_prompt = f"{prompt_template}\n{transcript}\n\nJSON:"

        # Tokenize
        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        if self.config.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self._tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )

        # Parse
        return self._parse_response(generated_text, transcript, workflow)

    def _parse_response(
        self, response_text: str, transcript: str, workflow: str
    ) -> ClinicalEntities:
        """Parse response into ClinicalEntities."""
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
            else:
                data = {}
        except json.JSONDecodeError:
            data = {}

        entities = ClinicalEntities(
            workflow=workflow,
            raw_transcript=transcript,
            extraction_metadata={
                "model": str(self.config.model_path),
                "backend": "local",
                "device": self.config.device,
            },
        )

        # Parse patient
        if "patient" in data and data["patient"]:
            p = data["patient"]
            entities.patient = PatientDemographics(
                name=p.get("name"),
                date_of_birth=p.get("date_of_birth"),
                gender=p.get("gender"),
            )

        # Parse conditions
        for c in data.get("conditions", []):
            entities.conditions.append(
                Condition(
                    description=c.get("description", ""),
                    severity=ConditionSeverity(c.get("severity", "unknown")),
                    onset=c.get("onset"),
                )
            )

        # Mark chief complaint
        if data.get("chief_complaint") and entities.conditions:
            entities.conditions[0].is_chief_complaint = True

        # Parse observations
        for o in data.get("observations", []):
            entities.observations.append(
                Observation(
                    name=o.get("name", ""),
                    value=o.get("value", ""),
                    unit=o.get("unit"),
                )
            )

        # Parse allergies
        for a in data.get("allergies", []):
            entities.allergies.append(
                Allergy(
                    substance=a.get("substance", ""),
                    reaction=a.get("reaction"),
                )
            )

        # Parse medications
        for m in data.get("current_medications", []):
            entities.current_medications.append(
                Medication(
                    name=m.get("name", ""),
                    dose=m.get("dose"),
                    frequency=m.get("frequency"),
                    status=MedicationStatus.ACTIVE,
                )
            )

        for m in data.get("new_medications", []):
            entities.new_medications.append(
                Medication(
                    name=m.get("name", ""),
                    dose=m.get("dose"),
                    frequency=m.get("frequency"),
                    is_new_order=True,
                )
            )

        return entities

    def warmup(self) -> None:
        """Warm up model with dummy inference."""
        self._ensure_initialized()
        _ = self.extract("Patient reports feeling well. No complaints.")
