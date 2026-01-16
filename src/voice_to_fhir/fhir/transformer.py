"""
FHIR Transformer

Transform extracted clinical entities to FHIR R4 resources.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import json
import uuid

from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Observation,
    Allergy,
)


@dataclass
class FHIRConfig:
    """Configuration for FHIR transformation."""

    fhir_version: str = "R4"
    base_url: str = "http://example.org/fhir"
    validate_output: bool = True
    include_text_narrative: bool = True
    terminology_service_url: str | None = None


class FHIRTransformer:
    """Transform clinical entities to FHIR R4 resources."""

    def __init__(self, config: FHIRConfig | None = None):
        """Initialize transformer."""
        self.config = config or FHIRConfig()

    def transform(self, entities: ClinicalEntities) -> dict[str, Any]:
        """Transform ClinicalEntities to FHIR Bundle."""
        bundle = {
            "resourceType": "Bundle",
            "id": str(uuid.uuid4()),
            "type": "transaction",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "entry": [],
        }

        # Track references for linking
        patient_ref = None
        encounter_ref = None

        # Create Patient if demographics present
        if entities.patient:
            patient_resource = self._create_patient(entities.patient)
            patient_ref = f"Patient/{patient_resource['id']}"
            bundle["entry"].append(self._wrap_entry(patient_resource, "POST"))

        # Create Encounter
        encounter_resource = self._create_encounter(entities.workflow, patient_ref)
        encounter_ref = f"Encounter/{encounter_resource['id']}"
        bundle["entry"].append(self._wrap_entry(encounter_resource, "POST"))

        # Create Conditions
        for condition in entities.conditions:
            resource = self._create_condition(
                condition, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create Observations (vital signs, etc.)
        for observation in entities.observations:
            resource = self._create_observation(
                observation, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create AllergyIntolerances
        for allergy in entities.allergies:
            resource = self._create_allergy_intolerance(allergy, patient_ref)
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create MedicationStatements (current medications)
        for med in entities.current_medications:
            resource = self._create_medication_statement(
                med, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create MedicationRequests (new medications)
        for med in entities.new_medications:
            resource = self._create_medication_request(
                med, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        return bundle

    def _wrap_entry(
        self, resource: dict[str, Any], method: str = "POST"
    ) -> dict[str, Any]:
        """Wrap resource in bundle entry."""
        return {
            "fullUrl": f"urn:uuid:{resource['id']}",
            "resource": resource,
            "request": {
                "method": method,
                "url": resource["resourceType"],
            },
        }

    def _create_patient(self, demographics) -> dict[str, Any]:
        """Create Patient resource."""
        resource = {
            "resourceType": "Patient",
            "id": str(uuid.uuid4()),
        }

        if demographics.name:
            resource["name"] = [{"text": demographics.name}]

        if demographics.date_of_birth:
            resource["birthDate"] = demographics.date_of_birth

        if demographics.gender:
            gender_map = {
                "male": "male",
                "female": "female",
                "m": "male",
                "f": "female",
            }
            resource["gender"] = gender_map.get(
                demographics.gender.lower(), "unknown"
            )

        if demographics.mrn:
            resource["identifier"] = [
                {
                    "type": {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                                "code": "MR",
                            }
                        ]
                    },
                    "value": demographics.mrn,
                }
            ]

        return resource

    def _create_encounter(
        self, workflow: str, patient_ref: str | None
    ) -> dict[str, Any]:
        """Create Encounter resource."""
        resource = {
            "resourceType": "Encounter",
            "id": str(uuid.uuid4()),
            "status": "in-progress",
            "class": {
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                "code": "AMB",
                "display": "ambulatory",
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        # Map workflow to encounter type
        workflow_types = {
            "intake": ("intake", "Patient intake"),
            "hpi": ("hpi", "History of present illness"),
            "assessment": ("assessment", "Clinical assessment"),
            "general": ("general", "General encounter"),
        }

        if workflow in workflow_types:
            code, display = workflow_types[workflow]
            resource["type"] = [
                {
                    "coding": [
                        {
                            "system": "http://example.org/encounter-type",
                            "code": code,
                            "display": display,
                        }
                    ]
                }
            ]

        return resource

    def _create_condition(
        self,
        condition: Condition,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create Condition resource."""
        resource = {
            "resourceType": "Condition",
            "id": str(uuid.uuid4()),
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active",
                    }
                ]
            },
            "code": {
                "text": condition.description,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add SNOMED code if available
        if condition.code and condition.code.system == "SNOMED":
            resource["code"]["coding"] = [
                {
                    "system": "http://snomed.info/sct",
                    "code": condition.code.code,
                    "display": condition.code.display,
                }
            ]

        # Add severity
        if condition.severity.value != "unknown":
            severity_codes = {
                "mild": ("255604002", "Mild"),
                "moderate": ("6736007", "Moderate"),
                "severe": ("24484000", "Severe"),
            }
            if condition.severity.value in severity_codes:
                code, display = severity_codes[condition.severity.value]
                resource["severity"] = {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": code,
                            "display": display,
                        }
                    ]
                }

        # Add category for chief complaint
        if condition.is_chief_complaint:
            resource["category"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                            "code": "encounter-diagnosis",
                            "display": "Encounter Diagnosis",
                        }
                    ],
                    "text": "Chief Complaint",
                }
            ]

        return resource

    def _create_observation(
        self,
        observation: Observation,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create Observation resource."""
        resource = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "final",
            "code": {
                "text": observation.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add LOINC code if available
        if observation.code and observation.code.system == "LOINC":
            resource["code"]["coding"] = [
                {
                    "system": "http://loinc.org",
                    "code": observation.code.code,
                    "display": observation.code.display,
                }
            ]

        # Try to parse numeric value
        try:
            numeric_value = float(observation.value.replace(",", ""))
            resource["valueQuantity"] = {
                "value": numeric_value,
                "unit": observation.unit or "",
            }
        except ValueError:
            resource["valueString"] = observation.value

        # Add vital signs category
        vital_signs = [
            "blood pressure",
            "heart rate",
            "pulse",
            "temperature",
            "respiratory rate",
            "oxygen saturation",
            "weight",
            "height",
            "bmi",
        ]
        if any(vs in observation.name.lower() for vs in vital_signs):
            resource["category"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs",
                        }
                    ]
                }
            ]

        return resource

    def _create_allergy_intolerance(
        self, allergy: Allergy, patient_ref: str | None
    ) -> dict[str, Any]:
        """Create AllergyIntolerance resource."""
        resource = {
            "resourceType": "AllergyIntolerance",
            "id": str(uuid.uuid4()),
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
                        "code": "active",
                    }
                ]
            },
            "code": {
                "text": allergy.substance,
            },
        }

        if patient_ref:
            resource["patient"] = {"reference": patient_ref}

        if allergy.reaction:
            resource["reaction"] = [
                {
                    "manifestation": [{"text": allergy.reaction}],
                }
            ]

        if allergy.severity:
            resource["reaction"] = resource.get("reaction", [{}])
            resource["reaction"][0]["severity"] = allergy.severity.lower()

        return resource

    def _create_medication_statement(
        self,
        medication: Medication,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create MedicationStatement resource."""
        resource = {
            "resourceType": "MedicationStatement",
            "id": str(uuid.uuid4()),
            "status": medication.status.value,
            "medicationCodeableConcept": {
                "text": medication.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["context"] = {"reference": encounter_ref}

        # Add RxNorm code if available
        if medication.code and medication.code.system == "RxNorm":
            resource["medicationCodeableConcept"]["coding"] = [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": medication.code.code,
                    "display": medication.code.display,
                }
            ]

        # Add dosage
        if medication.dose or medication.frequency:
            dosage = {}
            if medication.dose:
                dosage["text"] = medication.dose
            if medication.frequency:
                dosage["timing"] = {"code": {"text": medication.frequency}}
            if medication.route:
                dosage["route"] = {"text": medication.route}
            resource["dosage"] = [dosage]

        return resource

    def _create_medication_request(
        self,
        medication: Medication,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create MedicationRequest resource."""
        resource = {
            "resourceType": "MedicationRequest",
            "id": str(uuid.uuid4()),
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "text": medication.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add RxNorm code if available
        if medication.code and medication.code.system == "RxNorm":
            resource["medicationCodeableConcept"]["coding"] = [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": medication.code.code,
                    "display": medication.code.display,
                }
            ]

        # Add dosage instruction
        if medication.dose or medication.frequency:
            dosage = {}
            if medication.dose:
                dosage["text"] = medication.dose
            if medication.frequency:
                dosage["timing"] = {"code": {"text": medication.frequency}}
            if medication.route:
                dosage["route"] = {"text": medication.route}
            resource["dosageInstruction"] = [dosage]

        return resource

    def to_json(self, bundle: dict[str, Any], indent: int = 2) -> str:
        """Serialize bundle to JSON string."""
        return json.dumps(bundle, indent=indent)

    def to_ndjson(self, bundle: dict[str, Any]) -> str:
        """Serialize bundle entries as NDJSON (one resource per line)."""
        lines = []
        for entry in bundle.get("entry", []):
            resource = entry.get("resource", {})
            lines.append(json.dumps(resource))
        return "\n".join(lines)
