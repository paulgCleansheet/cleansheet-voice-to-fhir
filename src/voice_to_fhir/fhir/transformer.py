"""
FHIR Transformer

Transform extracted clinical entities to FHIR R4 resources.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import json
import uuid

from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Vital,
    LabResult,
    LabOrder,
    Allergy,
    FamilyHistory,
    SocialHistory,
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
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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

        # Create Observations for vitals
        for vital in entities.vitals:
            resource = self._create_observation_from_vital(
                vital, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create Observations for lab results
        for lab in entities.lab_results:
            resource = self._create_observation_from_lab(
                lab, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create Observations for lab orders (pending)
        for lab_order in entities.lab_orders:
            resource = self._create_observation_from_lab_order(
                lab_order, patient_ref, encounter_ref
            )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create AllergyIntolerances
        for allergy in entities.allergies:
            resource = self._create_allergy_intolerance(allergy, patient_ref)
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create MedicationStatements/MedicationRequests for medications
        for med in entities.medications:
            if med.is_new_order:
                resource = self._create_medication_request(
                    med, patient_ref, encounter_ref
                )
            else:
                resource = self._create_medication_statement(
                    med, patient_ref, encounter_ref
                )
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create FamilyMemberHistory resources
        for fh in entities.family_history:
            resource = self._create_family_member_history(fh, patient_ref)
            bundle["entry"].append(self._wrap_entry(resource, "POST"))

        # Create social history Observations
        if entities.social_history:
            for resource in self._create_social_history_observations(
                entities.social_history, patient_ref, encounter_ref
            ):
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
            "emergency": ("emergency", "Emergency encounter"),
            "followup": ("followup", "Follow-up visit"),
            "procedure": ("procedure", "Procedure"),
            "discharge": ("discharge", "Discharge"),
            "radiology": ("radiology", "Radiology"),
            "lab_review": ("lab_review", "Lab review"),
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
                        "code": condition.status or "active",
                    }
                ]
            },
            "code": {
                "text": condition.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add ICD-10 code if available
        if condition.icd10:
            resource["code"]["coding"] = [
                {
                    "system": "http://hl7.org/fhir/sid/icd-10",
                    "code": condition.icd10,
                    "display": condition.name,
                }
            ]

        # Add SNOMED code if available
        if condition.snomed:
            codings = resource["code"].get("coding", [])
            codings.append({
                "system": "http://snomed.info/sct",
                "code": condition.snomed,
                "display": condition.name,
            })
            resource["code"]["coding"] = codings

        # Add severity
        if condition.severity and condition.severity != "unknown":
            severity_codes = {
                "mild": ("255604002", "Mild"),
                "moderate": ("6736007", "Moderate"),
                "severe": ("24484000", "Severe"),
            }
            if condition.severity in severity_codes:
                code, display = severity_codes[condition.severity]
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

    def _create_observation_from_vital(
        self,
        vital: Vital,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create Observation resource from vital sign."""
        resource = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "vital-signs",
                            "display": "Vital Signs",
                        }
                    ]
                }
            ],
            "code": {
                "text": vital.type,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add LOINC code if available
        if vital.loinc:
            resource["code"]["coding"] = [
                {
                    "system": "http://loinc.org",
                    "code": vital.loinc,
                    "display": vital.type,
                }
            ]

        # Try to parse numeric value
        try:
            numeric_value = float(vital.value.replace(",", "").split("/")[0])
            resource["valueQuantity"] = {
                "value": numeric_value,
                "unit": vital.unit or "",
            }
        except ValueError:
            resource["valueString"] = vital.value

        return resource

    def _create_observation_from_lab(
        self,
        lab: LabResult,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create Observation resource from lab result."""
        # Map lab status to FHIR Observation status
        # pending → registered (ordered but no result yet)
        # completed → final (result available)
        fhir_status = "registered" if lab.status == "pending" else "final"

        resource = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": fhir_status,
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "text": lab.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add LOINC code if available
        if lab.loinc:
            resource["code"]["coding"] = [
                {
                    "system": "http://loinc.org",
                    "code": lab.loinc,
                    "display": lab.name,
                }
            ]

        # Add value if available (not for pending labs)
        # Also treat string "null" as no value (LLM sometimes returns this)
        if lab.value and lab.value.lower() != "null":
            # Try to parse numeric value
            try:
                numeric_value = float(lab.value.replace(",", ""))
                resource["valueQuantity"] = {
                    "value": numeric_value,
                    "unit": lab.unit or "",
                }
            except ValueError:
                resource["valueString"] = lab.value

        # Add reference range if available
        if lab.reference_range:
            resource["referenceRange"] = [{"text": lab.reference_range}]

        # Add interpretation if available
        if lab.interpretation:
            interpretation_map = {
                "normal": ("N", "Normal"),
                "high": ("H", "High"),
                "low": ("L", "Low"),
                "critical": ("AA", "Critical abnormal"),
            }
            if lab.interpretation.lower() in interpretation_map:
                code, display = interpretation_map[lab.interpretation.lower()]
                resource["interpretation"] = [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation",
                                "code": code,
                                "display": display,
                            }
                        ]
                    }
                ]

        return resource

    def _create_observation_from_lab_order(
        self,
        lab_order: LabOrder,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> dict[str, Any]:
        """Create Observation resource from lab order (pending, no value)."""
        resource = {
            "resourceType": "Observation",
            "id": str(uuid.uuid4()),
            "status": "registered",  # Ordered but no result yet
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "laboratory",
                            "display": "Laboratory",
                        }
                    ]
                }
            ],
            "code": {
                "text": lab_order.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["encounter"] = {"reference": encounter_ref}

        # Add LOINC code if available
        if lab_order.loinc:
            resource["code"]["coding"] = [
                {
                    "system": "http://loinc.org",
                    "code": lab_order.loinc,
                    "display": lab_order.name,
                }
            ]

        # No value for pending orders
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
            if not resource["reaction"]:
                resource["reaction"] = [{}]
            resource["reaction"][0]["severity"] = allergy.severity.lower()

        return resource

    def _create_family_member_history(
        self, fh: FamilyHistory, patient_ref: str | None
    ) -> dict[str, Any]:
        """Create FamilyMemberHistory resource."""
        # Map relationship to HL7 v3-RoleCode
        relationship_codes = {
            "mother": ("MTH", "Mother"),
            "father": ("FTH", "Father"),
            "brother": ("BRO", "Brother"),
            "sister": ("SIS", "Sister"),
            "son": ("SON", "Son"),
            "daughter": ("DAU", "Daughter"),
            "maternal grandmother": ("MGRMTH", "Maternal Grandmother"),
            "maternal grandfather": ("MGRFTH", "Maternal Grandfather"),
            "paternal grandmother": ("PGRMTH", "Paternal Grandmother"),
            "paternal grandfather": ("PGRFTH", "Paternal Grandfather"),
            "aunt": ("AUNT", "Aunt"),
            "uncle": ("UNCLE", "Uncle"),
            "cousin": ("COUSN", "Cousin"),
            "spouse": ("SPS", "Spouse"),
        }

        rel_lower = fh.relationship.lower()
        code, display = relationship_codes.get(rel_lower, ("FAMMEMB", fh.relationship))

        resource = {
            "resourceType": "FamilyMemberHistory",
            "id": str(uuid.uuid4()),
            "status": "completed",
            "relationship": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode",
                        "code": code,
                        "display": display,
                    }
                ]
            },
            "condition": [
                {
                    "code": {
                        "text": fh.condition,
                    }
                }
            ],
        }

        if patient_ref:
            resource["patient"] = {"reference": patient_ref}

        # Add age of onset if available
        if fh.age_of_onset:
            try:
                age_value = int(fh.age_of_onset.split()[0])
                resource["condition"][0]["onsetAge"] = {
                    "value": age_value,
                    "unit": "years",
                    "system": "http://unitsofmeasure.org",
                    "code": "a",
                }
            except (ValueError, IndexError):
                resource["condition"][0]["onsetString"] = fh.age_of_onset

        # Add deceased status if known
        if fh.deceased is not None:
            resource["deceasedBoolean"] = fh.deceased

        return resource

    def _create_social_history_observations(
        self,
        social: SocialHistory,
        patient_ref: str | None,
        encounter_ref: str | None,
    ) -> list[dict[str, Any]]:
        """Create Observation resources for social history."""
        observations = []

        # LOINC codes for social history components
        social_history_codes = {
            "tobacco": ("72166-2", "Tobacco smoking status"),
            "alcohol": ("74013-4", "Alcohol use status"),
            "drugs": ("74204-9", "Drug use"),
            "occupation": ("11341-5", "History of occupation"),
            "living_situation": ("63512-8", "Living situation"),
        }

        def create_observation(code_key: str, value: str) -> dict[str, Any]:
            loinc_code, display = social_history_codes[code_key]
            resource = {
                "resourceType": "Observation",
                "id": str(uuid.uuid4()),
                "status": "final",
                "category": [
                    {
                        "coding": [
                            {
                                "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                "code": "social-history",
                                "display": "Social History",
                            }
                        ]
                    }
                ],
                "code": {
                    "coding": [
                        {
                            "system": "http://loinc.org",
                            "code": loinc_code,
                            "display": display,
                        }
                    ],
                    "text": display,
                },
                "valueString": value,
            }

            if patient_ref:
                resource["subject"] = {"reference": patient_ref}

            if encounter_ref:
                resource["encounter"] = {"reference": encounter_ref}

            return resource

        # Create observation for each non-null social history field
        if social.tobacco:
            observations.append(create_observation("tobacco", social.tobacco))

        if social.alcohol:
            observations.append(create_observation("alcohol", social.alcohol))

        if social.drugs:
            observations.append(create_observation("drugs", social.drugs))

        if social.occupation:
            observations.append(create_observation("occupation", social.occupation))

        if social.living_situation:
            observations.append(create_observation("living_situation", social.living_situation))

        return observations

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
            "status": medication.status or "active",
            "medicationCodeableConcept": {
                "text": medication.name,
            },
        }

        if patient_ref:
            resource["subject"] = {"reference": patient_ref}

        if encounter_ref:
            resource["context"] = {"reference": encounter_ref}

        # Add RxNorm code if available
        if medication.rxnorm:
            resource["medicationCodeableConcept"]["coding"] = [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": medication.rxnorm,
                    "display": medication.name,
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
        if medication.rxnorm:
            resource["medicationCodeableConcept"]["coding"] = [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": medication.rxnorm,
                    "display": medication.name,
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
