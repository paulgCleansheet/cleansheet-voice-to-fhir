"""
Extraction Data Types

Data models for extracted clinical entities.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum


class ConditionSeverity(str, Enum):
    """Condition severity levels."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class MedicationStatus(str, Enum):
    """Medication status."""

    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    ON_HOLD = "on-hold"
    UNKNOWN = "unknown"


@dataclass
class CodedConcept:
    """A coded clinical concept."""

    display: str
    code: str | None = None
    system: str | None = None  # SNOMED, ICD-10, RxNorm, LOINC, etc.
    confidence: float = 1.0


@dataclass
class Condition:
    """An extracted condition/diagnosis."""

    description: str
    code: CodedConcept | None = None
    severity: ConditionSeverity = ConditionSeverity.UNKNOWN
    onset: str | None = None  # "2 days ago", "chronic", etc.
    is_chief_complaint: bool = False
    confidence: float = 1.0


@dataclass
class Medication:
    """An extracted medication."""

    name: str
    code: CodedConcept | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    status: MedicationStatus = MedicationStatus.UNKNOWN
    is_new_order: bool = False
    confidence: float = 1.0


@dataclass
class Observation:
    """An extracted observation (vital sign, lab result, etc.)."""

    name: str
    value: str
    unit: str | None = None
    code: CodedConcept | None = None
    interpretation: str | None = None  # "normal", "high", "low", etc.
    confidence: float = 1.0


@dataclass
class Procedure:
    """An extracted procedure."""

    description: str
    code: CodedConcept | None = None
    status: str = "completed"  # planned, in-progress, completed
    performed_date: str | None = None
    confidence: float = 1.0


@dataclass
class Allergy:
    """An extracted allergy."""

    substance: str
    code: CodedConcept | None = None
    reaction: str | None = None
    severity: str | None = None  # mild, moderate, severe
    confidence: float = 1.0


@dataclass
class PatientDemographics:
    """Extracted patient demographics."""

    name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None
    mrn: str | None = None


@dataclass
class Assessment:
    """Clinical assessment/impression."""

    summary: str
    diagnoses: list[Condition] = field(default_factory=list)
    differential: list[str] = field(default_factory=list)


@dataclass
class Plan:
    """Treatment plan."""

    summary: str
    medications: list[Medication] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    follow_up: str | None = None
    instructions: list[str] = field(default_factory=list)


@dataclass
class ClinicalEntities:
    """Container for all extracted clinical entities."""

    # Patient info
    patient: PatientDemographics | None = None

    # Clinical findings
    conditions: list[Condition] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    allergies: list[Allergy] = field(default_factory=list)

    # Medications
    current_medications: list[Medication] = field(default_factory=list)
    new_medications: list[Medication] = field(default_factory=list)

    # Procedures
    procedures: list[Procedure] = field(default_factory=list)

    # Assessment & Plan
    assessment: Assessment | None = None
    plan: Plan | None = None

    # Metadata
    workflow: str = "general"
    raw_transcript: str = ""
    extraction_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chief_complaint(self) -> Condition | None:
        """Get the chief complaint if identified."""
        for condition in self.conditions:
            if condition.is_chief_complaint:
                return condition
        return self.conditions[0] if self.conditions else None

    @property
    def all_medications(self) -> list[Medication]:
        """Get all medications (current + new)."""
        return self.current_medications + self.new_medications

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patient": self.patient.__dict__ if self.patient else None,
            "conditions": [
                {
                    "description": c.description,
                    "severity": c.severity.value,
                    "onset": c.onset,
                    "is_chief_complaint": c.is_chief_complaint,
                }
                for c in self.conditions
            ],
            "observations": [
                {
                    "name": o.name,
                    "value": o.value,
                    "unit": o.unit,
                }
                for o in self.observations
            ],
            "allergies": [
                {
                    "substance": a.substance,
                    "reaction": a.reaction,
                }
                for a in self.allergies
            ],
            "current_medications": [
                {
                    "name": m.name,
                    "dose": m.dose,
                    "frequency": m.frequency,
                }
                for m in self.current_medications
            ],
            "new_medications": [
                {
                    "name": m.name,
                    "dose": m.dose,
                    "frequency": m.frequency,
                }
                for m in self.new_medications
            ],
            "workflow": self.workflow,
        }
