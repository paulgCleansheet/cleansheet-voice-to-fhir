"""
Extraction Data Types

Data models for extracted clinical entities.
"""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import json


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

    name: str
    status: str = "active"
    icd10: str | None = None
    snomed: str | None = None
    onset: str | None = None  # "2 days ago", "chronic", etc.
    severity: str | None = None
    is_chief_complaint: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "icd10": self.icd10,
            "snomed": self.snomed,
            "onset": self.onset,
            "severity": self.severity,
            "is_chief_complaint": self.is_chief_complaint,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Condition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            status=data.get("status", "active"),
            icd10=data.get("icd10"),
            snomed=data.get("snomed"),
            onset=data.get("onset"),
            severity=data.get("severity"),
            is_chief_complaint=data.get("is_chief_complaint", False),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Medication:
    """An extracted medication."""

    name: str
    dose: str | None = None
    frequency: str | None = None
    route: str | None = None
    rxnorm: str | None = None
    status: str = "active"
    is_new_order: bool = False
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dose": self.dose,
            "frequency": self.frequency,
            "route": self.route,
            "rxnorm": self.rxnorm,
            "status": self.status,
            "is_new_order": self.is_new_order,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Medication":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            dose=data.get("dose"),
            frequency=data.get("frequency"),
            route=data.get("route"),
            rxnorm=data.get("rxnorm"),
            status=data.get("status", "active"),
            is_new_order=data.get("is_new_order", False),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Vital:
    """An extracted vital sign."""

    type: str
    value: str
    unit: str | None = None
    timestamp: str | None = None
    interpretation: str | None = None
    loinc: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "interpretation": self.interpretation,
            "loinc": self.loinc,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Vital":
        """Create from dictionary."""
        return cls(
            type=data["type"],
            value=data["value"],
            unit=data.get("unit"),
            timestamp=data.get("timestamp"),
            interpretation=data.get("interpretation"),
            loinc=data.get("loinc"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class LabResult:
    """An extracted lab result."""

    name: str
    value: str | None = None  # None for pending/ordered labs
    unit: str | None = None
    reference_range: str | None = None
    interpretation: str | None = None  # "normal", "high", "low", "critical"
    loinc: str | None = None
    timestamp: str | None = None
    status: str = "completed"  # "pending", "completed", "cancelled"
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "interpretation": self.interpretation,
            "loinc": self.loinc,
            "timestamp": self.timestamp,
            "status": self.status,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabResult":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            value=data.get("value"),
            unit=data.get("unit"),
            reference_range=data.get("reference_range"),
            interpretation=data.get("interpretation"),
            loinc=data.get("loinc"),
            timestamp=data.get("timestamp"),
            status=data.get("status", "completed"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class LabOrder:
    """An ordered lab test (no results yet)."""

    name: str
    loinc: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "loinc": self.loinc,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabOrder":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            loinc=data.get("loinc"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class MedicationOrder:
    """A new medication being prescribed."""

    name: str
    dose: str | None = None
    frequency: str | None = None
    instructions: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dose": self.dose,
            "frequency": self.frequency,
            "instructions": self.instructions,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MedicationOrder":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            dose=data.get("dose"),
            frequency=data.get("frequency"),
            instructions=data.get("instructions"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ReferralOrder:
    """A referral or consult order."""

    specialty: str
    reason: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "specialty": self.specialty,
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferralOrder":
        """Create from dictionary."""
        return cls(
            specialty=data.get("specialty", data.get("name", "")),
            reason=data.get("reason"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ProcedureOrder:
    """A procedure or study being ordered."""

    name: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcedureOrder":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ImagingOrder:
    """An imaging study being ordered."""

    name: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImagingOrder":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Observation:
    """An extracted observation (generic clinical finding)."""

    name: str
    value: str
    unit: str | None = None
    code: CodedConcept | None = None
    interpretation: str | None = None
    confidence: float = 1.0


@dataclass
class Procedure:
    """An extracted procedure."""

    name: str
    status: str = "completed"  # planned, in-progress, completed
    cpt: str | None = None
    snomed: str | None = None
    performed_date: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "cpt": self.cpt,
            "snomed": self.snomed,
            "performed_date": self.performed_date,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Procedure":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            status=data.get("status", "completed"),
            cpt=data.get("cpt"),
            snomed=data.get("snomed"),
            performed_date=data.get("performed_date"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Allergy:
    """An extracted allergy."""

    substance: str
    reaction: str | None = None
    severity: str | None = None  # mild, moderate, severe
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "substance": self.substance,
            "reaction": self.reaction,
            "severity": self.severity,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Allergy":
        """Create from dictionary."""
        return cls(
            substance=data["substance"],
            reaction=data.get("reaction"),
            severity=data.get("severity"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class FamilyHistory:
    """An extracted family history item."""

    relationship: str  # mother, father, sibling, etc.
    condition: str
    age_of_onset: str | None = None
    deceased: bool | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "relationship": self.relationship,
            "condition": self.condition,
            "age_of_onset": self.age_of_onset,
            "deceased": self.deceased,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FamilyHistory":
        """Create from dictionary."""
        return cls(
            relationship=data["relationship"],
            condition=data["condition"],
            age_of_onset=data.get("age_of_onset"),
            deceased=data.get("deceased"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class SocialHistory:
    """Extracted social history."""

    tobacco: str | None = None  # current|former|never + details
    alcohol: str | None = None  # usage description
    drugs: str | None = None  # usage description or "none"
    occupation: str | None = None
    living_situation: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tobacco": self.tobacco,
            "alcohol": self.alcohol,
            "drugs": self.drugs,
            "occupation": self.occupation,
            "living_situation": self.living_situation,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SocialHistory":
        """Create from dictionary."""
        return cls(
            tobacco=data.get("tobacco"),
            alcohol=data.get("alcohol"),
            drugs=data.get("drugs"),
            occupation=data.get("occupation"),
            living_situation=data.get("living_situation"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class PatientDemographics:
    """Extracted patient demographics."""

    name: str | None = None
    date_of_birth: str | None = None
    gender: str | None = None
    mrn: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "date_of_birth": self.date_of_birth,
            "gender": self.gender,
            "mrn": self.mrn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PatientDemographics":
        """Create from dictionary."""
        return cls(
            name=data.get("name"),
            date_of_birth=data.get("date_of_birth"),
            gender=data.get("gender"),
            mrn=data.get("mrn"),
        )


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
    vitals: list[Vital] = field(default_factory=list)
    lab_results: list[LabResult] = field(default_factory=list)
    lab_orders: list["LabOrder"] = field(default_factory=list)
    allergies: list[Allergy] = field(default_factory=list)

    # Medications
    medications: list[Medication] = field(default_factory=list)

    # Orders (new prescriptions, consults, procedures, imaging)
    medication_orders: list["MedicationOrder"] = field(default_factory=list)
    referral_orders: list["ReferralOrder"] = field(default_factory=list)
    procedure_orders: list["ProcedureOrder"] = field(default_factory=list)
    imaging_orders: list["ImagingOrder"] = field(default_factory=list)

    # Procedures
    procedures: list[Procedure] = field(default_factory=list)

    # Family history
    family_history: list[FamilyHistory] = field(default_factory=list)

    # Social history
    social_history: SocialHistory | None = None

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

    def add_condition(self, condition: Condition) -> None:
        """Add a condition to the list."""
        self.conditions.append(condition)

    def add_medication(self, medication: Medication) -> None:
        """Add a medication to the list."""
        self.medications.append(medication)

    def add_vital(self, vital: Vital) -> None:
        """Add a vital sign to the list."""
        self.vitals.append(vital)

    def add_allergy(self, allergy: Allergy) -> None:
        """Add an allergy to the list."""
        self.allergies.append(allergy)

    def add_procedure(self, procedure: Procedure) -> None:
        """Add a procedure to the list."""
        self.procedures.append(procedure)

    def add_lab_result(self, lab_result: LabResult) -> None:
        """Add a lab result to the list."""
        self.lab_results.append(lab_result)

    def add_family_history(self, fh: FamilyHistory) -> None:
        """Add a family history item to the list."""
        self.family_history.append(fh)

    def merge(self, other: "ClinicalEntities") -> "ClinicalEntities":
        """Merge another ClinicalEntities into a new combined instance."""
        merged = ClinicalEntities(
            patient=self.patient or other.patient,
            conditions=self.conditions + other.conditions,
            vitals=self.vitals + other.vitals,
            lab_results=self.lab_results + other.lab_results,
            lab_orders=self.lab_orders + other.lab_orders,
            allergies=self.allergies + other.allergies,
            medications=self.medications + other.medications,
            medication_orders=self.medication_orders + other.medication_orders,
            referral_orders=self.referral_orders + other.referral_orders,
            procedure_orders=self.procedure_orders + other.procedure_orders,
            imaging_orders=self.imaging_orders + other.imaging_orders,
            procedures=self.procedures + other.procedures,
            family_history=self.family_history + other.family_history,
            social_history=self.social_history or other.social_history,
            workflow=self.workflow,
            raw_transcript=self.raw_transcript + " " + other.raw_transcript,
        )
        return merged

    def summary(self) -> str:
        """Generate a text summary of the clinical entities."""
        parts = []
        if self.conditions:
            parts.append(f"{len(self.conditions)} conditions")
        if self.medications:
            parts.append(f"{len(self.medications)} medications")
        if self.allergies:
            parts.append(f"{len(self.allergies)} allergies")
        if self.vitals:
            parts.append(f"{len(self.vitals)} vitals")
        if self.lab_results:
            parts.append(f"{len(self.lab_results)} lab results")
        if self.lab_orders:
            parts.append(f"{len(self.lab_orders)} lab orders")
        if self.medication_orders:
            parts.append(f"{len(self.medication_orders)} medication orders")
        if self.referral_orders:
            parts.append(f"{len(self.referral_orders)} referrals")
        if self.procedure_orders:
            parts.append(f"{len(self.procedure_orders)} procedure orders")
        if self.imaging_orders:
            parts.append(f"{len(self.imaging_orders)} imaging orders")
        if self.procedures:
            parts.append(f"{len(self.procedures)} procedures")
        if self.family_history:
            parts.append(f"{len(self.family_history)} family history")
        if self.social_history:
            parts.append("social history")

        if not parts:
            return "No clinical entities extracted"
        return "Extracted: " + ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "patient": self.patient.to_dict() if self.patient else None,
            "conditions": [c.to_dict() for c in self.conditions],
            "vitals": [v.to_dict() for v in self.vitals],
            "lab_results": [lr.to_dict() for lr in self.lab_results],
            "lab_orders": [lo.to_dict() for lo in self.lab_orders],
            "medication_orders": [mo.to_dict() for mo in self.medication_orders],
            "referral_orders": [ro.to_dict() for ro in self.referral_orders],
            "procedure_orders": [po.to_dict() for po in self.procedure_orders],
            "imaging_orders": [io.to_dict() for io in self.imaging_orders],
            "allergies": [a.to_dict() for a in self.allergies],
            "medications": [m.to_dict() for m in self.medications],
            "procedures": [p.to_dict() for p in self.procedures],
            "family_history": [fh.to_dict() for fh in self.family_history],
            "social_history": self.social_history.to_dict() if self.social_history else None,
            "workflow": self.workflow,
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClinicalEntities":
        """Create from dictionary."""
        patient = None
        if data.get("patient"):
            patient = PatientDemographics.from_dict(data["patient"])

        social_history = None
        if data.get("social_history"):
            social_history = SocialHistory.from_dict(data["social_history"])

        return cls(
            patient=patient,
            conditions=[Condition.from_dict(c) for c in data.get("conditions", [])],
            vitals=[Vital.from_dict(v) for v in data.get("vitals", [])],
            lab_results=[LabResult.from_dict(lr) for lr in data.get("lab_results", [])],
            lab_orders=[LabOrder.from_dict(lo) for lo in data.get("lab_orders", [])],
            medication_orders=[MedicationOrder.from_dict(mo) for mo in data.get("medication_orders", [])],
            referral_orders=[ReferralOrder.from_dict(ro) for ro in data.get("referral_orders", [])],
            procedure_orders=[ProcedureOrder.from_dict(po) for po in data.get("procedure_orders", [])],
            imaging_orders=[ImagingOrder.from_dict(io) for io in data.get("imaging_orders", [])],
            allergies=[Allergy.from_dict(a) for a in data.get("allergies", [])],
            medications=[Medication.from_dict(m) for m in data.get("medications", [])],
            procedures=[Procedure.from_dict(p) for p in data.get("procedures", [])],
            family_history=[FamilyHistory.from_dict(fh) for fh in data.get("family_history", [])],
            social_history=social_history,
            workflow=data.get("workflow", "general"),
        )
