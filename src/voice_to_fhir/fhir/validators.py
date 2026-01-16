"""
FHIR Validators

Validate FHIR resources against R4 specification.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationError:
    """A single validation error."""

    path: str
    message: str
    severity: str = "error"  # error, warning, information


@dataclass
class ValidationResult:
    """Result of FHIR validation."""

    is_valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)


def validate_bundle(bundle: dict[str, Any]) -> ValidationResult:
    """Validate a FHIR Bundle."""
    errors: list[ValidationError] = []
    warnings: list[ValidationError] = []

    # Basic structure validation
    if bundle.get("resourceType") != "Bundle":
        errors.append(
            ValidationError(
                path="resourceType",
                message="resourceType must be 'Bundle'",
            )
        )

    if "type" not in bundle:
        errors.append(
            ValidationError(
                path="type",
                message="Bundle.type is required",
            )
        )

    # Validate entries
    entries = bundle.get("entry", [])
    for i, entry in enumerate(entries):
        entry_errors = _validate_entry(entry, i)
        errors.extend(entry_errors)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _validate_entry(entry: dict[str, Any], index: int) -> list[ValidationError]:
    """Validate a bundle entry."""
    errors: list[ValidationError] = []
    path_prefix = f"entry[{index}]"

    resource = entry.get("resource")
    if not resource:
        errors.append(
            ValidationError(
                path=f"{path_prefix}.resource",
                message="Entry must contain a resource",
            )
        )
        return errors

    # Validate resource has required fields
    resource_type = resource.get("resourceType")
    if not resource_type:
        errors.append(
            ValidationError(
                path=f"{path_prefix}.resource.resourceType",
                message="Resource must have resourceType",
            )
        )

    # Resource-specific validation
    validators = {
        "Patient": _validate_patient,
        "Encounter": _validate_encounter,
        "Condition": _validate_condition,
        "Observation": _validate_observation,
        "AllergyIntolerance": _validate_allergy_intolerance,
        "MedicationStatement": _validate_medication_statement,
        "MedicationRequest": _validate_medication_request,
    }

    if resource_type in validators:
        resource_errors = validators[resource_type](
            resource, f"{path_prefix}.resource"
        )
        errors.extend(resource_errors)

    return errors


def _validate_patient(resource: dict[str, Any], path: str) -> list[ValidationError]:
    """Validate Patient resource."""
    # Patient has no strictly required fields in R4
    return []


def _validate_encounter(resource: dict[str, Any], path: str) -> list[ValidationError]:
    """Validate Encounter resource."""
    errors: list[ValidationError] = []

    if "status" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.status",
                message="Encounter.status is required",
            )
        )

    if "class" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.class",
                message="Encounter.class is required",
            )
        )

    return errors


def _validate_condition(resource: dict[str, Any], path: str) -> list[ValidationError]:
    """Validate Condition resource."""
    errors: list[ValidationError] = []

    if "subject" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.subject",
                message="Condition.subject is required",
                severity="warning",
            )
        )

    return errors


def _validate_observation(
    resource: dict[str, Any], path: str
) -> list[ValidationError]:
    """Validate Observation resource."""
    errors: list[ValidationError] = []

    if "status" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.status",
                message="Observation.status is required",
            )
        )

    if "code" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.code",
                message="Observation.code is required",
            )
        )

    return errors


def _validate_allergy_intolerance(
    resource: dict[str, Any], path: str
) -> list[ValidationError]:
    """Validate AllergyIntolerance resource."""
    errors: list[ValidationError] = []

    if "patient" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.patient",
                message="AllergyIntolerance.patient is required",
                severity="warning",
            )
        )

    return errors


def _validate_medication_statement(
    resource: dict[str, Any], path: str
) -> list[ValidationError]:
    """Validate MedicationStatement resource."""
    errors: list[ValidationError] = []

    if "status" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.status",
                message="MedicationStatement.status is required",
            )
        )

    if "subject" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.subject",
                message="MedicationStatement.subject is required",
                severity="warning",
            )
        )

    return errors


def _validate_medication_request(
    resource: dict[str, Any], path: str
) -> list[ValidationError]:
    """Validate MedicationRequest resource."""
    errors: list[ValidationError] = []

    if "status" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.status",
                message="MedicationRequest.status is required",
            )
        )

    if "intent" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.intent",
                message="MedicationRequest.intent is required",
            )
        )

    if "subject" not in resource:
        errors.append(
            ValidationError(
                path=f"{path}.subject",
                message="MedicationRequest.subject is required",
                severity="warning",
            )
        )

    return errors
