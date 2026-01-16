"""
Tests for FHIR validators.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

from typing import Any

import pytest

from voice_to_fhir.fhir.validators import (
    validate_bundle,
    ValidationResult,
    ValidationError,
)


class TestValidationResult:
    """Tests for ValidationResult class."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])

        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0

    def test_invalid_result(self):
        """Test creating an invalid result."""
        errors = [
            ValidationError(path="Bundle.type", message="Missing required field"),
        ]
        result = ValidationResult(is_valid=False, errors=errors, warnings=[])

        assert result.is_valid is False
        assert result.error_count == 1

    def test_result_with_warnings(self):
        """Test result with warnings."""
        warnings = [
            ValidationError(
                path="Condition.subject",
                message="Subject reference recommended",
                severity="warning",
            ),
        ]
        result = ValidationResult(is_valid=True, errors=[], warnings=warnings)

        assert result.is_valid is True
        assert result.warning_count == 1


class TestValidationError:
    """Tests for ValidationError class."""

    def test_create_error(self):
        """Test creating a validation error."""
        error = ValidationError(
            path="Bundle.entry[0].resource.resourceType",
            message="resourceType is required",
        )

        assert error.path == "Bundle.entry[0].resource.resourceType"
        assert error.message == "resourceType is required"
        assert error.severity == "error"

    def test_create_warning(self):
        """Test creating a validation warning."""
        warning = ValidationError(
            path="Patient.name",
            message="Patient name recommended",
            severity="warning",
        )

        assert warning.severity == "warning"


class TestValidateBundle:
    """Tests for bundle validation."""

    def test_validate_valid_bundle(self, sample_fhir_bundle: dict[str, Any]):
        """Test validating a valid bundle."""
        result = validate_bundle(sample_fhir_bundle)

        assert result.is_valid is True
        assert result.error_count == 0

    def test_validate_missing_type(self):
        """Test validation catches missing type."""
        bundle = {
            "resourceType": "Bundle",
            # Missing "type" field
            "entry": [],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False
        assert any("type" in e.path for e in result.errors)

    def test_validate_wrong_resource_type(self):
        """Test validation catches wrong resourceType."""
        bundle = {
            "resourceType": "Patient",  # Wrong - should be Bundle
            "type": "collection",
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False
        assert any("resourceType" in e.path for e in result.errors)

    def test_validate_entry_missing_resource(self):
        """Test validation catches entry without resource."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {}  # Missing "resource" field
            ],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False
        assert any("resource" in e.message.lower() for e in result.errors)

    def test_validate_resource_missing_type(self):
        """Test validation catches resource without resourceType."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        # Missing resourceType
                        "id": "test-1",
                    }
                }
            ],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False

    def test_validate_encounter_missing_status(self):
        """Test validation catches Encounter without status."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Encounter",
                        "id": "enc-1",
                        # Missing "status" field
                        "class": {
                            "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                            "code": "AMB",
                        },
                    }
                }
            ],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False
        assert any("status" in e.path for e in result.errors)

    def test_validate_observation_missing_code(self):
        """Test validation catches Observation without code."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Observation",
                        "id": "obs-1",
                        "status": "final",
                        # Missing "code" field
                    }
                }
            ],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False
        assert any("code" in e.path for e in result.errors)

    def test_validate_medication_request_missing_intent(self):
        """Test validation catches MedicationRequest without intent."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "MedicationRequest",
                        "id": "med-1",
                        "status": "active",
                        # Missing "intent" field
                    }
                }
            ],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is False
        assert any("intent" in e.path for e in result.errors)

    def test_validate_empty_bundle(self):
        """Test validating empty bundle is valid."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [],
        }

        result = validate_bundle(bundle)

        assert result.is_valid is True

    def test_validate_patient_no_required_fields(self):
        """Test that Patient has no strictly required fields."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "patient-1",
                        # No other fields - should still be valid
                    }
                }
            ],
        }

        result = validate_bundle(bundle)

        # Patient has no strictly required fields in R4
        assert result.is_valid is True

    def test_validate_multiple_resources(self, sample_fhir_bundle: dict[str, Any]):
        """Test validating bundle with multiple resources."""
        result = validate_bundle(sample_fhir_bundle)

        assert result.is_valid is True
        # Should have validated all entries
        assert len(sample_fhir_bundle["entry"]) == 2

    def test_validate_condition_subject_warning(self):
        """Test that Condition without subject produces warning."""
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Condition",
                        "id": "cond-1",
                        # No subject - should generate warning
                        "code": {
                            "coding": [{"code": "I10", "display": "Hypertension"}]
                        },
                    }
                }
            ],
        }

        result = validate_bundle(bundle)

        # Subject is required but severity is warning
        # Check if it's in errors or warnings
        has_subject_issue = any(
            "subject" in (e.path + e.message).lower()
            for e in result.errors + result.warnings
        )
        assert has_subject_issue
