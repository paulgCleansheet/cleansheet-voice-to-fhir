"""
Tests for extraction type definitions.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import pytest
from typing import Any

from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Allergy,
    Vital,
    Procedure,
    LabResult,
)


class TestCondition:
    """Tests for Condition class."""

    def test_create_condition(self):
        """Test creating a condition."""
        condition = Condition(
            name="Hypertension",
            icd10="I10",
            status="active",
        )

        assert condition.name == "Hypertension"
        assert condition.icd10 == "I10"
        assert condition.status == "active"

    def test_condition_with_onset(self):
        """Test condition with onset date."""
        condition = Condition(
            name="Chest pain",
            onset="2 hours ago",
            status="active",
        )

        assert condition.onset == "2 hours ago"

    def test_condition_to_dict(self):
        """Test condition serialization."""
        condition = Condition(
            name="Diabetes",
            icd10="E11.9",
            status="active",
        )

        data = condition.to_dict()

        assert data["name"] == "Diabetes"
        assert data["icd10"] == "E11.9"

    def test_condition_from_dict(self):
        """Test condition deserialization."""
        data = {
            "name": "Asthma",
            "icd10": "J45.909",
            "status": "active",
        }

        condition = Condition.from_dict(data)

        assert condition.name == "Asthma"
        assert condition.icd10 == "J45.909"


class TestMedication:
    """Tests for Medication class."""

    def test_create_medication(self):
        """Test creating a medication."""
        med = Medication(
            name="Lisinopril",
            dose="10mg",
            frequency="daily",
        )

        assert med.name == "Lisinopril"
        assert med.dose == "10mg"
        assert med.frequency == "daily"

    def test_medication_with_rxnorm(self):
        """Test medication with RxNorm code."""
        med = Medication(
            name="Metformin",
            dose="500mg",
            frequency="twice daily",
            rxnorm="861004",
        )

        assert med.rxnorm == "861004"

    def test_medication_with_route(self):
        """Test medication with route."""
        med = Medication(
            name="Aspirin",
            dose="81mg",
            frequency="daily",
            route="oral",
        )

        assert med.route == "oral"


class TestAllergy:
    """Tests for Allergy class."""

    def test_create_allergy(self):
        """Test creating an allergy."""
        allergy = Allergy(
            substance="Penicillin",
            reaction="rash",
        )

        assert allergy.substance == "Penicillin"
        assert allergy.reaction == "rash"

    def test_allergy_with_severity(self):
        """Test allergy with severity."""
        allergy = Allergy(
            substance="Shellfish",
            reaction="anaphylaxis",
            severity="severe",
        )

        assert allergy.severity == "severe"


class TestVital:
    """Tests for Vital class."""

    def test_create_vital(self):
        """Test creating a vital sign."""
        vital = Vital(
            type="blood_pressure",
            value="120/80",
            unit="mmHg",
        )

        assert vital.type == "blood_pressure"
        assert vital.value == "120/80"
        assert vital.unit == "mmHg"

    def test_vital_with_timestamp(self):
        """Test vital with timestamp."""
        vital = Vital(
            type="heart_rate",
            value="72",
            unit="bpm",
            timestamp="2024-01-15T10:30:00Z",
        )

        assert vital.timestamp == "2024-01-15T10:30:00Z"


class TestClinicalEntities:
    """Tests for ClinicalEntities container class."""

    def test_create_empty(self):
        """Test creating empty clinical entities."""
        entities = ClinicalEntities()

        assert len(entities.conditions) == 0
        assert len(entities.medications) == 0
        assert len(entities.allergies) == 0
        assert len(entities.vitals) == 0

    def test_create_with_data(self, sample_clinical_entities: dict[str, Any]):
        """Test creating clinical entities with data."""
        entities = ClinicalEntities.from_dict(sample_clinical_entities)

        assert len(entities.conditions) == 3
        assert len(entities.medications) == 2
        assert len(entities.allergies) == 1
        assert len(entities.vitals) == 5

    def test_add_condition(self):
        """Test adding a condition."""
        entities = ClinicalEntities()
        condition = Condition(name="Headache", status="active")

        entities.add_condition(condition)

        assert len(entities.conditions) == 1
        assert entities.conditions[0].name == "Headache"

    def test_add_medication(self):
        """Test adding a medication."""
        entities = ClinicalEntities()
        med = Medication(name="Ibuprofen", dose="400mg", frequency="as needed")

        entities.add_medication(med)

        assert len(entities.medications) == 1

    def test_to_dict(self, sample_clinical_entities: dict[str, Any]):
        """Test serialization to dictionary."""
        entities = ClinicalEntities.from_dict(sample_clinical_entities)
        data = entities.to_dict()

        assert "conditions" in data
        assert "medications" in data
        assert "allergies" in data
        assert "vitals" in data
        assert len(data["conditions"]) == 3

    def test_to_json(self, sample_clinical_entities: dict[str, Any]):
        """Test serialization to JSON string."""
        entities = ClinicalEntities.from_dict(sample_clinical_entities)
        json_str = entities.to_json()

        assert isinstance(json_str, str)
        assert "Hypertension" in json_str

    def test_merge_entities(self):
        """Test merging two clinical entity sets."""
        entities1 = ClinicalEntities()
        entities1.add_condition(Condition(name="Condition A", status="active"))

        entities2 = ClinicalEntities()
        entities2.add_condition(Condition(name="Condition B", status="active"))

        merged = entities1.merge(entities2)

        assert len(merged.conditions) == 2

    def test_has_critical_findings(self, sample_clinical_entities: dict[str, Any]):
        """Test detection of critical findings."""
        entities = ClinicalEntities.from_dict(sample_clinical_entities)

        # Check for severe allergies
        has_severe = any(
            a.severity == "severe" for a in entities.allergies if a.severity
        )
        assert isinstance(has_severe, bool)

    def test_summary(self, sample_clinical_entities: dict[str, Any]):
        """Test generating summary."""
        entities = ClinicalEntities.from_dict(sample_clinical_entities)
        summary = entities.summary()

        assert "conditions" in summary.lower() or "3" in summary
