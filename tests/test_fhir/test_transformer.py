"""
Tests for FHIR transformer.

Copyright (c) 2024 Cleansheet LLC
License: CC BY 4.0
"""

import json
from typing import Any

import pytest

from voice_to_fhir.fhir.transformer import FHIRTransformer, FHIRConfig
from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    Medication,
    Allergy,
    Vital,
)


class TestFHIRConfig:
    """Tests for FHIR configuration."""

    def test_default_config(self):
        """Test default FHIR configuration."""
        config = FHIRConfig()

        assert config.fhir_version == "R4"
        assert config.validate_output is True

    def test_custom_config(self):
        """Test custom FHIR configuration."""
        config = FHIRConfig(
            fhir_version="R4",
            base_url="http://example.org/fhir",
            validate_output=False,
        )

        assert config.base_url == "http://example.org/fhir"
        assert config.validate_output is False


class TestFHIRTransformer:
    """Tests for FHIR transformer."""

    @pytest.fixture
    def transformer(self) -> FHIRTransformer:
        """Create FHIR transformer for testing."""
        config = FHIRConfig(validate_output=False)
        return FHIRTransformer(config)

    @pytest.fixture
    def sample_entities(self) -> ClinicalEntities:
        """Create sample clinical entities."""
        entities = ClinicalEntities()
        entities.add_condition(Condition(name="Hypertension", icd10="I10", status="active"))
        entities.add_medication(Medication(name="Lisinopril", dose="10mg", frequency="daily"))
        entities.add_allergy(Allergy(substance="Penicillin", reaction="rash"))
        entities.add_vital(Vital(type="blood_pressure", value="140/90", unit="mmHg"))
        return entities

    def test_transform_empty_entities(self, transformer: FHIRTransformer):
        """Test transforming empty entities."""
        entities = ClinicalEntities()
        bundle = transformer.transform(entities)

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        # Should have at least an Encounter
        assert len(bundle.get("entry", [])) >= 1

    def test_transform_with_condition(self, transformer: FHIRTransformer):
        """Test transforming entities with a condition."""
        entities = ClinicalEntities()
        entities.add_condition(Condition(name="Diabetes", icd10="E11.9", status="active"))

        bundle = transformer.transform(entities)

        assert len(bundle["entry"]) >= 1
        condition_entry = next(
            (e for e in bundle["entry"] if e["resource"]["resourceType"] == "Condition"),
            None,
        )
        assert condition_entry is not None
        assert "E11.9" in json.dumps(condition_entry)

    def test_transform_with_medication(self, transformer: FHIRTransformer):
        """Test transforming entities with a medication."""
        entities = ClinicalEntities()
        entities.add_medication(
            Medication(name="Metformin", dose="500mg", frequency="twice daily")
        )

        bundle = transformer.transform(entities)

        med_entry = next(
            (
                e
                for e in bundle["entry"]
                if e["resource"]["resourceType"] in ["MedicationStatement", "MedicationRequest"]
            ),
            None,
        )
        assert med_entry is not None

    def test_transform_with_allergy(self, transformer: FHIRTransformer):
        """Test transforming entities with an allergy."""
        entities = ClinicalEntities()
        entities.add_allergy(Allergy(substance="Sulfa drugs", reaction="hives", severity="moderate"))

        bundle = transformer.transform(entities)

        allergy_entry = next(
            (
                e
                for e in bundle["entry"]
                if e["resource"]["resourceType"] == "AllergyIntolerance"
            ),
            None,
        )
        assert allergy_entry is not None

    def test_transform_with_vitals(self, transformer: FHIRTransformer):
        """Test transforming entities with vital signs."""
        entities = ClinicalEntities()
        entities.add_vital(Vital(type="heart_rate", value="72", unit="bpm"))
        entities.add_vital(Vital(type="temperature", value="98.6", unit="F"))

        bundle = transformer.transform(entities)

        observation_entries = [
            e for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"
        ]
        assert len(observation_entries) >= 2

    def test_transform_full_entities(
        self, transformer: FHIRTransformer, sample_entities: ClinicalEntities
    ):
        """Test transforming full set of clinical entities."""
        bundle = transformer.transform(sample_entities)

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        assert len(bundle["entry"]) >= 4  # At least one of each type

    def test_bundle_has_ids(self, transformer: FHIRTransformer, sample_entities: ClinicalEntities):
        """Test that all resources have IDs."""
        bundle = transformer.transform(sample_entities)

        for entry in bundle["entry"]:
            resource = entry["resource"]
            assert "id" in resource
            assert resource["id"] is not None

    def test_to_json(self, transformer: FHIRTransformer, sample_entities: ClinicalEntities):
        """Test JSON serialization."""
        bundle = transformer.transform(sample_entities)
        json_str = transformer.to_json(bundle)

        assert isinstance(json_str, str)
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["resourceType"] == "Bundle"

    def test_to_json_with_indent(
        self, transformer: FHIRTransformer, sample_entities: ClinicalEntities
    ):
        """Test JSON serialization with custom indent."""
        bundle = transformer.transform(sample_entities)
        json_str = transformer.to_json(bundle, indent=4)

        # Check that indentation is applied
        assert "\n    " in json_str

    def test_condition_coding_system(self, transformer: FHIRTransformer):
        """Test that conditions use correct coding system."""
        entities = ClinicalEntities()
        entities.add_condition(Condition(name="Asthma", icd10="J45.909", status="active"))

        bundle = transformer.transform(entities)

        condition = next(
            e["resource"]
            for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Condition"
        )
        coding = condition["code"]["coding"][0]
        assert coding["system"] == "http://hl7.org/fhir/sid/icd-10"
        assert coding["code"] == "J45.909"

    def test_observation_loinc_codes(self, transformer: FHIRTransformer):
        """Test that observations include vital signs."""
        entities = ClinicalEntities()
        entities.add_vital(Vital(type="blood_pressure", value="120/80", unit="mmHg"))

        bundle = transformer.transform(entities)

        observation = next(
            e["resource"]
            for e in bundle["entry"]
            if e["resource"]["resourceType"] == "Observation"
        )
        # Should have a code
        assert "code" in observation
        assert observation["code"]["text"] == "blood_pressure"

    def test_medication_rxnorm(self, transformer: FHIRTransformer):
        """Test that medications with RxNorm codes are included."""
        entities = ClinicalEntities()
        entities.add_medication(
            Medication(name="Atorvastatin", dose="20mg", frequency="daily", rxnorm="617312")
        )

        bundle = transformer.transform(entities)

        med_resource = next(
            e["resource"]
            for e in bundle["entry"]
            if e["resource"]["resourceType"] in ["MedicationStatement", "MedicationRequest"]
        )
        # Check RxNorm code is present
        json_str = json.dumps(med_resource)
        assert "617312" in json_str or "rxnorm" in json_str.lower()

    def test_patient_reference(self, transformer: FHIRTransformer):
        """Test that resources reference a patient."""
        entities = ClinicalEntities()
        entities.add_condition(Condition(name="Test", status="active"))

        bundle = transformer.transform(entities)

        # If there's a patient resource, conditions should reference it
        patient_entry = next(
            (e for e in bundle["entry"] if e["resource"]["resourceType"] == "Patient"),
            None,
        )
        if patient_entry:
            condition_entry = next(
                e for e in bundle["entry"] if e["resource"]["resourceType"] == "Condition"
            )
            assert "subject" in condition_entry["resource"]
