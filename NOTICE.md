# Third-Party Notices

This software includes third-party components under various open source licenses.

## Google Health AI Developer Foundations (HAI-DEF)

### MedASR
- **Source:** https://developers.google.com/health-ai-developer-foundations
- **License:** Subject to HAI-DEF Terms of Use
- **Usage:** Medical speech recognition model

### MedGemma
- **Source:** https://developers.google.com/health-ai-developer-foundations
- **License:** Subject to HAI-DEF Terms of Use
- **Usage:** Medical vision-language model for structured extraction

**Note:** Use of MedASR and MedGemma requires acceptance of the HAI-DEF Terms of Use.
See https://developers.google.com/health-ai-developer-foundations for details.

---

## HL7 FHIR

- **Source:** https://www.hl7.org/fhir/
- **License:** Creative Commons CC0 1.0 Universal
- **Usage:** FHIR R4 resource definitions and specifications

---

## Python Dependencies

The following Python packages are used under their respective licenses:

### MIT License

- **huggingface_hub** - HuggingFace Hub client library
- **transformers** - HuggingFace Transformers library
- **pydantic** - Data validation library
- **typer** - CLI framework
- **rich** - Terminal formatting
- **loguru** - Logging library
- **pytest** - Testing framework

### Apache License 2.0

- **numpy** - Numerical computing library
- **scipy** - Scientific computing library
- **soundfile** - Audio file I/O

### BSD License

- **fhir.resources** - FHIR resource models for Python

---

## Medical Terminology Systems

### SNOMED CT
- **Source:** https://www.snomed.org/
- **License:** SNOMED CT Affiliate License (free for qualifying organizations)
- **Usage:** Clinical terminology codes

### ICD-10
- **Source:** https://www.who.int/standards/classifications/classification-of-diseases
- **License:** WHO copyright (free for use with attribution)
- **Usage:** Diagnosis codes

### RxNorm
- **Source:** https://www.nlm.nih.gov/research/umls/rxnorm/
- **License:** UMLS Metathesaurus License (free, registration required)
- **Usage:** Medication codes

### LOINC
- **Source:** https://loinc.org/
- **License:** LOINC License (free, registration required)
- **Usage:** Laboratory observation codes

---

## Attribution

When using this software, please include the following attribution:

> Voice-to-FHIR Pipeline by Cleansheet LLC
> https://github.com/CleansheetLLC/cleansheet-voice-to-fhir
>
> Uses MedASR and MedGemma from Google Health AI Developer Foundations
> Uses FHIR R4 from HL7 International

---

*Last updated: January 2026*
