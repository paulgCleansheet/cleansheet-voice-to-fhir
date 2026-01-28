# Known Issues

This document tracks known issues discovered during testing that are lower priority for the competition deadline but should be addressed for production use.

## MedGemma Extraction Issues

### 1. Duplicate Medication Orders
**Affected:** cardiology workflow
**Example:** `medication_orders` contains two identical "statin" entries
**Root Cause:** MedGemma extracts the same medication multiple times from transcript
**Potential Fix:** Add deduplication by name in `filter_medication_orders()` (partially implemented, may need case-insensitive matching improvement)

### 2. Duplicate Conditions
**Affected:** procedure workflow
**Example:** Two identical "sessile polyp" conditions
**Root Cause:** MedGemma extracts each polyp mention as separate condition
**Potential Fix:** Add deduplication in `filter_conditions()` by normalized name

### 3. Duplicate Family History Entries
**Affected:** intake workflow
**Example:** Father's MI appears twice (capitalized differently), brother's diabetes appears twice
**Root Cause:** MedGemma re-extracts family history with slight variations
**Potential Fix:** Add case-insensitive deduplication in `extract_family_history()` by (relationship, condition) key

### 4. Invalid Referral Orders
**Affected:** cardiology, discharge, hp, procedure, radiology workflows
**Examples:**
- `{"reason": null}` - missing specialty entirely
- `{"reason": "one week"}` - time duration instead of specialty
- `{"reason": "await pathology results"}` - instruction instead of referral

**Root Cause:** MedGemma conflates referral orders with follow-up instructions
**Potential Fix:**
- Filter out referral orders with null specialty
- Validate that `reason` or `specialty` contains a medical specialty keyword
- Consider renaming "referral_orders" to "follow_up_orders" if this is the intended semantics

### 5. Hallucinated Lab Orders
**Affected:** general workflow, complex patient
**Example:** `lab_orders: [{"name": "A1c"}]` but transcript has no mention of A1c
**Root Cause:** MedGemma infers labs based on conditions (diabetes -> A1c) rather than explicit orders
**Potential Fix:** Cross-reference lab_orders against transcript keywords; require explicit mention

### 6. Missing Referral Orders
**Affected:** complex patient workflow
**Example:** Transcript says "cardiology consult, diabetes consult, social work consult" but `referral_orders: []`
**Root Cause:** MedGemma doesn't extract "consult" as referral orders
**Potential Fix:** Add post-processor pattern to extract "X consult" patterns from transcript as referral orders

### 7. Wrong Family History Relationship
**Affected:** complex patient workflow
**Example:** "Paternal grandfather had heart failure" → `{"relationship": "father", "condition": "heart failure"}`
**Root Cause:** MedGemma misinterprets compound relationships like "paternal grandfather"
**Potential Fix:** Add relationship normalization/validation in post-processor

### 8. Missing Stated Conditions
**Affected:** complex patient workflow
**Example:** Transcript clearly states "hyperlipidemia" in past medical history but condition not extracted
**Root Cause:** MedGemma may truncate or miss conditions in very long transcripts
**Potential Fix:** Add deterministic extraction for conditions mentioned in "[PAST MEDICAL HISTORY]" section

### 9. Missing Lab Results
**Affected:** complex patient workflow
**Example:** "INR 2.4, therapeutic" in transcript but not in lab_results
**Root Cause:** MedGemma misses some lab values in dense lab sections
**Potential Fix:** Add regex patterns for common labs (INR, PT, PTT) in post-processor

## MedASR Transcription Issues

### 10. Drug Name Transcription Errors
**Affected:** complex patient workflow (uncommon drug names)
**Examples:**
- "spironolactone" → "pirro lactone"
- "tiotropium" → "Toroptooum"
- "tamsulosin" → "tamsosin"
- "sacubitril-valsartan" → "secubr valsartan"

**Root Cause:** MedASR phonetically transcribes uncommon drug names
**Potential Fix:** Add drug name normalization using fuzzy matching against RxNorm dictionary

## Lower Priority Issues

### 11. Ventilator Settings as Medication
**Affected:** respiratory workflow
**Example:** `medications: [{"name": "Ventilator settings", "dose": null}]`
**Root Cause:** MedGemma misclassifies equipment settings as medications
**Potential Fix:** Add "Ventilator settings" to non-medication filter list

### 12. ABG Values with Generic Units
**Affected:** respiratory workflow
**Example:** `{"type": "pH", "value": "7.38", "unit": "unit"}`
**Root Cause:** MedGemma doesn't know appropriate units for ABG values
**Potential Fix:** Add unit mapping by vital type (pH -> dimensionless, pCO2/pO2 -> mmHg, HCO3 -> mEq/L)

### 13. EEG Listed as Lab Result
**Affected:** neurology workflow
**Example:** `lab_results: [{"name": "EEG", "value": "deprivation"}]`
**Root Cause:** MedGemma categorizes diagnostic tests as lab results
**Potential Fix:** Move EEG, MRI, CT to procedure_orders or create diagnostic_orders category

---

## Resolution Status

| Issue | Priority | Status | Target |
|-------|----------|--------|--------|
| 1. Duplicate med orders | Medium | Open | Post-competition |
| 2. Duplicate conditions | Medium | Open | Post-competition |
| 3. Duplicate family history | Medium | Open | Post-competition |
| 4. Invalid referrals | Low | Open | Post-competition |
| 5. Hallucinated labs | Low | Open | Post-competition |
| 6. Missing referrals | Medium | Open | Post-competition |
| 7. Wrong family relationship | Low | Open | Post-competition |
| 8. Missing stated conditions | Medium | Open | Post-competition |
| 9. Missing lab results | Low | Open | Post-competition |
| 10. Drug name transcription | Low | Open | Post-competition |
| 11. Vent settings as med | Low | Open | Post-competition |
| 12. ABG units | Low | Open | Post-competition |
| 13. EEG categorization | Low | Open | Post-competition |

---

*Last updated: 2026-01-28*
