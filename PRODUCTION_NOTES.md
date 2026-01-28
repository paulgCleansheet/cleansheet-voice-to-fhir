# Production Notes

Operational guidance for practitioners integrating and tuning the voice-to-FHIR pipeline.

---

## Table of Contents

1. [Model Configuration](#model-configuration)
2. [MedASR Tuning](#medasr-tuning)
3. [MedGemma Tuning](#medgemma-tuning)
4. [Post-Processing](#post-processing)
5. [Architecture Considerations](#architecture-considerations)
6. [Known Limitations](#known-limitations)
7. [Monitoring and Debugging](#monitoring-and-debugging)

---

## Model Configuration

### Token Limits

| Parameter | Value | Location | Notes |
|-----------|-------|----------|-------|
| `max_tokens` | 8192 | `config.py:ExtractionConfig` | Required for complex patients with 15+ medications |
| `temperature` | 0.1 | `config.py:ExtractionConfig` | Low temperature for deterministic extraction |

**Critical**: The server uses `PipelineConfig.extraction.max_tokens` from `config.py`, not the `MedGemmaClientConfig` in `medgemma_client.py`. When tuning token limits, modify `config.py`.

For complex patients (12+ conditions, 15+ medications, comprehensive labs), JSON output can exceed 5000 characters. The default 2048 tokens causes truncation mid-JSON, resulting in parse failures.

**Sizing guidance:**
- Simple encounters (vitals, 1-2 meds, single condition): 1024-2048 tokens
- Standard encounters (5-8 meds, multiple conditions): 4096 tokens
- Complex patients (polypharmacy, multiple comorbidities): 8192 tokens

### Timeout Configuration

| Service | Timeout | Location | Rationale |
|---------|---------|----------|-----------|
| MedASR | 600s | `medasr_client.py` | 8-minute recordings need ~5-6 minutes to process |
| MedGemma | 300s | `medgemma_client.py` | Complex extractions with long transcripts |

**Rule of thumb**: Set MedASR timeout to roughly 1.5x the expected maximum recording duration.

---

## MedASR Tuning

### Speech Patterns

MedASR is trained on natural clinical speech. Avoid:

| Pattern | Problem | Solution |
|---------|---------|----------|
| Bracketed markers | `[CHIEF COMPLAINT]` → "ief complaint" | Use natural transitions: "Chief complaint is..." |
| Spelled abbreviations | "H-T-N" → garbled | Say full word: "hypertension" |
| Rapid enumeration | Numbers run together | Pause between list items |

### Drug Name Accuracy

MedASR struggles with uncommon drug names, transcribing phonetically:

| Actual | Transcribed |
|--------|-------------|
| spironolactone | pirro lactone |
| tiotropium | Toroptooum |
| tamsulosin | tamsosin |
| sacubitril-valsartan | secubr valsartan |

**Mitigation options:**
1. Fuzzy matching against RxNorm dictionary in post-processor
2. Pronunciation guide training data (future)
3. Spell-check against medication list from patient record

### Recording Quality

- Sample rate: 16kHz mono (matches MedASR training)
- Avoid background noise, HVAC, equipment alarms
- Consistent microphone distance (6-12 inches)
- Clip-on lavalier recommended for mobile dictation

---

## MedGemma Tuning

### Workflow Selection

The current implementation uses filename patterns to select workflow prompts:

```
cardiology*.wav → prompts/cardiology.txt
intake*.wav → prompts/intake.txt
default → prompts/general.txt
```

**This approach is fragile.** Production systems should:

1. **Explicit workflow parameter**: Pass workflow type in API request
2. **Context-based detection**: Analyze first 500 chars of transcript to detect specialty markers
3. **Provider profile**: Use provider's specialty as default workflow

### Prompt Engineering

Each workflow prompt in `prompts/` defines the JSON schema MedGemma should output. Key principles:

1. **Be explicit about field names**: MedGemma follows the schema exactly
2. **Provide examples**: Few-shot examples improve extraction accuracy
3. **Specify null handling**: Tell MedGemma to use `null` for missing values, not empty strings or "unknown"
4. **Limit categories**: Too many output fields causes extraction dilution

### Common Extraction Issues

| Issue | Root Cause | Mitigation |
|-------|------------|------------|
| Duplicate medications | Re-extraction from multiple mentions | Deduplication by normalized name in post-processor |
| Duplicate conditions | Each mention extracted separately | Case-insensitive deduplication |
| Hallucinated labs | Inferred from conditions (diabetes → A1c) | Cross-reference against transcript |
| Missing referrals | "Consult" not recognized as referral | Pattern extraction in post-processor |
| Wrong family relationships | "Paternal grandfather" → "father" | Relationship normalization |

### Structured Section Markers

If your transcription workflow supports it, use natural section markers that MedGemma can recognize:

```
"History of present illness..."
"Past medical history includes..."
"Current medications are..."
"Plan is as follows..."
```

These guide MedGemma's extraction without confusing MedASR.

---

## Post-Processing

### Important: Scope of Clinical Decision Support

The RxNorm verification, ICD-10 enrichment, and order-diagnosis linking features in this pipeline are **rudimentary first-stage processing** intended for demonstration and research purposes.

**These features are NOT substitutes for:**

- Commercial clinical decision support (CDS) systems (e.g., Wolters Kluwer, Elsevier)
- Pharmacy information systems with comprehensive drug databases
- Electronic health record (EHR) built-in coding assistance
- Professional medical coding services
- Drug interaction checkers with complete interaction databases
- Clinical terminology services (full RxNorm, SNOMED CT, ICD-10)

**Production environments should expect:**

1. **Integration with commercial CDS**: The extracted data should flow into hospital CDS systems that provide comprehensive drug-drug interaction checking, allergy cross-referencing, dose range validation, and contraindication alerts.

2. **Professional coding review**: ICD-10 codes suggested by this pipeline should be reviewed by certified medical coders before billing submission. The ~500 code lookup database covers common conditions but is not comprehensive.

3. **Pharmacy verification**: Medication orders should be verified against the patient's complete medication profile in the pharmacy information system, not just the limited RxNorm subset in this pipeline.

4. **Terminology service integration**: Production systems should integrate with full terminology services (UMLS, RxNorm API, ICD-10-CM API) rather than relying on the curated subsets included here.

The clinical rules (drug class → indication, specialty → condition) are based on common clinical patterns but do not account for:
- Patient-specific contraindications
- Drug-drug interactions
- Age/weight-based dosing considerations
- Renal/hepatic function adjustments
- Pregnancy/lactation considerations

**In summary**: This pipeline provides a starting point for structured data extraction. Commercial clinical information systems will substantially improve accuracy, completeness, and safety of the clinical decision support features.

---

### Deduplication Strategy

The post-processor (`post_processor.py`) handles:

1. **BP deduplication**: Track seen values, remove exact duplicates
2. **Medication deduplication**: By normalized name (case-insensitive)
3. **Condition deduplication**: By normalized name
4. **Null string cleanup**: Convert `"null"` strings to actual `null`

### Deterministic Fallback Extraction

For critical fields MedGemma misses, the post-processor uses regex patterns:

```python
# Medication dosage patterns
r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|mcg)\s*(daily|twice daily|tid|qid)?'

# Plan section medication orders
r'(?:start|prescribe|give)\s+(\w+)\s+(\d+)\s*(mg|g)'
```

**Trade-off**: Deterministic extraction is more reliable but less flexible than LLM extraction. Use for high-value fields where MedGemma is inconsistent.

### Unit Normalization

MedGemma sometimes uses inconsistent units:

| Vital | Expected Unit | Common Errors |
|-------|---------------|---------------|
| Blood pressure | mmHg | (usually correct) |
| Respiratory rate | breaths/min | "C" (copies from temperature) |
| Temperature | F or C | (usually correct) |
| SpO2 | % | (usually correct) |
| pH | (dimensionless) | "unit" |
| pCO2/pO2 | mmHg | "unit" |

Consider adding unit mapping by vital type in post-processor.

---

## Architecture Considerations

### Current Architecture

```
Browser → server.py → MedASR (HF Endpoint) → MedGemma (HF Endpoint) → FHIR
                                ↓
                         post_processor.py
```

### Workflow Detection Improvement

Current filename-based workflow detection should be replaced with:

**Option A: API Parameter**
```json
POST /process-audio
{
  "audio": "<base64>",
  "workflow": "cardiology"
}
```

**Option B: Transcript Analysis**
```python
def detect_workflow(transcript: str) -> str:
    """Detect workflow from transcript content."""
    markers = {
        "cardiology": ["ejection fraction", "stress test", "cardiac"],
        "neurology": ["seizure", "stroke", "neurological"],
        "respiratory": ["ventilator", "intubation", "FiO2"],
        "procedure": ["endoscopy", "colonoscopy", "biopsy"],
    }
    # Score by marker presence
    ...
```

**Option C: Provider Context**
```python
# Use provider specialty from session/auth
workflow = provider.specialty.lower()
```

### Scaling Considerations

1. **Queue long recordings**: 8-minute recordings block for 5+ minutes
2. **Chunked processing**: For very long recordings, process in segments
3. **Async notification**: Return job ID, poll or webhook for completion
4. **Caching**: Cache workflow prompts, don't reload on every request

---

## Known Limitations

See `KNOWN_ISSUES.md` for detailed issue tracking. Summary:

### High Impact
- Drug name transcription errors (MedASR phonetic issues)
- Missing referral orders ("consult" not recognized)
- Wrong family relationships (compound relationships)

### Medium Impact
- Duplicate extractions (partially mitigated by post-processor)
- Missing stated conditions in long transcripts
- Hallucinated lab orders (inference from conditions)

### Low Impact
- ABG values with generic "unit" instead of proper units
- EEG categorized as lab result instead of procedure
- Ventilator settings extracted as medication

---

## Monitoring and Debugging

### Key Metrics to Track

1. **Transcription latency**: MedASR processing time by audio duration
2. **Extraction latency**: MedGemma processing time by transcript length
3. **JSON parse failures**: Indicates token limit issues
4. **Empty extractions**: May indicate prompt mismatch or timeout
5. **Duplicate counts**: Pre/post deduplication entity counts

### Debug Logging

Current implementation uses `print()` statements. For production:

1. Replace with Python `logging` module
2. Include session IDs for traceability
3. Log extraction decisions (what was filtered and why)
4. Structured JSON format for log aggregation

### Audit Trail (Compliance)

For SOC2/FDA 510(k), implement:

1. Before/after snapshots of post-processing
2. Pattern match decisions logged
3. Timestamped, immutable audit entries
4. 6-year retention (HIPAA)

---

## Configuration Reference

### config.py Defaults

```python
@dataclass
class TranscriptionConfig:
    backend: str = "whisper"          # or "dedicated", "local"
    model_id: str = "google/medasr"
    local_url: str = "http://localhost:3002"
    device: str = "cuda"
    precision: str = "fp16"

@dataclass
class ExtractionConfig:
    backend: str = "dedicated"        # or "local", "serverless"
    model_id: str = "google/medgemma-4b-it"
    local_url: str = "http://localhost:3003"
    max_tokens: int = 8192            # Critical for complex patients
    temperature: float = 0.1
    workflow: str = "general"
    prompts_dir: str = "src/voice_to_fhir/extraction/prompts"
```

### Environment Variables

```bash
HF_TOKEN=hf_xxxxx              # HuggingFace API token
MEDASR_ENDPOINT=https://...    # Dedicated endpoint URL
MEDGEMMA_ENDPOINT=https://...  # Dedicated endpoint URL
```

---

*Last updated: 2026-01-28*
