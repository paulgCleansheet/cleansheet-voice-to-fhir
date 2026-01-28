"""
Post-processor for extracted clinical entities.

This module provides:
1. Transcript marker extraction - fills in missing fields from bracketed sections
2. Validation/filtering - removes placeholder values and invalid entries
3. ICD-10 code enrichment - adds verified codes from lookup database
"""

import re
from typing import Any
from voice_to_fhir.extraction.extraction_types import (
    ClinicalEntities,
    Condition,
    FamilyHistory,
    SocialHistory,
    Vital,
    Allergy,
    Medication,
    MedicationOrder,
)
from voice_to_fhir.extraction.icd10_lookup import enrich_conditions_with_icd10


# Placeholder values to filter out
PLACEHOLDER_VALUES = {
    "null", "none", "not mentioned", "not specified", "unknown", "n/a", "na",
    "not applicable", "not available", "not provided", "not stated",
    "not documented", "no information", "unspecified", ""
}


def is_placeholder(value: Any) -> bool:
    """Check if a value is a placeholder that should be filtered out."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.lower().strip() in PLACEHOLDER_VALUES
    return False


def extract_section(transcript: str, marker: str, first_sentence_only: bool = False) -> str | None:
    """
    Extract content following a bracketed marker until the next marker or end.

    Example: extract_section(text, "CHIEF COMPLAINT")
    Returns content after [CHIEF COMPLAINT] until next [MARKER] or end.

    Args:
        transcript: The transcript text
        marker: The section marker (e.g., "CHIEF COMPLAINT")
        first_sentence_only: If True, stop at first period followed by space
    """
    # Pattern: [MARKER] followed by content until next [SOMETHING] or end
    pattern = rf'\[{re.escape(marker)}\]\s*(?:[Ii]s\s+)?(.+?)(?=\[|$)'
    match = re.search(pattern, transcript, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()

        # For chief complaint, just get the first sentence
        if first_sentence_only:
            # Stop at first period followed by space and capital, or period at end
            sentence_match = re.match(r'^([^.]+\.?)(?:\s+[A-Z0-9]|$)', content)
            if sentence_match:
                content = sentence_match.group(1).strip()

        # Clean up: remove trailing punctuation and whitespace
        content = re.sub(r'[\.\s]+$', '', content)
        return content if content and not is_placeholder(content) else None
    return None


def extract_chief_complaint(transcript: str, entities: ClinicalEntities) -> str | None:
    """
    Extract chief complaint from transcript markers.

    PRIORITY ORDER (prefer symptoms over diagnoses):
    1. Explicit [CHIEF COMPLAINT] marker
    2. [CLINICAL HISTORY] marker (for radiology)
    3. [SUBJECTIVE] section start (for SOAP)
    4. "presents with" pattern
    5. "CC:" pattern
    6. "admitted with/for" pattern (for discharge)
    7. "follow-up for" / "return visit for" pattern
    8. "visit for" pattern
    9. MedGemma's extraction (fallback only if not a diagnosis)

    Looks for patterns like:
    - [CHIEF COMPLAINT] ...
    - [CLINICAL HISTORY] ...
    - [SUBJECTIVE] X presents with ...
    - Patient presents with ...
    - CC: ...
    - admitted with/for ...
    - follow-up visit for ...
    - visit for ...
    """
    # 1. Try explicit [CHIEF COMPLAINT] marker first
    cc = extract_section(transcript, "CHIEF COMPLAINT", first_sentence_only=True)
    if cc:
        print(f"[CC Extract] Found via [CHIEF COMPLAINT] marker: '{cc}'")
        return cc

    # 2. Try [CLINICAL HISTORY] marker (radiology reports)
    cc = extract_section(transcript, "CLINICAL HISTORY", first_sentence_only=True)
    if cc:
        # Extract just the symptoms part (after age/gender)
        # "67-year-old male with cough and shortness of breath" -> "cough and shortness of breath"
        match = re.search(r'(?:with|for|presents?\s+with)\s+(.+?)(?:\.|$)', cc, re.IGNORECASE)
        if match:
            symptoms = match.group(1).strip()
            print(f"[CC Extract] Found via [CLINICAL HISTORY] marker: '{symptoms}'")
            return symptoms
        # If no "with" pattern, return the whole thing
        print(f"[CC Extract] Found via [CLINICAL HISTORY] marker (full): '{cc}'")
        return cc

    # 3. Try [SUBJECTIVE] section for SOAP notes - look for "presents with"
    subj = extract_section(transcript, "SUBJECTIVE")
    if subj:
        match = re.search(r'presents?\s+with\s+(.+?)(?:for\s+\d|She\s+|He\s+|Patient\s+|No\s+|\.|$)', subj, re.IGNORECASE)
        if match:
            cc = match.group(1).strip().rstrip('.')
            print(f"[CC Extract] Found via [SUBJECTIVE] 'presents with': '{cc}'")
            return cc

    # 4. Try "presents with" pattern anywhere in transcript
    match = re.search(r'presents?\s+with\s+(.+?)(?:for\s+\d|She\s+|He\s+|Patient\s+|No\s+|\.|$)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip().rstrip('.')
        if not is_placeholder(cc):
            print(f"[CC Extract] Found via 'presents with' pattern: '{cc}'")
            return cc

    # 5. Try "CC:" pattern
    match = re.search(r'(?:^|\s)CC:\s*(.+?)(?:\.|$)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip()
        if not is_placeholder(cc):
            print(f"[CC Extract] Found via 'CC:' pattern: '{cc}'")
            return cc

    # 6. Try "admitted with/for" pattern (discharge summaries)
    match = re.search(r'admitted\s+(?:with|for)\s+(.+?)(?:\.|Chest|Started|$)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip().rstrip('.')
        if not is_placeholder(cc):
            print(f"[CC Extract] Found via 'admitted with/for' pattern: '{cc}'")
            return cc

    # 7. Try "follow-up/visit for X" pattern (follow-up visits)
    match = re.search(r'(?:follow-?up|f/u|return)\s+(?:visit\s+)?for\s+(.+?)(?:\.|Patient|She|He|$)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip().rstrip('.')
        if not is_placeholder(cc):
            # Append "follow-up" to clarify this is a follow-up visit
            cc_with_context = f"{cc} follow-up"
            print(f"[CC Extract] Found via 'follow-up for' pattern: '{cc_with_context}'")
            return cc_with_context

    # 8. Try "visit for X" pattern (general visits)
    match = re.search(r'(?:^|\.)\s*(?:office\s+)?visit\s+for\s+(.+?)(?:\.|Patient|She|He|$)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip().rstrip('.')
        if not is_placeholder(cc):
            print(f"[CC Extract] Found via 'visit for' pattern: '{cc}'")
            return cc

    # 9. Fallback: use MedGemma's chief complaint if it looks like a symptom (not a diagnosis)
    # Diagnoses often contain medical terms like "pharyngitis", "pneumonia", "diabetes"
    diagnosis_indicators = [
        'pharyngitis', 'pneumonia', 'diabetes', 'hypertension', 'syndrome',
        'disease', 'disorder', 'infection', 'mellitus', 'carcinoma', 'failure',
        'insufficiency', 'infarction', 'ischemia', 'fibrillation', 'embolism'
    ]

    for condition in entities.conditions:
        if condition.is_chief_complaint and not is_placeholder(condition.name):
            name_lower = condition.name.lower()
            is_diagnosis = any(indicator in name_lower for indicator in diagnosis_indicators)
            if not is_diagnosis:
                print(f"[CC Extract] Using MedGemma CC (looks like symptom): '{condition.name}'")
                return condition.name
            else:
                print(f"[CC Extract] Skipping MedGemma CC (looks like diagnosis): '{condition.name}'")

    print("[CC Extract] No chief complaint found")
    return None


def extract_family_history(transcript: str, entities: ClinicalEntities) -> list[FamilyHistory]:
    """
    Extract family history entries from transcript.

    Patterns:
    - [FAMILY HISTORY] ...
    - Family history significant for X in mother/father
    - Mother has/had X
    - Father died of X at age Y
    """
    # Start with existing valid entries
    result = [fh for fh in entities.family_history
              if not is_placeholder(fh.condition) and not is_placeholder(fh.relationship)]

    existing_conditions = {(fh.relationship.lower(), fh.condition.lower()) for fh in result}

    # Try bracketed section
    fh_section = extract_section(transcript, "FAMILY HISTORY")
    text_to_search = fh_section or transcript

    # Parse common patterns - each returns (relationship, condition, age_or_None)
    # Pattern 1: "Mother had stroke at age 58"
    pattern1 = r'(mother|father|brother|sister|sibling|parent|grandmother|grandfather|grandparent|aunt|uncle|cousin)\s+(?:had|has|with|died\s+(?:of|from))\s+([^,\.]+?)(?:[\s,]+(?:at\s+)?(?:age|onset)\s+(?:at\s+)?(\d+))?(?:,|\.|$)'
    for match in re.finditer(pattern1, text_to_search, re.IGNORECASE):
        relationship, condition, age = match.group(1), match.group(2), match.group(3)
        key = (relationship.lower(), condition.lower().strip())
        if key not in existing_conditions and not is_placeholder(condition):
            result.append(FamilyHistory(
                relationship=relationship.capitalize(),
                condition=condition.strip(),
                age_of_onset=age,
            ))
            existing_conditions.add(key)

    # Pattern 2: "stroke in mother" or "significant for stroke in mother"
    pattern2 = r'(?:significant\s+for\s+|history\s+of\s+)?([a-zA-Z\s]+?)\s+in\s+(mother|father|brother|sister|sibling|parent|grandmother|grandfather|cousin)(?:[\s,]+(?:at\s+)?(?:age|onset)\s+(?:at\s+)?(\d+))?'
    for match in re.finditer(pattern2, text_to_search, re.IGNORECASE):
        condition, relationship, age = match.group(1), match.group(2), match.group(3)
        condition = condition.strip()
        # Skip if condition looks like a non-condition phrase
        if condition.lower() in ('', 'history', 'family', 'significant'):
            continue
        key = (relationship.lower(), condition.lower())
        if key not in existing_conditions and not is_placeholder(condition):
            result.append(FamilyHistory(
                relationship=relationship.capitalize(),
                condition=condition.strip(),
                age_of_onset=age,
            ))
            existing_conditions.add(key)

    return result


def extract_social_history(transcript: str, entities: ClinicalEntities) -> SocialHistory | None:
    """
    Extract social history from transcript.

    Patterns:
    - [SOCIAL HISTORY] ...
    - Smoker / Former smoker / Never smoker
    - Drinks X per week/day
    - Works as X / Occupation: X
    """
    # Start with existing social history
    sh = entities.social_history or SocialHistory()

    # Try bracketed section first
    sh_section = extract_section(transcript, "SOCIAL HISTORY")
    text = sh_section or transcript

    # Extract tobacco
    if is_placeholder(sh.tobacco):
        tobacco_patterns = [
            (r'never\s+smok(?:er|ed)', 'Never smoker'),
            (r'former\s+smok(?:er|ed)', 'Former smoker'),
            (r'quit\s+(?:smoking\s+)?(\d+)\s+years?\s+ago', lambda m: f'Former smoker, quit {m.group(1)} years ago'),
            (r'current(?:ly)?\s+smok(?:er|es|ing)', 'Current smoker'),
            (r'smokes?\s+(\d+)\s+pack', lambda m: f'Current smoker, {m.group(1)} pack'),
            (r'(\d+)[- ]pack[- ]year', lambda m: f'{m.group(1)} pack-year history'),
            (r'denies\s+(?:tobacco|smoking)', 'Denies tobacco use'),
        ]
        for pattern, value in tobacco_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if callable(value):
                    sh.tobacco = value(match)
                else:
                    sh.tobacco = value
                break

    # Extract alcohol
    if is_placeholder(sh.alcohol):
        alcohol_patterns = [
            (r'denies\s+alcohol', 'Denies alcohol'),
            (r'no\s+alcohol', 'No alcohol use'),
            (r'social(?:ly)?\s+(?:drinks?|alcohol)', 'Social drinker'),
            (r'drinks?\s+(\d+)\s+(?:beers?|drinks?|glasses?)', lambda m: f'{m.group(1)} drinks'),
            (r'occasional\s+(?:alcohol|wine|beer)', 'Occasional alcohol use'),
        ]
        for pattern, value in alcohol_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if callable(value):
                    sh.alcohol = value(match)
                else:
                    sh.alcohol = value
                break

    # Extract occupation
    if is_placeholder(sh.occupation):
        # Words that are NOT occupations (smoking, drinking status)
        non_occupation_words = {'smoker', 'drinker', 'alcoholic', 'addict', 'user'}

        occupation_patterns = [
            r'(?:works?\s+as\s+(?:a\s+|an\s+)?)([^,\.]+)',
            r'occupation[:\s]+([^,\.]+)',
            # "retired teacher" but not "former smoker"
            r'(?:retired|former)\s+(?!smoker|drinker)([^,\.]+)',
            r'([^,\.]+)\s+by\s+(?:profession|occupation)',
        ]
        for pattern in occupation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                occupation = match.group(1).strip()
                # Skip if it's a non-occupation word or placeholder
                if occupation.lower() in non_occupation_words:
                    continue
                if not is_placeholder(occupation):
                    sh.occupation = occupation
                    break

    # Extract drugs
    if is_placeholder(sh.drugs):
        if re.search(r'denies\s+(?:recreational\s+)?drug', text, re.IGNORECASE):
            sh.drugs = "Denies drug use"
        elif re.search(r'no\s+(?:recreational\s+)?drug', text, re.IGNORECASE):
            sh.drugs = "No drug use"

    # Only return if we have any data
    if any([sh.tobacco, sh.alcohol, sh.drugs, sh.occupation, sh.living_situation]):
        return sh
    return None


def is_valid_vital_value(value: str, vital_type: str = "") -> bool:
    """Check if a vital value is valid (numeric or blood pressure format)."""
    if not value or not isinstance(value, str):
        return False
    value = str(value).strip()

    # Blood pressure format: "120/80" or "120 / 80"
    if "/" in value or vital_type.lower() in ("blood_pressure", "bp"):
        bp_match = re.match(r'^(\d+)\s*/\s*(\d+)$', value)
        if bp_match:
            return True

    # Regular numeric value
    try:
        float(value.replace(',', ''))
        return True
    except (ValueError, AttributeError):
        return False


def extract_blood_pressure_from_transcript(transcript: str) -> Vital | None:
    """
    Extract blood pressure from transcript text.

    Patterns:
    - "blood pressure 120/80"
    - "BP 120/80"
    - "blood pressure today 128/78"
    - "blood pressure is 118 over 74"
    - "BP 118 over 74"
    - "120/80 mm Hg"
    """
    bp_patterns = [
        # "blood pressure 120/80" or "blood pressure today 128/78"
        r'blood\s+pressure\s+(?:today\s+|is\s+|of\s+)?(\d{2,3})\s*/\s*(\d{2,3})',
        # "BP 120/80"
        r'\bBP\s+(\d{2,3})\s*/\s*(\d{2,3})',
        # "blood pressure is 118 over 74" or "blood pressure 118 over 74"
        r'blood\s+pressure\s+(?:is\s+|today\s+)?(\d{2,3})\s+over\s+(\d{2,3})',
        # "BP 118 over 74"
        r'\bBP\s+(\d{2,3})\s+over\s+(\d{2,3})',
        # "120/80 mm Hg" or "120/80 mmHg"
        r'(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mm\s*Hg|mmHg)',
    ]

    for pattern in bp_patterns:
        match = re.search(pattern, transcript, re.IGNORECASE)
        if match:
            systolic = match.group(1)
            diastolic = match.group(2)
            bp_value = f"{systolic}/{diastolic}"
            print(f"[BP Extract] Found blood pressure from transcript: {bp_value}")
            return Vital(
                type="blood_pressure",
                value=bp_value,
                unit="mmHg"
            )

    return None


def ensure_blood_pressure_from_transcript(entities: "ClinicalEntities", transcript: str) -> None:
    """
    SIMPLE BP EXTRACTION: Extract BP directly from transcript and fix vitals.

    This function:
    1. Extracts BP from transcript using reliable patterns
    2. Removes any partial/incorrect BP values from vitals (single mmHg values)
    3. Adds the complete BP to vitals

    Called at the START of post_process to ensure BP is always correct.
    """
    # Extract BP from transcript
    bp_from_transcript = extract_blood_pressure_from_transcript(transcript)

    if not bp_from_transcript:
        print("[BP Ensure] No blood pressure found in transcript")
        return

    print(f"[BP Ensure] Extracted BP from transcript: {bp_from_transcript.value}")

    # Remove ALL existing BP values (both partial and complete)
    # The deterministic extraction is the source of truth for BP
    cleaned_vitals = []
    for v in entities.vitals:
        value_str = str(v.value) if v.value else ""
        unit_lower = (v.unit or "").lower()
        type_lower = (v.type or "").lower()

        # Skip any blood pressure vitals (by type or by mmHg unit)
        if type_lower in ("blood_pressure", "bp"):
            print(f"[BP Ensure] Removing existing BP: {value_str} {v.unit}")
            continue

        # Also skip mmHg values that look like BP (partial or complete)
        if unit_lower in ("mmhg", "mm hg"):
            # Check if it's a BP-like value (contains "/" or is in BP numeric range)
            if "/" in value_str:
                print(f"[BP Ensure] Removing existing BP: {value_str} {v.unit}")
                continue
            try:
                numeric_val = float(value_str.replace(',', ''))
                if 40 <= numeric_val <= 220:
                    print(f"[BP Ensure] Removing partial BP value: {value_str} {v.unit}")
                    continue
            except (ValueError, TypeError):
                pass

        cleaned_vitals.append(v)

    # Add the single authoritative BP from transcript
    cleaned_vitals.insert(0, bp_from_transcript)
    entities.vitals = cleaned_vitals
    print(f"[BP Ensure] Added complete BP: {bp_from_transcript.value} mmHg")


def detect_resolved_status_from_transcript(entities: "ClinicalEntities", transcript: str) -> None:
    """
    Detect 'resolved' status for conditions mentioned as resolved in the transcript.

    This handles discharge summaries where conditions are marked as resolved:
    - "[DISCHARGE DIAGNOSIS] Community-acquired pneumonia, resolved."
    - "Diagnosis: pneumonia (resolved)"

    Called early in post_process to ensure status is correctly set even if
    MedGemma misses the "resolved" qualifier.
    """
    transcript_lower = transcript.lower()

    # Patterns that indicate a condition is resolved
    resolved_patterns = [
        r'\[discharge\s+diagnosis\]\s*([^,\[\]]+),?\s*resolved',
        r'diagnosis[:\s]+([^,\[\]]+),?\s*resolved',
        r'([^,\[\]]+)\s*\(resolved\)',
        r'([^,\[\]]+)\s*-\s*resolved',
        r'resolved\s+([^,\[\]]+)',
    ]

    resolved_conditions = set()
    for pattern in resolved_patterns:
        matches = re.finditer(pattern, transcript_lower)
        for match in matches:
            condition_name = match.group(1).strip()
            # Clean up the name (remove trailing punctuation)
            condition_name = re.sub(r'[\.\s]+$', '', condition_name)
            if condition_name:
                resolved_conditions.add(condition_name)
                print(f"[Status Detect] Found resolved condition in transcript: '{condition_name}'")

    if not resolved_conditions:
        return

    # Update matching conditions to resolved status
    for condition in entities.conditions:
        condition_lower = condition.name.lower().strip()
        for resolved_name in resolved_conditions:
            # Check for fuzzy match (resolved name contained in condition name or vice versa)
            if resolved_name in condition_lower or condition_lower in resolved_name:
                if condition.status != "resolved":
                    print(f"[Status Detect] Setting '{condition.name}' status to resolved (was: {condition.status})")
                    condition.status = "resolved"
                break


def merge_split_blood_pressure(vitals: list[Vital], transcript: str) -> tuple[list[Vital], Vital | None]:
    """
    Detect and merge split blood pressure values.

    MedGemma sometimes returns systolic and diastolic as separate entries:
      [{"value": 142, "unit": "mmHg"}, {"value": 88, "unit": "mmHg"}]

    This function detects this pattern and merges them into a single BP.

    Returns:
        Tuple of (non-BP vitals, merged BP vital or None)
    """
    non_bp_vitals = []
    potential_bp_values = []

    for v in vitals:
        value_str = str(v.value) if v.value else ""
        unit_lower = (v.unit or "").lower()

        # Already proper BP format - return as-is
        if "/" in value_str:
            return [vv for vv in vitals if vv != v], Vital(
                type="blood_pressure",
                value=value_str,
                unit="mmHg"
            )

        # Check if this looks like a BP component (mmHg in typical BP range)
        try:
            numeric_value = float(value_str.replace(',', ''))
            is_bp_unit = unit_lower in ("mmhg", "mm hg")
            is_bp_range = 40 <= numeric_value <= 220  # Expanded range for diastolic

            if is_bp_unit and is_bp_range:
                potential_bp_values.append(numeric_value)
            else:
                non_bp_vitals.append(v)
        except (ValueError, TypeError):
            non_bp_vitals.append(v)

    # If we have exactly 2 potential BP values, merge them
    if len(potential_bp_values) == 2:
        # Sort to get systolic (higher) and diastolic (lower)
        sorted_vals = sorted(potential_bp_values, reverse=True)
        systolic, diastolic = sorted_vals

        # Validate: systolic should be higher than diastolic by reasonable margin
        if systolic > diastolic and 20 <= (systolic - diastolic) <= 120:
            bp_value = f"{int(systolic)}/{int(diastolic)}"
            print(f"[BP Merge] Merged split BP values: {bp_value}")
            return non_bp_vitals, Vital(
                type="blood_pressure",
                value=bp_value,
                unit="mmHg"
            )

    # If we have potential BP values but couldn't merge, try transcript extraction
    if potential_bp_values:
        bp_from_transcript = extract_blood_pressure_from_transcript(transcript)
        if bp_from_transcript:
            print(f"[BP Merge] Using transcript BP instead of split values: {bp_from_transcript.value}")
            return non_bp_vitals, bp_from_transcript

        # Fallback: if we have exactly 1 value, keep it as partial BP
        if len(potential_bp_values) == 1:
            return non_bp_vitals, Vital(
                type="blood_pressure",
                value=str(int(potential_bp_values[0])),
                unit="mmHg"
            )

    return non_bp_vitals, None


def normalize_vitals(vitals: list[Vital], transcript: str) -> list[Vital]:
    """
    Normalize vitals to ensure proper types and formats.

    - Detects and merges split BP values (systolic/diastolic as separate entries)
    - Assigns types based on value ranges if missing
    - Ensures BP has proper format and type
    - Adds BP from transcript if missing
    """
    # First, try to detect and merge split blood pressure values
    non_bp_vitals, merged_bp = merge_split_blood_pressure(vitals, transcript)

    normalized = []
    has_bp = merged_bp is not None

    if merged_bp:
        normalized.append(merged_bp)

    for v in non_bp_vitals:
        vital = v  # May modify

        # Check if this looks like blood pressure
        value_str = str(v.value) if v.value else ""

        # If value contains "/" it's blood pressure
        if "/" in value_str:
            vital = Vital(
                type="blood_pressure",
                value=value_str,
                unit="mmHg"
            )
            has_bp = True
            print(f"[Vitals Normalize] Detected BP format: {value_str}")

        # If type is blood_pressure but value is single number, try to find full BP
        elif v.type and v.type.lower() in ("blood_pressure", "bp"):
            if not has_bp:
                # Try to find the full BP in transcript
                bp_from_transcript = extract_blood_pressure_from_transcript(transcript)
                if bp_from_transcript:
                    vital = bp_from_transcript
                    has_bp = True
                else:
                    # Keep as systolic-only if we can't find diastolic
                    vital = Vital(
                        type="blood_pressure",
                        value=value_str,
                        unit="mmHg"
                    )
                    has_bp = True
            else:
                # Already have BP, skip this duplicate
                continue

        # Infer type from unit if type is missing
        elif not v.type or is_placeholder(v.type):
            unit_lower = (v.unit or "").lower()
            try:
                numeric_value = float(value_str.replace(',', ''))

                # Infer type from unit
                if unit_lower in ("bpm", "beats per minute", "beats/min"):
                    vital = Vital(type="heart_rate", value=value_str, unit="bpm")
                elif unit_lower in ("f", "fahrenheit", "°f"):
                    vital = Vital(type="temperature", value=value_str, unit="F")
                elif unit_lower in ("c", "celsius", "°c"):
                    vital = Vital(type="temperature", value=value_str, unit="C")
                elif unit_lower in ("lbs", "lb", "pounds"):
                    vital = Vital(type="weight", value=value_str, unit="lbs")
                elif unit_lower in ("kg", "kilograms"):
                    vital = Vital(type="weight", value=value_str, unit="kg")
                elif unit_lower in ("%", "percent"):
                    vital = Vital(type="oxygen_saturation", value=value_str, unit="%")
                elif unit_lower in ("breaths/min", "breaths per minute", "/min"):
                    vital = Vital(type="respiratory_rate", value=value_str, unit="breaths/min")
                elif unit_lower in ("mmhg", "mm hg"):
                    # mmHg already handled by merge_split_blood_pressure
                    # This shouldn't normally be reached, but handle edge cases
                    if not has_bp and 70 <= numeric_value <= 200:
                        vital = Vital(type="blood_pressure", value=value_str, unit="mmHg")
                    else:
                        continue  # Skip - likely a duplicate BP component
                else:
                    # Infer from value ranges
                    if 90 <= numeric_value <= 110:
                        vital = Vital(type="temperature", value=value_str, unit="F")
                    elif 140 <= numeric_value <= 300:
                        vital = Vital(type="weight", value=value_str, unit="lbs")
                    elif 30 <= numeric_value <= 180:
                        vital = Vital(type="heart_rate", value=value_str, unit="bpm")

            except (ValueError, TypeError):
                pass  # Keep original if can't parse

        normalized.append(vital)

    # If no BP found in vitals, try to extract from transcript
    if not has_bp:
        bp_from_transcript = extract_blood_pressure_from_transcript(transcript)
        if bp_from_transcript:
            normalized.append(bp_from_transcript)
            print(f"[Vitals Normalize] Added BP from transcript: {bp_from_transcript.value}")

    return normalized


def filter_vitals(vitals: list[Vital]) -> list[Vital]:
    """Filter out vitals with placeholder or invalid values."""
    filtered = []
    for v in vitals:
        # Skip if value is placeholder
        if is_placeholder(v.value):
            continue
        # Skip if value is "not mentioned" or similar string placeholder
        if isinstance(v.value, str) and v.value.lower() in PLACEHOLDER_VALUES:
            continue
        # Check if value is valid (numeric or BP format)
        if isinstance(v.value, str) and not is_valid_vital_value(str(v.value), v.type or ""):
            continue
        filtered.append(v)
    return filtered


def filter_allergies(allergies: list[Allergy]) -> list[Allergy]:
    """Filter out allergies with placeholder substances."""
    return [a for a in allergies if not is_placeholder(a.substance)]


def filter_medications(medications: list[Medication]) -> list[Medication]:
    """Filter out medications with placeholder names."""
    return [m for m in medications if not is_placeholder(m.name)]


# Patterns for extracting medication dosages from transcript
MEDICATION_DOSE_PATTERNS = [
    # "metformin 1000 mg twice daily" - common drug suffixes
    r'(\b\w*(?:cillin|pril|statin|formin|olol|sartan|pam|zole|ine|ide|mab|nib)\b)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)\s*(?:(once|twice|three times|four times|every \d+ hours?|daily|bid|tid|qid|prn|as needed|per day))?',
    # "aspirin 81 mg daily" - common OTC meds
    r'(aspirin|ibuprofen|acetaminophen|tylenol|advil|motrin)\s+(\d+(?:\.\d+)?)\s*(mg|g)\s*(?:(daily|twice daily|as needed|prn|every \d+ hours?))?',
    # "amoxicillin 500 mg three times daily for 7 days"
    r'(amoxicillin|azithromycin|ciprofloxacin|levofloxacin|doxycycline|cephalexin|augmentin)\s+(\d+(?:\.\d+)?)\s*(mg|g)\s*(?:(once|twice|three times|four times|every \d+ hours?|daily|bid|tid|qid))?(?:\s+(?:daily|a day|per day))?(?:\s+for\s+\d+\s*days?)?',
    # Generic: "start/continue/prescribe DRUG DOSE UNIT FREQUENCY"
    r'(?:start|continue|prescribe|give|administer|take)\s+(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg|units?)\s*(?:(once|twice|three times|every \d+ hours?|daily|bid|tid|qid))?',
    # "DRUG DOSE UNIT FREQUENCY" - standalone pattern
    r'\b([a-z]+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)\s+(once|twice|three times|daily|bid|tid|qid|every \d+ hours?)',
]


def extract_medication_dosages_from_transcript(
    medications: list[Medication],
    transcript: str
) -> list[Medication]:
    """
    Fill in missing dose/frequency for medications by parsing transcript.

    For each medication with null dose, search the transcript for patterns
    that match the medication name followed by dosage information.

    Args:
        medications: List of medications (may have null dose/frequency)
        transcript: The transcript text to search

    Returns:
        Updated list of medications with doses filled in where found
    """
    if not medications or not transcript:
        return medications

    transcript_lower = transcript.lower()
    print(f"[MedDose DEBUG] Processing {len(medications)} medications")
    print(f"[MedDose DEBUG] Transcript preview: {transcript_lower[:200]}...")

    for med in medications:
        print(f"[MedDose DEBUG] Checking '{med.name}' - current dose: {med.dose}")
        # Skip if already has dose
        if med.dose:
            print(f"[MedDose DEBUG]   -> Skipping (already has dose)")
            continue

        med_name_lower = med.name.lower()

        # Try each pattern
        for i, pattern in enumerate(MEDICATION_DOSE_PATTERNS):
            matches = list(re.finditer(pattern, transcript_lower, re.IGNORECASE))
            if matches:
                print(f"[MedDose DEBUG]   Pattern {i} found {len(matches)} matches")
            for match in matches:
                # Check if this match is for our medication
                matched_drug = match.group(1)
                print(f"[MedDose DEBUG]   Matched drug: '{matched_drug}' vs med: '{med_name_lower}'")
                if matched_drug.lower() in med_name_lower or med_name_lower in matched_drug.lower():
                    # Extract dose
                    dose_value = match.group(2)
                    dose_unit = match.group(3)
                    med.dose = f"{dose_value} {dose_unit}"
                    print(f"[MedDose DEBUG]   -> MATCHED! Setting dose to: {med.dose}")

                    # Extract frequency if captured
                    if len(match.groups()) >= 4 and match.group(4):
                        freq = match.group(4).strip()
                        # Normalize frequency
                        freq_map = {
                            'bid': 'twice daily',
                            'tid': 'three times daily',
                            'qid': 'four times daily',
                            'once': 'once daily',
                            'twice': 'twice daily',
                            'three times': 'three times daily',
                            'four times': 'four times daily',
                        }
                        med.frequency = freq_map.get(freq.lower(), freq)

                    print(f"[Post-process] Extracted dosage for {med.name}: {med.dose} {med.frequency or ''}")
                    break

            if med.dose:
                break

    return medications


def extract_plan_medication_orders(
    transcript: str,
    existing_orders: list[MedicationOrder]
) -> list[MedicationOrder]:
    """
    Extract medication orders from [PLAN] section of SOAP notes.

    Parses the Plan section for new medication prescriptions and adds them
    to medication_orders if not already present.

    Args:
        transcript: The transcript text (may contain [PLAN] section)
        existing_orders: Current medication orders (for deduplication)

    Returns:
        Updated list of medication orders
    """
    # Extract PLAN section
    plan_text = extract_section(transcript, "PLAN")
    print(f"[PLAN Extract DEBUG] Looking for [PLAN] in transcript...")
    print(f"[PLAN Extract DEBUG] Found PLAN text: '{plan_text[:100] if plan_text else 'NONE'}...'")
    if not plan_text:
        print(f"[PLAN Extract DEBUG] No PLAN section found, returning existing orders")
        return existing_orders

    print(f"[Post-process] Extracting medication orders from PLAN section...")

    # Patterns for plan medications
    plan_med_patterns = [
        # "Amoxicillin 500 mg three times daily for 10 days" - standalone med at start of PLAN
        r'^([A-Za-z]+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)\s+(once daily|twice daily|three times daily|four times daily|once|twice|three times|four times|daily|every \d+ hours?)(?:\s+for\s+(\d+)\s*days?)?',
        # "Start amoxicillin 500 mg three times daily for 7 days"
        r'(?:start|prescribe|give|order|initiate|continue)\s+(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)\s*(?:(once daily|twice daily|three times daily|four times daily|once|twice|three times|every \d+ hours?|daily|bid|tid|qid))?(?:\s+(?:daily|a day|per day))?(?:\s+for\s+(\d+)\s*days?)?',
        # "Amoxicillin 500 mg TID x 7 days"
        r'\b([a-z]+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml)\s*(bid|tid|qid|daily)(?:\s*(?:x|for)\s*(\d+)\s*days?)?',
        # "prescribing penicillin 500 mg"
        r'(?:prescribing|ordering|starting)\s+(\w+(?:\s+\w+)?)\s+(\d+(?:\.\d+)?)\s*(mg|g)',
    ]

    # Get existing order names for deduplication
    existing_names = {o.name.lower().strip() for o in existing_orders}

    new_orders = list(existing_orders)

    for i, pattern in enumerate(plan_med_patterns):
        matches = list(re.finditer(pattern, plan_text, re.IGNORECASE))
        print(f"[PLAN Extract DEBUG] Pattern {i}: {len(matches)} matches")
        for match in matches:
            print(f"[PLAN Extract DEBUG]   Match: '{match.group(0)}'")
            drug_name = match.group(1)

            # Skip if already exists
            if drug_name.lower().strip() in existing_names:
                continue

            # Build order
            dose_value = match.group(2) if len(match.groups()) >= 2 else None
            dose_unit = match.group(3) if len(match.groups()) >= 3 else None
            frequency = match.group(4) if len(match.groups()) >= 4 else None

            dose = f"{dose_value} {dose_unit}" if dose_value and dose_unit else None

            # Normalize frequency
            if frequency:
                freq_map = {
                    'bid': 'twice daily',
                    'tid': 'three times daily',
                    'qid': 'four times daily',
                    'daily': 'once daily',
                }
                frequency = freq_map.get(frequency.lower(), frequency)

            # Build instructions
            instructions = None
            if len(match.groups()) >= 5 and match.group(5):
                instructions = f"for {match.group(5)} days"

            order = MedicationOrder(
                name=drug_name.capitalize(),
                dose=dose,
                frequency=frequency,
                instructions=instructions
            )

            new_orders.append(order)
            existing_names.add(drug_name.lower().strip())
            print(f"[Post-process] Added medication order from PLAN: {order.name} {order.dose or ''}")

    return new_orders


def filter_medication_orders(orders: list[MedicationOrder]) -> list[MedicationOrder]:
    """Filter out medication orders that aren't actually medications."""
    non_medication_patterns = [
        r'^await\s',
        r'^resume\s',
        r'^avoid\s',
        r'^continue\s',
        r'^stop\s',
        r'^follow\s*up',
        r'^return\s',
        r'^initiate\s',
        r'^proceed\s',
        r'pathology',
        r'diet',
        r'driving',
        r'activity',
        r'\beeg\b',           # EEG is a diagnostic, not medication
        r'\bmri\b',           # MRI is imaging
        r'\bct\b',            # CT is imaging
        r'\bx-?ray\b',        # X-ray is imaging
        r'\bultrasound\b',    # Ultrasound is imaging
        r'breathing\s+trial', # Spontaneous breathing trial
        r'counseling',        # Counseling is not a medication
    ]

    filtered = []
    seen_names = set()  # For deduplication
    for mo in orders:
        if is_placeholder(mo.name):
            continue
        # Check if it matches non-medication patterns
        is_non_med = False
        for pattern in non_medication_patterns:
            if re.search(pattern, mo.name, re.IGNORECASE):
                is_non_med = True
                break
        if is_non_med:
            continue
        # Deduplicate by lowercase name
        name_key = mo.name.lower().strip()
        if name_key in seen_names:
            continue
        seen_names.add(name_key)
        filtered.append(mo)
    return filtered


def filter_conditions(conditions: list[Condition]) -> list[Condition]:
    """Filter out conditions with placeholder names."""
    return [c for c in conditions if not is_placeholder(c.name)]


def filter_referral_orders(orders: list) -> list:
    """Filter out referral orders with placeholder or invalid data."""
    filtered = []
    for ro in orders:
        # Skip if specialty is placeholder
        if hasattr(ro, 'specialty') and is_placeholder(ro.specialty):
            continue
        # Skip if reason is "null" string
        if hasattr(ro, 'reason') and isinstance(ro.reason, str) and ro.reason.lower() == "null":
            ro.reason = None  # Convert "null" string to actual None
        filtered.append(ro)
    return filtered


def filter_lab_results(lab_results: list) -> list:
    """Filter out lab results with placeholder values."""
    filtered = []
    for lr in lab_results:
        # Skip if value is a placeholder like "not mentioned"
        if hasattr(lr, 'value') and is_placeholder(lr.value):
            continue
        filtered.append(lr)
    return filtered


def null_to_none(value: Any) -> Any:
    """Convert string 'null' to actual None."""
    if isinstance(value, str) and value.lower() == "null":
        return None
    return value


def clean_null_strings(entities: ClinicalEntities) -> None:
    """
    Convert all 'null' strings to actual None across all entity fields.

    MedGemma sometimes returns "null" as a string instead of actual null.
    This function cleans up those values in-place.
    """
    # Clean medications
    for med in entities.medications:
        med.dose = null_to_none(med.dose)
        med.frequency = null_to_none(med.frequency)
        med.route = null_to_none(med.route)

    # Clean medication orders
    for order in entities.medication_orders:
        order.dose = null_to_none(order.dose)
        order.frequency = null_to_none(order.frequency)
        if hasattr(order, 'route'):
            order.route = null_to_none(order.route)
        if hasattr(order, 'duration'):
            order.duration = null_to_none(order.duration)
        if hasattr(order, 'instructions'):
            order.instructions = null_to_none(order.instructions)

    # Clean conditions
    for cond in entities.conditions:
        cond.severity = null_to_none(cond.severity)
        cond.status = null_to_none(cond.status)

    # Clean vitals
    for vital in entities.vitals:
        vital.unit = null_to_none(vital.unit)

    # Clean allergies
    for allergy in entities.allergies:
        allergy.reaction = null_to_none(allergy.reaction)
        allergy.severity = null_to_none(allergy.severity)

    # Clean lab results
    for lab in entities.lab_results:
        lab.unit = null_to_none(lab.unit)
        lab.interpretation = null_to_none(lab.interpretation)

    # Clean family history
    for fh in entities.family_history:
        fh.age_of_onset = null_to_none(fh.age_of_onset)


def clean_social_history(sh: SocialHistory | None) -> SocialHistory | None:
    """Clean social history by converting 'null' strings to None."""
    if sh is None:
        return None

    # Convert "null" strings to actual None
    sh.tobacco = null_to_none(sh.tobacco)
    sh.alcohol = null_to_none(sh.alcohol)
    sh.drugs = null_to_none(sh.drugs)
    sh.occupation = null_to_none(sh.occupation)

    # Only return if we have any actual data
    if any([sh.tobacco, sh.alcohol, sh.drugs, sh.occupation, sh.living_situation]):
        return sh
    return None


def post_process(entities: ClinicalEntities, transcript: str) -> ClinicalEntities:
    """
    Apply all post-processing to extracted entities.

    1. Extract missing data from transcript markers
    2. Filter out placeholder values
    3. Validate and clean up entries

    Args:
        entities: The extracted clinical entities
        transcript: The original transcript text

    Returns:
        Enhanced and filtered ClinicalEntities
    """
    print(f"[Post-process DEBUG] Starting post_process...")
    print(f"[Post-process DEBUG] Transcript has [CHIEF COMPLAINT]: {'[CHIEF COMPLAINT]' in transcript}")
    print(f"[Post-process DEBUG] Transcript has [FAMILY HISTORY]: {'[FAMILY HISTORY]' in transcript}")

    # Store original transcript for marker extraction
    entities.raw_transcript = transcript

    # 0. CRITICAL: Extract blood pressure directly from transcript FIRST
    # This simple approach bypasses complex normalization and ensures BP is correct
    ensure_blood_pressure_from_transcript(entities, transcript)

    # 0.5: Detect resolved status from transcript (for discharge summaries)
    detect_resolved_status_from_transcript(entities, transcript)

    # 1. Extract chief complaint from markers if missing
    chief_complaint = extract_chief_complaint(transcript, entities)
    print(f"[Post-process DEBUG] Extracted chief complaint: '{chief_complaint}'")
    if chief_complaint:
        # Store the chief complaint text separately (reason for visit / presenting symptoms)
        entities.chief_complaint_text = chief_complaint

        # Mark first matching condition as chief complaint
        # DO NOT add symptoms as conditions - they're not diagnoses
        found = False
        for condition in entities.conditions:
            if condition.name.lower() == chief_complaint.lower():
                condition.is_chief_complaint = True
                found = True
                print(f"[Post-process DEBUG] Matched chief complaint to condition: '{condition.name}'")
                break
        if not found and entities.conditions:
            # No exact match - mark the first/primary condition as chief complaint
            # This preserves the chief_complaint property without adding symptoms as conditions
            entities.conditions[0].is_chief_complaint = True
            print(f"[Post-process DEBUG] No match - marked first condition as CC: '{entities.conditions[0].name}'")

    # 2. Extract family history from markers
    entities.family_history = extract_family_history(transcript, entities)
    print(f"[Post-process DEBUG] Family history extracted: {len(entities.family_history)} items")

    # 3. Extract social history from markers
    extracted_sh = extract_social_history(transcript, entities)
    if extracted_sh:
        entities.social_history = extracted_sh
        print(f"[Post-process DEBUG] Social history: tobacco={extracted_sh.tobacco}, occupation={extracted_sh.occupation}")

    # 4. Normalize and enhance vitals (extract BP from transcript if missing)
    entities.vitals = normalize_vitals(entities.vitals, transcript)
    print(f"[Post-process DEBUG] Vitals after normalization: {len(entities.vitals)} items")
    for v in entities.vitals:
        print(f"[Post-process DEBUG]   - {v.type}: {v.value} {v.unit}")

    # 5. Filter out placeholder values and invalid data
    entities.conditions = filter_conditions(entities.conditions)
    entities.vitals = filter_vitals(entities.vitals)
    entities.allergies = filter_allergies(entities.allergies)
    entities.medications = filter_medications(entities.medications)
    entities.medication_orders = filter_medication_orders(entities.medication_orders)
    entities.referral_orders = filter_referral_orders(entities.referral_orders)
    entities.lab_results = filter_lab_results(entities.lab_results)

    # 5.1. Clean "null" strings to actual None values across all entities
    clean_null_strings(entities)

    # 5.5. Extract medication dosages from transcript for meds with null dose
    entities.medications = extract_medication_dosages_from_transcript(entities.medications, transcript)
    print(f"[Post-process DEBUG] Medication dosages extracted for {sum(1 for m in entities.medications if m.dose)}/{len(entities.medications)} medications")

    # 5.6. Extract medication orders from [PLAN] section (for SOAP notes)
    entities.medication_orders = extract_plan_medication_orders(transcript, entities.medication_orders)
    print(f"[Post-process DEBUG] Medication orders after PLAN extraction: {len(entities.medication_orders)}")

    # 6. Clean social history (convert "null" strings to None)
    entities.social_history = clean_social_history(entities.social_history)

    # 7. Enrich conditions with verified ICD-10 codes from lookup database
    entities.conditions = enrich_conditions_with_icd10(entities.conditions)
    icd_coded = sum(1 for c in entities.conditions if c.icd10)
    print(f"[Post-process DEBUG] ICD-10 codes added: {icd_coded}/{len(entities.conditions)} conditions")

    return entities
