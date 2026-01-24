"""
Post-processor for extracted clinical entities.

This module provides:
1. Transcript marker extraction - fills in missing fields from bracketed sections
2. Validation/filtering - removes placeholder values and invalid entries
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


def extract_section(transcript: str, marker: str) -> str | None:
    """
    Extract content following a bracketed marker until the next marker or end.

    Example: extract_section(text, "CHIEF COMPLAINT")
    Returns content after [CHIEF COMPLAINT] until next [MARKER] or end.
    """
    # Pattern: [MARKER] followed by content until next [SOMETHING] or end
    pattern = rf'\[{re.escape(marker)}\]\s*(?:is\s+)?(.+?)(?=\[|$)'
    match = re.search(pattern, transcript, re.IGNORECASE | re.DOTALL)
    if match:
        content = match.group(1).strip()
        # Clean up: remove trailing punctuation and whitespace
        content = re.sub(r'[\.\s]+$', '', content)
        return content if content and not is_placeholder(content) else None
    return None


def extract_chief_complaint(transcript: str, entities: ClinicalEntities) -> str | None:
    """
    Extract chief complaint from transcript markers.

    Looks for patterns like:
    - [CHIEF COMPLAINT] ...
    - Chief complaint is ...
    - Patient presents with ...
    - CC: ...
    """
    # Check if we already have a valid chief complaint
    if entities.chief_complaint and not is_placeholder(entities.chief_complaint.name):
        return entities.chief_complaint.name

    # Try bracketed marker first
    cc = extract_section(transcript, "CHIEF COMPLAINT")
    if cc:
        return cc

    # Try "CC:" pattern
    match = re.search(r'(?:^|\s)CC:\s*(.+?)(?:\.|$)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip()
        if not is_placeholder(cc):
            return cc

    # Try "presents with" pattern
    match = re.search(r'presents?\s+with\s+(.+?)(?:\.|,|for)', transcript, re.IGNORECASE)
    if match:
        cc = match.group(1).strip()
        if not is_placeholder(cc):
            return cc

    # Fallback: first condition if marked as chief complaint
    for condition in entities.conditions:
        if condition.is_chief_complaint:
            return condition.name

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
    if fh_section:
        # Parse common patterns
        # Pattern: "relationship had/has condition (at age X)"
        patterns = [
            # "Father had myocardial infarction at age 65"
            r'(mother|father|brother|sister|sibling|parent|grandmother|grandfather|grandparent|aunt|uncle|cousin)\s+(?:had|has|with|died\s+(?:of|from))\s+([^,\.]+?)(?:\s+at\s+age\s+(\d+))?(?:,|\.|$)',
            # "family history of CAD in father"
            r'(?:family\s+history\s+(?:of|significant\s+for))\s+([^,\.]+?)\s+in\s+(mother|father|brother|sister|sibling|parent)',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, fh_section, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    # First pattern: relationship, condition, age
                    if groups[0].lower() in ('mother', 'father', 'brother', 'sister', 'sibling', 'parent', 'grandmother', 'grandfather', 'grandparent', 'aunt', 'uncle', 'cousin'):
                        relationship = groups[0]
                        condition = groups[1]
                        age = groups[2] if len(groups) > 2 else None
                    else:
                        # Second pattern: condition, relationship
                        condition = groups[0]
                        relationship = groups[1]
                        age = None

                    key = (relationship.lower(), condition.lower().strip())
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
        occupation_patterns = [
            r'(?:works?\s+as\s+(?:a|an)?\s*)([^,\.]+)',
            r'occupation[:\s]+([^,\.]+)',
            r'(?:retired|former)\s+([^,\.]+)',
            r'([^,\.]+)\s+by\s+(?:profession|occupation)',
        ]
        for pattern in occupation_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                occupation = match.group(1).strip()
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


def filter_vitals(vitals: list[Vital]) -> list[Vital]:
    """Filter out vitals with placeholder or invalid values."""
    filtered = []
    for v in vitals:
        # Skip if value is placeholder
        if is_placeholder(v.value):
            continue
        # Skip if value is "not mentioned" or similar
        if isinstance(v.value, str) and v.value.lower() in PLACEHOLDER_VALUES:
            continue
        filtered.append(v)
    return filtered


def filter_allergies(allergies: list[Allergy]) -> list[Allergy]:
    """Filter out allergies with placeholder substances."""
    return [a for a in allergies if not is_placeholder(a.substance)]


def filter_medications(medications: list[Medication]) -> list[Medication]:
    """Filter out medications with placeholder names."""
    return [m for m in medications if not is_placeholder(m.name)]


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
        r'pathology',
        r'diet',
        r'driving',
        r'activity',
    ]

    filtered = []
    for mo in orders:
        if is_placeholder(mo.name):
            continue
        # Check if it matches non-medication patterns
        is_non_med = False
        for pattern in non_medication_patterns:
            if re.search(pattern, mo.name, re.IGNORECASE):
                is_non_med = True
                break
        if not is_non_med:
            filtered.append(mo)
    return filtered


def filter_conditions(conditions: list[Condition]) -> list[Condition]:
    """Filter out conditions with placeholder names."""
    return [c for c in conditions if not is_placeholder(c.name)]


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
    # Store original transcript for marker extraction
    entities.raw_transcript = transcript

    # 1. Extract chief complaint from markers if missing
    chief_complaint = extract_chief_complaint(transcript, entities)
    if chief_complaint:
        # Mark first matching condition as chief complaint, or add new one
        found = False
        for condition in entities.conditions:
            if condition.name.lower() == chief_complaint.lower():
                condition.is_chief_complaint = True
                found = True
                break
        if not found:
            # Add as a new condition marked as chief complaint
            entities.conditions.insert(0, Condition(
                name=chief_complaint,
                is_chief_complaint=True,
                status="active"
            ))

    # 2. Extract family history from markers
    entities.family_history = extract_family_history(transcript, entities)

    # 3. Extract social history from markers
    extracted_sh = extract_social_history(transcript, entities)
    if extracted_sh:
        entities.social_history = extracted_sh

    # 4. Filter out placeholder values
    entities.conditions = filter_conditions(entities.conditions)
    entities.vitals = filter_vitals(entities.vitals)
    entities.allergies = filter_allergies(entities.allergies)
    entities.medications = filter_medications(entities.medications)
    entities.medication_orders = filter_medication_orders(entities.medication_orders)

    return entities
