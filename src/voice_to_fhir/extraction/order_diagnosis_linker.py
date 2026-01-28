"""
Order-Diagnosis Linker for Clinical Entity Association.

This module provides automatic linking of all order types to diagnoses
based on clinical rules and patient conditions.

Supported Order Types:
    - Medication Orders: Drug class → typical indications
    - Lab Orders: Test name → monitoring indications
    - Consult/Referral Orders: Specialty → typical conditions
    - Procedure Orders: Procedure name → typical indications

Flow:
    Order -> Name/Class Lookup -> Clinical Rules -> ICD-10 Match
                               -> Patient Conditions Match
"""

from typing import NamedTuple


class DiagnosisLink(NamedTuple):
    """A linked diagnosis for an order."""
    icd10: str
    display: str
    confidence: float
    method: str  # 'rule', 'patient_condition', 'manual'


# =============================================================================
# Medication-to-Diagnosis Clinical Rules
# Maps drug classes to their typical indications (ICD-10 codes)
# =============================================================================

MEDICATION_DIAGNOSIS_RULES: dict[str, list[tuple[str, str, float]]] = {
    # Drug class -> list of (ICD-10, display, confidence)
    # Higher confidence = more common/primary indication

    # Cardiovascular - Lipid Management
    "statin": [
        ("E78.5", "Hyperlipidemia, unspecified", 0.95),
        ("E78.00", "Pure hypercholesterolemia", 0.90),
        ("I25.10", "Atherosclerotic heart disease", 0.80),
    ],

    # Cardiovascular - Blood Pressure
    "ace_inhibitor": [
        ("I10", "Essential hypertension", 0.90),
        ("I50.9", "Heart failure, unspecified", 0.85),
        ("N18.9", "Chronic kidney disease, unspecified", 0.75),
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.70),  # Diabetic nephropathy prevention
    ],
    "arb": [
        ("I10", "Essential hypertension", 0.90),
        ("I50.9", "Heart failure, unspecified", 0.85),
        ("N18.9", "Chronic kidney disease, unspecified", 0.75),
    ],
    "beta_blocker": [
        ("I10", "Essential hypertension", 0.85),
        ("I48.91", "Unspecified atrial fibrillation", 0.85),
        ("I50.9", "Heart failure, unspecified", 0.85),
        ("I25.10", "Atherosclerotic heart disease", 0.80),
        ("I20.9", "Angina pectoris, unspecified", 0.75),
    ],
    "ccb": [
        ("I10", "Essential hypertension", 0.90),
        ("I20.9", "Angina pectoris, unspecified", 0.80),
    ],
    "diuretic": [
        ("I10", "Essential hypertension", 0.85),
        ("I50.9", "Heart failure, unspecified", 0.90),
        ("R60.9", "Edema, unspecified", 0.75),
    ],

    # Anticoagulants
    "anticoagulant": [
        ("I48.91", "Unspecified atrial fibrillation", 0.90),
        ("I82.40", "Deep vein thrombosis of unspecified deep veins of lower extremity", 0.85),
        ("I26.99", "Other pulmonary embolism without acute cor pulmonale", 0.85),
    ],
    "antiplatelet": [
        ("I25.10", "Atherosclerotic heart disease", 0.85),
        ("I63.9", "Cerebral infarction, unspecified", 0.80),  # Stroke prevention
        ("I21.9", "Acute myocardial infarction, unspecified", 0.80),
    ],

    # Diabetes
    "diabetes_oral": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.95),
        ("E11.65", "Type 2 diabetes mellitus with hyperglycemia", 0.90),
    ],
    "glp1": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.90),
        ("E66.9", "Obesity, unspecified", 0.85),  # Weight management indication
    ],
    "insulin": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.85),
        ("E10.9", "Type 1 diabetes mellitus without complications", 0.85),
    ],

    # Gastrointestinal
    "ppi": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.90),
        ("K21.9", "Gastro-esophageal reflux disease without esophagitis", 0.85),
        ("K27.9", "Peptic ulcer, site unspecified", 0.80),
    ],
    "h2_blocker": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.85),
        ("K29.70", "Gastritis, unspecified, without bleeding", 0.80),
    ],

    # Antibiotics - typically linked to active infections
    "antibiotic_penicillin": [
        ("J06.9", "Acute upper respiratory infection, unspecified", 0.80),
        ("J02.9", "Acute pharyngitis, unspecified", 0.80),
        ("J18.9", "Pneumonia, unspecified organism", 0.75),
        ("L03.90", "Cellulitis, unspecified", 0.75),
    ],
    "antibiotic_cephalosporin": [
        ("J18.9", "Pneumonia, unspecified organism", 0.80),
        ("N39.0", "Urinary tract infection, site not specified", 0.80),
        ("L03.90", "Cellulitis, unspecified", 0.75),
    ],
    "antibiotic_fluoroquinolone": [
        ("N39.0", "Urinary tract infection, site not specified", 0.85),
        ("J18.9", "Pneumonia, unspecified organism", 0.80),
    ],
    "antibiotic_macrolide": [
        ("J06.9", "Acute upper respiratory infection, unspecified", 0.85),
        ("J18.9", "Pneumonia, unspecified organism", 0.85),
        ("J20.9", "Acute bronchitis, unspecified", 0.80),
    ],
    "antibiotic_tetracycline": [
        ("J18.9", "Pneumonia, unspecified organism", 0.75),
        ("L70.9", "Acne, unspecified", 0.80),  # Doxycycline for acne
    ],
    "antibiotic_other": [
        ("N39.0", "Urinary tract infection, site not specified", 0.80),
        ("A04.72", "Clostridium difficile colitis", 0.75),  # Metronidazole for C. diff
    ],

    # Psychiatric
    "ssri": [
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.90),
        ("F41.9", "Anxiety disorder, unspecified", 0.85),
    ],
    "snri": [
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.90),
        ("G89.29", "Other chronic pain", 0.75),  # Duloxetine for chronic pain
    ],
    "antidepressant_other": [
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.85),
    ],
    "tca": [
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.80),
        ("G89.29", "Other chronic pain", 0.75),
    ],
    "benzodiazepine": [
        ("F41.9", "Anxiety disorder, unspecified", 0.85),
        ("G47.00", "Insomnia, unspecified", 0.80),
        ("R56.9", "Unspecified convulsions", 0.75),
    ],
    "antipsychotic": [
        ("F20.9", "Schizophrenia, unspecified", 0.80),
        ("F31.9", "Bipolar disorder, unspecified", 0.80),
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.70),  # Augmentation
    ],
    "adhd": [
        ("F90.9", "Attention-deficit hyperactivity disorder, unspecified type", 0.95),
    ],

    # Pain Management
    "nsaid": [
        ("M25.50", "Pain in unspecified joint", 0.80),
        ("M54.50", "Low back pain, unspecified", 0.80),
        ("M19.90", "Unspecified osteoarthritis, unspecified site", 0.75),
    ],
    "analgesic": [
        ("R52", "Pain, unspecified", 0.85),
        ("R50.9", "Fever, unspecified", 0.80),
    ],
    "opioid": [
        ("G89.29", "Other chronic pain", 0.85),
        ("R52", "Pain, unspecified", 0.80),
    ],
    "muscle_relaxant": [
        ("M62.89", "Other specified disorders of muscle", 0.80),
        ("M54.50", "Low back pain, unspecified", 0.85),
    ],

    # Respiratory
    "bronchodilator": [
        ("J45.909", "Unspecified asthma, uncomplicated", 0.90),
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.90),
    ],
    "inhaled_steroid": [
        ("J45.909", "Unspecified asthma, uncomplicated", 0.90),
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.85),
    ],
    "inhaler_combo": [
        ("J45.909", "Unspecified asthma, uncomplicated", 0.90),
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.90),
    ],
    "anticholinergic": [
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.90),
    ],
    "leukotriene_inhibitor": [
        ("J45.909", "Unspecified asthma, uncomplicated", 0.90),
        ("J30.9", "Allergic rhinitis, unspecified", 0.75),
    ],

    # Thyroid
    "thyroid": [
        ("E03.9", "Hypothyroidism, unspecified", 0.95),
    ],
    "antithyroid": [
        ("E05.90", "Thyrotoxicosis, unspecified without thyrotoxic crisis", 0.95),
    ],

    # Corticosteroids
    "corticosteroid": [
        ("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation", 0.80),
        ("J45.901", "Unspecified asthma with acute exacerbation", 0.80),
        ("M06.9", "Rheumatoid arthritis, unspecified", 0.75),
    ],

    # Anticonvulsants
    "anticonvulsant": [
        ("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus", 0.85),
        ("G89.29", "Other chronic pain", 0.75),  # Gabapentin/pregabalin for neuropathic pain
        ("G62.9", "Polyneuropathy, unspecified", 0.75),
    ],

    # Antiemetics
    "antiemetic": [
        ("R11.2", "Nausea with vomiting, unspecified", 0.90),
        ("R11.0", "Nausea", 0.85),
    ],

    # Sleep
    "sleep_aid": [
        ("G47.00", "Insomnia, unspecified", 0.95),
    ],

    # Other
    "antihistamine": [
        ("J30.9", "Allergic rhinitis, unspecified", 0.85),
        ("L50.9", "Urticaria, unspecified", 0.80),
    ],
    "laxative": [
        ("K59.00", "Constipation, unspecified", 0.95),
    ],
    "alpha_blocker": [
        ("N40.0", "Benign prostatic hyperplasia without lower urinary tract symptoms", 0.90),
    ],
    "5ari": [
        ("N40.0", "Benign prostatic hyperplasia without lower urinary tract symptoms", 0.90),
    ],
    "pde5_inhibitor": [
        ("N52.9", "Male erectile dysfunction, unspecified", 0.90),
    ],
    "bisphosphonate": [
        ("M81.0", "Age-related osteoporosis without current pathological fracture", 0.95),
    ],
    "gout": [
        ("M10.9", "Gout, unspecified", 0.95),
        ("E79.0", "Hyperuricemia", 0.85),
    ],
    "electrolyte": [
        ("E87.6", "Hypokalemia", 0.80),
    ],
    "iron_supplement": [
        ("D50.9", "Iron deficiency anemia, unspecified", 0.90),
    ],
}


# =============================================================================
# Lab Order-to-Diagnosis Clinical Rules
# Maps lab test names to their typical monitoring indications (ICD-10 codes)
# =============================================================================

LAB_DIAGNOSIS_RULES: dict[str, list[tuple[str, str, float]]] = {
    # Lab name (lowercase, normalized) -> list of (ICD-10, display, confidence)

    # Metabolic Panels
    "bmp": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.80),
        ("N18.9", "Chronic kidney disease, unspecified", 0.80),
        ("E87.6", "Hypokalemia", 0.75),
    ],
    "basic metabolic panel": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.80),
        ("N18.9", "Chronic kidney disease, unspecified", 0.80),
        ("E87.6", "Hypokalemia", 0.75),
    ],
    "cmp": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.80),
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.75),
        ("N18.9", "Chronic kidney disease, unspecified", 0.75),
    ],
    "comprehensive metabolic panel": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.80),
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.75),
        ("N18.9", "Chronic kidney disease, unspecified", 0.75),
    ],

    # Lipid Panel
    "lipid panel": [
        ("E78.5", "Hyperlipidemia, unspecified", 0.90),
        ("I25.10", "Atherosclerotic heart disease", 0.80),
    ],
    "lipids": [
        ("E78.5", "Hyperlipidemia, unspecified", 0.90),
    ],
    "cholesterol": [
        ("E78.5", "Hyperlipidemia, unspecified", 0.90),
    ],

    # Diabetes Monitoring
    "hba1c": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.95),
    ],
    "hemoglobin a1c": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.95),
    ],
    "a1c": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.95),
    ],
    "glucose": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.85),
        ("R73.09", "Other abnormal glucose", 0.75),
    ],
    "fasting glucose": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.90),
    ],

    # Hematology
    "cbc": [
        ("D64.9", "Anemia, unspecified", 0.80),
        ("Z23", "Encounter for immunization", 0.70),  # Routine screening
    ],
    "complete blood count": [
        ("D64.9", "Anemia, unspecified", 0.80),
    ],
    "cbc with diff": [
        ("D64.9", "Anemia, unspecified", 0.80),
        ("D72.9", "Disorder of white blood cells, unspecified", 0.75),
    ],

    # Coagulation
    "pt/inr": [
        ("Z79.01", "Long term (current) use of anticoagulants", 0.90),
        ("I48.91", "Unspecified atrial fibrillation", 0.85),
    ],
    "inr": [
        ("Z79.01", "Long term (current) use of anticoagulants", 0.90),
    ],
    "ptt": [
        ("Z79.01", "Long term (current) use of anticoagulants", 0.85),
    ],
    "coagulation panel": [
        ("Z79.01", "Long term (current) use of anticoagulants", 0.85),
    ],

    # Thyroid
    "tsh": [
        ("E03.9", "Hypothyroidism, unspecified", 0.85),
        ("E05.90", "Thyrotoxicosis, unspecified without thyrotoxic crisis", 0.80),
    ],
    "thyroid panel": [
        ("E03.9", "Hypothyroidism, unspecified", 0.85),
    ],
    "free t4": [
        ("E03.9", "Hypothyroidism, unspecified", 0.85),
    ],
    "t3": [
        ("E05.90", "Thyrotoxicosis, unspecified without thyrotoxic crisis", 0.80),
    ],

    # Renal Function
    "creatinine": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.85),
    ],
    "bun": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.80),
    ],
    "gfr": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.90),
    ],
    "egfr": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.90),
    ],
    "urine albumin": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.85),
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.80),
    ],
    "microalbumin": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.85),
    ],

    # Liver Function
    "lft": [
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.80),
        ("K75.81", "Nonalcoholic steatohepatitis (NASH)", 0.75),
    ],
    "liver function": [
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.80),
    ],
    "hepatic panel": [
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.80),
    ],
    "ast": [
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.75),
    ],
    "alt": [
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.75),
    ],

    # Cardiac Markers
    "bnp": [
        ("I50.9", "Heart failure, unspecified", 0.90),
    ],
    "pro-bnp": [
        ("I50.9", "Heart failure, unspecified", 0.90),
    ],
    "nt-probnp": [
        ("I50.9", "Heart failure, unspecified", 0.90),
    ],
    "troponin": [
        ("I21.9", "Acute myocardial infarction, unspecified", 0.90),
        ("I20.9", "Angina pectoris, unspecified", 0.80),
    ],

    # Iron Studies
    "iron panel": [
        ("D50.9", "Iron deficiency anemia, unspecified", 0.90),
    ],
    "ferritin": [
        ("D50.9", "Iron deficiency anemia, unspecified", 0.85),
    ],
    "iron studies": [
        ("D50.9", "Iron deficiency anemia, unspecified", 0.90),
    ],

    # Vitamins
    "vitamin d": [
        ("E55.9", "Vitamin D deficiency, unspecified", 0.90),
    ],
    "25-hydroxy vitamin d": [
        ("E55.9", "Vitamin D deficiency, unspecified", 0.90),
    ],
    "vitamin b12": [
        ("E53.8", "Deficiency of other specified B group vitamins", 0.85),
        ("D51.9", "Vitamin B12 deficiency anemia, unspecified", 0.80),
    ],
    "folate": [
        ("D52.9", "Folate deficiency anemia, unspecified", 0.85),
    ],

    # Uric Acid
    "uric acid": [
        ("M10.9", "Gout, unspecified", 0.90),
        ("E79.0", "Hyperuricemia", 0.85),
    ],

    # Urinalysis
    "ua": [
        ("N39.0", "Urinary tract infection, site not specified", 0.80),
    ],
    "urinalysis": [
        ("N39.0", "Urinary tract infection, site not specified", 0.80),
    ],
    "urine culture": [
        ("N39.0", "Urinary tract infection, site not specified", 0.90),
    ],

    # Inflammatory Markers
    "crp": [
        ("R79.89", "Other specified abnormal findings of blood chemistry", 0.75),
    ],
    "esr": [
        ("R70.0", "Elevated erythrocyte sedimentation rate", 0.80),
    ],
    "sed rate": [
        ("R70.0", "Elevated erythrocyte sedimentation rate", 0.80),
    ],

    # PSA
    "psa": [
        ("N40.0", "Benign prostatic hyperplasia without lower urinary tract symptoms", 0.80),
        ("Z12.5", "Encounter for screening for malignant neoplasm of prostate", 0.85),
    ],

    # Magnesium
    "magnesium": [
        ("E83.42", "Hypomagnesemia", 0.80),
    ],
}


# =============================================================================
# Consult/Referral Order-to-Diagnosis Clinical Rules
# Maps specialty names to their typical indications (ICD-10 codes)
# =============================================================================

CONSULT_DIAGNOSIS_RULES: dict[str, list[tuple[str, str, float]]] = {
    # Specialty (lowercase) -> list of (ICD-10, display, confidence)

    # Cardiology
    "cardiology": [
        ("I25.10", "Atherosclerotic heart disease", 0.80),
        ("I50.9", "Heart failure, unspecified", 0.80),
        ("I48.91", "Unspecified atrial fibrillation", 0.80),
        ("I10", "Essential hypertension", 0.75),
    ],
    "cardiac": [
        ("I25.10", "Atherosclerotic heart disease", 0.80),
    ],
    "electrophysiology": [
        ("I48.91", "Unspecified atrial fibrillation", 0.90),
        ("I49.9", "Cardiac arrhythmia, unspecified", 0.85),
    ],

    # Pulmonology
    "pulmonology": [
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.85),
        ("J45.909", "Unspecified asthma, uncomplicated", 0.80),
        ("J18.9", "Pneumonia, unspecified organism", 0.75),
    ],
    "pulmonary": [
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.85),
    ],

    # Gastroenterology
    "gastroenterology": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.80),
        ("K57.30", "Diverticulosis of large intestine without perforation or abscess", 0.75),
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.75),
    ],
    "gi": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.80),
    ],

    # Endocrinology
    "endocrinology": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.85),
        ("E03.9", "Hypothyroidism, unspecified", 0.80),
        ("E05.90", "Thyrotoxicosis, unspecified without thyrotoxic crisis", 0.75),
    ],
    "endocrine": [
        ("E11.9", "Type 2 diabetes mellitus without complications", 0.85),
    ],

    # Nephrology
    "nephrology": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.90),
        ("I10", "Essential hypertension", 0.75),
    ],
    "renal": [
        ("N18.9", "Chronic kidney disease, unspecified", 0.90),
    ],

    # Neurology
    "neurology": [
        ("G43.909", "Migraine, unspecified, not intractable, without status migrainosus", 0.80),
        ("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus", 0.80),
        ("G20", "Parkinson's disease", 0.75),
        ("I63.9", "Cerebral infarction, unspecified", 0.75),
    ],

    # Rheumatology
    "rheumatology": [
        ("M06.9", "Rheumatoid arthritis, unspecified", 0.85),
        ("M32.9", "Systemic lupus erythematosus, unspecified", 0.80),
        ("M10.9", "Gout, unspecified", 0.80),
    ],

    # Orthopedics
    "orthopedics": [
        ("M19.90", "Unspecified osteoarthritis, unspecified site", 0.85),
        ("M54.50", "Low back pain, unspecified", 0.80),
        ("M25.50", "Pain in unspecified joint", 0.80),
    ],
    "ortho": [
        ("M19.90", "Unspecified osteoarthritis, unspecified site", 0.85),
    ],

    # Psychiatry
    "psychiatry": [
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.85),
        ("F41.9", "Anxiety disorder, unspecified", 0.80),
        ("F31.9", "Bipolar disorder, unspecified", 0.75),
    ],
    "mental health": [
        ("F32.9", "Major depressive disorder, single episode, unspecified", 0.85),
    ],

    # Oncology
    "oncology": [
        ("C80.1", "Malignant (primary) neoplasm, unspecified", 0.85),
        ("Z85.9", "Personal history of malignant neoplasm, unspecified", 0.80),
    ],

    # Hematology
    "hematology": [
        ("D64.9", "Anemia, unspecified", 0.85),
        ("D69.6", "Thrombocytopenia, unspecified", 0.80),
    ],

    # Infectious Disease
    "infectious disease": [
        ("B20", "Human immunodeficiency virus [HIV] disease", 0.75),
        ("A41.9", "Sepsis, unspecified organism", 0.80),
    ],
    "id": [
        ("A41.9", "Sepsis, unspecified organism", 0.75),
    ],

    # Dermatology
    "dermatology": [
        ("L40.9", "Psoriasis, unspecified", 0.80),
        ("L30.9", "Dermatitis, unspecified", 0.80),
        ("C44.90", "Unspecified malignant neoplasm of skin, unspecified", 0.75),
    ],

    # Ophthalmology
    "ophthalmology": [
        ("H40.9", "Unspecified glaucoma", 0.80),
        ("H25.9", "Unspecified age-related cataract", 0.80),
        ("E11.319", "Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema", 0.80),
    ],
    "eye": [
        ("H40.9", "Unspecified glaucoma", 0.80),
    ],

    # Urology
    "urology": [
        ("N40.0", "Benign prostatic hyperplasia without lower urinary tract symptoms", 0.85),
        ("N39.0", "Urinary tract infection, site not specified", 0.80),
        ("N20.0", "Calculus of kidney", 0.80),
    ],

    # Surgery
    "general surgery": [
        ("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", 0.80),
        ("K40.90", "Unilateral inguinal hernia, without obstruction or gangrene, not specified as recurrent", 0.80),
    ],
    "surgery": [
        ("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", 0.75),
    ],
    "vascular surgery": [
        ("I70.0", "Atherosclerosis of aorta", 0.85),
        ("I73.9", "Peripheral vascular disease, unspecified", 0.85),
    ],

    # Pain Management
    "pain management": [
        ("G89.29", "Other chronic pain", 0.90),
        ("M54.50", "Low back pain, unspecified", 0.85),
    ],
    "pain": [
        ("G89.29", "Other chronic pain", 0.90),
    ],

    # Sleep Medicine
    "sleep medicine": [
        ("G47.33", "Obstructive sleep apnea (adult) (pediatric)", 0.90),
        ("G47.00", "Insomnia, unspecified", 0.80),
    ],
    "sleep": [
        ("G47.33", "Obstructive sleep apnea (adult) (pediatric)", 0.90),
    ],

    # Allergy/Immunology
    "allergy": [
        ("J30.9", "Allergic rhinitis, unspecified", 0.85),
        ("T78.40XA", "Allergy, unspecified, initial encounter", 0.80),
    ],
    "immunology": [
        ("D89.9", "Disorder involving the immune mechanism, unspecified", 0.80),
    ],

    # Physical Therapy
    "physical therapy": [
        ("M54.50", "Low back pain, unspecified", 0.85),
        ("M25.50", "Pain in unspecified joint", 0.80),
        ("Z96.641", "Presence of right artificial hip joint", 0.75),
    ],
    "pt": [
        ("M54.50", "Low back pain, unspecified", 0.80),
    ],

    # Occupational Therapy
    "occupational therapy": [
        ("I63.9", "Cerebral infarction, unspecified", 0.80),
        ("Z96.641", "Presence of right artificial hip joint", 0.75),
    ],
    "ot": [
        ("I63.9", "Cerebral infarction, unspecified", 0.75),
    ],

    # Podiatry
    "podiatry": [
        ("E11.621", "Type 2 diabetes mellitus with foot ulcer", 0.85),
        ("L03.031", "Cellulitis of right toe", 0.80),
    ],

    # ENT
    "ent": [
        ("J32.9", "Chronic sinusitis, unspecified", 0.80),
        ("H66.90", "Unspecified otitis media, unspecified ear", 0.80),
    ],
    "otolaryngology": [
        ("J32.9", "Chronic sinusitis, unspecified", 0.80),
    ],
}


# =============================================================================
# Procedure Order-to-Diagnosis Clinical Rules
# Maps procedure names to their typical indications (ICD-10 codes)
# =============================================================================

PROCEDURE_DIAGNOSIS_RULES: dict[str, list[tuple[str, str, float]]] = {
    # Procedure name (lowercase) -> list of (ICD-10, display, confidence)

    # Cardiac Procedures
    "ecg": [
        ("I25.10", "Atherosclerotic heart disease", 0.80),
        ("I48.91", "Unspecified atrial fibrillation", 0.80),
        ("R00.0", "Tachycardia, unspecified", 0.75),
    ],
    "ekg": [
        ("I25.10", "Atherosclerotic heart disease", 0.80),
        ("I48.91", "Unspecified atrial fibrillation", 0.80),
    ],
    "electrocardiogram": [
        ("I25.10", "Atherosclerotic heart disease", 0.80),
    ],
    "echocardiogram": [
        ("I50.9", "Heart failure, unspecified", 0.85),
        ("I25.10", "Atherosclerotic heart disease", 0.80),
    ],
    "echo": [
        ("I50.9", "Heart failure, unspecified", 0.85),
    ],
    "stress test": [
        ("I25.10", "Atherosclerotic heart disease", 0.90),
        ("I20.9", "Angina pectoris, unspecified", 0.85),
    ],
    "cardiac stress test": [
        ("I25.10", "Atherosclerotic heart disease", 0.90),
    ],
    "holter monitor": [
        ("I48.91", "Unspecified atrial fibrillation", 0.85),
        ("I49.9", "Cardiac arrhythmia, unspecified", 0.85),
    ],
    "event monitor": [
        ("I49.9", "Cardiac arrhythmia, unspecified", 0.85),
    ],
    "cardiac catheterization": [
        ("I25.10", "Atherosclerotic heart disease", 0.90),
        ("I21.9", "Acute myocardial infarction, unspecified", 0.85),
    ],
    "cath": [
        ("I25.10", "Atherosclerotic heart disease", 0.90),
    ],

    # Imaging - Chest
    "chest x-ray": [
        ("J18.9", "Pneumonia, unspecified organism", 0.80),
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.75),
    ],
    "cxr": [
        ("J18.9", "Pneumonia, unspecified organism", 0.80),
    ],
    "chest ct": [
        ("J18.9", "Pneumonia, unspecified organism", 0.80),
        ("I26.99", "Other pulmonary embolism without acute cor pulmonale", 0.80),
    ],
    "ct chest": [
        ("J18.9", "Pneumonia, unspecified organism", 0.80),
    ],
    "ct angiography chest": [
        ("I26.99", "Other pulmonary embolism without acute cor pulmonale", 0.90),
    ],
    "cta chest": [
        ("I26.99", "Other pulmonary embolism without acute cor pulmonale", 0.90),
    ],

    # Imaging - Abdomen
    "abdominal ultrasound": [
        ("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", 0.80),
        ("K76.0", "Fatty (change of) liver, not elsewhere classified", 0.80),
    ],
    "abd ultrasound": [
        ("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", 0.80),
    ],
    "ct abdomen": [
        ("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", 0.75),
        ("R10.9", "Unspecified abdominal pain", 0.80),
    ],
    "abdominal ct": [
        ("R10.9", "Unspecified abdominal pain", 0.80),
    ],

    # Imaging - Head/Brain
    "ct head": [
        ("I63.9", "Cerebral infarction, unspecified", 0.85),
        ("R51.9", "Headache, unspecified", 0.80),
    ],
    "head ct": [
        ("I63.9", "Cerebral infarction, unspecified", 0.85),
    ],
    "mri brain": [
        ("I63.9", "Cerebral infarction, unspecified", 0.85),
        ("G43.909", "Migraine, unspecified, not intractable, without status migrainosus", 0.80),
    ],
    "brain mri": [
        ("I63.9", "Cerebral infarction, unspecified", 0.85),
    ],

    # Imaging - Spine
    "mri spine": [
        ("M54.50", "Low back pain, unspecified", 0.85),
        ("M51.16", "Intervertebral disc degeneration, lumbar region", 0.80),
    ],
    "mri lumbar": [
        ("M54.50", "Low back pain, unspecified", 0.90),
    ],
    "mri cervical": [
        ("M54.2", "Cervicalgia", 0.85),
    ],
    "spine x-ray": [
        ("M54.50", "Low back pain, unspecified", 0.80),
    ],

    # Imaging - Extremities
    "x-ray": [
        ("M25.50", "Pain in unspecified joint", 0.75),
        ("S82.90XA", "Unspecified fracture of unspecified lower leg, initial encounter for closed fracture", 0.75),
    ],
    "mri knee": [
        ("M23.90", "Unspecified internal derangement of unspecified knee", 0.85),
        ("M17.9", "Osteoarthritis of knee, unspecified", 0.80),
    ],
    "mri shoulder": [
        ("M75.100", "Unspecified rotator cuff tear or rupture of unspecified shoulder, not specified as traumatic", 0.85),
    ],

    # Vascular
    "doppler ultrasound": [
        ("I82.40", "Deep vein thrombosis of unspecified deep veins of lower extremity", 0.85),
        ("I73.9", "Peripheral vascular disease, unspecified", 0.80),
    ],
    "venous doppler": [
        ("I82.40", "Deep vein thrombosis of unspecified deep veins of lower extremity", 0.90),
    ],
    "arterial doppler": [
        ("I73.9", "Peripheral vascular disease, unspecified", 0.90),
    ],
    "carotid ultrasound": [
        ("I65.29", "Occlusion and stenosis of unspecified carotid artery", 0.85),
        ("I63.9", "Cerebral infarction, unspecified", 0.80),
    ],

    # Pulmonary Function
    "pulmonary function test": [
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.90),
        ("J45.909", "Unspecified asthma, uncomplicated", 0.85),
    ],
    "pft": [
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.90),
    ],
    "spirometry": [
        ("J44.9", "Chronic obstructive pulmonary disease, unspecified", 0.90),
    ],

    # GI Procedures
    "colonoscopy": [
        ("Z12.11", "Encounter for screening for malignant neoplasm of colon", 0.85),
        ("K57.30", "Diverticulosis of large intestine without perforation or abscess", 0.80),
    ],
    "egd": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.85),
        ("K25.9", "Gastric ulcer, unspecified as acute or chronic, without hemorrhage or perforation", 0.80),
    ],
    "upper endoscopy": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.85),
    ],
    "endoscopy": [
        ("K21.0", "Gastro-esophageal reflux disease with esophagitis", 0.80),
    ],

    # Sleep Study
    "sleep study": [
        ("G47.33", "Obstructive sleep apnea (adult) (pediatric)", 0.95),
    ],
    "polysomnography": [
        ("G47.33", "Obstructive sleep apnea (adult) (pediatric)", 0.95),
    ],
    "psg": [
        ("G47.33", "Obstructive sleep apnea (adult) (pediatric)", 0.95),
    ],

    # DEXA
    "dexa": [
        ("M81.0", "Age-related osteoporosis without current pathological fracture", 0.90),
    ],
    "dexa scan": [
        ("M81.0", "Age-related osteoporosis without current pathological fracture", 0.90),
    ],
    "bone density": [
        ("M81.0", "Age-related osteoporosis without current pathological fracture", 0.90),
    ],

    # Mammography
    "mammogram": [
        ("Z12.31", "Encounter for screening mammogram for malignant neoplasm of breast", 0.90),
    ],
    "mammography": [
        ("Z12.31", "Encounter for screening mammogram for malignant neoplasm of breast", 0.90),
    ],

    # EMG/NCS
    "emg": [
        ("G62.9", "Polyneuropathy, unspecified", 0.85),
        ("G56.00", "Carpal tunnel syndrome, unspecified upper limb", 0.85),
    ],
    "nerve conduction study": [
        ("G62.9", "Polyneuropathy, unspecified", 0.85),
    ],
    "ncs": [
        ("G62.9", "Polyneuropathy, unspecified", 0.85),
    ],

    # EEG
    "eeg": [
        ("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus", 0.90),
        ("R56.9", "Unspecified convulsions", 0.85),
    ],
    "electroencephalogram": [
        ("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus", 0.90),
    ],

    # Biopsies
    "biopsy": [
        ("R59.9", "Enlarged lymph nodes, unspecified", 0.75),
    ],
    "skin biopsy": [
        ("L98.9", "Disorder of the skin and subcutaneous tissue, unspecified", 0.80),
    ],

    # Injections
    "joint injection": [
        ("M19.90", "Unspecified osteoarthritis, unspecified site", 0.85),
        ("M25.50", "Pain in unspecified joint", 0.80),
    ],
    "steroid injection": [
        ("M54.50", "Low back pain, unspecified", 0.85),
        ("M25.50", "Pain in unspecified joint", 0.80),
    ],
    "epidural injection": [
        ("M54.50", "Low back pain, unspecified", 0.90),
    ],
}


# Import drug class lookup from rxnorm_lookup
try:
    from voice_to_fhir.extraction.rxnorm_lookup import get_drug_class
except ImportError:
    # Fallback if rxnorm_lookup not available
    def get_drug_class(medication_name: str) -> str | None:
        return None


def link_medication_to_diagnosis(
    medication_name: str,
    drug_class: str | None,
    patient_conditions: list
) -> DiagnosisLink | None:
    """
    Link a medication to a diagnosis based on clinical rules and patient conditions.

    Priority:
    1. Match to patient's existing conditions (highest confidence)
    2. Use clinical rules based on drug class

    Args:
        medication_name: The medication name
        drug_class: The drug class (from RxNorm lookup)
        patient_conditions: List of patient's conditions (with icd10 and name)

    Returns:
        DiagnosisLink or None if no link found
    """
    if not drug_class:
        drug_class = get_drug_class(medication_name)

    if not drug_class:
        return None

    # Get rules for this drug class
    rules = MEDICATION_DIAGNOSIS_RULES.get(drug_class, [])
    if not rules:
        return None

    # 1. Try to match against patient's conditions
    for icd10, display, confidence in rules:
        for condition in patient_conditions:
            condition_icd10 = getattr(condition, 'icd10', None) or condition.get('icd10') if isinstance(condition, dict) else None
            condition_name = getattr(condition, 'name', None) or condition.get('name', '') if isinstance(condition, dict) else ''

            # Match by ICD-10 code
            if condition_icd10 and condition_icd10 == icd10:
                return DiagnosisLink(
                    icd10=icd10,
                    display=display,
                    confidence=min(confidence + 0.05, 1.0),  # Boost confidence for patient match
                    method='patient_condition'
                )

            # Fuzzy match by name (check if rule display is in condition name or vice versa)
            condition_name_lower = condition_name.lower()
            display_lower = display.lower()
            if display_lower in condition_name_lower or condition_name_lower in display_lower:
                return DiagnosisLink(
                    icd10=icd10,
                    display=display,
                    confidence=min(confidence + 0.03, 1.0),
                    method='patient_condition'
                )

    # 2. Fall back to rule-based default (first/primary indication)
    if rules:
        icd10, display, confidence = rules[0]
        return DiagnosisLink(
            icd10=icd10,
            display=display,
            confidence=confidence,
            method='rule'
        )

    return None


def _normalize_name(name: str) -> str:
    """Normalize order name for rule lookup."""
    return name.lower().strip()


def _match_against_conditions(
    rules: list[tuple[str, str, float]],
    patient_conditions: list
) -> DiagnosisLink | None:
    """Try to match rules against patient's existing conditions."""
    for icd10, display, confidence in rules:
        for condition in patient_conditions:
            condition_icd10 = getattr(condition, 'icd10', None) or (condition.get('icd10') if isinstance(condition, dict) else None)
            condition_name = getattr(condition, 'name', None) or (condition.get('name', '') if isinstance(condition, dict) else '')

            # Match by ICD-10 code
            if condition_icd10 and condition_icd10 == icd10:
                return DiagnosisLink(
                    icd10=icd10,
                    display=display,
                    confidence=min(confidence + 0.05, 1.0),
                    method='patient_condition'
                )

            # Fuzzy match by name
            condition_name_lower = condition_name.lower()
            display_lower = display.lower()
            if display_lower in condition_name_lower or condition_name_lower in display_lower:
                return DiagnosisLink(
                    icd10=icd10,
                    display=display,
                    confidence=min(confidence + 0.03, 1.0),
                    method='patient_condition'
                )
    return None


def link_lab_to_diagnosis(
    lab_name: str,
    patient_conditions: list
) -> DiagnosisLink | None:
    """
    Link a lab order to a diagnosis based on clinical rules and patient conditions.

    Args:
        lab_name: The lab test name
        patient_conditions: List of patient's conditions (with icd10 and name)

    Returns:
        DiagnosisLink or None if no link found
    """
    normalized = _normalize_name(lab_name)

    # Find matching rules (try exact match, then partial match)
    rules = LAB_DIAGNOSIS_RULES.get(normalized, [])

    if not rules:
        # Try partial matching for common variations
        for key, key_rules in LAB_DIAGNOSIS_RULES.items():
            if key in normalized or normalized in key:
                rules = key_rules
                break

    if not rules:
        return None

    # 1. Try to match against patient's conditions
    link = _match_against_conditions(rules, patient_conditions)
    if link:
        return link

    # 2. Fall back to rule-based default
    icd10, display, confidence = rules[0]
    return DiagnosisLink(
        icd10=icd10,
        display=display,
        confidence=confidence,
        method='rule'
    )


def link_consult_to_diagnosis(
    specialty: str,
    patient_conditions: list
) -> DiagnosisLink | None:
    """
    Link a consult/referral order to a diagnosis based on clinical rules and patient conditions.

    Args:
        specialty: The specialty name
        patient_conditions: List of patient's conditions (with icd10 and name)

    Returns:
        DiagnosisLink or None if no link found
    """
    normalized = _normalize_name(specialty)

    # Find matching rules
    rules = CONSULT_DIAGNOSIS_RULES.get(normalized, [])

    if not rules:
        # Try partial matching
        for key, key_rules in CONSULT_DIAGNOSIS_RULES.items():
            if key in normalized or normalized in key:
                rules = key_rules
                break

    if not rules:
        return None

    # 1. Try to match against patient's conditions
    link = _match_against_conditions(rules, patient_conditions)
    if link:
        return link

    # 2. Fall back to rule-based default
    icd10, display, confidence = rules[0]
    return DiagnosisLink(
        icd10=icd10,
        display=display,
        confidence=confidence,
        method='rule'
    )


def link_procedure_to_diagnosis(
    procedure_name: str,
    patient_conditions: list
) -> DiagnosisLink | None:
    """
    Link a procedure order to a diagnosis based on clinical rules and patient conditions.

    Args:
        procedure_name: The procedure name
        patient_conditions: List of patient's conditions (with icd10 and name)

    Returns:
        DiagnosisLink or None if no link found
    """
    normalized = _normalize_name(procedure_name)

    # Find matching rules
    rules = PROCEDURE_DIAGNOSIS_RULES.get(normalized, [])

    if not rules:
        # Try partial matching
        for key, key_rules in PROCEDURE_DIAGNOSIS_RULES.items():
            if key in normalized or normalized in key:
                rules = key_rules
                break

    if not rules:
        return None

    # 1. Try to match against patient's conditions
    link = _match_against_conditions(rules, patient_conditions)
    if link:
        return link

    # 2. Fall back to rule-based default
    icd10, display, confidence = rules[0]
    return DiagnosisLink(
        icd10=icd10,
        display=display,
        confidence=confidence,
        method='rule'
    )


def enrich_orders_with_diagnoses(entities) -> object:
    """
    Enrich all order types with linked diagnoses.

    Args:
        entities: ClinicalEntities object

    Returns:
        The same entities object with linked_diagnosis added to all order types
    """
    # Get patient conditions
    patient_conditions = entities.conditions if hasattr(entities, 'conditions') else []

    # Process medication orders
    for order in entities.medication_orders:
        if order.linked_diagnosis:
            continue

        drug_class = None
        try:
            from voice_to_fhir.extraction.rxnorm_lookup import get_drug_class
            drug_class = get_drug_class(order.name)
        except ImportError:
            pass

        link = link_medication_to_diagnosis(
            medication_name=order.name,
            drug_class=drug_class,
            patient_conditions=patient_conditions
        )

        if link:
            order.linked_diagnosis = {
                'icd10': link.icd10,
                'display': link.display,
                'confidence': link.confidence,
                'method': link.method
            }

    # Process lab orders
    if hasattr(entities, 'lab_orders'):
        for order in entities.lab_orders:
            if getattr(order, 'linked_diagnosis', None):
                continue

            link = link_lab_to_diagnosis(
                lab_name=order.name,
                patient_conditions=patient_conditions
            )

            if link:
                order.linked_diagnosis = {
                    'icd10': link.icd10,
                    'display': link.display,
                    'confidence': link.confidence,
                    'method': link.method
                }

    # Process referral/consult orders
    if hasattr(entities, 'referral_orders'):
        for order in entities.referral_orders:
            if getattr(order, 'linked_diagnosis', None):
                continue

            link = link_consult_to_diagnosis(
                specialty=order.specialty,
                patient_conditions=patient_conditions
            )

            if link:
                order.linked_diagnosis = {
                    'icd10': link.icd10,
                    'display': link.display,
                    'confidence': link.confidence,
                    'method': link.method
                }

    # Process procedure orders
    if hasattr(entities, 'procedure_orders'):
        for order in entities.procedure_orders:
            if getattr(order, 'linked_diagnosis', None):
                continue

            link = link_procedure_to_diagnosis(
                procedure_name=order.name,
                patient_conditions=patient_conditions
            )

            if link:
                order.linked_diagnosis = {
                    'icd10': link.icd10,
                    'display': link.display,
                    'confidence': link.confidence,
                    'method': link.method
                }

    return entities
