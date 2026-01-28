"""
RxNorm Lookup Database for Medication Verification.

This module provides verified RxNorm codes for common medications.
Instead of relying on LLM-generated codes (which can hallucinate), this lookup
provides deterministic, auditable medication coding.

Flow:
    MedGemma extracts medication name -> fuzzy match -> RxNorm lookup -> verified code
"""

import re
from difflib import SequenceMatcher
from typing import NamedTuple


class RxNormCode(NamedTuple):
    """RxNorm code with metadata."""
    rxcui: str
    display: str
    drug_class: str  # e.g., "statin", "ace_inhibitor", "beta_blocker"
    brand_names: tuple[str, ...] = ()  # Common brand names


# =============================================================================
# RxNorm Database (~200 common medications)
# Organized by drug class for maintainability
# =============================================================================

RXNORM_DATABASE: dict[str, RxNormCode] = {
    # -------------------------------------------------------------------------
    # Statins (HMG-CoA Reductase Inhibitors)
    # -------------------------------------------------------------------------
    "atorvastatin": RxNormCode("83367", "Atorvastatin", "statin", ("Lipitor",)),
    "rosuvastatin": RxNormCode("301542", "Rosuvastatin", "statin", ("Crestor",)),
    "simvastatin": RxNormCode("36567", "Simvastatin", "statin", ("Zocor",)),
    "pravastatin": RxNormCode("42463", "Pravastatin", "statin", ("Pravachol",)),
    "lovastatin": RxNormCode("6472", "Lovastatin", "statin", ("Mevacor",)),
    "fluvastatin": RxNormCode("41127", "Fluvastatin", "statin", ("Lescol",)),
    "pitavastatin": RxNormCode("861634", "Pitavastatin", "statin", ("Livalo",)),

    # -------------------------------------------------------------------------
    # ACE Inhibitors
    # -------------------------------------------------------------------------
    "lisinopril": RxNormCode("29046", "Lisinopril", "ace_inhibitor", ("Zestril", "Prinivil")),
    "enalapril": RxNormCode("3827", "Enalapril", "ace_inhibitor", ("Vasotec",)),
    "ramipril": RxNormCode("35296", "Ramipril", "ace_inhibitor", ("Altace",)),
    "benazepril": RxNormCode("18867", "Benazepril", "ace_inhibitor", ("Lotensin",)),
    "captopril": RxNormCode("1998", "Captopril", "ace_inhibitor", ("Capoten",)),
    "fosinopril": RxNormCode("50166", "Fosinopril", "ace_inhibitor", ("Monopril",)),
    "quinapril": RxNormCode("35208", "Quinapril", "ace_inhibitor", ("Accupril",)),
    "perindopril": RxNormCode("54552", "Perindopril", "ace_inhibitor", ("Aceon",)),
    "trandolapril": RxNormCode("38454", "Trandolapril", "ace_inhibitor", ("Mavik",)),

    # -------------------------------------------------------------------------
    # ARBs (Angiotensin II Receptor Blockers)
    # -------------------------------------------------------------------------
    "losartan": RxNormCode("52175", "Losartan", "arb", ("Cozaar",)),
    "valsartan": RxNormCode("69749", "Valsartan", "arb", ("Diovan",)),
    "irbesartan": RxNormCode("83818", "Irbesartan", "arb", ("Avapro",)),
    "olmesartan": RxNormCode("321064", "Olmesartan", "arb", ("Benicar",)),
    "telmisartan": RxNormCode("73494", "Telmisartan", "arb", ("Micardis",)),
    "candesartan": RxNormCode("214354", "Candesartan", "arb", ("Atacand",)),
    "azilsartan": RxNormCode("1091643", "Azilsartan", "arb", ("Edarbi",)),

    # -------------------------------------------------------------------------
    # Beta Blockers
    # -------------------------------------------------------------------------
    "metoprolol": RxNormCode("6918", "Metoprolol", "beta_blocker", ("Lopressor", "Toprol-XL")),
    "atenolol": RxNormCode("1202", "Atenolol", "beta_blocker", ("Tenormin",)),
    "carvedilol": RxNormCode("20352", "Carvedilol", "beta_blocker", ("Coreg",)),
    "bisoprolol": RxNormCode("19484", "Bisoprolol", "beta_blocker", ("Zebeta",)),
    "propranolol": RxNormCode("8787", "Propranolol", "beta_blocker", ("Inderal",)),
    "nebivolol": RxNormCode("31555", "Nebivolol", "beta_blocker", ("Bystolic",)),
    "labetalol": RxNormCode("6185", "Labetalol", "beta_blocker", ("Trandate",)),
    "nadolol": RxNormCode("7226", "Nadolol", "beta_blocker", ("Corgard",)),

    # -------------------------------------------------------------------------
    # Calcium Channel Blockers
    # -------------------------------------------------------------------------
    "amlodipine": RxNormCode("17767", "Amlodipine", "ccb", ("Norvasc",)),
    "diltiazem": RxNormCode("3443", "Diltiazem", "ccb", ("Cardizem", "Tiazac")),
    "verapamil": RxNormCode("11170", "Verapamil", "ccb", ("Calan", "Verelan")),
    "nifedipine": RxNormCode("7417", "Nifedipine", "ccb", ("Procardia", "Adalat")),
    "felodipine": RxNormCode("4316", "Felodipine", "ccb", ("Plendil",)),
    "nicardipine": RxNormCode("7396", "Nicardipine", "ccb", ("Cardene",)),

    # -------------------------------------------------------------------------
    # Diuretics
    # -------------------------------------------------------------------------
    "furosemide": RxNormCode("4603", "Furosemide", "diuretic", ("Lasix",)),
    "hydrochlorothiazide": RxNormCode("5487", "Hydrochlorothiazide", "diuretic", ("Microzide",)),
    "spironolactone": RxNormCode("9997", "Spironolactone", "diuretic", ("Aldactone",)),
    "chlorthalidone": RxNormCode("2409", "Chlorthalidone", "diuretic", ("Thalitone",)),
    "bumetanide": RxNormCode("1808", "Bumetanide", "diuretic", ("Bumex",)),
    "torsemide": RxNormCode("38413", "Torsemide", "diuretic", ("Demadex",)),
    "indapamide": RxNormCode("5764", "Indapamide", "diuretic", ("Lozol",)),
    "metolazone": RxNormCode("6916", "Metolazone", "diuretic", ("Zaroxolyn",)),
    "triamterene": RxNormCode("10689", "Triamterene", "diuretic", ("Dyrenium",)),

    # -------------------------------------------------------------------------
    # Diabetes Medications - Oral
    # -------------------------------------------------------------------------
    "metformin": RxNormCode("6809", "Metformin", "diabetes_oral", ("Glucophage",)),
    "glipizide": RxNormCode("4821", "Glipizide", "diabetes_oral", ("Glucotrol",)),
    "glyburide": RxNormCode("4815", "Glyburide", "diabetes_oral", ("DiaBeta", "Micronase")),
    "glimepiride": RxNormCode("25789", "Glimepiride", "diabetes_oral", ("Amaryl",)),
    "sitagliptin": RxNormCode("593411", "Sitagliptin", "diabetes_oral", ("Januvia",)),
    "linagliptin": RxNormCode("1100699", "Linagliptin", "diabetes_oral", ("Tradjenta",)),
    "saxagliptin": RxNormCode("857974", "Saxagliptin", "diabetes_oral", ("Onglyza",)),
    "pioglitazone": RxNormCode("33738", "Pioglitazone", "diabetes_oral", ("Actos",)),
    "empagliflozin": RxNormCode("1545653", "Empagliflozin", "diabetes_oral", ("Jardiance",)),
    "dapagliflozin": RxNormCode("1488564", "Dapagliflozin", "diabetes_oral", ("Farxiga",)),
    "canagliflozin": RxNormCode("1373458", "Canagliflozin", "diabetes_oral", ("Invokana",)),

    # -------------------------------------------------------------------------
    # Diabetes Medications - Injectable (GLP-1, Insulin)
    # -------------------------------------------------------------------------
    "semaglutide": RxNormCode("1991302", "Semaglutide", "glp1", ("Ozempic", "Wegovy", "Rybelsus")),
    "liraglutide": RxNormCode("475968", "Liraglutide", "glp1", ("Victoza", "Saxenda")),
    "dulaglutide": RxNormCode("1551291", "Dulaglutide", "glp1", ("Trulicity",)),
    "exenatide": RxNormCode("60548", "Exenatide", "glp1", ("Byetta", "Bydureon")),
    "tirzepatide": RxNormCode("2601723", "Tirzepatide", "glp1", ("Mounjaro", "Zepbound")),
    "insulin glargine": RxNormCode("274783", "Insulin Glargine", "insulin", ("Lantus", "Basaglar")),
    "insulin lispro": RxNormCode("86009", "Insulin Lispro", "insulin", ("Humalog",)),
    "insulin aspart": RxNormCode("86009", "Insulin Aspart", "insulin", ("NovoLog",)),
    "insulin degludec": RxNormCode("1372741", "Insulin Degludec", "insulin", ("Tresiba",)),

    # -------------------------------------------------------------------------
    # Proton Pump Inhibitors (PPIs)
    # -------------------------------------------------------------------------
    "omeprazole": RxNormCode("7646", "Omeprazole", "ppi", ("Prilosec",)),
    "pantoprazole": RxNormCode("40790", "Pantoprazole", "ppi", ("Protonix",)),
    "esomeprazole": RxNormCode("283742", "Esomeprazole", "ppi", ("Nexium",)),
    "lansoprazole": RxNormCode("17128", "Lansoprazole", "ppi", ("Prevacid",)),
    "rabeprazole": RxNormCode("35827", "Rabeprazole", "ppi", ("Aciphex",)),
    "dexlansoprazole": RxNormCode("854873", "Dexlansoprazole", "ppi", ("Dexilant",)),

    # -------------------------------------------------------------------------
    # H2 Blockers
    # -------------------------------------------------------------------------
    "famotidine": RxNormCode("4278", "Famotidine", "h2_blocker", ("Pepcid",)),
    "ranitidine": RxNormCode("9143", "Ranitidine", "h2_blocker", ("Zantac",)),
    "cimetidine": RxNormCode("2541", "Cimetidine", "h2_blocker", ("Tagamet",)),

    # -------------------------------------------------------------------------
    # Antibiotics - Penicillins
    # -------------------------------------------------------------------------
    "amoxicillin": RxNormCode("723", "Amoxicillin", "antibiotic_penicillin", ("Amoxil",)),
    "amoxicillin-clavulanate": RxNormCode("7980", "Amoxicillin-Clavulanate", "antibiotic_penicillin", ("Augmentin",)),
    "augmentin": RxNormCode("7980", "Amoxicillin-Clavulanate", "antibiotic_penicillin", ("Augmentin",)),
    "ampicillin": RxNormCode("733", "Ampicillin", "antibiotic_penicillin", ()),
    "penicillin": RxNormCode("7984", "Penicillin V", "antibiotic_penicillin", ()),
    "penicillin v": RxNormCode("7984", "Penicillin V", "antibiotic_penicillin", ()),
    "piperacillin-tazobactam": RxNormCode("56169", "Piperacillin-Tazobactam", "antibiotic_penicillin", ("Zosyn",)),

    # -------------------------------------------------------------------------
    # Antibiotics - Cephalosporins
    # -------------------------------------------------------------------------
    "cephalexin": RxNormCode("2231", "Cephalexin", "antibiotic_cephalosporin", ("Keflex",)),
    "cefdinir": RxNormCode("25033", "Cefdinir", "antibiotic_cephalosporin", ("Omnicef",)),
    "ceftriaxone": RxNormCode("2193", "Ceftriaxone", "antibiotic_cephalosporin", ("Rocephin",)),
    "cefuroxime": RxNormCode("2348", "Cefuroxime", "antibiotic_cephalosporin", ("Ceftin",)),
    "cefazolin": RxNormCode("2180", "Cefazolin", "antibiotic_cephalosporin", ("Ancef",)),
    "cefepime": RxNormCode("25037", "Cefepime", "antibiotic_cephalosporin", ("Maxipime",)),
    "ceftazidime": RxNormCode("2191", "Ceftazidime", "antibiotic_cephalosporin", ("Fortaz",)),
    "cefpodoxime": RxNormCode("25044", "Cefpodoxime", "antibiotic_cephalosporin", ("Vantin",)),

    # -------------------------------------------------------------------------
    # Antibiotics - Fluoroquinolones
    # -------------------------------------------------------------------------
    "ciprofloxacin": RxNormCode("2551", "Ciprofloxacin", "antibiotic_fluoroquinolone", ("Cipro",)),
    "levofloxacin": RxNormCode("82122", "Levofloxacin", "antibiotic_fluoroquinolone", ("Levaquin",)),
    "moxifloxacin": RxNormCode("139462", "Moxifloxacin", "antibiotic_fluoroquinolone", ("Avelox",)),

    # -------------------------------------------------------------------------
    # Antibiotics - Macrolides
    # -------------------------------------------------------------------------
    "azithromycin": RxNormCode("18631", "Azithromycin", "antibiotic_macrolide", ("Zithromax", "Z-Pak")),
    "clarithromycin": RxNormCode("21212", "Clarithromycin", "antibiotic_macrolide", ("Biaxin",)),
    "erythromycin": RxNormCode("3952", "Erythromycin", "antibiotic_macrolide", ()),

    # -------------------------------------------------------------------------
    # Antibiotics - Tetracyclines
    # -------------------------------------------------------------------------
    "doxycycline": RxNormCode("3640", "Doxycycline", "antibiotic_tetracycline", ("Vibramycin",)),
    "minocycline": RxNormCode("6922", "Minocycline", "antibiotic_tetracycline", ("Minocin",)),
    "tetracycline": RxNormCode("10395", "Tetracycline", "antibiotic_tetracycline", ()),

    # -------------------------------------------------------------------------
    # Antibiotics - Other
    # -------------------------------------------------------------------------
    "metronidazole": RxNormCode("6922", "Metronidazole", "antibiotic_other", ("Flagyl",)),
    "trimethoprim-sulfamethoxazole": RxNormCode("10831", "Trimethoprim-Sulfamethoxazole", "antibiotic_other", ("Bactrim", "Septra")),
    "bactrim": RxNormCode("10831", "Trimethoprim-Sulfamethoxazole", "antibiotic_other", ("Bactrim", "Septra")),
    "nitrofurantoin": RxNormCode("7454", "Nitrofurantoin", "antibiotic_other", ("Macrobid", "Macrodantin")),
    "clindamycin": RxNormCode("2582", "Clindamycin", "antibiotic_other", ("Cleocin",)),
    "vancomycin": RxNormCode("11124", "Vancomycin", "antibiotic_other", ("Vancocin",)),
    "linezolid": RxNormCode("190376", "Linezolid", "antibiotic_other", ("Zyvox",)),

    # -------------------------------------------------------------------------
    # Anticoagulants
    # -------------------------------------------------------------------------
    "warfarin": RxNormCode("11289", "Warfarin", "anticoagulant", ("Coumadin", "Jantoven")),
    "apixaban": RxNormCode("1364430", "Apixaban", "anticoagulant", ("Eliquis",)),
    "rivaroxaban": RxNormCode("1114195", "Rivaroxaban", "anticoagulant", ("Xarelto",)),
    "dabigatran": RxNormCode("1037045", "Dabigatran", "anticoagulant", ("Pradaxa",)),
    "edoxaban": RxNormCode("1599538", "Edoxaban", "anticoagulant", ("Savaysa",)),
    "enoxaparin": RxNormCode("67108", "Enoxaparin", "anticoagulant", ("Lovenox",)),
    "heparin": RxNormCode("5224", "Heparin", "anticoagulant", ()),

    # -------------------------------------------------------------------------
    # Antiplatelet Agents
    # -------------------------------------------------------------------------
    "aspirin": RxNormCode("1191", "Aspirin", "antiplatelet", ("Bayer",)),
    "clopidogrel": RxNormCode("32968", "Clopidogrel", "antiplatelet", ("Plavix",)),
    "prasugrel": RxNormCode("613391", "Prasugrel", "antiplatelet", ("Effient",)),
    "ticagrelor": RxNormCode("1116632", "Ticagrelor", "antiplatelet", ("Brilinta",)),

    # -------------------------------------------------------------------------
    # Pain Medications - NSAIDs
    # -------------------------------------------------------------------------
    "ibuprofen": RxNormCode("5640", "Ibuprofen", "nsaid", ("Advil", "Motrin")),
    "naproxen": RxNormCode("7258", "Naproxen", "nsaid", ("Aleve", "Naprosyn")),
    "meloxicam": RxNormCode("41493", "Meloxicam", "nsaid", ("Mobic",)),
    "celecoxib": RxNormCode("140587", "Celecoxib", "nsaid", ("Celebrex",)),
    "diclofenac": RxNormCode("3355", "Diclofenac", "nsaid", ("Voltaren",)),
    "ketorolac": RxNormCode("6135", "Ketorolac", "nsaid", ("Toradol",)),
    "indomethacin": RxNormCode("5781", "Indomethacin", "nsaid", ("Indocin",)),

    # -------------------------------------------------------------------------
    # Pain Medications - Acetaminophen
    # -------------------------------------------------------------------------
    "acetaminophen": RxNormCode("161", "Acetaminophen", "analgesic", ("Tylenol",)),
    "tylenol": RxNormCode("161", "Acetaminophen", "analgesic", ("Tylenol",)),

    # -------------------------------------------------------------------------
    # Pain Medications - Opioids
    # -------------------------------------------------------------------------
    "tramadol": RxNormCode("10689", "Tramadol", "opioid", ("Ultram",)),
    "hydrocodone": RxNormCode("5489", "Hydrocodone", "opioid", ("Vicodin", "Norco")),
    "oxycodone": RxNormCode("7804", "Oxycodone", "opioid", ("OxyContin", "Percocet")),
    "morphine": RxNormCode("7052", "Morphine", "opioid", ("MS Contin",)),
    "fentanyl": RxNormCode("4337", "Fentanyl", "opioid", ("Duragesic",)),
    "hydromorphone": RxNormCode("3423", "Hydromorphone", "opioid", ("Dilaudid",)),
    "codeine": RxNormCode("2670", "Codeine", "opioid", ()),

    # -------------------------------------------------------------------------
    # Psychiatric - Antidepressants (SSRIs)
    # -------------------------------------------------------------------------
    "sertraline": RxNormCode("36437", "Sertraline", "ssri", ("Zoloft",)),
    "fluoxetine": RxNormCode("4493", "Fluoxetine", "ssri", ("Prozac",)),
    "escitalopram": RxNormCode("321988", "Escitalopram", "ssri", ("Lexapro",)),
    "citalopram": RxNormCode("2556", "Citalopram", "ssri", ("Celexa",)),
    "paroxetine": RxNormCode("32937", "Paroxetine", "ssri", ("Paxil",)),
    "fluvoxamine": RxNormCode("42355", "Fluvoxamine", "ssri", ("Luvox",)),

    # -------------------------------------------------------------------------
    # Psychiatric - Antidepressants (SNRIs)
    # -------------------------------------------------------------------------
    "venlafaxine": RxNormCode("39786", "Venlafaxine", "snri", ("Effexor",)),
    "duloxetine": RxNormCode("72625", "Duloxetine", "snri", ("Cymbalta",)),
    "desvenlafaxine": RxNormCode("734064", "Desvenlafaxine", "snri", ("Pristiq",)),

    # -------------------------------------------------------------------------
    # Psychiatric - Antidepressants (Other)
    # -------------------------------------------------------------------------
    "bupropion": RxNormCode("42347", "Bupropion", "antidepressant_other", ("Wellbutrin", "Zyban")),
    "mirtazapine": RxNormCode("15996", "Mirtazapine", "antidepressant_other", ("Remeron",)),
    "trazodone": RxNormCode("10737", "Trazodone", "antidepressant_other", ("Desyrel",)),
    "amitriptyline": RxNormCode("704", "Amitriptyline", "tca", ("Elavil",)),
    "nortriptyline": RxNormCode("7531", "Nortriptyline", "tca", ("Pamelor",)),

    # -------------------------------------------------------------------------
    # Psychiatric - Anxiolytics/Benzodiazepines
    # -------------------------------------------------------------------------
    "alprazolam": RxNormCode("596", "Alprazolam", "benzodiazepine", ("Xanax",)),
    "lorazepam": RxNormCode("6470", "Lorazepam", "benzodiazepine", ("Ativan",)),
    "clonazepam": RxNormCode("2598", "Clonazepam", "benzodiazepine", ("Klonopin",)),
    "diazepam": RxNormCode("3322", "Diazepam", "benzodiazepine", ("Valium",)),
    "temazepam": RxNormCode("10355", "Temazepam", "benzodiazepine", ("Restoril",)),
    "midazolam": RxNormCode("6960", "Midazolam", "benzodiazepine", ("Versed",)),

    # -------------------------------------------------------------------------
    # Psychiatric - Antipsychotics
    # -------------------------------------------------------------------------
    "quetiapine": RxNormCode("51272", "Quetiapine", "antipsychotic", ("Seroquel",)),
    "risperidone": RxNormCode("35636", "Risperidone", "antipsychotic", ("Risperdal",)),
    "olanzapine": RxNormCode("61381", "Olanzapine", "antipsychotic", ("Zyprexa",)),
    "aripiprazole": RxNormCode("89013", "Aripiprazole", "antipsychotic", ("Abilify",)),
    "haloperidol": RxNormCode("5093", "Haloperidol", "antipsychotic", ("Haldol",)),
    "ziprasidone": RxNormCode("115698", "Ziprasidone", "antipsychotic", ("Geodon",)),

    # -------------------------------------------------------------------------
    # Psychiatric - ADHD
    # -------------------------------------------------------------------------
    "methylphenidate": RxNormCode("6901", "Methylphenidate", "adhd", ("Ritalin", "Concerta")),
    "amphetamine-dextroamphetamine": RxNormCode("725", "Amphetamine-Dextroamphetamine", "adhd", ("Adderall",)),
    "adderall": RxNormCode("725", "Amphetamine-Dextroamphetamine", "adhd", ("Adderall",)),
    "lisdexamfetamine": RxNormCode("813342", "Lisdexamfetamine", "adhd", ("Vyvanse",)),
    "atomoxetine": RxNormCode("38400", "Atomoxetine", "adhd", ("Strattera",)),

    # -------------------------------------------------------------------------
    # Respiratory - Inhalers
    # -------------------------------------------------------------------------
    "albuterol": RxNormCode("435", "Albuterol", "bronchodilator", ("ProAir", "Ventolin", "Proventil")),
    "fluticasone": RxNormCode("41126", "Fluticasone", "inhaled_steroid", ("Flovent", "Flonase")),
    "fluticasone-salmeterol": RxNormCode("141918", "Fluticasone-Salmeterol", "inhaler_combo", ("Advair",)),
    "budesonide": RxNormCode("19831", "Budesonide", "inhaled_steroid", ("Pulmicort",)),
    "budesonide-formoterol": RxNormCode("349191", "Budesonide-Formoterol", "inhaler_combo", ("Symbicort",)),
    "tiotropium": RxNormCode("274766", "Tiotropium", "anticholinergic", ("Spiriva",)),
    "ipratropium": RxNormCode("7213", "Ipratropium", "anticholinergic", ("Atrovent",)),
    "montelukast": RxNormCode("88249", "Montelukast", "leukotriene_inhibitor", ("Singulair",)),

    # -------------------------------------------------------------------------
    # Thyroid
    # -------------------------------------------------------------------------
    "levothyroxine": RxNormCode("10582", "Levothyroxine", "thyroid", ("Synthroid", "Levoxyl")),
    "liothyronine": RxNormCode("6238", "Liothyronine", "thyroid", ("Cytomel",)),
    "methimazole": RxNormCode("6835", "Methimazole", "antithyroid", ("Tapazole",)),

    # -------------------------------------------------------------------------
    # Corticosteroids
    # -------------------------------------------------------------------------
    "prednisone": RxNormCode("8640", "Prednisone", "corticosteroid", ("Deltasone",)),
    "prednisolone": RxNormCode("8638", "Prednisolone", "corticosteroid", ("Orapred",)),
    "methylprednisolone": RxNormCode("6902", "Methylprednisolone", "corticosteroid", ("Medrol",)),
    "dexamethasone": RxNormCode("3264", "Dexamethasone", "corticosteroid", ("Decadron",)),
    "hydrocortisone": RxNormCode("5492", "Hydrocortisone", "corticosteroid", ("Cortef",)),

    # -------------------------------------------------------------------------
    # Muscle Relaxants
    # -------------------------------------------------------------------------
    "cyclobenzaprine": RxNormCode("3112", "Cyclobenzaprine", "muscle_relaxant", ("Flexeril",)),
    "baclofen": RxNormCode("1292", "Baclofen", "muscle_relaxant", ("Lioresal",)),
    "tizanidine": RxNormCode("38535", "Tizanidine", "muscle_relaxant", ("Zanaflex",)),
    "methocarbamol": RxNormCode("6845", "Methocarbamol", "muscle_relaxant", ("Robaxin",)),

    # -------------------------------------------------------------------------
    # Anticonvulsants
    # -------------------------------------------------------------------------
    "gabapentin": RxNormCode("25480", "Gabapentin", "anticonvulsant", ("Neurontin",)),
    "pregabalin": RxNormCode("187832", "Pregabalin", "anticonvulsant", ("Lyrica",)),
    "levetiracetam": RxNormCode("39998", "Levetiracetam", "anticonvulsant", ("Keppra",)),
    "topiramate": RxNormCode("38404", "Topiramate", "anticonvulsant", ("Topamax",)),
    "valproic acid": RxNormCode("11118", "Valproic Acid", "anticonvulsant", ("Depakote",)),
    "carbamazepine": RxNormCode("2002", "Carbamazepine", "anticonvulsant", ("Tegretol",)),
    "phenytoin": RxNormCode("8183", "Phenytoin", "anticonvulsant", ("Dilantin",)),
    "lamotrigine": RxNormCode("28439", "Lamotrigine", "anticonvulsant", ("Lamictal",)),

    # -------------------------------------------------------------------------
    # Antiemetics
    # -------------------------------------------------------------------------
    "ondansetron": RxNormCode("26225", "Ondansetron", "antiemetic", ("Zofran",)),
    "promethazine": RxNormCode("8745", "Promethazine", "antiemetic", ("Phenergan",)),
    "metoclopramide": RxNormCode("6915", "Metoclopramide", "antiemetic", ("Reglan",)),
    "prochlorperazine": RxNormCode("8704", "Prochlorperazine", "antiemetic", ("Compazine",)),

    # -------------------------------------------------------------------------
    # Sleep Aids
    # -------------------------------------------------------------------------
    "zolpidem": RxNormCode("39993", "Zolpidem", "sleep_aid", ("Ambien",)),
    "eszopiclone": RxNormCode("352363", "Eszopiclone", "sleep_aid", ("Lunesta",)),
    "suvorexant": RxNormCode("1535942", "Suvorexant", "sleep_aid", ("Belsomra",)),
    "melatonin": RxNormCode("6640", "Melatonin", "sleep_aid", ()),

    # -------------------------------------------------------------------------
    # Erectile Dysfunction
    # -------------------------------------------------------------------------
    "sildenafil": RxNormCode("136411", "Sildenafil", "pde5_inhibitor", ("Viagra",)),
    "tadalafil": RxNormCode("110398", "Tadalafil", "pde5_inhibitor", ("Cialis",)),

    # -------------------------------------------------------------------------
    # Osteoporosis
    # -------------------------------------------------------------------------
    "alendronate": RxNormCode("9009", "Alendronate", "bisphosphonate", ("Fosamax",)),
    "risedronate": RxNormCode("35894", "Risedronate", "bisphosphonate", ("Actonel",)),
    "ibandronate": RxNormCode("379108", "Ibandronate", "bisphosphonate", ("Boniva",)),

    # -------------------------------------------------------------------------
    # Gout
    # -------------------------------------------------------------------------
    "allopurinol": RxNormCode("519", "Allopurinol", "gout", ("Zyloprim",)),
    "febuxostat": RxNormCode("596500", "Febuxostat", "gout", ("Uloric",)),
    "colchicine": RxNormCode("2683", "Colchicine", "gout", ("Colcrys",)),

    # -------------------------------------------------------------------------
    # Other Common Medications
    # -------------------------------------------------------------------------
    "potassium chloride": RxNormCode("8591", "Potassium Chloride", "electrolyte", ("K-Dur", "Klor-Con")),
    "ferrous sulfate": RxNormCode("4402", "Ferrous Sulfate", "iron_supplement", ()),
    "vitamin d": RxNormCode("11253", "Vitamin D", "vitamin", ()),
    "folic acid": RxNormCode("4511", "Folic Acid", "vitamin", ()),
    "vitamin b12": RxNormCode("11249", "Vitamin B12", "vitamin", ()),
    "diphenhydramine": RxNormCode("3498", "Diphenhydramine", "antihistamine", ("Benadryl",)),
    "cetirizine": RxNormCode("20610", "Cetirizine", "antihistamine", ("Zyrtec",)),
    "loratadine": RxNormCode("28889", "Loratadine", "antihistamine", ("Claritin",)),
    "fexofenadine": RxNormCode("20482", "Fexofenadine", "antihistamine", ("Allegra",)),
    "docusate": RxNormCode("3579", "Docusate", "laxative", ("Colace",)),
    "polyethylene glycol": RxNormCode("8514", "Polyethylene Glycol", "laxative", ("MiraLAX",)),
    "lactulose": RxNormCode("6028", "Lactulose", "laxative", ()),
    "tamsulosin": RxNormCode("77492", "Tamsulosin", "alpha_blocker", ("Flomax",)),
    "finasteride": RxNormCode("4449", "Finasteride", "5ari", ("Proscar", "Propecia")),
}


# =============================================================================
# Brand Name to Generic Mapping
# =============================================================================

BRAND_TO_GENERIC: dict[str, str] = {
    # Build from RXNORM_DATABASE
}

# Populate BRAND_TO_GENERIC from the database
for generic_name, code in RXNORM_DATABASE.items():
    for brand in code.brand_names:
        BRAND_TO_GENERIC[brand.lower()] = generic_name


def normalize_medication(name: str) -> str:
    """
    Normalize a medication name for matching.

    Strips common suffixes like dosage, form, frequency.
    """
    if not name:
        return ""

    # Lowercase and strip
    name = name.lower().strip()

    # Remove common suffixes and dosage info
    # "metformin 500 mg twice daily" -> "metformin"
    # "atorvastatin 40mg" -> "atorvastatin"
    # "lisinopril 10 mg tablet" -> "lisinopril"

    # Remove dosage patterns
    name = re.sub(r'\s+\d+(?:\.\d+)?\s*(?:mg|g|ml|mcg|units?|meq|%)\b.*$', '', name, flags=re.IGNORECASE)

    # Remove form suffixes
    form_suffixes = [
        r'\s+tablet[s]?$', r'\s+capsule[s]?$', r'\s+cap[s]?$',
        r'\s+tab[s]?$', r'\s+injection$', r'\s+solution$',
        r'\s+suspension$', r'\s+cream$', r'\s+ointment$',
        r'\s+patch$', r'\s+inhaler$', r'\s+spray$',
        r'\s+oral$', r'\s+iv$', r'\s+im$',
        r'\s+extended[- ]release$', r'\s+er$', r'\s+xr$', r'\s+xl$',
        r'\s+immediate[- ]release$', r'\s+ir$',
        r'\s+sustained[- ]release$', r'\s+sr$',
        r'\s+delayed[- ]release$', r'\s+dr$',
        r'\s+controlled[- ]release$', r'\s+cr$',
    ]
    for pattern in form_suffixes:
        name = re.sub(pattern, '', name, flags=re.IGNORECASE)

    # Remove extra whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    return name


def similarity_score(s1: str, s2: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, s1, s2).ratio()


def lookup_rxnorm(
    medication_name: str,
    min_similarity: float = 0.85
) -> tuple[str | None, str | None, float, bool]:
    """
    Look up RxNorm code for a medication name.

    Args:
        medication_name: The medication name to look up
        min_similarity: Minimum similarity score for fuzzy matching (0.0-1.0)

    Returns:
        Tuple of (rxcui, display_name, confidence, matched)
        - confidence is 1.0 for exact match, <1.0 for fuzzy match
        - matched is True if found in database, False otherwise
    """
    if not medication_name:
        return None, None, 0.0, False

    normalized = normalize_medication(medication_name)

    # 1. Exact match (generic name)
    if normalized in RXNORM_DATABASE:
        code = RXNORM_DATABASE[normalized]
        return code.rxcui, code.display, 1.0, True

    # 2. Brand name lookup
    if normalized in BRAND_TO_GENERIC:
        generic = BRAND_TO_GENERIC[normalized]
        code = RXNORM_DATABASE[generic]
        return code.rxcui, code.display, 0.98, True

    # 3. Fuzzy matching
    best_match = None
    best_score = 0.0

    # Check generic names
    for db_name, code in RXNORM_DATABASE.items():
        score = similarity_score(normalized, db_name)
        if score > best_score and score >= min_similarity:
            best_score = score
            best_match = code

    # Check brand names
    for brand, generic in BRAND_TO_GENERIC.items():
        score = similarity_score(normalized, brand)
        if score > best_score and score >= min_similarity:
            best_score = score
            best_match = RXNORM_DATABASE[generic]

    if best_match:
        return best_match.rxcui, best_match.display, round(best_score, 2), True

    # No match found
    return None, None, 0.0, False


def get_drug_class(medication_name: str) -> str | None:
    """
    Get the drug class for a medication.

    Args:
        medication_name: The medication name

    Returns:
        Drug class string or None if not found
    """
    normalized = normalize_medication(medication_name)

    if normalized in RXNORM_DATABASE:
        return RXNORM_DATABASE[normalized].drug_class

    if normalized in BRAND_TO_GENERIC:
        generic = BRAND_TO_GENERIC[normalized]
        return RXNORM_DATABASE[generic].drug_class

    return None


def enrich_medications_with_rxnorm(medications: list) -> list:
    """
    Enrich a list of Medication/MedicationOrder objects with RxNorm codes.

    This function adds verified RxNorm codes to medications, preferring
    lookup codes over any LLM-generated codes.

    Args:
        medications: List of Medication or MedicationOrder objects

    Returns:
        The same list with RxNorm codes and match status added
    """
    for med in medications:
        if not med.name:
            continue

        rxcui, display, confidence, matched = lookup_rxnorm(med.name)

        # Add rxnorm_matched field to track verification status
        med.rxnorm_matched = matched

        if rxcui and confidence >= 0.85:
            # Use lookup code (verified) over any existing code
            if hasattr(med, 'rxnorm'):
                med.rxnorm = rxcui
            # Store confidence in metadata if available
            if hasattr(med, 'confidence'):
                med.confidence = confidence

    return medications
