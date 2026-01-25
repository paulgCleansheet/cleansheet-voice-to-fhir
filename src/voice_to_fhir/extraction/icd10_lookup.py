"""
ICD-10-CM Lookup Database for Clinical Entity Coding.

This module provides verified ICD-10-CM codes for common clinical conditions.
Instead of relying on LLM-generated codes (which can hallucinate), this lookup
provides deterministic, auditable coding.

Flow:
    MedGemma extracts condition name → fuzzy match → ICD-10 lookup → verified code
"""

import re
from difflib import SequenceMatcher
from typing import NamedTuple


class ICD10Code(NamedTuple):
    """ICD-10 code with metadata."""
    code: str
    display: str
    category: str  # e.g., "cardiovascular", "respiratory", "endocrine"


# =============================================================================
# ICD-10-CM Database (~500 common conditions)
# Organized by clinical category for maintainability
# =============================================================================

ICD10_DATABASE: dict[str, ICD10Code] = {
    # -------------------------------------------------------------------------
    # Cardiovascular Conditions
    # -------------------------------------------------------------------------
    "acute coronary syndrome": ICD10Code("I24.9", "Acute coronary syndrome", "cardiovascular"),
    "acute myocardial infarction": ICD10Code("I21.9", "Acute myocardial infarction, unspecified", "cardiovascular"),
    "myocardial infarction": ICD10Code("I21.9", "Acute myocardial infarction, unspecified", "cardiovascular"),
    "mi": ICD10Code("I21.9", "Acute myocardial infarction, unspecified", "cardiovascular"),
    "heart attack": ICD10Code("I21.9", "Acute myocardial infarction, unspecified", "cardiovascular"),
    "nstemi": ICD10Code("I21.4", "Non-ST elevation myocardial infarction", "cardiovascular"),
    "stemi": ICD10Code("I21.3", "ST elevation myocardial infarction", "cardiovascular"),
    "unstable angina": ICD10Code("I20.0", "Unstable angina", "cardiovascular"),
    "stable angina": ICD10Code("I20.9", "Angina pectoris, unspecified", "cardiovascular"),
    "angina": ICD10Code("I20.9", "Angina pectoris, unspecified", "cardiovascular"),
    "chest pain": ICD10Code("R07.9", "Chest pain, unspecified", "cardiovascular"),
    "atypical chest pain": ICD10Code("R07.89", "Other chest pain", "cardiovascular"),
    "hypertension": ICD10Code("I10", "Essential hypertension", "cardiovascular"),
    "essential hypertension": ICD10Code("I10", "Essential hypertension", "cardiovascular"),
    "high blood pressure": ICD10Code("I10", "Essential hypertension", "cardiovascular"),
    "htn": ICD10Code("I10", "Essential hypertension", "cardiovascular"),
    "hypertensive emergency": ICD10Code("I16.1", "Hypertensive emergency", "cardiovascular"),
    "hypertensive urgency": ICD10Code("I16.0", "Hypertensive urgency", "cardiovascular"),
    "hypotension": ICD10Code("I95.9", "Hypotension, unspecified", "cardiovascular"),
    "heart failure": ICD10Code("I50.9", "Heart failure, unspecified", "cardiovascular"),
    "congestive heart failure": ICD10Code("I50.9", "Heart failure, unspecified", "cardiovascular"),
    "chf": ICD10Code("I50.9", "Heart failure, unspecified", "cardiovascular"),
    "atrial fibrillation": ICD10Code("I48.91", "Unspecified atrial fibrillation", "cardiovascular"),
    "afib": ICD10Code("I48.91", "Unspecified atrial fibrillation", "cardiovascular"),
    "atrial flutter": ICD10Code("I48.92", "Unspecified atrial flutter", "cardiovascular"),
    "ventricular tachycardia": ICD10Code("I47.2", "Ventricular tachycardia", "cardiovascular"),
    "vtach": ICD10Code("I47.2", "Ventricular tachycardia", "cardiovascular"),
    "supraventricular tachycardia": ICD10Code("I47.1", "Supraventricular tachycardia", "cardiovascular"),
    "svt": ICD10Code("I47.1", "Supraventricular tachycardia", "cardiovascular"),
    "bradycardia": ICD10Code("R00.1", "Bradycardia, unspecified", "cardiovascular"),
    "tachycardia": ICD10Code("R00.0", "Tachycardia, unspecified", "cardiovascular"),
    "palpitations": ICD10Code("R00.2", "Palpitations", "cardiovascular"),
    "coronary artery disease": ICD10Code("I25.10", "Atherosclerotic heart disease", "cardiovascular"),
    "cad": ICD10Code("I25.10", "Atherosclerotic heart disease", "cardiovascular"),
    "peripheral artery disease": ICD10Code("I73.9", "Peripheral vascular disease, unspecified", "cardiovascular"),
    "pad": ICD10Code("I73.9", "Peripheral vascular disease, unspecified", "cardiovascular"),
    "deep vein thrombosis": ICD10Code("I82.40", "Acute embolism and thrombosis of unspecified deep veins of lower extremity", "cardiovascular"),
    "dvt": ICD10Code("I82.40", "Acute embolism and thrombosis of unspecified deep veins of lower extremity", "cardiovascular"),
    "pulmonary embolism": ICD10Code("I26.99", "Other pulmonary embolism without acute cor pulmonale", "cardiovascular"),
    "pe": ICD10Code("I26.99", "Other pulmonary embolism without acute cor pulmonale", "cardiovascular"),
    "aortic stenosis": ICD10Code("I35.0", "Nonrheumatic aortic valve stenosis", "cardiovascular"),
    "mitral regurgitation": ICD10Code("I34.0", "Nonrheumatic mitral valve insufficiency", "cardiovascular"),
    "cardiomyopathy": ICD10Code("I42.9", "Cardiomyopathy, unspecified", "cardiovascular"),
    "pericarditis": ICD10Code("I30.9", "Acute pericarditis, unspecified", "cardiovascular"),
    "endocarditis": ICD10Code("I33.9", "Acute and subacute endocarditis, unspecified", "cardiovascular"),

    # -------------------------------------------------------------------------
    # Cerebrovascular / Neurological
    # -------------------------------------------------------------------------
    "stroke": ICD10Code("I63.9", "Cerebral infarction, unspecified", "neurological"),
    "cerebrovascular accident": ICD10Code("I63.9", "Cerebral infarction, unspecified", "neurological"),
    "cva": ICD10Code("I63.9", "Cerebral infarction, unspecified", "neurological"),
    "ischemic stroke": ICD10Code("I63.9", "Cerebral infarction, unspecified", "neurological"),
    "hemorrhagic stroke": ICD10Code("I61.9", "Nontraumatic intracerebral hemorrhage, unspecified", "neurological"),
    "transient ischemic attack": ICD10Code("G45.9", "Transient cerebral ischemic attack, unspecified", "neurological"),
    "tia": ICD10Code("G45.9", "Transient cerebral ischemic attack, unspecified", "neurological"),
    "seizure": ICD10Code("R56.9", "Unspecified convulsions", "neurological"),
    "epilepsy": ICD10Code("G40.909", "Epilepsy, unspecified, not intractable, without status epilepticus", "neurological"),
    "headache": ICD10Code("R51.9", "Headache, unspecified", "neurological"),
    "migraine": ICD10Code("G43.909", "Migraine, unspecified, not intractable, without status migrainosus", "neurological"),
    "tension headache": ICD10Code("G44.209", "Tension-type headache, unspecified, not intractable", "neurological"),
    "cluster headache": ICD10Code("G44.009", "Cluster headache syndrome, unspecified, not intractable", "neurological"),
    "syncope": ICD10Code("R55", "Syncope and collapse", "neurological"),
    "dizziness": ICD10Code("R42", "Dizziness and giddiness", "neurological"),
    "vertigo": ICD10Code("R42", "Dizziness and giddiness", "neurological"),
    "dementia": ICD10Code("F03.90", "Unspecified dementia without behavioral disturbance", "neurological"),
    "alzheimer disease": ICD10Code("G30.9", "Alzheimer disease, unspecified", "neurological"),
    "parkinson disease": ICD10Code("G20", "Parkinson disease", "neurological"),
    "multiple sclerosis": ICD10Code("G35", "Multiple sclerosis", "neurological"),
    "neuropathy": ICD10Code("G62.9", "Polyneuropathy, unspecified", "neurological"),
    "peripheral neuropathy": ICD10Code("G62.9", "Polyneuropathy, unspecified", "neurological"),
    "diabetic neuropathy": ICD10Code("E11.40", "Type 2 diabetes mellitus with diabetic neuropathy, unspecified", "neurological"),
    "altered mental status": ICD10Code("R41.82", "Altered mental status, unspecified", "neurological"),
    "confusion": ICD10Code("R41.0", "Disorientation, unspecified", "neurological"),
    "delirium": ICD10Code("R41.0", "Disorientation, unspecified", "neurological"),
    "encephalopathy": ICD10Code("G93.40", "Encephalopathy, unspecified", "neurological"),
    "meningitis": ICD10Code("G03.9", "Meningitis, unspecified", "neurological"),

    # -------------------------------------------------------------------------
    # Respiratory Conditions
    # -------------------------------------------------------------------------
    "pneumonia": ICD10Code("J18.9", "Pneumonia, unspecified organism", "respiratory"),
    "community acquired pneumonia": ICD10Code("J18.9", "Pneumonia, unspecified organism", "respiratory"),
    "bacterial pneumonia": ICD10Code("J15.9", "Unspecified bacterial pneumonia", "respiratory"),
    "viral pneumonia": ICD10Code("J12.9", "Viral pneumonia, unspecified", "respiratory"),
    "aspiration pneumonia": ICD10Code("J69.0", "Pneumonitis due to inhalation of food and vomit", "respiratory"),
    "bronchitis": ICD10Code("J40", "Bronchitis, not specified as acute or chronic", "respiratory"),
    "acute bronchitis": ICD10Code("J20.9", "Acute bronchitis, unspecified", "respiratory"),
    "chronic bronchitis": ICD10Code("J42", "Unspecified chronic bronchitis", "respiratory"),
    "copd": ICD10Code("J44.9", "Chronic obstructive pulmonary disease, unspecified", "respiratory"),
    "chronic obstructive pulmonary disease": ICD10Code("J44.9", "Chronic obstructive pulmonary disease, unspecified", "respiratory"),
    "copd exacerbation": ICD10Code("J44.1", "Chronic obstructive pulmonary disease with acute exacerbation", "respiratory"),
    "asthma": ICD10Code("J45.909", "Unspecified asthma, uncomplicated", "respiratory"),
    "asthma exacerbation": ICD10Code("J45.901", "Unspecified asthma with acute exacerbation", "respiratory"),
    "acute asthma exacerbation": ICD10Code("J45.901", "Unspecified asthma with acute exacerbation", "respiratory"),
    "shortness of breath": ICD10Code("R06.02", "Shortness of breath", "respiratory"),
    "dyspnea": ICD10Code("R06.00", "Dyspnea, unspecified", "respiratory"),
    "respiratory distress": ICD10Code("R06.03", "Acute respiratory distress", "respiratory"),
    "acute respiratory distress syndrome": ICD10Code("J80", "Acute respiratory distress syndrome", "respiratory"),
    "ards": ICD10Code("J80", "Acute respiratory distress syndrome", "respiratory"),
    "respiratory failure": ICD10Code("J96.90", "Respiratory failure, unspecified, unspecified whether with hypoxia or hypercapnia", "respiratory"),
    "acute respiratory failure": ICD10Code("J96.00", "Acute respiratory failure, unspecified whether with hypoxia or hypercapnia", "respiratory"),
    "hypoxia": ICD10Code("R09.02", "Hypoxemia", "respiratory"),
    "hypoxemia": ICD10Code("R09.02", "Hypoxemia", "respiratory"),
    "cough": ICD10Code("R05.9", "Cough, unspecified", "respiratory"),
    "hemoptysis": ICD10Code("R04.2", "Hemoptysis", "respiratory"),
    "pleural effusion": ICD10Code("J90", "Pleural effusion, not elsewhere classified", "respiratory"),
    "pneumothorax": ICD10Code("J93.9", "Pneumothorax, unspecified", "respiratory"),
    "pulmonary fibrosis": ICD10Code("J84.10", "Pulmonary fibrosis, unspecified", "respiratory"),
    "sleep apnea": ICD10Code("G47.30", "Sleep apnea, unspecified", "respiratory"),
    "obstructive sleep apnea": ICD10Code("G47.33", "Obstructive sleep apnea", "respiratory"),
    "upper respiratory infection": ICD10Code("J06.9", "Acute upper respiratory infection, unspecified", "respiratory"),
    "uri": ICD10Code("J06.9", "Acute upper respiratory infection, unspecified", "respiratory"),
    "viral upper respiratory infection": ICD10Code("J06.9", "Acute upper respiratory infection, unspecified", "respiratory"),
    "sinusitis": ICD10Code("J32.9", "Chronic sinusitis, unspecified", "respiratory"),
    "acute sinusitis": ICD10Code("J01.90", "Acute sinusitis, unspecified", "respiratory"),
    "pharyngitis": ICD10Code("J02.9", "Acute pharyngitis, unspecified", "respiratory"),
    "strep throat": ICD10Code("J02.0", "Streptococcal pharyngitis", "respiratory"),
    "streptococcal pharyngitis": ICD10Code("J02.0", "Streptococcal pharyngitis", "respiratory"),
    "tonsillitis": ICD10Code("J03.90", "Acute tonsillitis, unspecified", "respiratory"),
    "influenza": ICD10Code("J11.1", "Influenza due to unidentified influenza virus with other respiratory manifestations", "respiratory"),
    "flu": ICD10Code("J11.1", "Influenza due to unidentified influenza virus with other respiratory manifestations", "respiratory"),
    "covid-19": ICD10Code("U07.1", "COVID-19", "respiratory"),
    "covid": ICD10Code("U07.1", "COVID-19", "respiratory"),
    "atelectasis": ICD10Code("J98.11", "Atelectasis", "respiratory"),

    # -------------------------------------------------------------------------
    # Endocrine / Metabolic
    # -------------------------------------------------------------------------
    "diabetes": ICD10Code("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    "diabetes mellitus": ICD10Code("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    "type 2 diabetes": ICD10Code("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    "type 2 diabetes mellitus": ICD10Code("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    "type two diabetes": ICD10Code("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    "type two diabetes mellitus": ICD10Code("E11.9", "Type 2 diabetes mellitus without complications", "endocrine"),
    "type 1 diabetes": ICD10Code("E10.9", "Type 1 diabetes mellitus without complications", "endocrine"),
    "type 1 diabetes mellitus": ICD10Code("E10.9", "Type 1 diabetes mellitus without complications", "endocrine"),
    "diabetic ketoacidosis": ICD10Code("E11.10", "Type 2 diabetes mellitus with ketoacidosis without coma", "endocrine"),
    "dka": ICD10Code("E11.10", "Type 2 diabetes mellitus with ketoacidosis without coma", "endocrine"),
    "hypoglycemia": ICD10Code("E16.2", "Hypoglycemia, unspecified", "endocrine"),
    "hyperglycemia": ICD10Code("R73.9", "Hyperglycemia, unspecified", "endocrine"),
    "prediabetes": ICD10Code("R73.03", "Prediabetes", "endocrine"),
    "hyperlipidemia": ICD10Code("E78.5", "Hyperlipidemia, unspecified", "endocrine"),
    "dyslipidemia": ICD10Code("E78.5", "Hyperlipidemia, unspecified", "endocrine"),
    "high cholesterol": ICD10Code("E78.00", "Pure hypercholesterolemia, unspecified", "endocrine"),
    "hypercholesterolemia": ICD10Code("E78.00", "Pure hypercholesterolemia, unspecified", "endocrine"),
    "hypothyroidism": ICD10Code("E03.9", "Hypothyroidism, unspecified", "endocrine"),
    "hyperthyroidism": ICD10Code("E05.90", "Thyrotoxicosis, unspecified without thyrotoxic crisis or storm", "endocrine"),
    "thyroid nodule": ICD10Code("E04.1", "Nontoxic single thyroid nodule", "endocrine"),
    "goiter": ICD10Code("E04.9", "Nontoxic goiter, unspecified", "endocrine"),
    "obesity": ICD10Code("E66.9", "Obesity, unspecified", "endocrine"),
    "morbid obesity": ICD10Code("E66.01", "Morbid (severe) obesity due to excess calories", "endocrine"),
    "metabolic syndrome": ICD10Code("E88.81", "Metabolic syndrome", "endocrine"),
    "gout": ICD10Code("M10.9", "Gout, unspecified", "endocrine"),
    "hyperuricemia": ICD10Code("E79.0", "Hyperuricemia without signs of inflammatory arthritis and tophaceous disease", "endocrine"),
    "hyponatremia": ICD10Code("E87.1", "Hypo-osmolality and hyponatremia", "endocrine"),
    "hypernatremia": ICD10Code("E87.0", "Hyperosmolality and hypernatremia", "endocrine"),
    "hypokalemia": ICD10Code("E87.6", "Hypokalemia", "endocrine"),
    "hyperkalemia": ICD10Code("E87.5", "Hyperkalemia", "endocrine"),
    "dehydration": ICD10Code("E86.0", "Dehydration", "endocrine"),
    "electrolyte imbalance": ICD10Code("E87.8", "Other disorders of electrolyte and fluid balance", "endocrine"),

    # -------------------------------------------------------------------------
    # Gastrointestinal
    # -------------------------------------------------------------------------
    "gerd": ICD10Code("K21.0", "Gastro-esophageal reflux disease with esophagitis", "gastrointestinal"),
    "gastroesophageal reflux disease": ICD10Code("K21.0", "Gastro-esophageal reflux disease with esophagitis", "gastrointestinal"),
    "acid reflux": ICD10Code("K21.9", "Gastro-esophageal reflux disease without esophagitis", "gastrointestinal"),
    "gastritis": ICD10Code("K29.70", "Gastritis, unspecified, without bleeding", "gastrointestinal"),
    "peptic ulcer": ICD10Code("K27.9", "Peptic ulcer, site unspecified, unspecified as acute or chronic, without hemorrhage or perforation", "gastrointestinal"),
    "gastric ulcer": ICD10Code("K25.9", "Gastric ulcer, unspecified as acute or chronic, without hemorrhage or perforation", "gastrointestinal"),
    "duodenal ulcer": ICD10Code("K26.9", "Duodenal ulcer, unspecified as acute or chronic, without hemorrhage or perforation", "gastrointestinal"),
    "gi bleed": ICD10Code("K92.2", "Gastrointestinal hemorrhage, unspecified", "gastrointestinal"),
    "gastrointestinal bleeding": ICD10Code("K92.2", "Gastrointestinal hemorrhage, unspecified", "gastrointestinal"),
    "upper gi bleed": ICD10Code("K92.0", "Hematemesis", "gastrointestinal"),
    "lower gi bleed": ICD10Code("K92.1", "Melena", "gastrointestinal"),
    "nausea": ICD10Code("R11.0", "Nausea", "gastrointestinal"),
    "vomiting": ICD10Code("R11.10", "Vomiting, unspecified", "gastrointestinal"),
    "nausea and vomiting": ICD10Code("R11.2", "Nausea with vomiting, unspecified", "gastrointestinal"),
    "diarrhea": ICD10Code("R19.7", "Diarrhea, unspecified", "gastrointestinal"),
    "constipation": ICD10Code("K59.00", "Constipation, unspecified", "gastrointestinal"),
    "abdominal pain": ICD10Code("R10.9", "Unspecified abdominal pain", "gastrointestinal"),
    "acute abdomen": ICD10Code("R10.0", "Acute abdomen", "gastrointestinal"),
    "appendicitis": ICD10Code("K35.80", "Unspecified acute appendicitis", "gastrointestinal"),
    "acute appendicitis": ICD10Code("K35.80", "Unspecified acute appendicitis", "gastrointestinal"),
    "cholecystitis": ICD10Code("K81.9", "Cholecystitis, unspecified", "gastrointestinal"),
    "acute cholecystitis": ICD10Code("K81.0", "Acute cholecystitis", "gastrointestinal"),
    "cholelithiasis": ICD10Code("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", "gastrointestinal"),
    "gallstones": ICD10Code("K80.20", "Calculus of gallbladder without cholecystitis without obstruction", "gastrointestinal"),
    "pancreatitis": ICD10Code("K85.9", "Acute pancreatitis, unspecified", "gastrointestinal"),
    "acute pancreatitis": ICD10Code("K85.9", "Acute pancreatitis, unspecified", "gastrointestinal"),
    "chronic pancreatitis": ICD10Code("K86.1", "Other chronic pancreatitis", "gastrointestinal"),
    "diverticulitis": ICD10Code("K57.92", "Diverticulitis of intestine, part unspecified, without perforation or abscess without bleeding", "gastrointestinal"),
    "diverticulosis": ICD10Code("K57.90", "Diverticulosis of intestine, part unspecified, without perforation or abscess without bleeding", "gastrointestinal"),
    "bowel obstruction": ICD10Code("K56.60", "Unspecified intestinal obstruction", "gastrointestinal"),
    "small bowel obstruction": ICD10Code("K56.60", "Unspecified intestinal obstruction", "gastrointestinal"),
    "ileus": ICD10Code("K56.7", "Ileus, unspecified", "gastrointestinal"),
    "irritable bowel syndrome": ICD10Code("K58.9", "Irritable bowel syndrome without diarrhea", "gastrointestinal"),
    "ibs": ICD10Code("K58.9", "Irritable bowel syndrome without diarrhea", "gastrointestinal"),
    "inflammatory bowel disease": ICD10Code("K50.90", "Crohn disease, unspecified, without complications", "gastrointestinal"),
    "crohn disease": ICD10Code("K50.90", "Crohn disease, unspecified, without complications", "gastrointestinal"),
    "ulcerative colitis": ICD10Code("K51.90", "Ulcerative colitis, unspecified, without complications", "gastrointestinal"),
    "celiac disease": ICD10Code("K90.0", "Celiac disease", "gastrointestinal"),
    "hepatitis": ICD10Code("K75.9", "Inflammatory liver disease, unspecified", "gastrointestinal"),
    "cirrhosis": ICD10Code("K74.60", "Unspecified cirrhosis of liver", "gastrointestinal"),
    "liver cirrhosis": ICD10Code("K74.60", "Unspecified cirrhosis of liver", "gastrointestinal"),
    "fatty liver": ICD10Code("K76.0", "Fatty (change of) liver, not elsewhere classified", "gastrointestinal"),
    "nafld": ICD10Code("K76.0", "Fatty (change of) liver, not elsewhere classified", "gastrointestinal"),
    "ascites": ICD10Code("R18.8", "Other ascites", "gastrointestinal"),
    "hemorrhoids": ICD10Code("K64.9", "Unspecified hemorrhoids", "gastrointestinal"),

    # -------------------------------------------------------------------------
    # Renal / Urological
    # -------------------------------------------------------------------------
    "acute kidney injury": ICD10Code("N17.9", "Acute kidney failure, unspecified", "renal"),
    "aki": ICD10Code("N17.9", "Acute kidney failure, unspecified", "renal"),
    "chronic kidney disease": ICD10Code("N18.9", "Chronic kidney disease, unspecified", "renal"),
    "ckd": ICD10Code("N18.9", "Chronic kidney disease, unspecified", "renal"),
    "chronic kidney disease stage 3": ICD10Code("N18.3", "Chronic kidney disease, stage 3", "renal"),
    "ckd stage 3": ICD10Code("N18.3", "Chronic kidney disease, stage 3", "renal"),
    "chronic kidney disease stage 4": ICD10Code("N18.4", "Chronic kidney disease, stage 4", "renal"),
    "chronic kidney disease stage 5": ICD10Code("N18.5", "Chronic kidney disease, stage 5", "renal"),
    "end stage renal disease": ICD10Code("N18.6", "End stage renal disease", "renal"),
    "esrd": ICD10Code("N18.6", "End stage renal disease", "renal"),
    "urinary tract infection": ICD10Code("N39.0", "Urinary tract infection, site not specified", "renal"),
    "uti": ICD10Code("N39.0", "Urinary tract infection, site not specified", "renal"),
    "pyelonephritis": ICD10Code("N10", "Acute tubulo-interstitial nephritis", "renal"),
    "acute pyelonephritis": ICD10Code("N10", "Acute tubulo-interstitial nephritis", "renal"),
    "cystitis": ICD10Code("N30.90", "Cystitis, unspecified without hematuria", "renal"),
    "kidney stones": ICD10Code("N20.0", "Calculus of kidney", "renal"),
    "nephrolithiasis": ICD10Code("N20.0", "Calculus of kidney", "renal"),
    "renal colic": ICD10Code("N23", "Unspecified renal colic", "renal"),
    "urinary retention": ICD10Code("R33.9", "Retention of urine, unspecified", "renal"),
    "hematuria": ICD10Code("R31.9", "Hematuria, unspecified", "renal"),
    "proteinuria": ICD10Code("R80.9", "Proteinuria, unspecified", "renal"),
    "benign prostatic hyperplasia": ICD10Code("N40.0", "Benign prostatic hyperplasia without lower urinary tract symptoms", "renal"),
    "bph": ICD10Code("N40.0", "Benign prostatic hyperplasia without lower urinary tract symptoms", "renal"),
    "urinary incontinence": ICD10Code("R32", "Unspecified urinary incontinence", "renal"),

    # -------------------------------------------------------------------------
    # Infectious Diseases
    # -------------------------------------------------------------------------
    "sepsis": ICD10Code("A41.9", "Sepsis, unspecified organism", "infectious"),
    "septic shock": ICD10Code("R65.21", "Severe sepsis with septic shock", "infectious"),
    "severe sepsis": ICD10Code("R65.20", "Severe sepsis without septic shock", "infectious"),
    "bacteremia": ICD10Code("R78.81", "Bacteremia", "infectious"),
    "cellulitis": ICD10Code("L03.90", "Cellulitis, unspecified", "infectious"),
    "abscess": ICD10Code("L02.91", "Cutaneous abscess, unspecified", "infectious"),
    "skin abscess": ICD10Code("L02.91", "Cutaneous abscess, unspecified", "infectious"),
    "wound infection": ICD10Code("T79.3XXA", "Post-traumatic wound infection, not elsewhere classified, initial encounter", "infectious"),
    "osteomyelitis": ICD10Code("M86.9", "Osteomyelitis, unspecified", "infectious"),
    "endocarditis": ICD10Code("I33.9", "Acute and subacute endocarditis, unspecified", "infectious"),
    "gastroenteritis": ICD10Code("K52.9", "Noninfective gastroenteritis and colitis, unspecified", "infectious"),
    "viral gastroenteritis": ICD10Code("A08.4", "Viral intestinal infection, unspecified", "infectious"),
    "c diff": ICD10Code("A04.72", "Enterocolitis due to Clostridium difficile, not specified as recurrent", "infectious"),
    "clostridium difficile": ICD10Code("A04.72", "Enterocolitis due to Clostridium difficile, not specified as recurrent", "infectious"),
    "mrsa": ICD10Code("A49.02", "Methicillin resistant Staphylococcus aureus infection, unspecified site", "infectious"),
    "herpes zoster": ICD10Code("B02.9", "Zoster without complications", "infectious"),
    "shingles": ICD10Code("B02.9", "Zoster without complications", "infectious"),
    "hiv": ICD10Code("B20", "Human immunodeficiency virus disease", "infectious"),
    "tuberculosis": ICD10Code("A15.9", "Respiratory tuberculosis unspecified", "infectious"),
    "tb": ICD10Code("A15.9", "Respiratory tuberculosis unspecified", "infectious"),
    "lyme disease": ICD10Code("A69.20", "Lyme disease, unspecified", "infectious"),

    # -------------------------------------------------------------------------
    # Musculoskeletal
    # -------------------------------------------------------------------------
    "osteoarthritis": ICD10Code("M19.90", "Unspecified osteoarthritis, unspecified site", "musculoskeletal"),
    "arthritis": ICD10Code("M19.90", "Unspecified osteoarthritis, unspecified site", "musculoskeletal"),
    "rheumatoid arthritis": ICD10Code("M06.9", "Rheumatoid arthritis, unspecified", "musculoskeletal"),
    "back pain": ICD10Code("M54.9", "Dorsalgia, unspecified", "musculoskeletal"),
    "low back pain": ICD10Code("M54.50", "Low back pain, unspecified", "musculoskeletal"),
    "lumbar pain": ICD10Code("M54.50", "Low back pain, unspecified", "musculoskeletal"),
    "neck pain": ICD10Code("M54.2", "Cervicalgia", "musculoskeletal"),
    "cervicalgia": ICD10Code("M54.2", "Cervicalgia", "musculoskeletal"),
    "sciatica": ICD10Code("M54.30", "Sciatica, unspecified side", "musculoskeletal"),
    "herniated disc": ICD10Code("M51.16", "Intervertebral disc degeneration, lumbar region", "musculoskeletal"),
    "spinal stenosis": ICD10Code("M48.00", "Spinal stenosis, site unspecified", "musculoskeletal"),
    "osteoporosis": ICD10Code("M81.0", "Age-related osteoporosis without current pathological fracture", "musculoskeletal"),
    "fracture": ICD10Code("T14.8XXA", "Other injury of unspecified body region, initial encounter", "musculoskeletal"),
    "hip fracture": ICD10Code("S72.90XA", "Unspecified fracture of unspecified femur, initial encounter for closed fracture", "musculoskeletal"),
    "fibromyalgia": ICD10Code("M79.7", "Fibromyalgia", "musculoskeletal"),
    "muscle strain": ICD10Code("M62.89", "Other specified disorders of muscle", "musculoskeletal"),
    "tendinitis": ICD10Code("M77.9", "Enthesopathy, unspecified", "musculoskeletal"),
    "bursitis": ICD10Code("M71.9", "Bursopathy, unspecified", "musculoskeletal"),
    "rotator cuff tear": ICD10Code("M75.100", "Unspecified rotator cuff tear or rupture of unspecified shoulder, not specified as traumatic", "musculoskeletal"),
    "carpal tunnel syndrome": ICD10Code("G56.00", "Carpal tunnel syndrome, unspecified upper limb", "musculoskeletal"),
    "joint pain": ICD10Code("M25.50", "Pain in unspecified joint", "musculoskeletal"),
    "knee pain": ICD10Code("M25.569", "Pain in unspecified knee", "musculoskeletal"),
    "shoulder pain": ICD10Code("M25.519", "Pain in unspecified shoulder", "musculoskeletal"),

    # -------------------------------------------------------------------------
    # Psychiatric / Mental Health
    # -------------------------------------------------------------------------
    "depression": ICD10Code("F32.9", "Major depressive disorder, single episode, unspecified", "psychiatric"),
    "major depressive disorder": ICD10Code("F32.9", "Major depressive disorder, single episode, unspecified", "psychiatric"),
    "anxiety": ICD10Code("F41.9", "Anxiety disorder, unspecified", "psychiatric"),
    "anxiety disorder": ICD10Code("F41.9", "Anxiety disorder, unspecified", "psychiatric"),
    "generalized anxiety disorder": ICD10Code("F41.1", "Generalized anxiety disorder", "psychiatric"),
    "panic disorder": ICD10Code("F41.0", "Panic disorder without agoraphobia", "psychiatric"),
    "panic attack": ICD10Code("F41.0", "Panic disorder without agoraphobia", "psychiatric"),
    "bipolar disorder": ICD10Code("F31.9", "Bipolar disorder, unspecified", "psychiatric"),
    "schizophrenia": ICD10Code("F20.9", "Schizophrenia, unspecified", "psychiatric"),
    "ptsd": ICD10Code("F43.10", "Post-traumatic stress disorder, unspecified", "psychiatric"),
    "post traumatic stress disorder": ICD10Code("F43.10", "Post-traumatic stress disorder, unspecified", "psychiatric"),
    "ocd": ICD10Code("F42.9", "Obsessive-compulsive disorder, unspecified", "psychiatric"),
    "obsessive compulsive disorder": ICD10Code("F42.9", "Obsessive-compulsive disorder, unspecified", "psychiatric"),
    "adhd": ICD10Code("F90.9", "Attention-deficit hyperactivity disorder, unspecified type", "psychiatric"),
    "attention deficit hyperactivity disorder": ICD10Code("F90.9", "Attention-deficit hyperactivity disorder, unspecified type", "psychiatric"),
    "insomnia": ICD10Code("G47.00", "Insomnia, unspecified", "psychiatric"),
    "alcohol use disorder": ICD10Code("F10.20", "Alcohol dependence, uncomplicated", "psychiatric"),
    "alcohol abuse": ICD10Code("F10.10", "Alcohol abuse, uncomplicated", "psychiatric"),
    "alcohol withdrawal": ICD10Code("F10.239", "Alcohol dependence with withdrawal, unspecified", "psychiatric"),
    "opioid use disorder": ICD10Code("F11.20", "Opioid dependence, uncomplicated", "psychiatric"),
    "opioid overdose": ICD10Code("T40.2X1A", "Poisoning by other opioids, accidental (unintentional), initial encounter", "psychiatric"),
    "substance abuse": ICD10Code("F19.10", "Other psychoactive substance abuse, uncomplicated", "psychiatric"),
    "suicidal ideation": ICD10Code("R45.851", "Suicidal ideations", "psychiatric"),

    # -------------------------------------------------------------------------
    # Dermatological
    # -------------------------------------------------------------------------
    "rash": ICD10Code("R21", "Rash and other nonspecific skin eruption", "dermatological"),
    "dermatitis": ICD10Code("L30.9", "Dermatitis, unspecified", "dermatological"),
    "eczema": ICD10Code("L30.9", "Dermatitis, unspecified", "dermatological"),
    "psoriasis": ICD10Code("L40.9", "Psoriasis, unspecified", "dermatological"),
    "urticaria": ICD10Code("L50.9", "Urticaria, unspecified", "dermatological"),
    "hives": ICD10Code("L50.9", "Urticaria, unspecified", "dermatological"),
    "skin ulcer": ICD10Code("L98.9", "Disorder of the skin and subcutaneous tissue, unspecified", "dermatological"),
    "pressure ulcer": ICD10Code("L89.90", "Pressure ulcer of unspecified site, unspecified stage", "dermatological"),
    "decubitus ulcer": ICD10Code("L89.90", "Pressure ulcer of unspecified site, unspecified stage", "dermatological"),
    "diabetic foot ulcer": ICD10Code("E11.621", "Type 2 diabetes mellitus with foot ulcer", "dermatological"),
    "acne": ICD10Code("L70.9", "Acne, unspecified", "dermatological"),
    "skin cancer": ICD10Code("C44.90", "Other and unspecified malignant neoplasm of skin, unspecified", "dermatological"),
    "melanoma": ICD10Code("C43.9", "Malignant melanoma of skin, unspecified", "dermatological"),

    # -------------------------------------------------------------------------
    # Hematological / Oncological
    # -------------------------------------------------------------------------
    "anemia": ICD10Code("D64.9", "Anemia, unspecified", "hematological"),
    "iron deficiency anemia": ICD10Code("D50.9", "Iron deficiency anemia, unspecified", "hematological"),
    "anemia of chronic disease": ICD10Code("D63.8", "Anemia in other chronic diseases classified elsewhere", "hematological"),
    "thrombocytopenia": ICD10Code("D69.6", "Thrombocytopenia, unspecified", "hematological"),
    "leukocytosis": ICD10Code("D72.829", "Elevated white blood cell count, unspecified", "hematological"),
    "leukopenia": ICD10Code("D72.819", "Decreased white blood cell count, unspecified", "hematological"),
    "neutropenia": ICD10Code("D70.9", "Neutropenia, unspecified", "hematological"),
    "coagulopathy": ICD10Code("D68.9", "Coagulation defect, unspecified", "hematological"),
    "lymphoma": ICD10Code("C85.90", "Non-Hodgkin lymphoma, unspecified, unspecified site", "hematological"),
    "leukemia": ICD10Code("C95.90", "Leukemia, unspecified not having achieved remission", "hematological"),
    "multiple myeloma": ICD10Code("C90.00", "Multiple myeloma not having achieved remission", "hematological"),
    "breast cancer": ICD10Code("C50.919", "Malignant neoplasm of unspecified site of unspecified female breast", "hematological"),
    "lung cancer": ICD10Code("C34.90", "Malignant neoplasm of unspecified part of unspecified bronchus or lung", "hematological"),
    "colon cancer": ICD10Code("C18.9", "Malignant neoplasm of colon, unspecified", "hematological"),
    "prostate cancer": ICD10Code("C61", "Malignant neoplasm of prostate", "hematological"),
    "pancreatic cancer": ICD10Code("C25.9", "Malignant neoplasm of pancreas, unspecified", "hematological"),
    "cancer": ICD10Code("C80.1", "Malignant (primary) neoplasm, unspecified", "hematological"),
    "malignancy": ICD10Code("C80.1", "Malignant (primary) neoplasm, unspecified", "hematological"),
    "metastatic disease": ICD10Code("C79.9", "Secondary malignant neoplasm of unspecified site", "hematological"),

    # -------------------------------------------------------------------------
    # Allergic / Immunological
    # -------------------------------------------------------------------------
    "allergic reaction": ICD10Code("T78.40XA", "Allergy, unspecified, initial encounter", "allergic"),
    "anaphylaxis": ICD10Code("T78.2XXA", "Anaphylactic shock, unspecified, initial encounter", "allergic"),
    "anaphylactic shock": ICD10Code("T78.2XXA", "Anaphylactic shock, unspecified, initial encounter", "allergic"),
    "drug allergy": ICD10Code("T88.7XXA", "Unspecified adverse effect of drug or medicament, initial encounter", "allergic"),
    "food allergy": ICD10Code("T78.1XXA", "Other adverse food reactions, not elsewhere classified, initial encounter", "allergic"),
    "seasonal allergies": ICD10Code("J30.2", "Other seasonal allergic rhinitis", "allergic"),
    "allergic rhinitis": ICD10Code("J30.9", "Allergic rhinitis, unspecified", "allergic"),
    "hay fever": ICD10Code("J30.1", "Allergic rhinitis due to pollen", "allergic"),
    "angioedema": ICD10Code("T78.3XXA", "Angioneurotic edema, initial encounter", "allergic"),
    "lupus": ICD10Code("M32.9", "Systemic lupus erythematosus, unspecified", "allergic"),
    "systemic lupus erythematosus": ICD10Code("M32.9", "Systemic lupus erythematosus, unspecified", "allergic"),

    # -------------------------------------------------------------------------
    # Other Common Conditions
    # -------------------------------------------------------------------------
    "fall": ICD10Code("W19.XXXA", "Unspecified fall, initial encounter", "other"),
    "trauma": ICD10Code("T14.90XA", "Injury, unspecified, initial encounter", "other"),
    "fever": ICD10Code("R50.9", "Fever, unspecified", "other"),
    "fatigue": ICD10Code("R53.83", "Other fatigue", "other"),
    "weakness": ICD10Code("R53.1", "Weakness", "other"),
    "weight loss": ICD10Code("R63.4", "Abnormal weight loss", "other"),
    "weight gain": ICD10Code("R63.5", "Abnormal weight gain", "other"),
    "edema": ICD10Code("R60.9", "Edema, unspecified", "other"),
    "peripheral edema": ICD10Code("R60.0", "Localized edema", "other"),
    "lymphedema": ICD10Code("I89.0", "Lymphedema, not elsewhere classified", "other"),
    "pain": ICD10Code("R52", "Pain, unspecified", "other"),
    "chronic pain": ICD10Code("G89.29", "Other chronic pain", "other"),
    "malnutrition": ICD10Code("E46", "Unspecified protein-calorie malnutrition", "other"),
    "failure to thrive": ICD10Code("R62.51", "Failure to thrive (child)", "other"),
}


# =============================================================================
# Synonym Mappings (for common alternate names)
# =============================================================================

SYNONYMS: dict[str, str] = {
    # Cardiovascular
    "heart disease": "coronary artery disease",
    "cardiac disease": "coronary artery disease",
    "ischemic heart disease": "coronary artery disease",
    "irregular heartbeat": "atrial fibrillation",
    "blood clot": "deep vein thrombosis",

    # Respiratory
    "breathing difficulty": "shortness of breath",
    "trouble breathing": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "sob": "shortness of breath",
    "lung infection": "pneumonia",
    "chest cold": "bronchitis",
    "common cold": "upper respiratory infection",
    "cold": "upper respiratory infection",

    # Endocrine
    "sugar diabetes": "type 2 diabetes",
    "dm": "diabetes",
    "dm2": "type 2 diabetes",
    "t2dm": "type 2 diabetes",
    "high blood sugar": "hyperglycemia",
    "low blood sugar": "hypoglycemia",
    "thyroid problems": "hypothyroidism",
    "overweight": "obesity",

    # Neurological
    "brain attack": "stroke",
    "mini stroke": "transient ischemic attack",
    "fits": "seizure",
    "convulsion": "seizure",
    "fainting": "syncope",
    "passed out": "syncope",
    "memory loss": "dementia",

    # GI
    "stomach flu": "viral gastroenteritis",
    "heartburn": "gerd",
    "indigestion": "gastritis",
    "upset stomach": "gastritis",
    "belly pain": "abdominal pain",
    "stomach pain": "abdominal pain",

    # Renal
    "kidney failure": "acute kidney injury",
    "kidney infection": "pyelonephritis",
    "bladder infection": "urinary tract infection",

    # Psychiatric
    "sad": "depression",
    "feeling down": "depression",
    "nervous": "anxiety",
    "worried": "anxiety",
    "can't sleep": "insomnia",
    "trouble sleeping": "insomnia",

    # Other
    "infection": "sepsis",
    "blood infection": "sepsis",
    "tiredness": "fatigue",
    "exhaustion": "fatigue",
    "swelling": "edema",
}


def normalize_condition(name: str) -> str:
    """Normalize a condition name for matching."""
    if not name:
        return ""
    # Lowercase
    name = name.lower().strip()
    # Remove common prefixes
    prefixes = ["acute ", "chronic ", "unspecified ", "history of ", "h/o "]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Remove punctuation and extra whitespace
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def similarity_score(s1: str, s2: str) -> float:
    """Calculate similarity between two strings (0.0 to 1.0)."""
    return SequenceMatcher(None, s1, s2).ratio()


def lookup_icd10(condition_name: str, min_similarity: float = 0.85) -> tuple[str | None, str | None, float]:
    """
    Look up ICD-10 code for a condition name.

    Args:
        condition_name: The condition name to look up
        min_similarity: Minimum similarity score for fuzzy matching (0.0-1.0)

    Returns:
        Tuple of (icd10_code, display_name, confidence)
        - confidence is 1.0 for exact match, <1.0 for fuzzy match, 0.0 if not found
    """
    if not condition_name:
        return None, None, 0.0

    normalized = normalize_condition(condition_name)

    # 1. Exact match
    if normalized in ICD10_DATABASE:
        code = ICD10_DATABASE[normalized]
        return code.code, code.display, 1.0

    # 2. Check synonyms
    if normalized in SYNONYMS:
        canonical = SYNONYMS[normalized]
        if canonical in ICD10_DATABASE:
            code = ICD10_DATABASE[canonical]
            return code.code, code.display, 0.95  # High confidence for synonym match

    # 3. Fuzzy matching
    best_match = None
    best_score = 0.0

    for db_name, code in ICD10_DATABASE.items():
        score = similarity_score(normalized, db_name)
        if score > best_score and score >= min_similarity:
            best_score = score
            best_match = code

    if best_match:
        return best_match.code, best_match.display, round(best_score, 2)

    return None, None, 0.0


def enrich_conditions_with_icd10(conditions: list) -> list:
    """
    Enrich a list of Condition objects with ICD-10 codes from lookup.

    This function adds verified ICD-10 codes to conditions, preferring
    lookup codes over any LLM-generated codes.

    Args:
        conditions: List of Condition objects

    Returns:
        The same list with ICD-10 codes added
    """
    for condition in conditions:
        if not condition.name:
            continue

        code, display, confidence = lookup_icd10(condition.name)

        if code and confidence >= 0.85:
            # Use lookup code (verified) over any existing code
            condition.icd10 = code
            # Store confidence in metadata if available
            if hasattr(condition, 'confidence'):
                condition.confidence = confidence

    return conditions
