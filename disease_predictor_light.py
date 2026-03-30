import re
from typing import Dict, List, Tuple

import numpy as np


DISEASE_NAMES = [
    "Vertigo", "AIDS", "Acne", "Alcoholic hepatitis", "Allergy", "Arthritis",
    "Bronchial Asthma", "Cervical spondylosis", "Chicken pox", "Chronic cholestasis",
    "Common Cold", "Dengue", "Diabetes", "Dimorphic hemorrhoids (piles)",
    "Drug Reaction", "Fungal infection", "GERD", "Gastroenteritis",
    "Heart attack", "Hepatitis B", "Hepatitis C", "Hepatitis D",
    "Hepatitis E", "Hypertension", "Hyperthyroidism", "Hypoglycemia",
    "Hypothyroidism", "Impetigo", "Jaundice", "Malaria", "Migraine",
    "Osteoarthritis", "Paralysis (brain hemorrhage)", "Peptic ulcer disease",
    "Pneumonia", "Psoriasis", "Tuberculosis", "Typhoid",
    "Urinary tract infection", "Varicose veins", "hepatitis A"
]


SYMPTOM_MAPPINGS = {
    "can't sleep": "insomnia",
    "trouble sleeping": "insomnia",
    "chest pain": "chest discomfort",
    "throwing up": "vomiting",
    "throwing-up": "vomiting",
    "dizzy": "dizziness",
    "feeling dizzy": "dizziness",
    "tired": "fatigue",
    "exhausted": "fatigue",
    "no energy": "fatigue",
    "sweating": "sweats",
    "can't breathe": "difficulty breathing",
    "hard to breathe": "difficulty breathing",
    "out of breath": "shortness of breath",
    "stomach ache": "abdominal pain",
    "belly pain": "abdominal pain",
    "tummy pain": "abdominal pain",
    "pee": "urination",
    "peeing": "urination",
}


SYMPTOM_DISEASE_MAP: Dict[str, Dict[str, float]] = {
    "fever": {"Malaria": 0.6, "Dengue": 0.65, "Typhoid": 0.55, "Common Cold": 0.5},
    "headache": {"Migraine": 0.75, "Malaria": 0.5, "Common Cold": 0.4},
    "cough": {"Common Cold": 0.7, "Pneumonia": 0.6, "Bronchial Asthma": 0.55, "Tuberculosis": 0.5},
    "chest pain": {"Heart attack": 0.8, "GERD": 0.4},
    "fatigue": {"Hypothyroidism": 0.6, "Diabetes": 0.5, "Malaria": 0.45},
    "dizziness": {"Vertigo": 0.8, "Hypertension": 0.5, "Hypoglycemia": 0.45},
    "nausea": {"Gastroenteritis": 0.7, "Migraine": 0.4},
    "vomiting": {"Gastroenteritis": 0.75, "Dengue": 0.5, "Migraine": 0.4},
    "pain": {"Arthritis": 0.5, "Osteoarthritis": 0.5},
    "joint pain": {"Arthritis": 0.8, "Osteoarthritis": 0.75},
    "rash": {"Chicken pox": 0.7, "Allergy": 0.65, "Drug Reaction": 0.6},
    "itching": {"Fungal infection": 0.7, "Allergy": 0.6, "Chicken pox": 0.5},
    "stomach pain": {"Gastroenteritis": 0.7, "GERD": 0.6, "Peptic ulcer disease": 0.65},
    "abdominal pain": {"Gastroenteritis": 0.65, "GERD": 0.55, "Peptic ulcer disease": 0.7},
    "diarrhea": {"Gastroenteritis": 0.8, "Typhoid": 0.6},
    "constipation": {"Hypothyroidism": 0.5, "Diabetes": 0.4},
    "breathlessness": {"Bronchial Asthma": 0.8, "Pneumonia": 0.65, "Heart attack": 0.6},
    "difficulty breathing": {"Bronchial Asthma": 0.8, "Pneumonia": 0.65, "Heart attack": 0.6},
    "sore throat": {"Common Cold": 0.75, "Allergy": 0.4},
    "yellow eyes": {"Jaundice": 0.85, "Hepatitis B": 0.7},
    "yellowing": {"Jaundice": 0.85, "Hepatitis B": 0.6, "hepatitis A": 0.6},
    "back pain": {"Cervical spondylosis": 0.7, "Arthritis": 0.5},
    "neck pain": {"Cervical spondylosis": 0.8, "Arthritis": 0.5},
    "knee pain": {"Osteoarthritis": 0.8, "Arthritis": 0.7},
    "frequent urination": {"Diabetes": 0.7, "Urinary tract infection": 0.65},
    "burning urination": {"Urinary tract infection": 0.8},
    "thirst": {"Diabetes": 0.7, "Hyperthyroidism": 0.5},
    "excessive thirst": {"Diabetes": 0.8, "Hyperthyroidism": 0.6},
    "weight loss": {"Diabetes": 0.6, "Hyperthyroidism": 0.7, "Tuberculosis": 0.6},
    "weight gain": {"Hypothyroidism": 0.7},
}


DISEASE_INFO = {
    "Migraine": {
        "description": "A neurological condition marked by recurring headaches, often with throbbing pain on one side of the head.",
        "symptoms": ["Severe headache", "Sensitivity to light and sound", "Nausea", "Visual disturbances"],
        "common_treatments": ["Pain relievers", "Triptans", "Anti-nausea medication", "Preventive therapy"],
    },
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract that usually affects the nose and throat.",
        "symptoms": ["Runny or stuffy nose", "Sore throat", "Cough", "Mild body aches"],
        "common_treatments": ["Rest", "Hydration", "Cold medicines", "Pain relievers"],
    },
    "Diabetes": {
        "description": "A chronic disease that affects how the body processes blood sugar.",
        "symptoms": ["Frequent urination", "Increased thirst", "Fatigue", "Unexplained weight loss"],
        "common_treatments": ["Diet changes", "Blood sugar monitoring", "Medication", "Exercise"],
    },
    "Hypertension": {
        "description": "A condition in which the force of blood against artery walls is consistently too high.",
        "symptoms": ["Often none", "Headaches in severe cases", "Shortness of breath", "Nosebleeds"],
        "common_treatments": ["Blood pressure medication", "Lower sodium intake", "Regular exercise", "Weight control"],
    },
    "Heart attack": {
        "description": "A blockage of blood flow to the heart muscle, often caused by a clot.",
        "symptoms": ["Chest pressure", "Shortness of breath", "Pain radiating to arm or jaw", "Cold sweat"],
        "common_treatments": ["Emergency care", "Clot-busting medication", "Surgery", "Cardiac rehabilitation"],
    },
}


def preprocess_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for colloquial, medical in SYMPTOM_MAPPINGS.items():
        if colloquial in normalized and medical not in normalized:
            normalized += f" {medical}"

    return normalized


def predict_diseases(symptoms_text: str, top_n: int = 5) -> List[Tuple[str, float]]:
    normalized = preprocess_text(symptoms_text)
    scores = np.full(len(DISEASE_NAMES), 0.02, dtype="float32")

    for symptom, disease_dict in SYMPTOM_DISEASE_MAP.items():
        if symptom in normalized:
            for disease, boost in disease_dict.items():
                if disease in DISEASE_NAMES:
                    scores[DISEASE_NAMES.index(disease)] += boost

    if scores.sum() > 0:
        scores = scores / scores.sum()

    results = [(DISEASE_NAMES[i], float(scores[i])) for i in range(len(DISEASE_NAMES))]
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:top_n]


def get_disease_information(disease_name: str) -> Dict[str, object]:
    return DISEASE_INFO.get(
        disease_name,
        {
            "description": f"A medical condition known as {disease_name}.",
            "symptoms": ["Varies by individual", "Consult a clinician for condition-specific symptoms"],
            "common_treatments": ["Consult a clinician for treatment options"],
        },
    )
