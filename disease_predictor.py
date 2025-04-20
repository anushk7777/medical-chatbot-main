import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

# Disease names for the prediction model
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


class DiseasePredictor:
    def __init__(self):
        self.model_name = "shanover/symps_disease_bert_v3_c41"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False

    def predict(self, symptoms_text):
        """
        Predict diseases based on symptoms description
        Returns: list of (disease, probability) tuples
        """
        # If model failed to load, use keyword-based prediction as fallback
        if not self.model_loaded:
            return self._get_keyword_based_prediction(symptoms_text)

        # Preprocess text
        symptoms_text = self._preprocess_text(symptoms_text)

        # Tokenize input
        inputs = self.tokenizer(symptoms_text, return_tensors="pt", truncation=True, padding=True)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Convert logits to probabilities
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().numpy()

        # Create list of (disease, probability) tuples
        results = [(DISEASE_NAMES[i], float(probabilities[i])) for i in range(len(DISEASE_NAMES))]

        # Sort by probability in descending order
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _preprocess_text(self, text):
        """Preprocess symptoms text for better prediction"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text).strip()

        # Add symptom keywords if they're implied but not explicitly stated
        symptom_mappings = {
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
            "peeing": "urination"
        }

        for colloquial, medical in symptom_mappings.items():
            if colloquial in text and medical not in text:
                text += f" {medical}"

        return text

    def _get_keyword_based_prediction(self, symptoms_text):
        """Fallback prediction method using keyword matching"""
        symptoms_text = symptoms_text.lower()

        # Base probabilities with small random values
        base_probs = np.random.uniform(0.01, 0.05, len(DISEASE_NAMES))

        # Dictionary of symptom keywords mapping to diseases and their probability boosts
        symptom_disease_map = {
            "fever": {"Malaria": 0.6, "Dengue": 0.65, "Typhoid": 0.55, "Common Cold": 0.5},
            "headache": {"Migraine": 0.75, "Malaria": 0.5, "Common Cold": 0.4},
            "cough": {"Common Cold": 0.7, "Pneumonia": 0.6, "Bronchial Asthma": 0.55, "Tuberculosis": 0.5},
            "chest pain": {"Heart attack": 0.8, "GERD": 0.4},
            "fatigue": {"Hypothyroidism": 0.6, "Diabetes": 0.5, "Malaria": 0.45},
            "dizziness": {"Vertigo": 0.8, "Hypertension": 0.5, "Hypoglycemia": 0.45},
            "nausea": {"Gastroenteritis": 0.7, "Food poisoning": 0.65, "Migraine": 0.4},
            "vomiting": {"Gastroenteritis": 0.75, "Food poisoning": 0.7, "Dengue": 0.5},
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
            "yellow eyes": {"Jaundice": 0.85, "Hepatitis": 0.7},
            "yellowing": {"Jaundice": 0.85, "Hepatitis B": 0.6, "Hepatitis A": 0.6},
            "back pain": {"Cervical spondylosis": 0.7, "Arthritis": 0.5},
            "neck pain": {"Cervical spondylosis": 0.8, "Arthritis": 0.5},
            "knee pain": {"Osteoarthritis": 0.8, "Arthritis": 0.7},
            "frequent urination": {"Diabetes": 0.7, "Urinary tract infection": 0.65},
            "burning urination": {"Urinary tract infection": 0.8},
            "calf pain": {"Varicose veins": 0.75, "Deep vein thrombosis": 0.6},
            "bleeding": {"Dimorphic hemorrhoids (piles)": 0.7, "Peptic ulcer disease": 0.5},
            "thirst": {"Diabetes": 0.7, "Hyperthyroidism": 0.5},
            "excessive thirst": {"Diabetes": 0.8, "Hyperthyroidism": 0.6},
            "weight loss": {"Diabetes": 0.6, "Hyperthyroidism": 0.7, "Tuberculosis": 0.6},
            "weight gain": {"Hypothyroidism": 0.7}
        }

        # Boost probabilities based on symptom keywords
        for symptom, disease_dict in symptom_disease_map.items():
            if symptom in symptoms_text:
                for disease, boost in disease_dict.items():
                    if disease in DISEASE_NAMES:
                        disease_idx = DISEASE_NAMES.index(disease)
                        base_probs[disease_idx] += boost

        # Normalize probabilities to sum to 1
        if np.sum(base_probs) > 0:
            base_probs = base_probs / np.sum(base_probs)

        # Create results list
        results = [(DISEASE_NAMES[i], float(base_probs[i])) for i in range(len(DISEASE_NAMES))]
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def get_top_predictions(self, symptoms_text, top_n=5):
        """Get top N predictions with probabilities"""
        all_predictions = self.predict(symptoms_text)
        return all_predictions[:top_n]

    def get_disease_information(self, disease_name):
        """Return information about a disease"""

        disease_info = {
            "Migraine": {
                "description": "A neurological condition characterized by recurrent headaches, often with throbbing pain on one side of the head.",
                "symptoms": ["Severe headache", "Sensitivity to light and sound", "Nausea", "Visual disturbances"],
                "common_treatments": ["Pain relievers", "Triptans", "Anti-nausea medications", "Preventive medications"]
            },
            "Common Cold": {
                "description": "A viral infection of the upper respiratory tract, primarily the nose and throat.",
                "symptoms": ["Runny or stuffy nose", "Sore throat", "Cough", "Congestion", "Mild body aches"],
                "common_treatments": ["Rest", "Hydration", "Over-the-counter cold medications", "Pain relievers"]
            },
            "Diabetes": {
                "description": "A chronic disease affecting how the body processes blood sugar (glucose).",
                "symptoms": ["Increased thirst", "Frequent urination", "Extreme hunger", "Unexplained weight loss",
                             "Fatigue"],
                "common_treatments": ["Insulin therapy", "Blood sugar monitoring", "Healthy diet", "Regular exercise"]
            },
            "Hypertension": {
                "description": "A condition in which the force of blood against artery walls is too high.",
                "symptoms": ["Usually asymptomatic", "Headaches (in severe cases)", "Shortness of breath",
                             "Nosebleeds"],
                "common_treatments": ["Blood pressure medications", "Lifestyle changes", "Reduced sodium intake",
                                      "Regular exercise"]
            },
            "Heart attack": {
                "description": "A blockage of blood flow to the heart muscle, often caused by a blood clot.",
                "symptoms": ["Chest pain or pressure", "Pain radiating to arm/jaw", "Shortness of breath", "Cold sweat",
                             "Nausea"],
                "common_treatments": ["Emergency medical care", "Medications to dissolve clots", "Surgery",
                                      "Lifestyle changes"]
            }
            # More diseases could be added here
        }

        # Return disease info if available, otherwise a generic message
        return disease_info.get(disease_name, {
            "description": f"A medical condition known as {disease_name}.",
            "symptoms": ["Varies by individual", "Consult a healthcare provider for specific symptoms"],
            "common_treatments": ["Consult a healthcare provider for appropriate treatment options"]
        })
