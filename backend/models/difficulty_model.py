# backend/models/difficulty_model.py
import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "difficulty_model.pkl"
SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "difficulty_scaler.pkl"

clf = None
scaler = None
try:
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded difficulty_model from", MODEL_PATH)
except Exception as e:
    print("Could not load difficulty_model; using fallback stub. Err:", e)

def predict_difficulty(student_id: str, attempts=1, time_spent=20, correct=1, concept_difficulty=2):
    if clf is None or scaler is None:
        # fallback rules
        score = concept_difficulty + (attempts-1)*0.5 - (1 if correct else -1)*0.3
        if score <= 1.5:
            return "easy"
        if score <= 2.5:
            return "medium"
        return "hard"
    X = np.array([[concept_difficulty, attempts, time_spent, correct]])
    Xs = scaler.transform(X)
    pred = clf.predict(Xs)[0]
    return str(pred)
