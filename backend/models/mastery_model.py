# backend/models/mastery_model.py
import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "mastery_model.pkl"
SCALER_PATH = Path(__file__).resolve().parents[1] / "models" / "mastery_scaler.pkl"

clf = None
scaler = None
try:
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded mastery_model from", MODEL_PATH)
except Exception as e:
    print("Could not load mastery_model; using fallback stub. Err:", e)

def predict_mastery(student_id: str, concept: str, attempts=1, time_spent=20, correct=1, concept_difficulty=2):
    """
    Inputs are optional; main API will call with values if available.
    Returns dict with probability.
    """
    if clf is None or scaler is None:
        # fallback stochastic but deterministic-ish
        p = min(max(0.5 + (2 - concept_difficulty)*0.05 - (attempts-1)*0.08 + (0.02 if correct else -0.1), 0.01), 0.99)
        return {"concept": concept, "mastery_probability": round(float(p), 2), "model":"stub"}
    import numpy as np
    X = np.array([[concept_difficulty, attempts, time_spent, correct]])
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[0,1]
    return {"concept": concept, "mastery_probability": round(float(prob), 2), "model":"logreg"}
