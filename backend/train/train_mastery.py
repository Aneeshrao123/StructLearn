# backend/train/train_mastery.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "structlearn_synth.csv"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(DATA)
    return df

def featurize(df):
    # features: concept_difficulty, attempts, time_spent, correct (previous attempt)
    X = df[["concept_difficulty", "attempts", "time_spent", "correct"]].copy()
    # you can add one-hot for concept if wanted; keep simple for now
    y = df["mastery_label"]
    return X, y

def train():
    df = load_data()
    X, y = featurize(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:,1]

    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    except:
        print("ROC-AUC: cannot compute (maybe single-class in test split)")

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, OUT_DIR / "mastery_model.pkl")
    joblib.dump(scaler, OUT_DIR / "mastery_scaler.pkl")
    print("Saved mastery_model.pkl and scaler to:", OUT_DIR)

if __name__ == "__main__":
    train()
