# backend/train/train_difficulty.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "structlearn_synth.csv"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

def load_data():
    df = pd.read_csv(DATA)
    # create a 3-class difficulty label from concept_difficulty
    # map 1 -> easy, 2 -> medium, 3+ -> hard
    def map_diff(x):
        if x <= 1: return "easy"
        if x == 2: return "medium"
        return "hard"
    df["diff_label"] = df["concept_difficulty"].apply(map_diff)
    return df

def featurize(df):
    X = df[["concept_difficulty", "attempts", "time_spent", "correct"]].copy()
    y = df["diff_label"]
    return X, y

def train():
    df = load_data()
    X, y = featurize(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, OUT_DIR / "difficulty_model.pkl")
    joblib.dump(scaler, OUT_DIR / "difficulty_scaler.pkl")
    # save feature importances
    importances = clf.feature_importances_
    np.savetxt(OUT_DIR / "difficulty_feature_importances.txt", importances)
    print("Saved difficulty_model, scaler and feature importances to:", OUT_DIR)

if __name__ == "__main__":
    train()
