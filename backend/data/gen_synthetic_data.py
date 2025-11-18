# backend/data/gen_synthetic_data.py
import csv
import random
import uuid
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT_FILE = OUT / "structlearn_synth.csv"

CONCEPTS = ["Variables", "Loops", "Functions", "Recursion", "Dynamic Programming"]
CONCEPT_DIFFICULTY = {"Variables": 1, "Loops": 2, "Functions": 2, "Recursion": 3, "Dynamic Programming": 4}

def gen_row(student_id):
    concept = random.choice(CONCEPTS)
    concept_diff = CONCEPT_DIFFICULTY[concept]
    attempts = random.choice([1,1,1,2,2,3])
    time_spent = max(5, int(random.gauss(20 + concept_diff*3 + attempts*2, 8)))
    # base mastery probability lower for hard concepts and more attempts reduces success prob
    base = 0.7 - 0.12*(concept_diff-1) - 0.1*(attempts-1)
    # some student variability
    student_skill = random.uniform(-0.15, 0.15)
    p_correct = min(max(base + student_skill, 0.05), 0.99)
    correct = 1 if random.random() < p_correct else 0
    mastery_label = 1 if p_correct > 0.6 and correct==1 else 0  # labelling heuristic
    return {
        "student_id": student_id,
        "concept": concept,
        "concept_difficulty": concept_diff,
        "attempts": attempts,
        "time_spent": time_spent,
        "correct": correct,
        "mastery_label": mastery_label
    }

def generate(n_students=200, rows=5000):
    rows_out = []
    student_ids = [str(uuid.uuid4())[:8] for _ in range(n_students)]
    for _ in range(rows):
        sid = random.choice(student_ids)
        rows_out.append(gen_row(sid))

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)
    print("Wrote synthetic dataset to:", OUT_FILE)

if __name__ == "__main__":
    generate()
