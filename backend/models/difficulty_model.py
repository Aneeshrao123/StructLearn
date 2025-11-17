import random

def predict_difficulty(student_id: str):
    levels = ["easy", "medium", "hard"]
    return random.choice(levels)
