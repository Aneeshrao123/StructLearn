# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from backend.graph.build_graph import load_graph
from backend.rag.retriever import retrieve_context
from backend.rag.generator import generate_answer
from backend.models.mastery_model import predict_mastery
from backend.models.difficulty_model import predict_difficulty
from backend.models.ranking_model import rank_resources

app = FastAPI(title="StructLearn API")

graph = load_graph()

class Query(BaseModel):
    question: str
    student_id: str = "default"
    level: str = "beginner"
    # optional features if the frontend provides them
    attempts: int = 1
    time_spent: int = 20
    correct: int = 1
    concept_difficulty: int = 2

@app.post("/ask")
def ask_question(data: Query):
    # 1. Retrieve relevant context
    context = retrieve_context(data.question)

    # 2. Rank resources
    ranked = rank_resources(data.question, context)

    # 3. Generate answer
    answer = generate_answer(data.question, ranked)

    # 4. Predict mastery update (real model if trained)
    mastery = predict_mastery(student_id=data.student_id,
                              concept=data.question,
                              attempts=data.attempts,
                              time_spent=data.time_spent,
                              correct=data.correct,
                              concept_difficulty=data.concept_difficulty)

    # 5. Suggest next difficulty
    difficulty = predict_difficulty(student_id=data.student_id,
                                    attempts=data.attempts,
                                    time_spent=data.time_spent,
                                    correct=data.correct,
                                    concept_difficulty=data.concept_difficulty)

    return {
        "answer": answer,
        "concept_mastery": mastery,
        "recommended_difficulty": difficulty,
        "retrieved_context": ranked[:2],
    }
