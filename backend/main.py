from fastapi import FastAPI
from pydantic import BaseModel

from graph.build_graph import load_graph
from rag.retriever import retrieve_context
from rag.generator import generate_answer
from models.mastery_model import predict_mastery
from models.difficulty_model import predict_difficulty
from models.ranking_model import rank_resources

app = FastAPI(title="EduGraph API")

graph = load_graph()

class Query(BaseModel):
    question: str
    student_id: str = "default"
    level: str = "beginner"

@app.post("/ask")
def ask_question(data: Query):
    # 1. Retrieve relevant context
    context = retrieve_context(data.question)

    # 2. Rank resources
    ranked = rank_resources(data.question, context)

    # 3. Generate answer
    answer = generate_answer(data.question, ranked)

    # 4. Predict mastery update
    mastery = predict_mastery(student_id=data.student_id, concept=data.question)

    # 5. Suggest next difficulty
    difficulty = predict_difficulty(student_id=data.student_id)

    return {
        "answer": answer,
        "concept_mastery": mastery,
        "recommended_difficulty": difficulty,
        "retrieved_context": ranked[:2],
    }
