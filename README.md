🚀 StructLearn
An AI-Powered Structured Learning Assistant using Knowledge Graphs + Machine Learning + RAG

StructLearn is an end-to-end adaptive learning system that combines Knowledge Graphs, Machine Learning, and Retrieval-Augmented Generation (RAG) to provide structured, personalized learning experiences.

Modern students are overwhelmed by the volume of online learning content. Large Language Models (LLMs) can answer questions, but they cannot track a student’s progress, adapt difficulty, or guide learning paths. StructLearn solves this.

🎯 Features
✅ 1. Knowledge Graph (KG)

Defines prerequisite relationships between concepts and helps structure the learning path.

✅ 2. RAG (Retrieval-Augmented Generation)

Uses SentenceTransformer (all-MiniLM-L6-v2) to embed content

Retrieves the most relevant explanations

Generates clean, grounded answers

Prevents hallucinations

✅ 3. REAL Machine Learning Models

We use a synthetic educational dataset (5000 rows) to train:

Mastery Prediction Model (Logistic Regression)

Predicts whether a student has mastered a concept based on:

concept difficulty

attempts

time spent

correctness

Model Performance:

Accuracy: 0.909

ROC-AUC: 0.965

Difficulty Prediction Model (RandomForest)

Predicts the ideal next difficulty (easy / medium / hard).

Model Performance:

Accuracy: 1.00

Perfect confusion matrix

✅ 4. FastAPI Backend

The /ask endpoint integrates:

RAG retriever

RAG generator

Knowledge Graph

ML mastery prediction

ML difficulty adaptation

📁 Project Structure
StructLearn/
│
├── backend/
│   ├── main.py                # FastAPI app
│   │
│   ├── data/
│   │   ├── gen_synthetic_data.py
│   │   └── structlearn_synth.csv
│   │
│   ├── train/
│   │   ├── train_mastery.py
│   │   └── train_difficulty.py
│   │
│   ├── models/
│   │   ├── mastery_model.py
│   │   ├── difficulty_model.py
│   │   ├── mastery_model.pkl
│   │   └── difficulty_model.pkl
│   │
│   ├── rag/
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   └── graph/
│       ├── graph_data.json
│       └── build_graph.py
│
└── tests/                     # optional test suite

🔧 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/Aneeshrao123/StructLearn.git
cd StructLearn

2️⃣ Create a virtual environment
python3 -m venv myenv
source myenv/bin/activate

3️⃣ Install requirements
pip install -r backend/requirements.txt

🧪 Train the ML Models
1. Generate synthetic dataset:
python backend/data/gen_synthetic_data.py

2. Train Mastery Prediction Model:
python backend/train/train_mastery.py

3. Train Difficulty Prediction Model:
python backend/train/train_difficulty.py


This creates:

backend/models/mastery_model.pkl
backend/models/difficulty_model.pkl

🚀 Run the FastAPI Server

From project root:

uvicorn backend.main:app --reload


Visit the interactive API docs:

👉 http://127.0.0.1:8000/docs

📡 Example API Call
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{
    "question": "Explain recursion",
    "student_id": "s1",
    "attempts": 2,
    "time_spent": 25,
    "correct": 1,
    "concept_difficulty": 3
}'


Example Response:

{
  "answer": "Explanation about recursion...",
  "concept_mastery": {
      "concept": "recursion",
      "mastery_probability": 0.46,
      "model": "logreg"
  },
  "recommended_difficulty": "medium",
  "retrieved_context": [
      ["Recursion", "Recursion is when a function calls itself.", 0.82]
  ]
}

📊 Evaluation
Mastery Model

Accuracy: 0.909

ROC-AUC: 0.965

Strong classification on synthetic learning logs

Difficulty Model

Accuracy: 1.00

Perfect separation of difficulty classes

🔥 Why StructLearn Matters

Brings structure to learning

Provides personalized difficulty adaptation

Tracks student understanding

Uses ML for real-time estimation

Uses RAG for accurate explanations

Avoids hallucinations

Supports scalable integration with future UIs
