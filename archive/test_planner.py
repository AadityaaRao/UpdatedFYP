"""Quick test: load the trained planner and classify unseen questions."""
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from backend.edu.planner import load_planner, classify_question

# Load models
planner = load_planner("models/edu_planner.pt")
tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert.eval()

# Test questions (NOT in training set)
tests = [
    "Explain the concept of recursion",
    "How do you sort a linked list?",
    "What was covered after arrays?",
    "What is shown on the slide?",
    "Summarize the main topics",
    "What is polymorphism in OOP?",
    "What steps do I follow to debug code?",
    "When was the formula introduced?",
]

print("Planner loaded from checkpoint — classifying unseen questions:\n")
for q in tests:
    inputs = tok(q, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        emb = bert(**inputs).last_hidden_state[:, 0, :]
    result = classify_question(planner, emb)
    route = result.route
    conf = result.confidence
    source = result.source
    print(f"  [{route:10s}] (conf={conf:.3f}, src={source:8s})  {q}")
