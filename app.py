# app.py

from fastapi import FastAPI, Request
from classification_pipeline import SklearnClassifier

# Load the model
model = SklearnClassifier("text_rf_pipeline.pkl")

# Initialize FastAPI app
app = FastAPI()

@app.post("/predict/")
async def predict(request: Request):
    payload = await request.json()
    query = payload.get("text", "")
    
    if not query:
        return {"error": "No 'text' provided"}

    result, _ = model.run(query=query)
    doc = result["documents"][0]

    return {
        "label": doc["meta"].get("predicted_label", "N/A"),
        "probability": doc["meta"].get("probability", 0.0)
    }

