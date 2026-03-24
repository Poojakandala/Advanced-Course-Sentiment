from fastapi import FastAPI
from pydantic import BaseModel
from app.processor import get_prediction

app = FastAPI(title="Advanced Course Sentiment API")

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict(request: ReviewRequest):
    sentiment = get_prediction(request.text)
    return {"review": request.text, "sentiment": sentiment}

@app.get("/")
def home():
    return {"message": "Deep Learning Sentiment API is Running"}
