from fastapi import FastAPI
from inference import predict_mental_health

app = FastAPI()

#Health endpoint to check if the API is running
@app.get("/")
def home():
    return {
        "message": "Mental Health NLP API is running"
    }
    
@app.post("/predict")
def predict(data: dict):
    text = data.get("text", "")
    if not text:
        return {
            "error": "No text provided for prediction"
        }
    
    result = predict_mental_health(text)
    return result