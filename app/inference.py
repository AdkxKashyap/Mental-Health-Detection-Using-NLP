import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from app.config import (
    MODEL_PATH,
    MAX_LENGTH,
    ID_TO_LABEL
)

#Use GPU if available, otherwise use CPU for inference
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#Load the tokenizer and model from the specified path
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
).to(device) #Move the model to the appropriate device (GPU or CPU)

def predict_mental_health(text):
    #Tokenize the input text and convert it to tensors
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    ).to(device) #Move the input tensors to the appropriate device (GPU or CPU)

#Perform inference without calculating gradients to save memory and improve performance
    with torch.no_grad():
        outputs = model(**inputs)

#Apply softmax to the model's output logits to get probabilities for each class, and determine the predicted class. Logits are the raw output values from the model before applying any activation function. Softmax converts these logits into probabilities that sum to 1, making it easier to interpret the model's predictions. The predicted class is determined by finding the index of the highest probability.
    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )
    
    predicted_class = torch.argmax(
    probabilities,
    dim=1
    ).item()
    
    probs = probabilities[0].cpu().numpy()
    probability_dict = {
    ID_TO_LABEL[i]: float(probs[i])
    for i in range(len(probs))
    }
    
    return {
    "prediction": ID_TO_LABEL[int(predicted_class)],

    "probabilities": probability_dict
    }

    