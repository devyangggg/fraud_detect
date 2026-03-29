import torch
from model import Anomaly_Detect
from fastapi import FastAPI
from pydantic import BaseModel

class OrderBookInput(BaseModel):
    data: list

model = Anomaly_Detect()
model.load_state_dict(torch.load('main_model.pt', weights_only=True))

app = FastAPI()

@app.post("/predict")
async def predict(res : OrderBookInput):
    data =  torch.tensor(res.data, dtype=torch.float32).reshape(1, 100, 18)
    output = model(data)
    prob = torch.sigmoid(output)
    return {"Prediction": int(prob > 0.5), "Confidence": float(prob)}
    
