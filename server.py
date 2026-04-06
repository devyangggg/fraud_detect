import torch
from model import Anomaly_Detect
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

class OrderBookInput(BaseModel):
    data: list

model = Anomaly_Detect()
model.load_state_dict(torch.load('main_model.pt', weights_only=True))

app = FastAPI()

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(res : OrderBookInput):
    if len(res.data) != 100:
        raise HTTPException(status_code=400, detail="Expected exactly 100 snapshots")
    for snapshot in res.data:
        if len(snapshot) != 6:
            raise HTTPException(status_code=400, detail="Each snapshot must have exactly 6 rows")
        for row in snapshot:
            if len(row) != 3:
                raise HTTPException(status_code=400, detail="Each row must have exactly 3 values")
    data =  torch.tensor(res.data, dtype=torch.float32).reshape(1, 100, 18)
    output = model(data)
    prob = torch.sigmoid(output)
    return {"Prediction": int(prob > 0.5), "Confidence": float(prob)}
    
