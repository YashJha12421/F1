from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict_one

app = FastAPI(title="F1 Podium Prediction API", version="1.0", description= "a novel podium predictor based on an XGBoost model which takes as input the driver's age, his current form etc")

dic={
    'harry': 'lion',
    'hermione': 'raven',
    'ron': 'beaver',
    'draco': 'snake'

}


class PredictionInput(BaseModel):
    grid: int = Field(..., ge=1, le=30)
    driverId: int
    constructorId: int
    year: int = Field(..., ge=1950, le=2030)
    round: int = Field(..., ge=1, le=30)
    circuitId: int
    form_last3: float = Field(..., ge=0)
    age: int = Field(..., ge=16, le=60)
    team_strength: float = Field(..., ge=0)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(inp: PredictionInput):
    result = predict_one(inp.dict())
    return result
@app.get("/HP")
def yn(charac:str):
    for a in dic:
        if charac.lower().strip()==a:
            return(f'{a} is in {dic[a]}')