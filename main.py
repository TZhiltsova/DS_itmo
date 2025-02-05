import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("model.pkl", 'rb') as file:
    model = joblib.load(file)


class ModelRequestData(BaseModel):
    rooms: int
    lon: float
    lat: float



class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "Свет внутри меня угас, но я всё ещё работаю"}, status_code=200)

@app.get("/predict_get", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)

@app.post("/predict_post", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.0", port=8000)
