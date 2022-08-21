from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np
import torch
import os

app = FastAPI()

@app.get('/')
def index():
    return {"Message": "This is Index"}

@app.get('/load/')
async def load():
    path = "model/"
    ckpt = []
    for pt in os.listdir(path):
        tensor = torch.load(os.path.join(path, pt))
        ckpt.append(pt[4:-3])
        ckpt.append(tensor.tolist())
    ckpt = jsonable_encoder(ckpt)
    return JSONResponse(ckpt)