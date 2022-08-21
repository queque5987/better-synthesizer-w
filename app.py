from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np
import torch
import os
import collections
app = FastAPI()

@app.get('/')
def index():
    return {"Message": "This is Index"}

@app.get('/load/')
async def load():
    path = "model/"
    checkpoint = collections.OrderedDict()
    # ckpt = []
    # print(len(os.listdir(path)))
    for pt in os.listdir(path):
        tensor = torch.load(os.path.join(path, pt))
        checkpoint[pt[4:-3]] = tensor
        # ckpt.append(pt[4:-3])
        # ckpt.append(tensor.tolist())
    checkpoint = jsonable_encoder(checkpoint)
    return JSONResponse(checkpoint)