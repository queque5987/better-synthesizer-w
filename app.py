from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np
import torch
import os

import rtvc_main

app = FastAPI()

class UserInput(BaseModel):
    chars: list
    batched_embeds: list

@app.get('/')
def index():
    return {"Message": "This is Index"}

@app.get('/generate/')
async def generate(userinput: UserInput):
    _model = rtvc_main.taco()
    userinput = userinput.dict()
    chars = userinput["chars"]
    chars = [np.array(c)for c in chars]
    chars = np.stack(chars)
    chars = torch.tensor(chars).long().to('cpu')

    batched_embeds = userinput["batched_embeds"]
    batched_embeds = [np.array(b)for b in batched_embeds]
    speaker_embeds = np.stack(batched_embeds)
    speaker_embeddings = torch.tensor(speaker_embeds).float().to('cpu')

    _, mel, _ = _model.generate(chars, speaker_embeddings)
    print("--mel--")
    print(mel)
    print("--mel--")
    mel = mel.tolist()
    # alignments = alignments.tolist()
    mel = jsonable_encoder(mel)
    return JSONResponse(mel)
    #     {'mel': mel,
    #     'alignments': alignments
    #     }
    # )