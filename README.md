Better-API (synthesizer-w)
=============
https://github.com/queque5987/better-encoder   
https://github.com/queque5987/better-synthesizer   
https://github.com/queque5987/better-vocoder   

### Better-API generates a voice that cloning user's voice from a text.
    1.encoder recieves a user voice and gives an embedding to synthesizer.
    2.synthesizer recieves an embedding and a text to generate speech and gives mel spectrogram to vocoder.   
    3.vocoder recieves a mel spectrogram and gives generated wav file.   
       
*Encoder speaker embedding Model*   
*Synthesizer uses TACOTRON2 Model; it is on better-synthesizer-w API*   
*Vocoder uses waveRNN Model*   
    
## available on
https://better-synthesizer.herokuapp.com/
## to inference, send request on
https://better-synthesizer.herokuapp.com/inference/
### Request JSON
    chars @type {list}
    batched_embeds @type {list}
**receives user voice embedding to generate mel spectrogram*   
**chars[*index*] {tensor} must be converted into {list}*   
**batched_embeds[*index*] {tensor} must be converted into {list}*
### Response JSON
    mel @type {list}   
**return mel spectrogram*   
**convert mel{list} to {ndarray} to use*   

* * *
# used libraries
## Real-Time-Voice-Cloning
https://github.com/CorentinJ/Real-Time-Voice-Cloning

## FastAPI   
developed with FastAPI   
to install librosa : https://github.com/heroku/heroku-buildpack-apt   
source : https://fastapi.tiangolo.com/   

## Heroku
deployed with FastAPI   
https://dashboard.heroku.com/

## requirements.txt
### For deployment
    fastapi
    pydantic
    uvicorn
    favicon
    gunicorn
### For Voice clonning   
    -f https://download.pytorch.org/whl/torch_stable.html   
    torch==1.12.1+cpu
    fastapi
    pydantic
    numpy
    uvicorn
    favicon
    gunicorn
    librosa
    scipy
    Unidecode
    pymysql
**first line enableds install torch for cpu when deploying server to heroku*
