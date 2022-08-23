import shutil
from fastapi import FastAPI,File,UploadFile
from cnn import recognition
app = FastAPI()

@app.post("/api/get_char")
async def api_get_char(file: UploadFile = File(...)):
    with open('tmp/user_input.jpg','wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"Estimated Symbol":recognition.pred('tmp/user_input.jpg')}
