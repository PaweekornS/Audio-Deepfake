from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.utils import audio_to_spectrogram
from app.model import SpeechClassifier

from pathlib import Path
import shutil

app = FastAPI(
    title="Speech AI Detection API",
    version="1.0"
)

classifier = SpeechClassifier(
    model_path="model/yolo11n-best.pt"
)

ALLOWED_EXTENSIONS = {".wav", ".mp3"}
UPLOAD_DIR = Path("mp3_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .wav and .mp3 are supported"
        )
        
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        spec_img = audio_to_spectrogram(file_path)
        y_prob = classifier.predict(spec_img)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    return JSONResponse(
        content={
            "filename": file.filename,
            "y_prob": y_prob
        }
    )


@app.post("/upload-mp3/")
async def upload_mp3_file(file: UploadFile = File(...)):
    """
    Receives an uploaded file, validates it as an MP3, and saves it.
    """
    # Optional: Basic file extension check
    if not file.filename.lower().endswith('.mp3'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only MP3 files are allowed.")

    file_path = UPLOAD_DIR / file.filename

    # Save the file to disk in chunks for efficient handling of large files
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"There was an error uploading the file: {e}")
    finally:
        await file.close() # Ensure the SpooledTemporaryFile is closed

    return {"filename": file.filename, "message": f"Successfully uploaded {file.filename}"}
