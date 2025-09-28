from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
from app.model_optimizer import optimize_model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],  # frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
OPTIMIZED_DIR = Path("optimized")
UPLOAD_DIR.mkdir(exist_ok=True)
OPTIMIZED_DIR.mkdir(exist_ok=True)

@app.post("/api/optimize")
async def optimize_ml_model(
    file: UploadFile = File(...),
    target_device: str = Form(...)
):
    try:
        print("Received file:", file.filename)
        print("Target device:", target_device)

        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        upload_path = UPLOAD_DIR / unique_filename

        # Save uploaded file
        with open(upload_path, "wb") as buffer:
            buffer.write(await file.read())

        # Run optimization
        optimized_model_path, metrics = await optimize_model(
            str(upload_path),
            target_device,
            str(OPTIMIZED_DIR / f"opt_{unique_filename}")
        )

        print("Optimization metrics:", metrics)

        return {
            "optimized_model": Path(optimized_model_path).name,
            "metrics": metrics
        }

    except Exception as e:
        print("ERROR during optimization:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_model(filename: str):
    file_path = OPTIMIZED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)
