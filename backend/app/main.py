from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import uuid
from app.model_optimizer import optimize_model
from app.auth_middleware import get_current_user
from app.firebase_config import save_upload_metadata, get_user_uploads

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
    target_device: str = Form(...),
    current_user = Depends(get_current_user)
):
    try:
        print("Received file:", file.filename)
        print("Target device:", target_device)
        print("User ID:", current_user['uid'])

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

        # Save metadata to Firestore
        optimized_filename = Path(optimized_model_path).name
        save_upload_metadata(
            current_user['uid'],
            file.filename,
            target_device,
            optimized_filename,
            metrics
        )

        return {
            "optimized_model": optimized_filename,
            "metrics": metrics
        }

    except Exception as e:
        print("ERROR during optimization:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/download/{filename}")
async def download_model(filename: str, current_user = Depends(get_current_user)):
    file_path = OPTIMIZED_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    
    # In a production environment, you should verify that the user owns this file
    # by checking the Firestore database
    
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)

@app.get("/api/uploads")
async def get_user_uploads_endpoint(current_user = Depends(get_current_user)):
    """Get all uploads for the current user"""
    try:
        uploads = get_user_uploads(current_user['uid'])
        return {"uploads": uploads}
    except Exception as e:
        print(f"Error fetching uploads: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch uploads")
