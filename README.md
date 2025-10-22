## ModelFlex — ML Model Optimization

ModelFlex is a small full‑stack application that helps you optimize ML models for deployment. It supports TensorFlow, PyTorch and ONNX formats and applies device‑specific optimizations (CPU, GPU, TPU when applicable).

This README focuses on how to run the project locally, what the backend API exposes, and troubleshooting tips specific to this repository layout.

## Project layout (relevant files)

- `frontend/` — React + Vite frontend (dev server runs on :5173)
- `backend/` — FastAPI backend (ASGI app at `app.main:app`)
  - `backend/app/main.py` — API endpoints: `/api/optimize`, `/api/download/{filename}`
  - `backend/app/model_optimizer.py` — model optimization logic
  - `backend/run.py` — simple runner that starts Uvicorn
  - `backend/requirements.txt` — Python dependencies
- `README.md` — this file

## Quick start (Windows / PowerShell)

1) Backend — create venv and install dependencies

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Start the backend (from `backend/`)

```powershell
# either using the run helper
python run.py

# or directly with uvicorn
python -m uvicorn app.main:app --reload
```

Backend will be available at: http://127.0.0.1:8000

3) Frontend

```powershell
cd frontend
npm install
npm run dev
```

Frontend dev server defaults to http://localhost:5173. The frontend expects the backend at `http://127.0.0.1:8000` (CORS is configured for `:5173`).

## Backend API

POST /api/optimize
- Content-Type: multipart/form-data
- Form fields:
  - `file` — uploaded model file
  - `target_device` — string (one of `cpu`, `gpu`, `tpu`)

Response (200):
```
{
  "optimized_model": "opt_<uuid>.<ext>",
  "metrics": {
    "original_size_mb": float,
    "optimized_size_mb": float,
    "size_reduction_percent": float,
    "original_latency_ms": float,
    "optimized_latency_ms": float
  }
}
```

GET /api/download/{filename}
- Returns the optimized model file as an octet stream.

## Notes about the optimizer

- `backend/app/model_optimizer.py` currently attempts real optimizations for TensorFlow, PyTorch and ONNX. Depending on the model and installed packages, conversion/quantization may be slow and require additional system libraries (for GPU support or specific TensorFlow builds).
- TPU optimization requires representative data for full integer quantization. The implementation falls back to FP16 quantization when full integer quantization fails.

## Troubleshooting

- If `uvicorn` fails to import the app as `app.main:app`, confirm you are running the command from the `backend/` folder and that `backend` is the current working directory.
- On Windows PowerShell use `.\.venv\Scripts\Activate.ps1` to activate the virtualenv before installing or running.
- TensorFlow/PyTorch/ONNX installations can be large and sometimes fail on Windows without the right wheel for your Python version. If `pip install -r requirements.txt` errors on a package, try installing the package separately or use the CPU-only variants (for example `pip install tensorflow-cpu` if available for your platform).
- If quantization fails with "representative_dataset must be specified", the optimizer now includes a synthetic representative dataset, but for best results provide a real sample dataset or adjust the optimizer to accept a dataset path.

## Local testing recommendations

- Start the backend and confirm the OpenAPI docs at `http://127.0.0.1:8000/docs` load.
- Use the frontend to perform an end‑to‑end flow or test the backend using `curl` / `httpie`:

```powershell
# example using curl (PowerShell)
curl -F "file=@C:\path\to\model.onnx" -F "target_device=cpu" http://127.0.0.1:8000/api/optimize
```

## Development notes

- The FastAPI app accepts files to the `uploads/` folder and writes optimized artifacts to `optimized/` inside `backend/`.
- The frontend expects the optimized filename in the JSON response and requests the download from `/api/download/<filename>`.

## Contributing

If you want to help improve ModelFlex:

1. Open a new branch for your change
2. Add tests for new behavior (if applicable)
3. Run frontend and backend locally to verify end‑to‑end
4. Create a PR with a clear description of changes

## License

MIT

---
Timestamp: October 22, 2025