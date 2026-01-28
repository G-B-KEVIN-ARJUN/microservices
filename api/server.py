# api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import sys
from pathlib import Path

app = FastAPI(title="Model Shrinker API")

# --- 1. DYNAMIC PATH SETUP ---
# Automatically finds the project root (C:\model-shrinker) 
# regardless of where you run this file from.
BASE_DIR = Path(__file__).resolve().parent.parent 

# Define paths relative to the root.
# NOTE: Check if your venv folders are named 'shrink' or 'venv'. 
# Based on your previous snippet, I am using 'shrink' for encoder and 'shrink_decoder' for decoder.
ENCODER_PYTHON = BASE_DIR / "worker_encoder" / "shrink" / "Scripts" / "python.exe"
DECODER_PYTHON = BASE_DIR / "worker_decoder" / "shrink_decoder" / "Scripts" / "python.exe"

ENCODER_SCRIPT = BASE_DIR / "worker_encoder" / "scripts" / "shrink_onnx.py"
DECODER_SCRIPT = BASE_DIR / "worker_decoder" / "scripts" / "shrink_awq.py"

class ShrinkRequest(BaseModel):
    model_id: str
    type: str  # 'encoder' or 'decoder'

@app.post("/shrink")
def shrink_model(request: ShrinkRequest):
    """
    Routes the request to the correct environment.
    """
    # Verify the paths exist before trying to run them
    if request.type == "encoder":
        worker_python = str(ENCODER_PYTHON)
        script_path = str(ENCODER_SCRIPT)
        if not ENCODER_PYTHON.exists():
            return {"status": "error", "logs": f"Python interpreter not found at: {ENCODER_PYTHON}"}
    elif request.type == "decoder":
        worker_python = str(DECODER_PYTHON)
        script_path = str(DECODER_SCRIPT)
        if not DECODER_PYTHON.exists():
            return {"status": "error", "logs": f"Python interpreter not found at: {DECODER_PYTHON}"}
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Use 'encoder' or 'decoder'.")

    # --- 2. EXECUTION ---
    try:
        # We start the subprocess
        result = subprocess.run(
            [worker_python, script_path, "--model_id", request.model_id],
            capture_output=True, # Captures both print() and errors
            text=True
        )
        
        # --- 3. LOG CAPTURE FIX ---
        # Combine Output (stdout) and Warnings (stderr)
        # This ensures you see "Saving to..." logs even if the status is "Success"
        full_logs = f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
        
        # Check if the worker reported a failure (Exit Code != 0)
        if result.returncode != 0:
            return {"status": "failed", "logs": full_logs}
            
        return {"status": "success", "logs": full_logs}

    except Exception as e:
        # Capture system-level crashes (e.g., Python not found)
        return {"status": "critical_error", "logs": str(e)}

# Run command: python -m uvicorn server:app --reload