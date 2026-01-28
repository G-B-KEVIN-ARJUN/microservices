# worker_decoder/debug_awq.py
import sys
import os

print("--- DIAGNOSTIC PROBE STARTING ---")

print("1. Testing Basic Imports...")
try:
    import argparse
    from pathlib import Path
    print("   [PASS] Basic libraries ok.")
except Exception as e:
    print(f"   [FAIL] Basic libraries crashed: {e}")
    sys.exit(1)

print("2. Testing PyTorch Import...")
try:
    import torch
    print(f"   [PASS] Torch version: {torch.__version__}")
    print(f"   [INFO] CUDA Available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"   [FAIL] Torch Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Torch Crashed: {e}")
    sys.exit(1)

print("3. Testing Transformers Import...")
try:
    import transformers
    print(f"   [PASS] Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"   [FAIL] Transformers Import Error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   [FAIL] Transformers Crashed: {e}")
    sys.exit(1)

print("4. Testing AutoAWQ Import (Most likely failure point)...")
try:
    # We try to import the specific class your script uses
    from awq import AutoAWQForCausalLM
    print("   [PASS] AutoAWQ loaded successfully!")
except ImportError as e:
    print(f"   [FAIL] AutoAWQ Import Error: {e}")
    print("   HINT: This usually means a version mismatch or missing DLL.")
    sys.exit(1)
except Exception as e:
    print(f"   [FAIL] AutoAWQ Crashed: {e}")
    sys.exit(1)

print("--- DIAGNOSTIC PROBE COMPLETE: ALL SYSTEMS GO ---")