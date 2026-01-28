# worker_encoder/scripts/shrink_onnx.py
import argparse
import sys
import os
from pathlib import Path

# --- 1. ROBUST PATH SETUP ---
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    # Attempt 1: Standard
    from modules.onnx_converter import ONNXConverter
    from modules.quantizer import Quantizer
    from modules.calibrator import BertDataReader
except ImportError:
    try:
        # Attempt 2: Nested (VS Code style)
        from modules.optimizer.onnx_converter import ONNXConverter
        from modules.optimizer.quantizer import Quantizer
        from modules.optimizer.calibrator import BertDataReader
    except ImportError as e:
        print(f"CRITICAL ERROR: Could not find modules. {e}")
        sys.exit(1)

def get_dir_size(path):
    """Calculates total size of a directory in MB"""
    total = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total / (1024 * 1024) # Convert bytes to MB

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--mode", default="dynamic")
    args = parser.parse_args()

    print(f"[Worker-Encoder] Processing: {args.model_id}")
    
    project_root = Path(__file__).resolve().parent.parent.parent
    safe_name = args.model_id.replace("/", "_")
    output_dir = project_root / "output_models" / f"{safe_name}_onnx"

    print(f"[Worker-Encoder] Saving to: {output_dir}")

    try:
        # 1. Convert (FP32)
        print("   [ONNX] Converting to ONNX...")
        converter = ONNXConverter(args.model_id)
        onnx_file = converter.convert(str(output_dir))
        
        # Measure Original Size (FP32)
        original_size = os.path.getsize(onnx_file) / (1024 * 1024)

        # 2. Quantize (INT8)
        print("   [ONNX] Quantizing...")
        quantizer = Quantizer(onnx_file)
        
        if args.mode == "dynamic":
            final_model = quantizer.quantize_dynamic(str(output_dir))
        else:
            reader = BertDataReader(args.model_id)
            final_model = quantizer.quantize_static(str(output_dir), reader)
        
        # Measure Final Size
        final_size = os.path.getsize(final_model) / (1024 * 1024)
        
        # --- THE METRIC CALCULATION ---
        reduction = (1 - (final_size / original_size)) * 100
        
        print("-" * 40)
        print(f"   [METRICS] COMPRESSION COMPLETE")
        print(f"   Original Size: {original_size:.2f} MB")
        print(f"   Optimized Size: {final_size:.2f} MB")
        print(f"   🔻 Reduction: {reduction:.1f}%")
        print("-" * 40)

        print(f"[Worker-Encoder] SUCCESS")
        
    except Exception as e:
        print(f"[Worker-Encoder] FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()