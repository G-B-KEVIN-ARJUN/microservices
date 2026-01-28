# worker_decoder/scripts/shrink_awq.py
import argparse
import sys
import os
from pathlib import Path

# Fix import path to find the 'modules' folder
sys.path.append(str(Path(__file__).parent))

try:
    from modules.awq_quantizer import AWQQuantizer
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import AWQQuantizer. Check your file structure.\nError: {e}")
    sys.exit(1)

def get_dir_size_mb(path):
    """Calculates total size of a directory in MB"""
    total = 0
    try:
        # Walk through the directory to count all model shards
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 * 1024)
    except Exception:
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", required=True)
    args = parser.parse_args()

    # Define Output Path
    project_root = Path(__file__).resolve().parent.parent.parent
    safe_name = args.model_id.replace("/", "_")
    output_dir = project_root / "output_models" / f"{safe_name}_awq"

    print(f"[Worker-Decoder] Starting Job for: {args.model_id}")
    print(f"[Worker-Decoder] Saving to: {output_dir}")

    # Run the Shrinker
    try:
        quantizer = AWQQuantizer(args.model_id)
        quantizer.quantize(str(output_dir))
        
        # --- NEW: Final Size Verification ---
        final_size = get_dir_size_mb(output_dir)
        
        print("-" * 40)
        print(f"   [Worker-Decoder] FINAL STATUS REPORT")
        print(f"   📂 Output Location: {output_dir}")
        print(f"   💾 Disk Usage:      {final_size:.2f} MB")
        print("-" * 40)

        print("[Worker-Decoder] SUCCESS")
        
    except Exception as e:
        print(f"[Worker-Decoder] FAILED: {e}")
        # Print the crash log for the server to see
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()