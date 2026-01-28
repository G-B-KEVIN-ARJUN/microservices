# test_step2.py
from pathlib import Path
from optimizer.quantizer import Quantizer

# We point to the file created in Step 1
INPUT_MODEL = Path("models/bert-tiny-onnx/model.onnx")

def test_quantization():
    if not INPUT_MODEL.exists():
        print("❌ Error: You must run test_step1.py first!")
        return

    print(f"🧪 Testing Quantizer on {INPUT_MODEL}...")
    
    quantizer = Quantizer(INPUT_MODEL)
    
    # Save the result in the same folder
    quantizer.quantize("models/bert-tiny-onnx")

if __name__ == "__main__":
    test_quantization()