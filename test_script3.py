# test_step3.py
from evaluator.accuracy import AccuracyChecker

# Inputs from previous steps
MODEL_ID = "prajjwal1/bert-tiny"
QUANTIZED_MODEL = "models/bert-tiny-onnx/model_quantized.onnx"

def test_accuracy():
    checker = AccuracyChecker(MODEL_ID, QUANTIZED_MODEL)
    
    # We ask it to verify a simple sentence
    checker.check_fidelity("The quick brown fox jumps over the lazy dog.")

if __name__ == "__main__":
    test_accuracy()