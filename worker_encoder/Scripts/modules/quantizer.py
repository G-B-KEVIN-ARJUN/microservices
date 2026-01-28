# worker_encoder/scripts/modules/quantizer.py
from onnxruntime.quantization import quantize_dynamic, QuantType
import os

class Quantizer:
    def __init__(self, model_path: str):
        """
        model_path: The full path to the .onnx file we just created.
        """
        self.model_path = model_path

    def quantize_dynamic(self, output_dir: str):
        """
        Shrinks the model weights from 32-bit float to 8-bit integer.
        """
        # Define the output name
        output_model_path = os.path.join(output_dir, "model_quantized.onnx")

        print(f"   [Quantizer] Optimizing {os.path.basename(self.model_path)}...")
        
        # The Magic: ONNX Runtime's built-in shrinker
        quantize_dynamic(
            model_input=self.model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8  # This reduces size by ~4x
        )
        
        print(f"   [Quantizer] Success! Optimized model saved to: {output_model_path}")
        return output_model_path

    def quantize_static(self, output_dir, calibration_reader):
        # Placeholder for future advanced features
        print("   [Quantizer] Static quantization not yet implemented for this worker.")
        pass