# optimizer/calibrator.py
import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from transformers import AutoTokenizer
from datasets import load_dataset

class BertDataReader(CalibrationDataReader):
    def __init__(self, model_id: str, batch_size: int = 10):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load a tiny slice of the "wikitext" dataset for calibration
        # This is used only if you select --mode static (for ONNX)
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        self.data = dataset.filter(lambda x: len(x["text"]) > 50)["text"][:batch_size]
        
        self.enum_data = iter(self.data)

    def get_next(self):
        """
        Generates the next batch of data for the calibrator to 'watch'.
        """
        try:
            text = next(self.enum_data)
        except StopIteration:
            return None

        # Convert text to ONNX-ready numbers
        inputs = self.tokenizer(text, return_tensors="np", padding="max_length", max_length=128, truncation=True)
        
        # ONNX Calibration expects a dictionary: { "input_name": numpy_array }
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs["token_type_ids"]
        }

    def rewind(self):
        """Resets the iterator (required by ONNX)"""
        self.enum_data = iter(self.data)