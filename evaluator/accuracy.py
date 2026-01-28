# evaluator/accuracy.py
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel  # <--- Changed to AutoModel (Base)
import torch
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console

console = Console()

class AccuracyChecker:
    def __init__(self, model_id: str, onnx_model_path: str):
        self.model_id = model_id
        self.onnx_path = onnx_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def check_fidelity(self, test_text: str = "This is a sample sentence to test the model accuracy."):
        """
        Compares the output vectors of the original PyTorch model vs the Quantized ONNX model.
        """
        console.print(f"[bold blue]Running Quality Control Check...[/bold blue]")

        # 1. Run Original PyTorch Model (Base Model - No Heads)
        pt_model = AutoModel.from_pretrained(self.model_id)  # <--- Load Base Model
        pt_inputs = self.tokenizer(test_text, return_tensors="pt")
        with torch.no_grad():
            # Grab the 'last_hidden_state' (the raw features)
            pt_outputs = pt_model(**pt_inputs).last_hidden_state.numpy()

        # 2. Run Quantized ONNX Model
        ort_session = ort.InferenceSession(str(self.onnx_path))
        
        onnx_inputs = {
            "input_ids": pt_inputs["input_ids"].numpy(),
            "attention_mask": pt_inputs["attention_mask"].numpy(),
            "token_type_ids": pt_inputs["token_type_ids"].numpy()
        }
        
        # Run inference
        onnx_outputs = ort_session.run(None, onnx_inputs)[0]

        # 3. Compare Results
        # Flatten both to 1D vectors to compare the entire feature map
        similarity = cosine_similarity(pt_outputs.flatten().reshape(1, -1), 
                                     onnx_outputs.flatten().reshape(1, -1))[0][0]

        console.print(f"[bold green]Fidelity Check Complete![/bold green]")
        console.print(f"   Similarity Score: [bold yellow]{similarity:.4f}[/bold yellow] (Max 1.0)")
        
        if similarity > 0.99:
            console.print("   Status: [bold green]Excellent Retention[/bold green]")
        elif similarity > 0.95:
            console.print("   Status: [bold yellow]Acceptable Quality Loss[/bold yellow]")
        else:
            console.print("   Status: [bold red]CRITICAL QUALITY DROP[/bold red]")

        return similarity