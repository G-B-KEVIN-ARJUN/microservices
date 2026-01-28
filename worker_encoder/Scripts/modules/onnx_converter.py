# optimizer/onnx_converter.py
import os
from pathlib import Path
from optimum.exporters.onnx import main_export
from rich.console import Console

console = Console()

class ONNXConverter:
    def __init__(self, model_id: str, task: str = "auto"):
        """
        Initializes the converter.
        :param model_id: HuggingFace Hub ID (e.g., 'distilbert-base-uncased')
        :param task: The AI task (e.g., 'text-classification', 'text-generation', or 'auto')
        """
        self.model_id = model_id
        self.task = task

    def convert(self, output_dir: str):
        """
        Converts the PyTorch model to ONNX format.
        """
        output_path = Path(output_dir)
        
        # Create directory if it doesn't exist
        if not output_path.exists():
            output_path.mkdir(parents=True)

        console.print(f"[bold blue]🔄 Loading and Exporting [white]{self.model_id}[/white]...[/bold blue]")
        console.print(f"[dim]   Task: {self.task} | Target: {output_path}[/dim]")

        try:
            # The Magic Function: This handles loading, tracing, and config validation automatically
            main_export(
                model_name_or_path=self.model_id,
                output=output_path,
                task=self.task,
                opset=None,      # Use default ONNX opset
                device="cpu",    # CPU is more stable for export tracing
                fp16=False,      # We will handle quantization in the next step
                do_validation=False # Skip validation for speed (we will validate later)
            )
            
            onnx_file = output_path / "model.onnx"
            if onnx_file.exists():
                size_mb = onnx_file.stat().st_size / (1024 * 1024)
                console.print(f"[bold green]✅ Conversion Complete![/bold green]")
                console.print(f"   📂 File: {onnx_file}")
                console.print(f"   ⚖️  Size: [bold yellow]{size_mb:.2f} MB[/bold yellow]")
                return onnx_file
            else:
                raise FileNotFoundError("Export finished but model.onnx not found.")

        except Exception as e:
            console.print(f"[bold red]❌ Critical Error during export:[/bold red] {e}")
            raise e