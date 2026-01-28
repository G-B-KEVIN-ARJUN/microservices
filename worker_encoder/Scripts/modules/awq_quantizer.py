# optimizer/awq_quantizer.py
import os
from pathlib import Path
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from rich.console import Console

console = Console()

class AWQQuantizer:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def quantize(self, output_dir: str):
        """
        Runs the AWQ pipeline: Load -> Calibrate -> Quantize -> Save.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        console.print(f"[bold blue]Loading LLM for AWQ: {self.model_id}[/bold blue]")

        # 1. Load Model & Tokenizer
        # strict=False allows loading models that might have minor config mismatches
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_id, 
            safetensors=True,
            strict=False,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # 2. Define AWQ Config
        # w_bit=4 means 4-bit quantization (Standard for AWQ)
        # q_group_size=128 is the industry standard block size
        quant_config = {
            "zero_point": True, 
            "q_group_size": 128, 
            "w_bit": 4, 
            "version": "GEMM"
        }

        # 3. Quantize (This includes Calibration automatically)
        console.print("[bold yellow]Running AWQ Calibration & Quantization...[/bold yellow]")
        console.print("[dim]Note: This uses the 'wikitext' dataset by default for calibration.[/dim]")
        
        model.quantize(
            tokenizer, 
            quant_config=quant_config
        )

        # 4. Save
        save_path = output_path / "awq_model"
        console.print(f"[bold blue]Saving 4-bit Model to: {save_path}[/bold blue]")
        
        model.save_quantized(save_path)
        tokenizer.save_pretrained(save_path)

        console.print("[bold green]AWQ Pipeline Complete![/bold green]")
        
        return save_path