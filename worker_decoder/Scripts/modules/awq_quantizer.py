# worker_decoder/scripts/modules/awq_quantizer.py
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import os

class AWQQuantizer:
    def __init__(self, model_id: str):
        self.model_id = model_id

    def quantize(self, output_dir: str):
        print(f"   [AWQ] Loading Model: {self.model_id}")
        
        # Load model and tokenizer
        # FIXED: Removed 'strict=False' which causes errors on older transformers
        model = AutoAWQForCausalLM.from_pretrained(
            self.model_id, 
            safetensors=True,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        # Define 4-bit configuration
        quant_config = {
            "zero_point": True, 
            "q_group_size": 128, 
            "w_bit": 4, 
            "version": "GEMM"
        }

        # Run Quantization
        print("   [AWQ] Quantizing (This may take a while)...")
        model.quantize(tokenizer, quant_config=quant_config)

        # Save results
        print(f"   [AWQ] Saving to: {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        return output_dir