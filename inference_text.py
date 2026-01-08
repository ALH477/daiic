#!/usr/bin/env python3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="LLM Text Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="HuggingFace Model ID")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Text Pipeline on {device}...")

    try:
        # Configure quantization (4-bit) for efficient loading on consumer GPUs
        quantization_config = None
        if device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        print(f"Loading Model: {args.model}")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Load model with quantization if available
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Tokenize
        inputs = tokenizer(args.prompt, return_tensors="pt")
        if device == "cuda":
            inputs = inputs.to("cuda")

        print("Generating response...")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=args.temp,
                pad_token_id=tokenizer.eos_token_id
            )
            
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "="*40)
        print("OUTPUT:")
        print("="*40)
        print(decoded_output)
        print("="*40)

    except Exception as e:
        print(f"Error during text generation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
