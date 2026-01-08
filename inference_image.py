#!/usr/bin/env python3
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="HuggingFace Model ID")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the generated image")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale (CFG)")
    
    args = parser.parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Image Pipeline on {device}...")
    
    try:
        # Initialize pipeline
        # Use float16 for GPU to save VRAM, float32 for CPU
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=dtype,
            use_safetensors=True
        )

        # Optimize Scheduler for speed
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Move to device
        pipe = pipe.to(device)
        
        # Enable memory optimizations if on GPU
        if device == "cuda":
            pipe.enable_attention_slicing()
            
        print(f"Generating image for prompt: '{args.prompt}'")
        
        # Run Inference
        with torch.inference_mode():
            image = pipe(
                args.prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance
            ).images[0]
        
        image.save(args.output)
        print(f"Success! Image saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error during generation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
