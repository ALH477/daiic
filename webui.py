import gradio as gr
import torch
import os
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

# --- Configuration ---
SD_MODEL_ID = os.getenv("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "mistralai/Mistral-7B-v0.1")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Starting WebUI on {DEVICE}")
print(f"SD Model: {SD_MODEL_ID}")
print(f"LLM Model: {LLM_MODEL_ID}")

# --- Lazy Loading Global Variables ---
sd_pipe = None
llm_pipeline = None

def get_sd_pipe():
    global sd_pipe
    if sd_pipe is None:
        print("Loading Stable Diffusion Model...")
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            use_safetensors=True
        )
        pipe = pipe.to(DEVICE)
        if DEVICE == "cuda":
            pipe.enable_attention_slicing()
        sd_pipe = pipe
    return sd_pipe

def get_llm_pipeline():
    global llm_pipeline
    if llm_pipeline is None:
        print("Loading LLM Model...")
        quant_config = None
        if DEVICE == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            quantization_config=quant_config,
            device_map="auto" if DEVICE == "cuda" else None,
            use_safetensors=True
        )
        
        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    return llm_pipeline

# --- Generation Functions ---

def generate_image(prompt, steps, guidance):
    try:
        pipe = get_sd_pipe()
        image = pipe(prompt, num_inference_steps=steps, guidance_scale=guidance).images[0]
        return image
    except Exception as e:
        return None

def generate_text(prompt, max_length, temperature):
    try:
        generator = get_llm_pipeline()
        result = generator(
            prompt, 
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        return result[0]['generated_text']
    except Exception as e:
        return f"Error: {str(e)}"

# --- Gradio Interface ---

with gr.Blocks(title="NixOS AI Inference Hub") as demo:
    gr.Markdown("# AI Inference Hub")
    
    with gr.Tab("Image Generation"):
        with gr.Row():
            with gr.Column():
                img_prompt = gr.Textbox(label="Prompt", placeholder="A cyberpunk city in rain...")
                steps = gr.Slider(1, 100, value=25, label="Inference Steps")
                guidance = gr.Slider(1.0, 20.0, value=7.5, label="Guidance Scale")
                img_btn = gr.Button("Generate Image", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Generated Result")
        
        img_btn.click(generate_image, inputs=[img_prompt, steps, guidance], outputs=output_image)

    with gr.Tab("Text Generation"):
        with gr.Row():
            with gr.Column():
                text_prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Explain quantum physics...")
                max_len = gr.Slider(10, 1000, value=200, label="Max New Tokens")
                temp = gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
                text_btn = gr.Button("Generate Text", variant="primary")
            with gr.Column():
                output_text = gr.Textbox(label="Response", lines=10)
        
        text_btn.click(generate_text, inputs=[text_prompt, max_len, temp], outputs=output_text)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
