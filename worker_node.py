import argparse
import time
import logging
import threading
import os
import torch
import io
from diffusers import StableDiffusionPipeline
from dcf_common import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s [WORKER] %(message)s')
logger = logging.getLogger("WorkerNode")

LISTEN_PORT = 7779

class LocalSafeTensorEngine:
    def __init__(self, model_path):
        logger.info(f"Loading Local SafeTensor: {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            load_safety_checker=False
        ).to("cuda")

        self.pipe.enable_sequential_cpu_offload()
        self.pipe.enable_attention_slicing()
        logger.info("Local Model Loaded Successfully.")

    def run(self, prompt: str):
        image = self.pipe(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

def heartbeat_loop(sock, head_ip, head_port):
    target = (head_ip, head_port)
    while True:
        payload = str(LISTEN_PORT).encode()
        msg = DCFMessage(MSG_HEARTBEAT, 0, DCFMessage.current_timestamp_micros(), payload)
        sock.send(msg, target)
        time.sleep(2.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-ip", required=True)
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    engine = LocalSafeTensorEngine(args.model_path)
    sock = AsyncUDPSocket(LISTEN_PORT)
    
    hb_thread = threading.Thread(target=heartbeat_loop, args=(sock, args.head_ip, 7778), daemon=True)
    hb_thread.start()

    logger.info(f"Worker Online. Listening on {LISTEN_PORT}. Head at {args.head_ip}:7778")

    while True:
        packet = sock.recv()
        if packet:
            msg, _ = packet
            if msg.msg_type == MSG_TASK:
                try:
                    prompt = msg.payload.decode('utf-8')
                    logger.info(f"Generating: {prompt[:20]}...")
                    start_ts = time.time()
                    png_bytes = engine.run(prompt)
                    duration = time.time() - start_ts
                    
                    resp = DCFMessage(MSG_RESULT, msg.sequence, DCFMessage.current_timestamp_micros(), png_bytes)
                    sock.send(resp, (args.head_ip, 7778))
                    logger.info(f"Task {msg.sequence} Done ({duration:.2f}s). Sent {len(png_bytes)} bytes.")
                except Exception as e:
                    logger.error(f"Inference Failed: {e}")
        time.sleep(0.001)

if __name__ == "__main__":
    main()
