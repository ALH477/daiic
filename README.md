# HydraMesh DCF: Distributed AI Inference Cluster

## Overview

HydraMesh is a high-performance, low-latency distributed inference framework built on the **DeMoD Communications Framework (DCF) v5.0.0**. It leverages a handshakeless, 17-byte binary UDP protocol to facilitate real-time data exchange between nodes. The system is built using NixOS flakes to provide bit-for-bit reproducible environments for development, containerized deployment, and bare-metal ISO execution.

---

## System Architecture

The cluster is composed of three primary layers to ensure a modular and scalable design:

1. 
**Ingress Layer (DCF Shim)**: A high-speed Rust bridge that receives external UDP traffic and encapsulates it into the 17-byte DCF header.


2. 
**Routing Layer (Head Node)**: A central controller that manages worker health via heartbeats and distributes tasks using round-robin logic.


3. **Inference Layer (Worker Nodes)**: GPU-accelerated endpoints supporting both NVIDIA (CUDA) and AMD (ROCm) backends.

---

## Use Cases and Deployment Modes

### 1. Local Development

**Use Case:** Development and testing of models or protocol extensions.

* 
**Command**: `nix develop` for NVIDIA/General or `nix develop .#rocm` for AMD.


* 
**Features**: Provides a full toolchain including Python 3.11 with ML stack, Rust stable, and hardware-specific SDKs.



### 2. Containerized Deployment

**Use Case:** Scaling across modern cloud or local Docker infrastructure.

* **Build Head**: `nix build .#container-head`
* **Build Worker (NVIDIA)**: `nix build .#container-worker-nvidia`
* **Build Worker (AMD)**: `nix build .#container-worker-rocm`

### 3. Bare-Metal All-in-One (ISO)

**Use Case:** Deploying a "headless" GPU server or a portable development station without installing a persistent OS.

* 
**Command**: `nix build .#iso`.


* 
**Function**: Creates a bootable NixOS ISO pre-loaded with NVIDIA and ROCm drivers, the full DCF SDK, and automated worker logic.



---

## Sourcing Models

The system supports both remote sourcing from the Hugging Face Hub and local hosting of Safe Tensor files.

### Remote Models (Hugging Face)

The cluster requires a Hugging Face Access Token to download and cache models.

* **Environment Variable**: Set `HF_TOKEN` in your shell or `.env` file.
* 
**NixOS Option**: Set `services.hydramesh.hfToken` in your configuration.



### Local Models (Safe Tensor)

**Use Case:** Loading custom .safetensors files (e.g., from CivitAI) directly from disk.

* **Setup**: Place `.safetensors` files in the `./local-models` directory.
* 
**Loading**: The worker uses the `from_single_file` loader to map weights directly into memory with maximum security and efficiency.



---

## Docker Compose Configuration

This setup utilizes the official DCF Shim and your locally built images.

```yaml
version: '3.8'

services:
  # Official DCF Bridge from alh477
  shim:
    image: alh477/dcf-shim:latest
    network_mode: host
    environment:
      - SHIM_INGRESS_PORT=9999
      - SHIM_NODE_TARGET=127.0.0.1:7777
    restart: always

  # Local Mesh Router
  head:
    image: demod/mesh-head:latest
    network_mode: host
    restart: always

  # Local GPU Worker (NVIDIA Example)
  worker:
    image: demod/mesh-worker-nvidia:latest
    network_mode: host
    command: [
      "python3", "/bin/worker_node.py", 
      "--head-ip", "127.0.0.1",
      "--model-path", "/models/my_custom_model.safetensors"
    ]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./local-models:/models
      - ./model-cache:/data/huggingface
    restart: always

```

---

## Protocol Definition (17-Byte Header)

All cluster traffic adheres to the following binary structure:

| Size | Type | Field | Description |
| --- | --- | --- | --- |
| 1B | u8 | `msg_type` | 0x01: Heartbeat, 0x02: Task, 0x03: Result 

 |
| 4B | u32 | `sequence` | Big-Endian packet identifier |
| 8B | u64 | `timestamp` | Microseconds since Epoch 

 |
| 4B | u32 | `payload_len` | Length of the subsequent data payload |

---

## Resource Management and Stability

**RAM Offloading**: Prevents Out-Of-Memory (OOM) crashes by spilling model weights from VRAM to System RAM using Accelerate.

**Precision**: Defaults to `torch.float16` for GPU backends to reduce memory footprint by 50%.

**Hardened Isolation**: The NixOS module utilizes `ProtectSystem` and `PrivateTmp` to secure the inference environment.

**Priority Handling**: Workers are assigned an `OOMScoreAdjust` of -500 to protect the inference process from kernel termination during heavy load.
