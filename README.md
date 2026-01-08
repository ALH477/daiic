# DAIIC: DCF AI Inference Cluster

A high-performance, hardware-agnostic distributed AI inference framework built on the DeMoD Communications Framework (DCF) v5.1.0. Deploy Stable Diffusion and LLM inference across heterogeneous GPU clusters with sub-10ms routing latency.

## Features

- **Multi-Backend Support**: NVIDIA CUDA, AMD ROCm, and CPU inference in a single cluster
- **Zero-Config Networking**: UDP-based mesh with automatic worker discovery
- **Production-Ready**: Health checks, metrics endpoints, graceful shutdown, OOM protection
- **NixOS Native**: Declarative deployment via flake with reproducible builds
- **Chunked Transfer**: Automatic fragmentation for large inference results (images)

## Quick Start

### Docker Compose (Recommended)

```bash
# Clone and enter directory
git clone https://github.com/demod-llc/daiic.git
cd daiic

# Set HuggingFace token (required for gated models)
export HF_TOKEN="hf_your_token_here"

# Start with NVIDIA GPU
docker compose --profile nvidia up -d

# Or AMD GPU
docker compose --profile rocm up -d

# Or CPU-only
docker compose --profile cpu up -d

# Check status
curl http://localhost:8080/metrics | jq
```

### NixOS Module

```nix
# configuration.nix
{
  imports = [ inputs.daiic.nixosModules.default ];
  
  services.hydramesh = {
    enable = true;
    role = "worker";           # or "head"
    backend = "nvidia";        # or "rocm" / "cpu"
    headIp = "192.168.1.100";
    modelId = "runwayml/stable-diffusion-v1-5";
  };
}
```

### Development Shell

```bash
# Auto-detect GPU
nix develop

# Force specific backend
nix develop .#cuda
nix develop .#rocm
nix develop .#default  # CPU

# Legacy nix-shell
nix-shell
nix-shell --arg backend '"cuda"'
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│    Shim     │────▶│    Head     │
│  (External) │     │  (Ingress)  │     │ (Router)    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              ┌─────▼─────┐            ┌───────▼───────┐          ┌───────▼───────┐
              │  Worker   │            │    Worker     │          │    Worker     │
              │  (NVIDIA) │            │    (ROCm)     │          │    (CPU)      │
              └───────────┘            └───────────────┘          └───────────────┘
```

### Components

| Component | Port | Description |
|-----------|------|-------------|
| Shim | UDP 9999 | External ingress bridge (Rust) |
| Head | UDP 7777/7778, TCP 8080 | Cluster router + health API |
| Worker | UDP 7779, TCP 8081 | Inference engine + health API |
| WebUI | TCP 7860 | Gradio interface |

## Protocol (DCF v5.1)

18-byte binary header for minimal overhead:

```
┌─────────┬──────┬──────────┬───────────┬─────────────┬─────────┐
│ Version │ Type │ Sequence │ Timestamp │ PayloadLen  │ Payload │
│  1 byte │ 1 b  │  4 bytes │  8 bytes  │   4 bytes   │  N bytes│
└─────────┴──────┴──────────┴───────────┴─────────────┴─────────┘
```

Message Types:
- `0x01` HEARTBEAT — Worker liveness
- `0x02` TASK — Inference request
- `0x03` RESULT — Inference response
- `0x06` CHUNK — Fragmented payload
- `0xFF` ERROR — Error with code

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | HuggingFace API token |
| `SD_MODEL_ID` | `runwayml/stable-diffusion-v1-5` | Default image model |
| `LLM_MODEL_ID` | `mistralai/Mistral-7B-v0.1` | Default text model |
| `TORCH_DEVICE` | auto | Force `cuda` or `cpu` |
| `DCF_WORKER_TIMEOUT` | `10` | Worker heartbeat timeout (s) |
| `DCF_REQUEST_TIMEOUT` | `300` | Inference timeout (s) |

### Local Models

Mount `.safetensors` files to `/models`:

```yaml
volumes:
  - ./my-models:/models:ro

environment:
  - LOCAL_MODEL_PATH=custom-sd.safetensors
```

## API Endpoints

### Health Checks

```bash
# Head controller
curl http://localhost:8080/health    # Liveness
curl http://localhost:8080/ready     # Readiness (has workers?)
curl http://localhost:8080/metrics   # Prometheus-compatible

# Worker
curl http://localhost:8081/health
curl http://localhost:8081/metrics
```

### Metrics Response

```json
{
  "node": {
    "messages_sent": 1542,
    "messages_received": 1538,
    "tasks_processed": 127,
    "avg_latency_ms": 2847.3,
    "uptime_seconds": 3600.0
  },
  "registry": {
    "total_workers": 3,
    "healthy_workers": 3,
    "busy_workers": 1
  }
}
```

## Building Containers

```bash
# Build all containers
nix build .#container-head
nix build .#container-worker-nvidia
nix build .#container-worker-rocm
nix build .#container-worker-cpu

# Load into Docker
docker load < result

# Build bootable ISO
nix build .#iso
```

## Production Deployment

### Systemd Hardening

The NixOS module applies:
- `ProtectSystem=strict` — Read-only filesystem
- `PrivateTmp=true` — Isolated temp directory
- `NoNewPrivileges=true` — No privilege escalation
- `OOMScoreAdjust=-500` — Protected from OOM killer
- `MemoryMax=90%` — Memory limits

### Recommended Architecture

For production clusters:
1. Run 1 head per availability zone
2. Use dedicated GPU nodes for workers
3. Place shim behind load balancer for external access
4. Enable TLS termination at ingress layer

## Troubleshooting

### No workers available

```bash
# Check worker registration
curl http://localhost:8080/metrics | jq '.registry'

# Verify worker heartbeats
journalctl -u hydramesh -f
```

### CUDA not detected

```bash
# Verify driver
nvidia-smi

# Check container GPU access
docker run --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### ROCm permission denied

```bash
# Add user to required groups
sudo usermod -aG video,render $USER

# Verify device access
ls -la /dev/kfd /dev/dri/*
```

### Model download fails

```bash
# Verify HF token
echo $HF_TOKEN

# Test HuggingFace access
huggingface-cli whoami
```

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.

DCF protocol specification is GPL-3.0 for transparency.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Run tests: `nix flake check`
4. Submit a pull request

---

© 2026 DeMoD LLC. All rights reserved.
