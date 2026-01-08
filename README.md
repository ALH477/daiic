# NixOS Stable Diffusion Inference Environment

This repository provides a comprehensive Nix flake for deploying and running Stable Diffusion inference environments. It is designed for high-performance workloads using CUDA acceleration and emphasizes reproducibility through the Nix package manager.

---

## Copyright and License

Copyright (c) 2026, DeMoD LLC. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of DeMoD LLC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## Technical Specifications

The environment is built upon the following stack:

| Component | Version / Specification |
| --- | --- |
| **Operating System** | NixOS (Unstable Branch) |
| **Language Runtime** | Python 3.11 |
| **ML Framework** | PyTorch with CUDA Support |
| **Model Format** | SafeTensors (Default) |
| **Acceleration** | xformers, CUDA Toolkit, cuDNN |
| **Interface** | Gradio (Web UI) and Argparse (CLI) |

---

## Prerequisites

1. **Nix Package Manager**: Must have Flakes and Nix Command enabled.
2. **NVIDIA GPU**: Required for hardware acceleration.
3. **Nix Config**: Ensure `allowUnfree = true` is set in your Nix configuration to permit CUDA driver usage.

---

## Usage Instructions

### Development Environment

To enter an isolated shell with all dependencies, model paths, and library paths pre-configured:

```bash
nix develop

```

### Command Line Inference

The CLI tool allows for direct image generation from the terminal. By default, it uses the Stable Diffusion v1-5 model.

```bash
nix run .#inference -- --prompt "High resolution architectural render of a modern library"

```

Available arguments:

* `--prompt`: The text description for generation (Required).
* `--model`: HuggingFace model ID or local path.
* `--output`: Filename for the generated image.
* `--steps`: Number of inference steps (Default: 25).

### Web Interface

To launch the Gradio-based web UI for interactive generation:

```bash
nix run .#webui

```

The interface will be accessible at `http://localhost:7860`.

---

## NixOS System Deployment

This flake includes a NixOS module for system-wide service deployment. This is ideal for headless servers or dedicated inference nodes.

### Configuration Example

Add the flake to your inputs and include the module in your system configuration:

```nix
{
  inputs.sd-env.url = "github:demod-llc/stable-diffusion-flake";

  outputs = { nixpkgs, sd-env, ... }: {
    nixosConfigurations.server = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        sd-env.nixosModules.default
        {
          services.stable-diffusion = {
            enable = true;
            model = "runwayml/stable-diffusion-v1-5";
            port = 7860;
          };
        }
      ];
    };
  };
}

```

---

## Data Management

The environment dynamically handles model caching based on the execution context:

* **System Service**: Models are stored in `/var/lib/stable-diffusion-models`.
* **User Execution**: Models are stored in `$HOME/.cache/stable-diffusion-models`.

This ensures that development activities do not interfere with system-level service persistence and prevents permission conflicts.

---

## Optimization Notes

The pipeline utilizes `torch.float16` and `enable_attention_slicing()` when CUDA is detected to minimize VRAM consumption. For users with restricted hardware, further memory optimizations can be configured within the `inference.py` script by adjusting the `pipe.enable_sequential_cpu_offload()` settings.
