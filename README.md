# DAIIC: DCF AI Inference Cluster

The HydraMesh Distributed AI Inference Cluster is a high-performance, low-latency framework designed for distributed data exchange and real-time AI synchronization. It implements a Distributed Actor System that utilizes the 17-byte binary UDP transport protocol defined by the DeMoD Communications Framework (DCF) v5.0.0. This version ensures hardware-agnostic deployments, supporting resources from edge devices to high-performance GPU clusters on NixOS.

---

## Copyright and License

Copyright (c) 2026 DeMoD LLC. All rights reserved. This software is licensed under the BSD-3-Clause License. The underlying DCF design specifications are provided under the GPL-3.0 License. This ensures transparency and community-driven development while protecting the proprietary implementations of the shim and driver logic.

---

## System Architecture

The environment implements a Distributed DCF Mesh Orchestrator, which effectively decouples high-level routing from intensive GPU computation.

The Ingress Layer uses the DCF Standalone Shim, a high-performance bidirectional UDP bridge written in Rust, which serves as the entry point for external clients. The Routing Layer consists of the Head Controller, a central router that manages the worker registry via UDP heartbeats and load-balances tasks across the mesh using 17-byte DCF messages. The Inference Layer is comprised of Worker Nodes, which are GPU-accelerated endpoints that execute LLM and Stable Diffusion tasks using 4-bit quantization and RAM offloading for hardware efficiency.

---

## Sourcing Models

The system is designed to pull models dynamically using your Hugging Face credentials, which is essential for accessing gated models like Llama-3 or Mistral. To configure this, you must obtain a "Read" access token from your Hugging Face settings and provide it to the cluster through the `HF_TOKEN` environment variable or the `hfToken` Nix option.

For local hosting, the cluster supports loading custom `.safetensors` files directly from a designated directory. Users should place their model checkpoints in the `./local-models` folder, and the worker node will use the `from_single_file` loader to map weights into memory with high security and efficiency.

---

## Deployment Artifacts (alh477)

The flake generates OCI-compliant containers under the `alh477` namespace to suit various hardware requirements.

The `alh477/mesh-head` image is a lightweight cluster router designed to run on any standard CPU. For NVIDIA users, the `alh477/mesh-worker-nvidia` image provides an inference engine optimized for CUDA. AMD users can utilize the `alh477/mesh-worker-rocm` image, which is built specifically for ROCm backends. Finally, the `alh477/mesh-worker-cpu` image offers a universal, hardware-agnostic inference engine that runs on any machine without requiring specific GPU drivers.

---

## Bare-Metal All-in-One (ISO)

The All-in-One ISO is a bootable NixOS environment designed for field deployment or rapid development. It creates a portable "Swiss Army Knife" that pre-configures the entire toolchain, including Rust, Python, and CUDA, along with the project source code.

When booted, the ISO automatically initializes kernel drivers for both NVIDIA and AMD hardware. It allows the machine to join the cluster as a worker or act as a head node without requiring a persistent OS installation. Developers can also use this environment for real-time coding and performance profiling using included tools like `nvtop` and `rust-analyzer`.

---

## Protocol Definition (17-Byte Header)

All internal cluster traffic adheres to the DCF 17-byte handshakeless binary header to ensure wire compatibility and minimal overhead.

The structure begins with a 1-byte `msg_type` (0x01 for Heartbeat, 0x02 for Task, 0x03 for Result), followed by a 4-byte Big-Endian `sequence` identifier for client tracking. An 8-byte Big-Endian `timestamp` records microseconds since the Epoch. The header concludes with a 4-byte Big-Endian `payload_len`, which specifies the length of the subsequent data payload.

---

## Resource Management and Stability

Professional stability is ensured through aggressive resource management strategies. RAM Offloading prevents Out-Of-Memory (OOM) crashes by spilling model weights from VRAM to System RAM when capacity is reached.

The NixOS module applies hardened isolation using `ProtectSystem` and `PrivateTmp` to secure the inference environment. Furthermore, workers are assigned an `OOMScoreAdjust` of -500, which prioritizes the inference process and protects it from kernel termination during heavy load spikes. The Head Node also implements self-healing by pruning workers that fail to send heartbeats within a 10-second window.

---

## Developer Experience (DevX)

The project provides a dedicated development shell that can be accessed via `nix develop` or `nix-shell`. This environment is equipped with auto-detection logic that scans the host system for NVIDIA or AMD drivers before falling back to a CPU-only mode.

The shell pre-loads all necessary compilers, ML libraries, and hardware-specific SDKs. It includes aliases and environment variables that point to the correct dynamic libraries, ensuring that tools like `bitsandbytes` function correctly across different hardware backends. This unified interface allows developers to switch between local testing and cluster deployment seamlessly.

---

## Recommended: Ultra Intelligent Head System (HydraMesh)

The **Lisp HydraMesh** implementation is the recommended "Intelligent Head" for complex DCF clusters. It provides a feature-rich environment optimized for real-time decision-making, high-speed serialization, and hardware-level transport management.

### Core Intelligence Features
 
**High-Speed Serialization**: Replaces standard JSON with binary Protocol Buffers, achieving 10–100x faster serialization for real-time state synchronization.
 
**Modular DSL Architecture**: Employs independent, composable Domain Specific Languages for game networking, audio streaming, and IoT sensor management.
 
**Low-Latency Transport**: Optimized for <10ms latency in high-demand gaming and audio scenarios using native UDP transport.
 
**Advanced Reliability**: Features a built-in redundancy module with RTT-based peer grouping and self-healing logic.
 
**Local Persistence**: Integrates with StreamDB for high-performance, key-value storage of cluster states and metrics.

### HydraMesh Container Variants

The system is delivered via hardened OCI-compliant images designed for different operational environments:

| Variant | Base Image | Security Level | Use Case |
| --- | --- | --- | --- |
| **`runtime`** | Debian Bookworm | High | Production deployments |
| **`runtime-dev`** | fukamachi/sbcl | Medium | Development with interactive REPL |
| **`runtime-usb4`** | runtime (ext) | High | High-speed hardware transport (Thunderbolt) 

 |

### Hardened Container Security

* **Non-Root Execution**: Containers run under the unprivileged `hydramesh` user (UID 10001) to mitigate escape risks.
* **Minimal Attack Surface**: Production variants use minimal base images with all Linux capabilities dropped except `NET_BIND_SERVICE`.
* **Read-Only Integrity**: The root filesystem is set to read-only, preventing unauthorized modifications to the binary environment.
* **Automated Health Monitoring**: Includes built-in health checks that verify node status every 30 seconds.

### Hardware-Accelerated Transport (Thunderbolt/USB4)

For ultra-low latency requirements, the HydraMesh head system includes a beta **Thunderbolt/USB4 transport plugin**. This allows for point-to-point PCIe-level communication between nodes, reaching throughputs of 20–40 Gbps and RTTs as low as 5μs.

**Configuration**: Enable via the `HYDRAMESH_USB4_ENABLED=true` environment variable.

**Requirements**: Requires the `runtime-usb4` image variant and `--privileged` or `--cap-add` flags for hardware access.
-----

## Theoretical Capability and Practical Use Cases

The aggregation of eight Framework desktops into a unified HydraMesh cluster represents a paradigm shift in local computing power, specifically when leveraging the Ryzen AI Max platform. By interconnecting these nodes via the high-speed USB4 transport plugin, the system establishes a theoretical memory pool exceeding one terabyte of unified VRAM. This massive addressable space allows the cluster to load and execute the most demanding open-source models, such as DeepSeek R1 or Llama 3.1 405B, at high precision levels that are typically reserved for enterprise-grade data centers. The 17-byte DCF protocol facilitates this by ensuring that the inter-node coordination happens with a round-trip time as low as 5μs, effectively eliminating the networking bottlenecks found in traditional Ethernet-based clusters.

Beyond raw memory capacity, the theoretical throughput of this eight-node mesh is substantial, providing a combined 128 CPU cores and 320 GPU Compute Units. This hardware-agnostic architecture allows the cluster to utilize NVIDIA, AMD, and even CPU-only workers simultaneously, optimizing the workload based on the specific requirements of the task. Because the HydraMesh head system utilizes binary Protocol Buffers for serialization, it can handle state synchronization at speeds 10 to 100 times faster than standard JSON-based systems, enabling the cluster to maintain coherence even during high-frequency data bursts. The system's stability is further reinforced by an OOMScoreAdjust of -500 and hardened containerization, ensuring that even under maximum theoretical load, the inference processes remain protected from kernel-level termination.

### Advanced Use Case Scenarios

One primary use case for this eight-node configuration is the hosting of private, local-first Large Language Model (LLM) services for sensitive data environments. Because the entire 900 GB+ VRAM pool is contained within a private USB4 mesh, organizations can process massive datasets without ever exposing information to a cloud provider. The HydraMesh head node manages this by utilizing its modular DSL architecture to route complex queries across the cluster, ensuring that each node's memory is utilized efficiently while providing a unified API to the end-user. This allows for the execution of "monster" models that would be physically impossible to load on any single consumer device.

Another significant use case involves real-time multi-modal synchronization for gaming and advanced simulation. By utilizing the HydraMesh Audio and Game-Net DSLs, the cluster can manage thousands of concurrent actor updates and high-fidelity audio streams with less than 10ms of latency. The system's self-healing capabilities ensure that even if a desktop in the 8-node chain fails, the registry reroutes tasks in less than 10 seconds, maintaining a seamless experience for connected clients. This makes it an ideal foundation for hosting distributed virtual worlds or real-time AI-driven NPCs that require constant, low-latency interaction with a high-parameter brain.

A third application is the deployment of a high-density IoT and sensor network processing hub. In this scenario, the 8-node cluster acts as an "Ultra-Intelligent Head" that aggregates data from thousands of edge sensors using the Sensor-Net DSL. The cluster performs real-time aggregation and threshold monitoring, utilizing its combined compute power to run local anomaly detection models on the incoming data streams. With StreamDB persistence, every metric is stored with transaction-level integrity, providing a robust and searchable history of the entire network's performance and health.


