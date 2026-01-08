{
  description = "HydraMesh DAIIC: DCF AI Inference Cluster (CUDA, ROCm, CPU)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    nixos-generators = {
      url = "github:nix-community/nixos-generators";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
      "https://rocm.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "rocm.cachix.org-1:6QLNg1NbVzc7Q87rK0Kqa+4L5+jO2r+q7Q+O9X+vI4Q="
    ];
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, nixos-generators, ... }:
    let
      # ═══════════════════════════════════════════════════════════════════════
      # System-Independent Definitions
      # ═══════════════════════════════════════════════════════════════════════
      
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
      
      # NixOS module (system-independent)
      nixosModule = { config, lib, pkgs, ... }:
        let 
          cfg = config.services.hydramesh;
        in {
          options.services.hydramesh = {
            enable = lib.mkEnableOption "HydraMesh DCF Service";
            
            role = lib.mkOption {
              type = lib.types.enum [ "head" "worker" ];
              default = "worker";
              description = "Node role in the cluster";
            };
            
            backend = lib.mkOption {
              type = lib.types.enum [ "nvidia" "rocm" "cpu" ];
              default = "cpu";
              description = "Compute backend for inference";
            };
            
            headIp = lib.mkOption {
              type = lib.types.str;
              default = "127.0.0.1";
              description = "IP address of the head controller (for workers)";
            };
            
            hfToken = lib.mkOption {
              type = lib.types.str;
              default = "";
              description = "HuggingFace token for gated models";
            };
            
            modelPath = lib.mkOption {
              type = lib.types.nullOr lib.types.str;
              default = null;
              description = "Path to local model file (.safetensors)";
            };
            
            modelId = lib.mkOption {
              type = lib.types.str;
              default = "runwayml/stable-diffusion-v1-5";
              description = "HuggingFace model ID (if modelPath not set)";
            };
            
            clientPort = lib.mkOption {
              type = lib.types.port;
              default = 7777;
              description = "UDP port for client connections";
            };
            
            workerPort = lib.mkOption {
              type = lib.types.port;
              default = 7778;
              description = "UDP port for worker bus";
            };
            
            healthPort = lib.mkOption {
              type = lib.types.port;
              default = 8080;
              description = "HTTP port for health checks";
            };
          };

          config = lib.mkIf cfg.enable {
            # Assertions
            assertions = [
              {
                assertion = cfg.role == "head" || cfg.modelPath != null || cfg.modelId != "";
                message = "Workers must have either modelPath or modelId configured";
              }
            ];

            # User/Group
            users.users.hydramesh = {
              isSystemUser = true;
              group = "hydramesh";
              home = "/var/lib/hydramesh";
              createHome = true;
              extraGroups = lib.optionals (cfg.backend == "rocm") [ "video" "render" ];
            };
            users.groups.hydramesh = {};

            # Firewall
            networking.firewall.allowedUDPPorts = [ 
              cfg.clientPort 
              cfg.workerPort 
              7779  # Worker listen port
              9999  # Shim ingress
            ];
            networking.firewall.allowedTCPPorts = [ 
              cfg.healthPort 
              8081  # Worker health
              7860  # WebUI
            ];

            # SystemD Service
            systemd.services.hydramesh = {
              description = "HydraMesh ${cfg.role} (${cfg.backend})";
              wantedBy = [ "multi-user.target" ];
              after = [ "network.target" ];
              
              environment = lib.filterAttrs (n: v: v != "") {
                HF_TOKEN = cfg.hfToken;
                HF_HOME = "/var/lib/hydramesh/huggingface";
                SD_MODEL_ID = cfg.modelId;
                DCF_CLIENT_PORT = toString cfg.clientPort;
                DCF_WORKER_PORT = toString cfg.workerPort;
                DCF_HEALTH_PORT = toString cfg.healthPort;
                HSA_OVERRIDE_GFX_VERSION = if cfg.backend == "rocm" then "10.3.0" else "";
                TORCH_DEVICE = if cfg.backend == "cpu" then "cpu" else "cuda";
              };

              serviceConfig = {
                User = "hydramesh";
                Group = "hydramesh";
                WorkingDirectory = "/var/lib/hydramesh";
                StateDirectory = "hydramesh";
                
                # Hardening
                ProtectSystem = "strict";
                ProtectHome = true;
                PrivateTmp = true;
                NoNewPrivileges = true;
                ProtectKernelTunables = true;
                ProtectKernelModules = true;
                ProtectControlGroups = true;
                RestrictSUIDSGID = true;
                
                # Allow GPU access
                SupplementaryGroups = lib.optionals (cfg.backend == "rocm") [ "video" "render" ];
                
                # Resource Management
                Restart = "always";
                RestartSec = 5;
                OOMScoreAdjust = -500;
                
                # Memory limits
                MemoryMax = "90%";
                MemoryHigh = "80%";
                
                ExecStart =
                  let
                    modelArg = if cfg.modelPath != null 
                               then "--model-path ${cfg.modelPath}"
                               else "";
                  in
                  if cfg.role == "head" then
                    "${pkgs.python311}/bin/python3 ${./src/head_controller.py}"
                  else
                    "${pkgs.python311}/bin/python3 ${./src/worker_node.py} --head-ip ${cfg.headIp} ${modelArg}";
              };
            };
          };
        };

    in
    flake-utils.lib.eachSystem supportedSystems (system:
      let
        # ═══════════════════════════════════════════════════════════════════════
        # Package Sets per Backend
        # ═══════════════════════════════════════════════════════════════════════
        
        pkgsCuda = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config = { allowUnfree = true; cudaSupport = true; };
        };

        pkgsRocm = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config = { allowUnfree = true; rocmSupport = true; cudaSupport = false; };
        };

        pkgsCpu = import nixpkgs {
          inherit system;
          overlays = [ (import rust-overlay) ];
          config = { allowUnfree = true; cudaSupport = false; rocmSupport = false; };
        };

        # ═══════════════════════════════════════════════════════════════════════
        # Shared Assets
        # ═══════════════════════════════════════════════════════════════════════

        rustToolchain = pkgsCpu.rust-bin.stable.latest.default.override {
          extensions = [ "rust-src" "rust-analyzer" "clippy" ];
        };

        meshSource = pkgsCpu.stdenv.mkDerivation {
          name = "hydramesh-src";
          src = ./src;
          installPhase = ''
            mkdir -p $out/bin
            cp *.py $out/bin/
            chmod +x $out/bin/*.py
          '';
        };

        # ═══════════════════════════════════════════════════════════════════════
        # Python Environments
        # ═══════════════════════════════════════════════════════════════════════

        commonPyPkgs = ps: with ps; [
          pydantic
          requests
          numpy
          pillow
          omegaconf
          protobuf
          sentencepiece
          gradio
        ];

        pythonML_Nvidia = pkgsCuda.python311.withPackages (ps: (commonPyPkgs ps) ++ (with ps; [
          torch
          accelerate
          safetensors
          huggingface-hub
          transformers
          diffusers
          bitsandbytes
        ]));

        pythonML_Rocm = pkgsRocm.python311.withPackages (ps: (commonPyPkgs ps) ++ (with ps; [
          torch
          accelerate
          safetensors
          huggingface-hub
          transformers
          diffusers
        ]));

        pythonML_Cpu = pkgsCpu.python311.withPackages (ps: (commonPyPkgs ps) ++ (with ps; [
          torch
          accelerate
          safetensors
          huggingface-hub
          transformers
          diffusers
        ]));

      in {
        # ═══════════════════════════════════════════════════════════════════════
        # Packages
        # ═══════════════════════════════════════════════════════════════════════

        packages = {
          # Head Node Container
          container-head = pkgsCpu.dockerTools.buildLayeredImage {
            name = "alh477/mesh-head";
            tag = "latest";
            contents = [ pythonML_Cpu meshSource pkgsCpu.bash pkgsCpu.coreutils pkgsCpu.curl ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/head_controller.py" ];
              Env = [ 
                "PYTHONPATH=${pythonML_Cpu}/${pythonML_Cpu.sitePackages}"
                "DCF_CLIENT_PORT=7777"
                "DCF_WORKER_PORT=7778"
                "DCF_HEALTH_PORT=8080"
              ];
              ExposedPorts = { 
                "7777/udp" = {}; 
                "7778/udp" = {}; 
                "8080/tcp" = {};
              };
              WorkingDir = "/var/lib/hydramesh";
              Labels = {
                "org.opencontainers.image.title" = "HydraMesh Head Controller";
                "org.opencontainers.image.version" = "5.1.0";
                "org.opencontainers.image.vendor" = "DeMoD LLC";
              };
            };
          };

          # NVIDIA Worker Container
          container-worker-nvidia = pkgsCuda.dockerTools.buildLayeredImage {
            name = "alh477/mesh-worker-nvidia";
            tag = "latest";
            maxLayers = 120;
            contents = [ 
              pythonML_Nvidia 
              meshSource 
              pkgsCuda.cudaPackages.cudatoolkit 
              pkgsCuda.bash 
              pkgsCuda.curl
            ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/worker_node.py" ];
              Env = [
                "PYTHONPATH=${pythonML_Nvidia}/${pythonML_Nvidia.sitePackages}"
                "LD_LIBRARY_PATH=${pkgsCuda.cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib"
                "HF_HOME=/data/huggingface"
                "TORCH_DEVICE=cuda"
              ];
              ExposedPorts = { "7779/udp" = {}; "8081/tcp" = {}; };
              WorkingDir = "/data";
              Volumes = { "/data" = {}; "/models" = {}; };
              Labels = {
                "org.opencontainers.image.title" = "HydraMesh Worker (NVIDIA)";
                "org.opencontainers.image.version" = "5.1.0";
              };
            };
          };

          # ROCm Worker Container
          container-worker-rocm = pkgsRocm.dockerTools.buildLayeredImage {
            name = "alh477/mesh-worker-rocm";
            tag = "latest";
            maxLayers = 120;
            contents = [ 
              pythonML_Rocm 
              meshSource 
              pkgsRocm.rocmPackages.clr 
              pkgsRocm.bash 
              pkgsRocm.curl
            ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/worker_node.py" ];
              Env = [
                "PYTHONPATH=${pythonML_Rocm}/${pythonML_Rocm.sitePackages}"
                "LD_LIBRARY_PATH=${pkgsRocm.rocmPackages.clr}/lib:/run/opengl-driver/lib"
                "HSA_OVERRIDE_GFX_VERSION=10.3.0"
                "HF_HOME=/data/huggingface"
                "TORCH_DEVICE=cuda"
              ];
              ExposedPorts = { "7779/udp" = {}; "8081/tcp" = {}; };
              WorkingDir = "/data";
              Volumes = { "/data" = {}; "/models" = {}; };
              Labels = {
                "org.opencontainers.image.title" = "HydraMesh Worker (ROCm)";
                "org.opencontainers.image.version" = "5.1.0";
              };
            };
          };

          # CPU Worker Container
          container-worker-cpu = pkgsCpu.dockerTools.buildLayeredImage {
            name = "alh477/mesh-worker-cpu";
            tag = "latest";
            contents = [ pythonML_Cpu meshSource pkgsCpu.bash pkgsCpu.curl ];
            config = {
              Cmd = [ "python3" "${meshSource}/bin/worker_node.py" ];
              Env = [
                "PYTHONPATH=${pythonML_Cpu}/${pythonML_Cpu.sitePackages}"
                "HF_HOME=/data/huggingface"
                "TORCH_DEVICE=cpu"
              ];
              ExposedPorts = { "7779/udp" = {}; "8081/tcp" = {}; };
              WorkingDir = "/data";
              Volumes = { "/data" = {}; "/models" = {}; };
              Labels = {
                "org.opencontainers.image.title" = "HydraMesh Worker (CPU)";
                "org.opencontainers.image.version" = "5.1.0";
              };
            };
          };

          # ISO
          iso = nixos-generators.nixosGenerate {
            inherit system;
            modules = [
              self.nixosModules.default
              ./iso/configuration.nix
              ({ pkgs, ... }: {
                environment.systemPackages = [
                  pythonML_Nvidia
                  rustToolchain
                  meshSource
                  pkgs.rocmPackages.rocminfo
                  pkgs.pciutils
                  pkgs.curl
                  pkgs.jq
                ];
              })
            ];
            format = "install-iso";
          };
        };

        # ═══════════════════════════════════════════════════════════════════════
        # Development Shell
        # ═══════════════════════════════════════════════════════════════════════

        devShells = {
          default = pkgsCpu.mkShell {
            name = "hydramesh-dev";
            
            buildInputs = [
              pythonML_Cpu
              rustToolchain
              pkgsCpu.git
              pkgsCpu.pkg-config
              pkgsCpu.openssl
              pkgsCpu.jq
              pkgsCpu.htop
              pkgsCpu.curl
            ];

            shellHook = ''
              export RUST_LOG=info
              export HF_HOME="$PWD/model-cache"
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              
              echo "────────────────────────────────────────────────────────"
              echo "  HydraMesh DCF Development Shell"
              echo "────────────────────────────────────────────────────────"
              echo "  Python:  $(python3 --version)"
              echo "  Rust:    $(rustc --version)"
              echo "  Device:  CPU (use nix develop .#cuda for GPU)"
              echo "────────────────────────────────────────────────────────"
            '';
          };

          cuda = pkgsCuda.mkShell {
            name = "hydramesh-cuda";
            
            buildInputs = [
              pythonML_Nvidia
              rustToolchain
              pkgsCuda.cudaPackages.cudatoolkit
              pkgsCuda.git
              pkgsCuda.nvtopPackages.full
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgsCuda.cudaPackages.cudatoolkit}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
              export CUDA_PATH=${pkgsCuda.cudaPackages.cudatoolkit}
              export HF_HOME="$PWD/model-cache"
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              
              echo "────────────────────────────────────────────────────────"
              echo "  HydraMesh DCF Development Shell (CUDA)"
              echo "────────────────────────────────────────────────────────"
            '';
          };

          rocm = pkgsRocm.mkShell {
            name = "hydramesh-rocm";
            
            buildInputs = [
              pythonML_Rocm
              rustToolchain
              pkgsRocm.rocmPackages.clr
              pkgsRocm.rocmPackages.rocminfo
              pkgsRocm.git
            ];

            shellHook = ''
              export LD_LIBRARY_PATH=${pkgsRocm.rocmPackages.clr}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH
              export HSA_OVERRIDE_GFX_VERSION=10.3.0
              export ROCM_PATH=${pkgsRocm.rocmPackages.clr}
              export HF_HOME="$PWD/model-cache"
              export PYTHONPATH="$PWD/src:$PYTHONPATH"
              
              echo "────────────────────────────────────────────────────────"
              echo "  HydraMesh DCF Development Shell (ROCm)"
              echo "────────────────────────────────────────────────────────"
            '';
          };
        };
      }
    ) // {
      # ═══════════════════════════════════════════════════════════════════════
      # System-Independent Outputs
      # ═══════════════════════════════════════════════════════════════════════
      
      nixosModules.default = nixosModule;
      nixosModules.hydramesh = nixosModule;
    };
}
