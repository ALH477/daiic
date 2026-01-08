{
  description = "HydraMesh Native Inference Cluster (UDP/DCF)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # Binary caches for PyTorch/CUDA
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; cudaSupport = true; };
        };

        pythonEnv = pkgs.python311.withPackages (ps: with ps; [
          # Networking & Core
          pydantic
          requests
          numpy
          
          # ML Stack
          torch
          accelerate
          safetensors
          huggingface-hub
          transformers
          diffusers
          bitsandbytes
          sentencepiece
          protobuf
        ]);

        # Shared Environment Setup
        envWrapper = ''
          export BASE_DIR="/var/lib/hydramesh"
          export HF_HOME="$BASE_DIR/huggingface"
          mkdir -p "$BASE_DIR"

          # Link System CUDA for bitsandbytes
          export LD_LIBRARY_PATH=/run/opengl-driver/lib:${pkgs.cudaPackages.cudatoolkit}/lib:$LD_LIBRARY_PATH
          export PYTHONPATH=${pythonEnv}/lib/python3.11/site-packages:$PYTHONPATH
        '';

      in {
        packages = {
          # Head Node Package
          mesh-head = pkgs.writeShellApplication {
            name = "mesh-head";
            runtimeInputs = [ pythonEnv pkgs.cudaPackages.cudatoolkit ];
            text = ''
              ${envWrapper}
              python ${./head_controller.py} "$@"
            '';
          };

          # Worker Node Package
          mesh-worker = pkgs.writeShellApplication {
            name = "mesh-worker";
            runtimeInputs = [ pythonEnv pkgs.cudaPackages.cudatoolkit ];
            text = ''
              ${envWrapper}
              python ${./worker_node.py} "$@"
            '';
          };
          
          default = self.packages.${system}.mesh-head;
        };

        # --- NIXOS MODULE ---
        nixosModules.default = { config, lib, pkgs, ... }: 
          let
            cfg = config.services.hydramesh;
          in {
            options.services.hydramesh = {
              enable = lib.mkEnableOption "HydraMesh DCF Cluster";
              
              role = lib.mkOption {
                type = lib.types.enum [ "head" "worker" ];
                default = "worker";
              };

              headIp = lib.mkOption {
                type = lib.types.str;
                default = "127.0.0.1";
                description = "IP of Head Node (Required for Workers)";
              };

              hfToken = lib.mkOption {
                type = lib.types.str;
                default = "";
                description = "HuggingFace Token";
              };
            };

            config = lib.mkIf cfg.enable {
              # 1. System User
              users.users.hydramesh = {
                isSystemUser = true;
                group = "hydramesh";
                home = "/var/lib/hydramesh";
                createHome = true;
              };
              users.groups.hydramesh = {};

              # 2. Firewall (UDP Ports)
              networking.firewall.allowedUDPPorts = [ 
                7777 # Head Ingress (Client -> Head)
                7778 # Head Internal (Worker -> Head)
                7779 # Worker Internal (Head -> Worker)
              ];

              # 3. Service Definition
              systemd.services.hydramesh = {
                description = "HydraMesh DCF Node (${cfg.role})";
                wantedBy = [ "multi-user.target" ];
                after = [ "network.target" ];
                environment = {
                  HF_TOKEN = cfg.hfToken;
                };
                
                serviceConfig = {
                  User = "hydramesh";
                  Group = "hydramesh";
                  WorkingDirectory = "/var/lib/hydramesh";
                  Restart = "always";
                  RestartSec = "3s";
                  LimitNOFILE = 65536;
                  
                  # Dynamic ExecStart based on role
                  ExecStart = if cfg.role == "head" then
                    "${self.packages.${system}.mesh-head}/bin/mesh-head"
                  else
                    "${self.packages.${system}.mesh-worker}/bin/mesh-worker --head-ip ${cfg.headIp}";
                };
              };
            };
        };
      }
    );
}
