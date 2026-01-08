{ 
  # Allow overriding the auto-detection if needed
  # Usage: nix-shell --arg overrideBackend '"rocm"'
  overrideBackend ? null 
}:

let
  # --- 1. Impure Hardware Detection ---
  # We check the host system for driver markers to determine the default backend.
  # This works in standard nix-shell.
  sysHasNvidia = builtins.pathExists "/proc/driver/nvidia/version";
  sysHasAmd    = builtins.pathExists "/dev/kfd";

  # Logic: Override -> NVIDIA -> AMD -> CPU
  backend = 
    if overrideBackend != null then overrideBackend
    else if sysHasNvidia then "cuda"
    else if sysHasAmd then "rocm"
    else "cpu";

  # --- 2. Configure Nixpkgs based on Detection ---
  pkgs = import <nixpkgs> { 
    config = { 
      allowUnfree = true;
      cudaSupport = (backend == "cuda");
      rocmSupport = (backend == "rocm");
    }; 
  };

  # --- 3. Define Python Environment ---
  # We dynamically build the python package set based on the backend
  pythonEnv = pkgs.python311.withPackages (ps: with ps; 
    # Common Deps
    ([ pydantic requests numpy pillow omegaconf protobuf sentencepiece ]) ++
    # Backend Specific Deps
    (if backend == "cuda" then [
       torch accelerate safetensors huggingface-hub 
       transformers diffusers bitsandbytes # BitsAndBytes works best on CUDA
     ]
     else if backend == "rocm" then [
       torch accelerate safetensors huggingface-hub
       transformers diffusers # BitsAndBytes often skipped on ROCm due to compat issues
     ]
     else [ # CPU
       torch accelerate safetensors huggingface-hub
       transformers diffusers
     ])
  );

  # --- 4. Define Rust & Tools ---
  rustEnv = [
    pkgs.cargo pkgs.rustc pkgs.rust-analyzer pkgs.clippy pkgs.rustfmt
  ];

  # --- 5. Hardware Libraries (LD_LIBRARY_PATH) ---
  hardwareLibs = 
    if backend == "cuda" then [
      pkgs.cudaPackages.cudatoolkit
      pkgs.cudaPackages.cudnn
      pkgs.cudaPackages.libcublas
      pkgs.cudaPackages.libcufft
    ]
    else if backend == "rocm" then [
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocminfo
      pkgs.rocmPackages.rocm-smi
    ]
    else [];

  libPath = pkgs.lib.makeLibraryPath (hardwareLibs ++ [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
    pkgs.glib
  ]);

in pkgs.mkShell {
  name = "hydramesh-${backend}-shell";

  buildInputs = [ pythonEnv ] ++ rustEnv ++ hardwareLibs ++ [
    pkgs.git pkgs.pkg-config pkgs.openssl pkgs.jq pkgs.htop pkgs.nvtopPackages.full
  ];

  shellHook = ''
    # --- Link Libraries Dynamically ---
    export LD_LIBRARY_PATH=${libPath}:/run/opengl-driver/lib:$LD_LIBRARY_PATH
    
    # --- Environment Setup ---
    export RUST_LOG=info
    export HF_HOME="$PWD/model-cache"
    export PYTHONPATH="$PWD/src:$PYTHONPATH"
    
    # --- ROCm Specific Overrides ---
    ${if backend == "rocm" then ''
      export HSA_OVERRIDE_GFX_VERSION=10.3.0 # Common fix for RDNA2 cards
      export ROCM_PATH=${pkgs.rocmPackages.clr}
    '' else ""}

    # --- CUDA Specific Overrides ---
    ${if backend == "cuda" then ''
      export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    '' else ""}

    # --- Banner ---
    echo "────────────────────────────────────────────────────────"
    echo "  HydraMesh DCF - Auto-Detected Environment"
    echo "────────────────────────────────────────────────────────"
    echo "  Detected Backend:  ${backend^^}"
    
    if [ "${backend}" == "cuda" ]; then
       echo "  > Driver Check:    Found /proc/driver/nvidia/version"
       echo "  > BitsAndBytes:    ENABLED"
    elif [ "${backend}" == "rocm" ]; then
       echo "  > Driver Check:    Found /dev/kfd"
       echo "  > BitsAndBytes:    DISABLED (Stability)"
    else
       echo "  > Driver Check:    No GPU found. Falling back to CPU."
    fi
    echo "────────────────────────────────────────────────────────"
  '';
}
