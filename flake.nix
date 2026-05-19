{
  description = "claude code environment";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    devshell.url = "github:numtide/devshell";
  };
  outputs = { nixpkgs, devshell, fenix, ... }:
    let
      systems = [ "x86_64-linux" ];
      system = "x86_64-linux";

      # Pin claude-code to a specific version from GitHub ahead of nixpkgs.
      # Update the tag here, then rebuild: nix will fail with the correct npmDepsHash.
      claude-code-rev = "v2.1.144";

      claude-code-overlay = final: prev:
        let
          stdenv = final.stdenvNoCC;
          baseUrl = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases";
          platformKey = "${stdenv.hostPlatform.node.platform}-${stdenv.hostPlatform.node.arch}";
        in
        {
          claude-code =
            prev.claude-code.overrideAttrs
              (old: rec {
                version = final.lib.removePrefix "v" claude-code-rev;
                src = final.fetchurl {
                  url = "${baseUrl}/${version}/${platformKey}/claude";
                  sha256 = "sha256-FHSAd0Ry5XIP1eg2F7PpKZNE5yE++oTDJrJb1aDyC04=";
                };
              });
        };

      pkgsFor = system: import nixpkgs {
        config.allowUnfree = true;
        inherit system; overlays = [
        devshell.overlays.default
        fenix.overlays.default
        claude-code-overlay
      ];
      };

      # CUDA overlay: enable parallel building + pin cuDNN for Pascal cc 6.1
      cudaOverlay = final: prev: {
        cudaPackages_12_9 = prev.cudaPackages_12_9.overrideScope (cFinal: cPrev:
          let
            parallelPkgs = [
              "cuda_nvcc"
              "cuda_cudart"
              "cuda_cccl"
              "nccl"
              "libcublas"
              "libcufft"
              "libcusolver"
              "libcurand"
              "cuda_nvrtc"
              "cudnn"
            ];
            overriddenCudnn = cPrev.cudnn.overrideAttrs (old: rec {
              version = "9.11.1.4";
              src = prev.fetchurl {
                url = "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-${version}_cuda12-archive.tar.xz";
                hash = "sha256-YJrEikSORTMoek18YgVr8TD66MOx6yohgIDingAm7Bg=";
              };
            });
            makeParallel = name: {
              inherit name;
              value = (if name == "cudnn" then overriddenCudnn else cPrev.${name}).overrideAttrs (_: { enableParallelBuilding = true; });
            };
          in
          cPrev // (builtins.listToAttrs (map makeParallel parallelPkgs))
        );
      };

      # CUDA-enabled nixpkgs for GPU build (Pascal cc 6.1)
      pkgsCuda = import nixpkgs {
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapability = [ "6.1" ];
        };
        enableCUDA = true;
        cudaVersion = "12.9";
        system = "x86_64-linux";
        overlays = [
          devshell.overlays.default
          fenix.overlays.default
          cudaOverlay
          claude-code-overlay
        ];
      };

      # CUDA toolkit symlinkJoin: one derivation with all needed libs
      cudaToolkit = pkgsCuda.symlinkJoin {
        name = "cuda-toolkit-ph2";
        paths = with pkgsCuda.cudaPackages_12_9; [
          cuda_nvcc
          cuda_cudart
          cuda_cccl
          (cuda_nvrtc.include or cuda_nvrtc)
          cuda_nvrtc.lib
          (libcublas.include or libcublas)
          libcublas.lib
          (libcufft.include or libcufft)
          libcufft.lib
          (libcusolver.include or libcusolver)
          libcusolver.lib
          (libcurand.include or libcurand)
          libcurand.lib
        ];
      };

      rustPackages = (fenix.packages.${system}.stable.withComponents [
        "cargo"
        "clippy"
        "rust-src"
        "rustc"
        "rustfmt"
        "rust-analyzer"
      ]);
    in
    {
      devShells = {
        "${system}".default = pkgsCuda.devshell.mkShell {
          packages = with pkgsCuda; [
            rustPackages
            fish
            uv
            ty
            claude-code
            cudaToolkit
          ];
          env = [
            { name = "CUDA_HOME"; value = "${cudaToolkit}"; }
            { name = "CUDA_INCLUDE"; value = "${cudaToolkit}/include"; }
            { name = "CUDA_LIB"; value = "${cudaToolkit}/lib"; }
            { name = "LD_LIBRARY_PATH"; value = "${cudaToolkit}/lib:/run/opengl-driver/lib"; }
          ];
          commands = [
            {
              name = "claude-qwen3.6-nix";
              command = ''
                ANTHROPIC_BASE_URL=http://127.0.0.1:4000 \
                CLAUDE_CODE_ATTRIBUTION_HEADER="0" \
                ANTHROPIC_DEFAULT_OPUS_MODEL=qwen3.6-apex-think \
                ANTHROPIC_DEFAULT_SONNET_MODEL=qwen3.6-apex-think \
                ANTHROPIC_DEFAULT_HAIKU_MODEL=qwen3.6-apex \
                claude
              '';
            }
            {
              name = "claude-deepseek";
              command = ''
                ANTHROPIC_BASE_URL=$DEEPSEEK_BASE_URL \
                ANTHROPIC_AUTH_TOKEN=$DEEPSEEK_TOKEN \
                CLAUDE_CODE_ATTRIBUTION_HEADER="0" \
                ANTHROPIC_DEFAULT_OPUS_MODEL=deepseek-v4-pro[1m] \
                ANTHROPIC_DEFAULT_SONNET_MODEL=deepseek-v4-flash[1m] \
                ANTHROPIC_DEFAULT_HAIKU_MODEL=deepseek-v4-flash \
                claude --model "opusplan"
              '';
            }
            {
              name = "claude-fox";
              command = ''
                ANTHROPIC_BASE_URL=https://code.newcli.com/claude/ultra \
                ANTHROPIC_AUTH_TOKEN=$FOXCODE_TOKEN \
                claude
              '';
            }
          ];
        };
      };
    };
}
