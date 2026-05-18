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
      systems = [ "x86_64-linux" "aarch64-darwin" ];

      # Pin claude-code to a specific version from GitHub ahead of nixpkgs.
      # Update the tag here, then rebuild: nix will fail with the correct npmDepsHash.
      claude-code-rev = "v2.1.138";

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
                  sha256 = "sha256-w8Vv+8Es8W5AwzaHyf5jYe0lDDWp4XGNDDjUkEn1+MM=";
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
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
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
          default = pkgs.devshell.mkShell {
            packages = with pkgs; [
              rustPackages
              fish
              uv
              ty
              claude-code
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
        }
      );
    };
}

