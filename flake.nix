{
  description = "Korean 100M LLM from Scratch Dev Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            python311
            uv
            stdenv.cc.cc.lib
            libGL
            glib
          ];
          shellHook = ''
            # 시스템 NVIDIA 드라이버 경로 추가 (Nix에서 GPU 인식을 위한 필수 설정)
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
            
            echo "✅ Nix GPU Environment Loaded"
            echo "💡 If CUDA is still not detected, ensure you have run 'uv sync' after pyproject.toml update."
          '';
        };
      });
}
