{
  description = "Name Ranking Application Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfreePredicate = pkg: builtins.elem (nixpkgs.lib.getName pkg) [
            "claude-code"
          ];
        };
        python = pkgs.python313;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python
            uv
            gcc
            git
            prek
            playwright-test
          ];


          shellHook = ''
            export PLAYWRIGHT_BROWSERS_PATH=${pkgs.playwright-driver.browsers}
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
            export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            echo "Development environment ready"
            echo "Playwright browsers available at: ${pkgs.playwright-driver.browsers}"
            echo "To install Python dependencies: uv sync"
          '';
        };
      }
    );
}
