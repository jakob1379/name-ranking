{
  description = "Name Ranking Application Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python314;
        # Use playwright 1.56.1 from nixpkgs to match Python package version
        playwright-pkgs = pkgs.playwright-driver.browsers;
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Python and package management
            python
            uv
            gcc

            # Git and development tools
            git
            pre-commit

            # Browsers for Playwright - use the browsers package
            playwright-driver.browsers

            # system dependencies for playwright browsers
            stdenv.cc.cc.lib
            libxkbcommon
            libGL
            libuuid
            libappindicator
            libdrm
            mesa
            nss
            nspr
            atk
            at-spi2-atk
            cups
            dbus
            expat
            libx11
            libxcomposite
            libxdamage
            libxext
            libxfixes
            libxrandr
            libxcb
            libxshmfence
            pango
            cairo
            gdk-pixbuf
            glib
            gtk3
            alsa-lib
            at-spi2-core
          ];

          shellHook = ''
            # Set Playwright browser path to use Nix-provided browsers
            export PLAYWRIGHT_BROWSERS_PATH=${playwright-pkgs}
            # Skip browser download and host validation
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
            export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true

            echo "Development environment ready"
            echo "Playwright browsers available at: ${playwright-pkgs}"
            echo "To install Python dependencies: uv sync"
            echo "Note: Playwright browsers already installed via Nix"
          '';
        };
      }
    );
}
