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

            # Browsers for Playwright
            playwright-driver
            playwright-test
            # Access the overridable browsers attribute
            (playwright-test {
              withChromium = true;
              withFirefox = true;
              withWebkit = true;
            })
            # system dependencies for playwright browsers
            stdenv.cc.cc.lib
            libxkbcommon
            libgl
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
            # Install playwright browsers
            export PLAYWRIGHT_BROWSERS_PATH=${pkgs.playwright-driver}/bin
            echo "Development environment ready"
            echo "To install Python dependencies: uv sync"
            echo "To install Playwright browsers: playwright install"
          '';
        };
      }
    );
}
