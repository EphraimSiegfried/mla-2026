{
  description = "Python Data Science Development Environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  outputs =
    inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      perSystem =
        { pkgs, ... }:
        let
          pythonEnv = pkgs.python3.withPackages (
            pythonPackages: with pythonPackages; [
              # Core / Math
              numpy
              scipy
              sympy

              # Data Processing
              pandas
              polars

              # Machine Learning & Stats
              scikit-learn
              statsmodels

              # Visualization
              matplotlib
              seaborn
              plotly

              # Notebooks / Interactive Environment
              jupyterlab
              notebook
              ipython

              # Code Quality / Utilities
              black
              ruff
              pytest
            ]
          );
        in
        {
          devShells.default = pkgs.mkShell {
            name = "data-science-shell";

            packages = [
              pythonEnv

              # Useful C/System dependencies for native compilation or graphics
              pkgs.zlib
              pkgs.glibcLocales
            ];
          };
        };
    };
}
