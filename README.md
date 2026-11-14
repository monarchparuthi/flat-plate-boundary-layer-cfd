# Laminar Flat-Plate Boundary-Layer Solver (Python)

A from-scratch, two-dimensional, steady, incompressible **laminar boundary-layer marching solver** implemented with NumPy/SciPy and validated against an independently computed Blasius similarity solution.

> **Model scope:** this solves the Prandtl boundary-layer equations, not the full incompressible Navier–Stokes equations. The model is appropriate for attached, laminar, zero-pressure-gradient flow over a smooth flat plate. It does not predict transition, turbulence, separation, or pressure-gradient effects.

## Key results

The default case uses \(U_\infty=10\ \mathrm{m\,s^{-1}}\), \(L=1\ \mathrm m\), \(\rho=1.225\ \mathrm{kg\,m^{-3}}\), \(\mu=1.81\times10^{-5}\ \mathrm{Pa\,s}\), and a \(401\times201\) grid. This gives \(Re_L\approx6.77\times10^5\).

| Validation quantity | Baseline (`bdf1`, uniform) | Enhanced (`bdf2`, clustered) |
|---|---:|---:|
| Velocity-profile RMSE | 0.11–0.18% of \(U_\infty\) | 0.04–0.10% |
| Maximum profile error | 0.44–1.35% of \(U_\infty\) | 0.11–0.51% |
| Skin-friction error | 0.03–1.19% | 0.20–0.50% |
| Maximum \(\delta_{99}\) error | 9.85% | 3.72% |
| \(\delta_{99}\) error at \(x/L=1\) | 1.72% | 0.52% |

The enhanced case meets the listed profile, skin-friction, displacement-thickness and momentum-thickness tolerances at the sampled stations. Its \(\delta_{99}\) error at \(x/L=0.1\) is 3.72%, slightly above the 3% target. The baseline has larger leading-edge thickness errors. These misses are reported rather than concealed or tuned away.

## Physics and numerical method

The governing boundary-layer equations are

\[
\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0,
\qquad
u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}
=\nu\frac{\partial^2u}{\partial y^2},
\qquad \frac{dp}{dx}=0.
\]

The solver marches in the streamwise direction. Each station requires a tridiagonal solve in the wall-normal direction. The code supports first-order (`bdf1`) and second-order (`bdf2`) backward differencing in \(x\), uniform or wall-clustered \(y\)-grids, and SciPy-banded or hand-written Thomas linear solvers. The wall-normal velocity is recovered by integrating continuity.

The Blasius reference is obtained independently by integrating its similarity ODE with SciPy and shooting for the wall-shear parameter. It is not used to initialise or prescribe the CFD solution.

## Quick start

Requirements: Python 3.10 or newer. Install Git separately if you intend to publish the repository.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe main.py
```

If GNU Make is installed, the equivalent shortcuts are `make setup`, `make test`, `make run`, and `make lint`. The Makefile selects the Windows virtual-environment paths automatically.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
make test
make run
```

If GNU Make is unavailable, replace `make test` and `make run` with `.venv/bin/python -m pytest` and `.venv/bin/python main.py`.

### Useful commands

```bash
python main.py --help
python main.py --Nx 201 --Ny 101 --U_inf 5
python main.py --skip-investigations
```

On Windows, use `.\.venv\Scripts\python.exe` instead of `python` if the environment is not activated. The `main.py` command-line flags correspond to the configurable physical and numerical parameters in `config.py`.

## Repository structure

| Path | Purpose |
|---|---|
| `config.py` | Physical and numerical parameters |
| `grid.py` | Structured grid and non-uniform derivative stencils |
| `solver.py` | Finite-difference boundary-layer solver |
| `blasius.py` | Independent Blasius ODE reference |
| `postprocess.py` | Boundary-layer thickness, skin friction and continuity diagnostics |
| `validation.py` | Error metrics, mesh study and GCI calculations |
| `investigations.py` | Reynolds-number, grid-clustering, scheme and performance studies |
| `visualization.py` | Plot generation |
| `main.py` | End-to-end simulation and output generation |
| `tests/` | Automated numerical and regression tests |
| `results/` | Included figures, CSV data and summary output |
| `REPORT.md` / `REPORT.pdf` | Detailed technical report |
| `docs/DEVELOPMENT.md` | Developer notes and numerical implementation details |

## Included figures

The repository contains generated plots for velocity contours, velocity profiles, Blasius similarity collapse, skin friction, boundary-layer thickness, mesh refinement, wall clustering and scheme comparison. Re-run `main.py` to regenerate the figures and numerical data for the current code and configuration.

## Mesh-convergence interpretation

The coarse/medium/fine study uses nested grids of \(101\times51\), \(201\times101\), and \(401\times201\). Several quantities improve monotonically under refinement. However, a monotone sequence alone does not establish that all quantities are in the asymptotic convergence range.

The reported GCI diagnostics explicitly flag implausible observed orders and non-monotone sequences. For example, the baseline trailing-edge skin-friction sequence produces an observed order outside the configured plausibility range, while the displacement-thickness sequence is non-monotone. The enhanced mid-station profile RMSE also gives a very large fine-grid GCI estimate. Treat these GCI values as diagnostics, not as a blanket claim of mesh independence. A stronger study would add at least one finer grid and demonstrate asymptotic behaviour for each quantity of interest.

## Physical limitations and interpretation

- The default \(Re_L\approx6.77\times10^5\) is above the often-quoted order-of-magnitude flat-plate transition value of \(Re_x\sim5\times10^5\). Transition is disturbance- and environment-dependent; this solver **imposes laminar flow** and must not be interpreted as a physical prediction of a fully laminar plate at every downstream station.
- The leading edge is a singular start-up region for the similarity solution. The first sampled station has the largest relative thickness error; the enhanced \(\delta_{99}\) error there is 3.72%.
- The `bdf2` solution has a small, documented near-wall start-up wiggle. The test suite constrains its magnitude and location; it is not evidence of a higher-fidelity physical model.
- The outer boundary is finite and prescribed as \(u=U_\infty\). The default domain height is several times the trailing-edge Blasius \(\delta_{99}\); users changing Reynolds number or domain height should check this ratio.
- The solver does not model compressibility, heat transfer, roughness, three-dimensionality, transition, turbulence, separation, or non-zero pressure gradients.

## Reproducibility and validation

Run the tests and the complete pipeline from the repository root:

```bash
python -m pytest
python main.py
```

The pipeline writes CSV diagnostics and a `summary.json` file under `results/data/`. Existing numerical tables in this README and the technical report are snapshots for the stated default configuration; regenerate and inspect the CSVs after changing the code or parameters.

## License

No licence has been selected for this repository yet. Until one is added, assume the source is shared for review but is not granted an explicit open-source reuse licence.
