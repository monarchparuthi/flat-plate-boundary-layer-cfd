# Developer guide

For users, see the top-level `README.md`. The numerical results are discussed in `REPORT.md`/`REPORT.pdf`.
The original specification is kept in `docs/project_brief.pdf`.

## Setup

Windows PowerShell (without GNU Make):

```powershell
py -3 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe -m pytest
.\\.venv\\Scripts\\python.exe main.py
```

macOS/Linux, or any platform with GNU Make:

```bash
make setup        # create .venv and install requirements
make test         # pytest
make lint         # ruff (pyflakes rules + syntax errors)
make run          # full pipeline -> results/
make report       # REPORT.md -> REPORT.pdf (needs pandoc + pdflatex)
```

The Makefile selects `.venv/Scripts/` on Windows and `.venv/bin/` on macOS/Linux.

Requirements: Python >= 3.10 (developed on 3.14), numpy >= 2.0 (`np.trapezoid`), scipy, matplotlib, pandas, pytest.
Only 3.14 has actually been run locally; the 3.10/3.12 CI matrix in `.github/workflows/ci.yml` has not been exercised yet.

The code is a set of flat top-level modules (no package), run from the repository root. `tests/conftest.py`
puts the root on `sys.path`.

## Architecture and data flow

```
config.Config ──► grid.make_grid ──► solver.solve ──► Result(u, v, grid, cfg)
                                                         │
          blasius.solve_blasius (independent) ──────┐    ├─► postprocess.* (δ99, δ*, θ, Cf, residuals)
                                                    ▼    ▼
                                             validation.* (errors, mesh study, GCI)
                                                         │
                          investigations.*  ◄────────────┤
                          visualization.*   ◄────────────┘
                          main.py orchestrates everything and writes results/
```

| Module | Notes |
|---|---|
| `config.py` | Frozen dataclass. Variants via `dataclasses.replace(cfg, ...)`; `with_Re_L()` rescales `U_inf` and `H`. Add new parameters here; `main.py` exposes every field as a `--flag` automatically (tuples excluded). |
| `grid.py` | `Grid` holds `x`, `y`, `dx`; `y_stencil_weights` returns non-uniform central-difference weights (nodes 1..Ny-2). |
| `solver.py` | The only place the PDE is discretised. `solve(cfg) -> Result`; arrays are `(Nx, Ny)`, index `[i, j]` = (x, y). |
| `blasius.py` | Must stay independent of `solver.py`: nothing from the CFD may be imported or used here. Result is cached (`lru_cache`). |
| `postprocess.py` | Pure functions of `u`, `v`, `y`. Wall gradient uses a second-order one-sided stencil. |
| `validation.py` | Error metrics, `TARGETS` (the brief's tolerances), mesh study, `gci()`. |
| `investigations.py` | Parametric studies returning DataFrames. |
| `visualization.py` | Plotting only; takes computed data, writes PNGs. Uses the `Agg` backend. |

## The numerical scheme (derivation)

Unknown at station *i*: `u_{i,j}`, `j = 1..Ny-2`. With the x-operator
`D_x u = (c0 u_i + c1 u_{i-1} + c2 u_{i-2})/dx` (bdf1: `(1,-1,0)`; bdf2: `(3/2,-2,1/2)`), central y-operators
`D_y`, `D_yy` and frozen coefficients `ũ, ṽ`:

```
ũ_j c0/dx · u_j + ṽ_j (D_y u)_j − ν (D_yy u)_j = −ũ_j (c1 u_{i-1,j} + c2 u_{i-2,j}) / dx
```

Row *j* is `lower_j u_{j-1} + diag_j u_j + upper_j u_{j+1} = rhs_j`. The wall node (`u=0`) contributes nothing; the
top Dirichlet value `U` is moved to the right-hand side of the last row. `v` then follows from continuity using the *same* `D_x`
(this is why the discrete continuity residual is zero to round-off, see `postprocess.continuity_residual`).

Why frozen coefficients: substituting `u u_x` directly gives a quadratic system in `u_i`, not a tridiagonal one. Lagging is
first-order consistent; Picard sweeps (`n_picard > 1`) re-evaluate `ũ, ṽ` at the current station until the nonlinear discrete equation is satisfied.
For bdf2, the first sweep extrapolates `ũ = 2u_{i-1} − u_{i-2}`.

Stability notes: the diagonal is `ũ c0/dx + ...`, positive since `u ≥ 0`, so the system is well posed. Near the leading edge the cell Péclet
number `|v| dy/ν` can exceed 2 (central convection), which is the likely origin of the small bdf2 start-up wiggle documented in the README.
Replacing central y-convection by a blended/upwind form is an untested candidate fix.

## Conventions

* SI units everywhere. Percent errors are `100·|CFD − ref|/ref`, except profile errors which are percent of `U∞`.
* Station selection uses the nearest node (`postprocess.station_index`); the standard meshes are nested so `x/L = 0.1…1.0` fall exactly on nodes.
* Reference quantities come from the ODE solution (`BlasiusSolution`), never from the rounded textbook constants.
* The validation targets are *reported*, not enforced as test failures that would force tuning: `tests/test_solver.py` asserts only the targets the baseline genuinely meets.
  Do not loosen the metrics or hand-tune parameters to make a target pass; the brief explicitly forbids it.

## Tests

`pytest` runs in under a second. Mapping to the brief's section 13:

| Brief test | Where |
|---|---|
| 1 Blasius `f''(0)` | `tests/test_blasius.py` |
| 2 Wall BC | `test_wall_condition` |
| 3 Free stream | `test_far_field_and_inlet` |
| 4 Continuity | `test_continuity` |
| 5 Positivity | `test_positivity_no_overshoot` |
| 6 Grid convergence | `test_grid_convergence` |

Also covered: stencil exactness, Thomas vs `solve_banded`, √x growth, domain-height check, independence from Blasius.

## Adding things

* **New streamwise scheme:** add its `(c0, c1, c2)` to `solver._COEFFS`, handle start-up in `solve()`, add it to `investigations.SCHEMES` and a test.
* **New parameter:** add to `Config`; it appears as a CLI flag. If it affects results, add a test.
* **New figure:** add a function to `visualization.py` and call it from `main.py`.
* After changing numerics, rerun `make run`, review `results/data/validation_metrics_*.csv`, and update the numbers in `README.md` and `REPORT.md`
  (they are typed in by hand and are *not* regenerated; this is a known maintenance cost).

## Known issues / ideas

* bdf2 near-wall start-up wiggle (<0.6 % of `U∞`, first ~10 stations).
* GCI is unreliable when the observed order is implausible; flagged via `order_plausible`.
* Errors near the leading edge dominate everything; non-uniform x-spacing there is the obvious next improvement.
* The report/README tables are hand-copied from the CSVs; generating them from `summary.json` would remove that step.
* No license file has been chosen yet.
