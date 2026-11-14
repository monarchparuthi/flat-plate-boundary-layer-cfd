---
title: "Laminar Boundary Layer Solver: A Python-Based CFD Implementation with Blasius Validation"
author: "Technical report"
geometry: margin=2.3cm
fontsize: 11pt
colorlinks: true
header-includes:
  - \usepackage{float}
  - \usepackage{graphicx}
  - \floatplacement{figure}{H}
---

# 1. Abstract

A two-dimensional incompressible laminar boundary-layer solver was written from scratch in NumPy/SciPy and
validated against the Blasius similarity solution, which is computed independently with an ODE solver and
a shooting method. The solver marches the boundary-layer equations along a flat plate with a semi-implicit
finite-difference scheme (a tridiagonal system per station, with $v$ recovered from continuity). With the
brief's baseline parameters ($U_\infty=10$ m/s, $L=1$ m, $Re_L=6.77\times10^5$, $401\times201$ uniform grid,
first-order streamwise differencing) the normalised velocity-profile RMSE is 0.11--0.18 % of $U_\infty$
and the maximum error 0.44--1.35 %. Skin friction is within 1.2 % for $x/L\ge0.1$.
Thickness metrics converge from above towards the analytical values: at the trailing edge
$\delta_{99}$ is 1.7 % high and $\delta^*$ and $\theta$ are about 1 % high. Errors are concentrated near the leading edge,
where the start-up is singular; the baseline misses the brief's 3 % target for $\delta_{99}$ at $x/L\le0.5$ and for
$\delta^*$ at $x/L=0.1$. A second-order, wall-clustered variant on the same mesh meets every target at $x/L\ge0.25$.
Mesh-independence, Reynolds-number, wall-clustering, scheme-accuracy and performance studies are reported.

# 2. Introduction

A flat plate aligned with a uniform stream develops a thin viscous layer in which the velocity rises from zero at the wall
to the free-stream value. Prandtl's boundary-layer approximation ($\partial^2u/\partial x^2\ll\partial^2u/\partial y^2$,
$\partial p/\partial y\approx0$) reduces the Navier--Stokes equations to a parabolic system. For zero pressure gradient it
has an exact similarity solution (Blasius), making it the standard first validation case for a CFD code.
The objective here is a *working solver* whose discretisation is derived and implemented by hand, then
quantitatively compared to that benchmark. **The solver is a boundary-layer marching solver, not a full
pressure--velocity Navier--Stokes solver.**

# 3. Mathematical formulation

Assumptions: 2-D, steady, laminar, incompressible Newtonian flow with constant properties, zero pressure gradient,
no body forces, smooth stationary plate, and validity of the boundary-layer approximation. The governing equations are

$$\frac{\partial u}{\partial x}+\frac{\partial v}{\partial y}=0,\qquad
u\frac{\partial u}{\partial x}+v\frac{\partial u}{\partial y}=\nu\frac{\partial^2u}{\partial y^2}.$$

Physical parameters: $U_\infty=10$ m/s, $\rho=1.225$ kg/m$^3$, $\mu=1.81\times10^{-5}$ Pa s
($\nu=1.4776\times10^{-5}$ m$^2$/s), $L=1$ m, $H=0.05$ m. $Re_L=6.77\times10^5$. The brief calls this below the usual transition threshold, but
$5\times10^5$ is the commonly quoted value, so the trailing edge is marginal and real transition depends on disturbances; the model simply imposes laminar flow.
The domain height is $H/\delta_{99}(L)=8.4$, so the upper boundary does not interact with the layer.

# 4. Blasius analytical solution

With $\eta=y\sqrt{U_\infty/(\nu x)}$ and $\psi=\sqrt{\nu U_\infty x}\,f(\eta)$ the system reduces to
$f'''+\tfrac12ff''=0$, $f(0)=f'(0)=0$, $f'(\infty)=1$, with $u=U_\infty f'$ and
$v=\tfrac12\sqrt{\nu U_\infty/x}\,(\eta f'-f)$. `blasius.py` integrates it with `scipy.integrate.solve_ivp`
(DOP853, rtol $10^{-12}$) and finds the missing wall value by Brent root-finding on $f'(\eta_{max})-1$:

| Quantity | Computed | Classical |
|---|---|---|
| $f''(0)$ | 0.332057336 | 0.332057336 |
| $\eta_{99}$ ($\delta_{99}/\sqrt{\nu x/U}$) | 4.90999 | 4.91 |
| $\delta^*/\sqrt{\nu x/U}$ | 1.72079 | 1.7208 |
| $\theta/\sqrt{\nu x/U}$ (quadrature) | 0.66411 | 0.664 |

The coefficients are obtained from the ODE solution (the momentum thickness by numerical integration, not by
assuming $2f''(0)$), and $C_{f,x}=2f''(0)/\sqrt{Re_x}$. This module shares no code with the CFD solver and is used only for validation.

# 5. Numerical methodology

**Grid.** Uniform in $x$ ($\Delta x=2.5$ mm for 401 points). In $y$ the grid is uniform ($\Delta y=0.25$ mm) or clustered
at the wall with $y=H[1-\tanh(\beta(1-s))/\tanh\beta]$. Derivatives on non-uniform grids use second-order three-point
stencils (verified exact for quadratics).

**Marching scheme.** At station $i$, with $\partial u/\partial x\approx(c_0u_i+c_1u_{i-1}+c_2u_{i-2})/\Delta x$
(first order: $(1,-1,0)$; second order: $(3/2,-2,1/2)$), the momentum equation with *frozen* convection velocities
$(\tilde u,\tilde v)$ becomes linear and tridiagonal in $u_{i,j}$:

$$\tilde u_j\frac{c_0}{\Delta x}u_j+\tilde v_j\left(\frac{\partial u}{\partial y}\right)_j-\nu\left(\frac{\partial^2u}{\partial y^2}\right)_j
=-\tilde u_j\frac{c_1u_{i-1,j}+c_2u_{i-2,j}}{\Delta x}.$$

The frozen values are lagged ($u_{i-1},v_{i-1}$) for first order and linearly extrapolated for second order. Optional Picard sweeps
re-evaluate them at the current station, so the converged result solves the full non-linear discrete equation; this addresses the brief's
requirement that the non-linear convective terms be treated consistently rather than substituted blindly into a tridiagonal system.
The system is solved with `scipy.linalg.solve_banded` or a hand-written Thomas algorithm (identical to $10^{-12}$). The wall-normal velocity follows
from continuity, integrated from the wall with the trapezoid rule and the same $x$-operator as the momentum equation:
$v_{i,j}=v_{i,j-1}-\tfrac12\Delta y_j(\dot u_{i,j}+\dot u_{i,j-1})$, where $\dot u=\partial u/\partial x$.
(The brief suggests a rectangle rule; the trapezoid rule is second order in $y$.)

# 6. Boundary conditions

Inlet: $u=U_\infty$, $v=0$, with the wall node of the inlet column set to $u=0$. Wall: $u=v=0$. Top: $u=U_\infty$ (Dirichlet), $v$ free and
set by continuity, which produces the outward displacement. Outlet: zero streamwise gradient is automatic in a marching method -- no condition is imposed and
the last station is simply the last step -- so nothing can reflect upstream. Tests confirm the wall and top conditions are satisfied exactly.

# 7. Results

\begin{figure}
\centering
\includegraphics[width=0.95\textwidth]{results/contours/velocity_magnitude.png}
\caption{Velocity magnitude (baseline). Dashed white: CFD $\delta_{99}$; dotted red: Blasius.}
\end{figure}

The contours show uniform free-stream flow upstream, zero velocity at the wall, a layer thickening downstream, and a smooth transition to $U_\infty$.
The wall-normal velocity (`results/contours/wall_normal_velocity.png`) is positive and grows with the layer, i.e. the displacement of the outer flow.

\begin{figure}
\centering
\includegraphics[width=\textwidth]{results/velocity_profiles/profiles_u_vs_y.png}
\caption{$u/U_\infty$ vs $y$ at $x/L=0.1,0.25,0.5,0.75,1.0$: CFD (markers) vs Blasius (line).}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.5\textwidth]{results/validation/similarity_collapse.png}
\caption{All five CFD profiles collapse onto the Blasius curve when plotted against $\eta$.}
\end{figure}

# 8. Validation

Errors are normalised as in the brief. Profile errors use all $N_y$ nodes of the column and are in percent of $U_\infty$;
the other columns are percent of the Blasius value. Targets are those of the brief.

**Baseline** (spec parameters, first-order, uniform):

| $x/L$ | RMSE | $E_{max}$ | $E_{\delta_{99}}$ | $E_{C_f}$ | $E_{\delta^*}$ | $E_\theta$ |
|---|---|---|---|---|---|---|
| 0.10 | 0.18 | 1.35 | **9.85** | 0.03 | **3.36** | 2.99 |
| 0.25 | 0.18 | 0.96 | **5.32** | 1.19 | 2.36 | 2.22 |
| 0.50 | 0.15 | 0.67 | **3.07** | 1.02 | 1.56 | 1.50 |
| 0.75 | 0.13 | 0.52 | 2.08 | 0.84 | 1.19 | 1.15 |
| 1.00 | 0.11 | 0.44 | 1.72 | 0.72 | 0.98 | 0.94 |
| *target* | $<2$ | $<5$ | $<3$ | $<5$ | $<3$ | $<3$ |

**Enhanced** (second-order, wall clustering $\beta=1$, same $401\times201$ mesh):

| $x/L$ | RMSE | $E_{max}$ | $E_{\delta_{99}}$ | $E_{C_f}$ | $E_{\delta^*}$ | $E_\theta$ |
|---|---|---|---|---|---|---|
| 0.10 | 0.10 | 0.51 | **3.72** | 0.50 | 1.34 | 1.17 |
| 0.25 | 0.07 | 0.30 | 1.69 | 0.45 | 0.75 | 0.67 |
| 0.50 | 0.05 | 0.19 | 0.90 | 0.31 | 0.46 | 0.41 |
| 0.75 | 0.04 | 0.14 | 0.67 | 0.24 | 0.34 | 0.31 |
| 1.00 | 0.04 | 0.11 | 0.52 | 0.20 | 0.27 | 0.25 |

Bold entries miss the target. The baseline meets RMSE, $E_{max}$, $C_f$ and $\theta$ everywhere, but not $\delta_{99}$ at $x/L\le0.5$ nor $\delta^*$ at $x/L=0.1$.
The enhanced scheme misses only $\delta_{99}$ at $x/L=0.1$. The brief says to investigate rather than adjust results; the diagnosis is as follows.

* $\delta_{99}$ is intrinsically ill-conditioned: near $\eta_{99}$ the profile is almost flat ($f''\approx0.02$), so a velocity error of $10^{-3}U_\infty$ shifts $\delta_{99}$ by roughly 1 %. A profile RMSE of 0.15 % therefore translates to a 3 % thickness error.
* The error is a leading-edge effect that decays with $x$. The start-up at the singular leading edge is first-order accurate, and relative error scales roughly like $\Delta x/x$ (the stretched-grid solution converges at about first order in $\Delta x$).
* Refinement removes it: with $1601\times801$ points the second-order scheme gives $E_{\delta_{99}}=0.97$ % at $x/L=0.1$.

\begin{figure}
\centering
\includegraphics[width=\textwidth]{results/validation/error_vs_x.png}
\caption{Signed errors vs $x$ for the baseline and the enhanced scheme (leading-edge region excluded).}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.95\textwidth]{results/validation/boundary_layer_thickness.png}
\caption{$\delta_{99}(x)$ and $\delta_{99}/\sqrt x$: the CFD thickness follows the expected $\sqrt x$ law (fitted exponent within 0.05 of 0.5 for $x/L\ge0.25$, tested).}
\end{figure}

\begin{figure}
\centering
\includegraphics[width=0.48\textwidth]{results/validation/skin_friction.png}
\includegraphics[width=0.48\textwidth]{results/validation/displacement_momentum_thickness.png}
\caption{Skin friction against $0.664/\sqrt{Re_x}$ (left); displacement and momentum thickness (right).}
\end{figure}

**Numerical-correctness checks.** (i) Wall: $u=v=0$ exactly. (ii) Top: $u=U_\infty$; the flow beyond $1.5\delta_{99}$ is within $2\times10^{-3}$ of $U_\infty$.
(iii) Continuity: the residual of the scheme's own discretisation is $\le1.1\times10^{-13}$ s$^{-1}$ (round-off, by construction); an *independent*
central-difference residual has RMS 0.98 s$^{-1}$ (baseline; scale $U_\infty/L=10$ s$^{-1}$) and a maximum of 122 s$^{-1}$, confined to the leading-edge nodes. (iv) Positivity: $u_{min}=0$, overshoot $\sim10^{-15}U_\infty$.
The first-order solution is strictly monotone in $x$ at fixed $y$; the second-order solution shows a start-up wiggle of at most 0.056 m/s ($0.56\%\,U_\infty$)
at two near-wall nodes of station 6, with no negative velocities. (v) Domain height $H/\delta_{99}(L)=8.4$.

# 9. Mesh independence

Three nested meshes ($101\times51$, $201\times101$, $401\times201$; refinement ratio 2) at the trailing edge and, for profiles, at $x/L=0.5$.

| Mesh | RMSE($x/L{=}0.5$) [%] | $E_{\delta_{99}}(L)$ [%] | $E_{C_f}(L)$ [%] | $E_{\delta^*}(L)$ [%] |
|---|---|---|---|---|
| coarse | 0.461 | 10.54 | 2.87 | 2.19 |
| medium | 0.184 | 4.29 | 0.69 | 1.71 |
| fine | 0.145 | 1.72 | 0.72 | 0.98 |

The profile RMSE and $\delta_{99}$ decrease monotonically. The solution-to-solution maximum difference of $u/U_\infty$ at shared nodes drops from 0.081 (medium vs coarse) to
0.036 (fine vs medium). Richardson/GCI analysis (Roache, safety factor 1.25) of the baseline gives an observed order $p=2.9$ for the RMSE
(GCI 5.2 %) and $p=1.3$ for $\delta_{99}(L)$ (GCI 2.2 %, extrapolated 5.96 mm against the Blasius 5.97 mm). $C_f(L)$ is monotone but gives $p=6.9$, outside a plausible range, so its GCI is not meaningful (the grids are
not in the asymptotic range, and the near-zero change between the medium and fine grid is partly error cancellation between $x$ and $y$ truncation). $\delta^*(L)$ is not monotone in the baseline for the same reason. For the enhanced
scheme $\delta_{99}(L)$ and $\delta^*(L)$ converge monotonically with $p=2.1$ (GCI 0.31 %) and $p=1.4$ (GCI 0.27 %); $C_f(L)$ is non-monotone at the $<0.2$ % level.
The trailing-edge $C_f$ is already within 0.2--0.7 % even on the medium grid.

\begin{figure}
\centering
\includegraphics[width=0.46\textwidth]{results/validation/mesh_profiles.png}
\includegraphics[width=0.53\textwidth]{results/validation/mesh_convergence.png}
\caption{Mesh study: profile at $x/L=0.5$ (left) and trailing-edge quantities vs grid points (right).}
\end{figure}

# 10. Additional investigations

**Reynolds number.** $Re_L=10^4,10^5,10^6$ via $U_\infty$ at fixed $\nu,L$, with $H=10\,\delta_{99}(L)$ so the layer fills the same fraction of the domain
(the fixed $H=0.05$ m would be thinner than the layer at $Re_L=10^4$, where $\delta_{99}(L)=49$ mm). The CFD thickness follows $\delta_{99}/L\propto Re_L^{-1/2}$; the errors are identical at all three
($E_{\delta_{99}}(L)=2.05$ %, $E_{C_f}=0.69$ %, $E_{\delta^*}=1.04$ %) since the problem is self-similar and the grid is scaled with the layer. The $Re_L=10^6$ run lies above the transition range and is a purely numerical test.

**Wall clustering** ($\beta=0,1,2,3$, $N_y=51$ and $201$). On the coarse $N_y=51$ grid clustering matters a great deal: the mean $C_f$ error over $x/L\ge0.25$ falls from 12.3 % (uniform) to 0.84 % ($\beta=1$), and the trailing-edge wall velocity gradient error from 2.9 % to 0.46 %.
On $N_y=201$ the uniform grid is already adequate (wall-gradient error 0.7 %), and stronger clustering (larger $\beta$) *worsens* the result slightly (1.2 % at $\beta=3$) because the outer part of the layer
is then under-resolved and, at this $\Delta x$, the error is dominated by the streamwise truncation. Moderate clustering ($\beta\approx1$) is the best compromise.

**Numerical accuracy.** All variants use central differences for wall-normal diffusion (the brief's third item); the streamwise scheme is what changes.
Mean errors for $x/L\ge0.25$ (baseline mesh): 

| Scheme | mean $E_{\delta_{99}}$ | mean $E_{\theta}$ | mean $E_{C_f}$ | time [ms] |
|---|---|---|---|---|
| first-order | 2.76 | 1.38 | 0.94 | 16 |
| second-order | 1.62 | 0.45 | 0.25 | 21 |
| first-order + 3 Picard | 2.60 | 1.22 | 0.76 | 48 |
| second-order + 3 Picard | 1.65 | 0.49 | 0.29 | 48 |

Refining $N_x$ at $N_y=801$, the observed order in $x$ (error in $\theta(L)$) is 0.93--0.95 for the first-order scheme and rises from 1.5 to 2.0 for the second-order scheme.
Re-evaluating the convection coefficients (Picard) changes the first-order result by only about 10 % and the second-order result by nothing useful: the lagging error is small against truncation error.
The second-order scheme costs about 30 % more time per run and gives roughly half the error; its drawbacks are the start-up wiggle noted above and the need to extrapolate coefficients.

**Performance.** (Median of 3 runs; peak memory from `tracemalloc`.) 

| Mesh | points | Thomas [ms] | `solve_banded` [ms] | peak memory [MB] |
|---|---|---|---|---|
| coarse | 5,151 | 5.9 | 3.3 | 0.10 |
| medium | 20,301 | 19.9 | 6.7 | 0.35 |
| fine | 80,601 | 73.0 | 15.7 | 1.33 |

Cost scales linearly with the number of grid points (one $O(N_y)$ solve per station). The hand-written Thomas loop is about 4.7 times slower than the LAPACK banded solver on the fine mesh because of Python-loop overhead.
The marching has no iteration (one pass, or $k$ passes with Picard), and no convergence criterion is needed.

# 11. Discussion

*Accuracy.* The solver reproduces the Blasius solution to within a fraction of a percent in velocity and about 1 % in the integral thicknesses over most of the plate.
The remaining error is first order in $\Delta x$ with a leading-edge origin, which is typical of marching schemes started from a singular
inlet; the second-order variant converges at its formal order once $\Delta x$ is small enough.
*Metric sensitivity.* $\delta_{99}$ is a poor validation metric on its own: it amplifies profile errors by roughly 10--20 and was the only metric to miss the brief's targets on the baseline mesh.
$C_f$, $\delta^*$ and $\theta$ are integral or wall quantities and are much more stable.
*Cancelling errors.* On the baseline grid the $x$ and $y$ truncation errors have opposite signs for some quantities, so refining only one of them can increase the error (e.g. $C_f$ with
$N_y=801$, $N_x=401$ is worse at $x/L=0.1$ than the baseline); refine both together, as in the mesh study.
*Limitations.* The boundary-layer approximation cannot represent the leading-edge region, where $\partial p/\partial x\neq0$ and $\partial^2u/\partial x^2$ is not negligible; the Blasius solution is also only valid downstream of it.
Both the model and the benchmark share this limitation, so the comparison measures numerical error, not modelling error.
The solver cannot handle separation (marching breaks down), pressure gradients without extension, or turbulence.

# 12. Conclusion

A hand-derived finite-difference marching solver for the laminar flat-plate boundary layer was implemented, tested with 19 automated checks, and
validated quantitatively against an independently solved Blasius solution. On the brief's baseline it meets the RMSE, maximum-error, skin-friction and momentum-thickness targets everywhere,
and meets $\delta_{99}$ and $\delta^*$ targets away from the leading edge; with second-order differencing and mild wall clustering all targets are met for $x/L\ge0.25$. Grid
refinement drives the errors towards zero. Possible extensions: streamwise grid clustering at the leading edge (the dominant remaining error source), imposed pressure gradients (Falkner--Skan), a conservative
momentum-integral form, heat transfer (the brief's references), and an implicit coupled treatment of $u$ and $v$.


## Reproducibility note: mesh convergence and model scope

The coarse/medium/fine sequence is useful evidence that the solution improves under refinement, but it does **not** establish asymptotic mesh independence for every reported quantity. The GCI implementation marks sequences with non-monotone differences or implausible observed order. In the supplied default results, the baseline trailing-edge skin-friction order is implausible and the baseline displacement-thickness sequence is non-monotone. For the enhanced case, the mid-station profile-RMSE GCI is unusually large. These cases should be reported as inconclusive GCI diagnostics, not as validated uncertainty estimates. A stronger study should add a finer grid and verify asymptotic behaviour separately for each quantity of interest.

The default plate Reynolds number is approximately $6.77\\times10^5$, above the often-quoted order-of-magnitude transition value of $Re_x\\sim5\\times10^5$. Transition depends on free-stream disturbance, surface condition and other environmental factors. This code imposes a laminar boundary-layer model; it does not predict transition or turbulence. Therefore the high-Reynolds-number parametric cases are numerical studies of the laminar equations, not claims that the physical flow remains laminar.

The enhanced case reduces the leading-edge errors substantially, but its $\\delta_{99}$ error at $x/L=0.1$ remains 3.72%, above the 3% target. This miss is retained transparently. Do not describe the enhanced case as meeting every target at every station.

# Reproducing

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py && .venv/bin/python -m pytest -q
```
