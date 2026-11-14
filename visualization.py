"""All figures.  Pure plotting: every function receives already-computed data."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import postprocess as pp
from blasius import BlasiusSolution
from solver import Result

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
CFD, REF = "#1f77b4", "#d62728"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def velocity_contours(res: Result, b: BlasiusSolution, outdir):
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    d99 = pp.summary_curves(res)["delta99"]
    ymax_zoom = 2.0 * float(b.delta99(cfg.L, U, nu))
    X, Y = np.meshgrid(g.x, g.y, indexing="ij")
    speed = np.hypot(res.u, res.v)
    paths = []

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), constrained_layout=True)
    for ax, ylim, title in ((axes[0], cfg.H, "full domain"),
                            (axes[1], ymax_zoom, "zoom on the boundary layer")):
        cs = ax.contourf(X, Y * 1e3, speed / U, levels=np.linspace(0, 1, 21),
                         cmap="viridis")
        ax.plot(g.x[1:], d99[1:] * 1e3, "w--", lw=1.4, label=r"CFD $\delta_{99}$")
        ax.plot(g.x[1:], b.delta99(g.x[1:], U, nu) * 1e3, "r:", lw=1.4,
                label=r"Blasius $\delta_{99}$")
        ax.set_ylim(0, ylim * 1e3)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [mm]")
        ax.set_title(f"Velocity magnitude |V|/U$_\\infty$ - {title}")
        ax.grid(False)
    axes[0].legend(loc="upper left", facecolor="0.2", labelcolor="w", framealpha=0.7)
    fig.colorbar(cs, ax=axes, label=r"|V| / U$_\infty$")
    paths.append(_save(fig, Path(outdir) / "contours" / "velocity_magnitude.png"))

    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    cs = ax.contourf(X, Y * 1e3, res.u / U, levels=np.linspace(0, 1, 21), cmap="RdYlBu_r")
    ax.contour(X, Y * 1e3, res.u / U, levels=[0.5, 0.9, 0.99], colors="k", linewidths=0.7)
    ax.set_ylim(0, ymax_zoom * 1e3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [mm]"); ax.grid(False)
    ax.set_title(r"Streamwise velocity u/U$_\infty$ (black lines: 0.5, 0.9, 0.99)")
    fig.colorbar(cs, ax=ax, label=r"u / U$_\infty$")
    paths.append(_save(fig, Path(outdir) / "contours" / "streamwise_velocity.png"))

    fig, ax = plt.subplots(figsize=(10, 3.4), constrained_layout=True)
    cs = ax.contourf(X, Y * 1e3, res.v * 1e3, levels=21, cmap="magma")
    ax.plot(g.x[1:], d99[1:] * 1e3, "w--", lw=1.2, label=r"$\delta_{99}$")
    ax.set_ylim(0, ymax_zoom * 1e3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [mm]"); ax.grid(False)
    ax.legend(loc="upper left", facecolor="0.2", labelcolor="w")
    ax.set_title("Wall-normal velocity v (from continuity): outward displacement")
    fig.colorbar(cs, ax=ax, label="v [mm/s]")
    paths.append(_save(fig, Path(outdir) / "contours" / "wall_normal_velocity.png"))
    return paths


def velocity_profiles(res: Result, b: BlasiusSolution, outdir):
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    st = cfg.profile_stations
    fig, axes = plt.subplots(1, len(st), figsize=(3.0 * len(st), 3.8), sharey=False,
                             constrained_layout=True)
    for ax, s in zip(axes, st):
        i = pp.station_index(g.x, s * cfg.L)
        x = g.x[i]
        ymax = 2.5 * float(b.delta99(x, U, nu))
        yy = np.linspace(0, ymax, 400)
        ax.plot(b.u_over_U(x, yy, U, nu), yy * 1e3, color=REF, lw=1.8, label="Blasius")
        m = g.y <= ymax
        ax.plot(res.u[i][m] / U, g.y[m] * 1e3, "o", ms=3, color=CFD, label="CFD")
        ax.set_title(f"x/L = {s:g}")
        ax.set_xlabel(r"u / U$_\infty$")
    axes[0].set_ylabel("y [mm]")
    axes[0].legend(loc="lower right")
    return _save(fig, Path(outdir) / "velocity_profiles" / "profiles_u_vs_y.png")


def similarity_profiles(res: Result, b: BlasiusSolution, outdir):
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    fig, ax = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    eta = np.linspace(0, 10, 400)
    ax.plot(b.fprime(eta), eta, color=REF, lw=2.2, label="Blasius f'(η)")
    markers = "osv^D"
    for s, mk in zip(cfg.profile_stations, markers):
        i = pp.station_index(g.x, s * cfg.L)
        e = b.eta(g.x[i], g.y, U, nu)
        m = e <= 10
        ax.plot(res.u[i][m] / U, e[m], mk, ms=3.2, mfc="none", label=f"CFD x/L={s:g}")
    ax.set_xlabel(r"u / U$_\infty$")
    ax.set_ylabel(r"η = y $\sqrt{U_\infty/(\nu x)}$")
    ax.set_title("Similarity collapse")
    ax.legend(fontsize=8)
    return _save(fig, Path(outdir) / "validation" / "similarity_collapse.png")


def development_plots(res: Result, b: BlasiusSolution, outdir):
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    c = pp.summary_curves(res)
    x = g.x[1:]
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x, b.delta99(x, U, nu) * 1e3, color=REF, lw=2, label="Blasius 4.91√(νx/U)")
    axes[0].plot(x, c["delta99"][1:] * 1e3, color=CFD, ls="--", lw=1.6, label="CFD")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel(r"$\delta_{99}$ [mm]"); axes[0].legend()
    axes[0].set_title(r"Boundary-layer thickness")
    axes[1].plot(x, (c["delta99"][1:] / np.sqrt(x)) * 1e3, color=CFD, label="CFD")
    axes[1].axhline(float(b.eta99) * np.sqrt(nu / U) * 1e3, color=REF, label="Blasius (constant)")
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel(r"$\delta_{99}/\sqrt{x}$ [mm/√m]")
    axes[1].set_title(r"$\delta_{99}\propto\sqrt{x}$ check"); axes[1].legend()
    paths.append(_save(fig, Path(outdir) / "validation" / "boundary_layer_thickness.png"))

    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    Rex = c["Re_x"][1:]
    ax.loglog(Rex, b.cf(x, U, nu), color=REF, lw=2, label=r"0.664/$\sqrt{Re_x}$")
    ax.loglog(Rex[1:], c["cf"][2:], color=CFD, ls="--", lw=1.6, label="CFD")
    ax.set_xlabel(r"Re$_x$"); ax.set_ylabel(r"C$_{f,x}$"); ax.legend()
    ax.set_title("Local skin-friction coefficient")
    ax.set_xlim(Rex[3], Rex[-1]); ax.set_ylim(top=b.cf(x[3], U, nu))
    paths.append(_save(fig, Path(outdir) / "validation" / "skin_friction.png"))

    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    ax.plot(x, b.delta_star(x, U, nu) * 1e3, color=REF, lw=2, label=r"Blasius $\delta^*$")
    ax.plot(x, c["delta_star"][1:] * 1e3, color=CFD, ls="--", label=r"CFD $\delta^*$")
    ax.plot(x, b.theta(x, U, nu) * 1e3, color="#9467bd", lw=2, label=r"Blasius $\theta$")
    ax.plot(x, c["theta"][1:] * 1e3, color="#2ca02c", ls="--", label=r"CFD $\theta$")
    ax.set_xlabel("x [m]"); ax.set_ylabel("thickness [mm]"); ax.legend()
    ax.set_title("Displacement and momentum thickness")
    paths.append(_save(fig, Path(outdir) / "validation" / "displacement_momentum_thickness.png"))
    return paths


def v_profile_plot(res: Result, b: BlasiusSolution, outdir):
    cfg, g = res.cfg, res.grid
    U, nu = cfg.U_inf, cfg.nu
    fig, ax = plt.subplots(figsize=(5.4, 4.4), constrained_layout=True)
    eta = np.linspace(0, 10, 300)
    ax.plot(0.5 * (eta * b.fprime(eta) - b.f(eta)), eta, color=REF, lw=2, label="Blasius")
    for s, mk in zip(res.cfg.profile_stations, "osv^D"):
        i = pp.station_index(g.x, s * cfg.L)
        e = b.eta(g.x[i], g.y, U, nu)
        m = e <= 10
        vn = res.v[i] / U * np.sqrt(U * g.x[i] / nu)
        ax.plot(vn[m], e[m], mk, ms=3.2, mfc="none", label=f"CFD x/L={s:g}")
    ax.set_xlabel(r"$v\sqrt{Re_x}/U_\infty$"); ax.set_ylabel("η")
    ax.set_title("Wall-normal velocity (similarity scaling)"); ax.legend(fontsize=8)
    return _save(fig, Path(outdir) / "validation" / "wall_normal_velocity_similarity.png")


def error_vs_x_plot(curves: dict, outdir, name="error_vs_x.png"):
    """curves: label -> dict from validation.error_curves."""
    keys = [("delta99", r"$\delta_{99}$"), ("delta_star", r"$\delta^*$"),
            ("theta", r"$\theta$"), ("cf", r"$C_f$")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), constrained_layout=True, sharex=True)
    for ax, (k, lab) in zip(axes, keys):
        for label, c in curves.items():
            ax.plot(c["x"], c[k], label=label)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylim(-12, 12)
        ax.set_title(f"{lab}: signed error vs Blasius [%]")
        ax.set_xlabel("x [m]")
    axes[0].legend(fontsize=8)
    return _save(fig, Path(outdir) / "validation" / name)


def mesh_study_plots(table, profiles, b, cfg, outdir):
    U, nu = cfg.U_inf, cfg.nu
    paths = []
    fig, ax = plt.subplots(figsize=(5.6, 4.4), constrained_layout=True)
    x = 0.5 * cfg.L
    yy = np.linspace(0, 2.5 * float(b.delta99(x, U, nu)), 400)
    ax.plot(b.u_over_U(x, yy, U, nu), yy * 1e3, "k-", lw=2.0, label="Blasius")
    for (name, (y, u)), mk in zip(profiles.items(), "o^s"):
        m = y <= yy[-1]
        ax.plot(u[m] / U, y[m] * 1e3, mk, ms=3.5, mfc="none", label=name)
    ax.set_xlabel(r"u/U$_\infty$"); ax.set_ylabel("y [mm]")
    ax.set_title("Mesh study: velocity profile at x/L = 0.5"); ax.legend()
    paths.append(_save(fig, Path(outdir) / "validation" / "mesh_profiles.png"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    n = table["Nx"] * table["Ny"]
    for ax, (col, lab, ref) in zip(axes, [
            ("delta99_TE_m", r"$\delta_{99}$(L) [mm]", float(b.delta99(cfg.L, U, nu))),
            ("cf_TE", r"$C_f$(L)", float(b.cf(cfg.L, U, nu))),
            ("delta_star_TE_m", r"$\delta^*$(L) [mm]", float(b.delta_star(cfg.L, U, nu)))]):
        s = 1e3 if col.endswith("_m") else 1.0
        ax.semilogx(n, table[col] * s, "o-", color=CFD, label="CFD")
        ax.axhline(ref * s, color=REF, ls="--", label="Blasius")
        ax.set_xlabel("grid points Nx·Ny"); ax.set_ylabel(lab)
    axes[0].legend()
    fig.suptitle("Mesh independence at the trailing edge")
    paths.append(_save(fig, Path(outdir) / "validation" / "mesh_convergence.png"))
    return paths


def reynolds_plot(table, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    ax = axes[0]
    ax.loglog(table["Re_L"], table["delta99_over_L_blasius"], "-", color=REF, lw=2, label="Blasius")
    ax.loglog(table["Re_L"], table["delta99_over_L_cfd"], "o", color=CFD, label="CFD")
    ax.set_xlabel(r"Re$_L$"); ax.set_ylabel(r"$\delta_{99}(L)/L$"); ax.legend()
    p = np.polyfit(np.log(table["Re_L"]), np.log(table["delta99_over_L_cfd"]), 1)[0]
    ax.set_title(f"Trailing-edge thickness (fitted slope {p:.3f}, theory -0.5)")
    ax = axes[1]
    ax.semilogx(table["Re_L"], table["err_delta99_pct"], "o-", label=r"$\delta_{99}$")
    ax.semilogx(table["Re_L"], table["err_cf_pct"], "s-", label=r"$C_f$")
    ax.semilogx(table["Re_L"], table["err_delta_star_pct"], "^-", label=r"$\delta^*$")
    ax.set_xlabel(r"Re$_L$"); ax.set_ylabel("error at x = L [%]"); ax.legend()
    ax.set_title("Error vs Blasius (domain height = 10 δ99(L))")
    return _save(fig, Path(outdir) / "validation" / "reynolds_variation.png")


def clustering_plot(curves_by_label, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, c in curves_by_label.items():
        axes[0].plot(c["x"], c["cf"], label=label)
        axes[1].plot(c["x"], c["delta99"], label=label)
    for ax, t in zip(axes, [r"$C_f$ error [%]", r"$\delta_{99}$ error [%]"]):
        ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("x [m]"); ax.set_title(t)
        ax.set_ylim(-15, 15)
    axes[0].legend(fontsize=8)
    return _save(fig, Path(outdir) / "validation" / "wall_clustering.png")


def performance_plot(table, outdir):
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for solver, sub in table.groupby("linear_solver"):
        ax.loglog(sub["grid_points"], sub["time_s"], "o-", label=solver)
    ax.set_xlabel("grid points"); ax.set_ylabel("marching time [s]")
    ax.set_title("Cost: hand-written Thomas vs scipy solve_banded"); ax.legend()
    return _save(fig, Path(outdir) / "validation" / "performance.png")
