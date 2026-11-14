"""Run the complete flat-plate boundary-layer pipeline.

    python main.py                     # everything, default (brief) parameters
    python main.py --Nx 201 --Ny 101   # override any physical/numerical parameter
    python main.py --skip-investigations

Results (figures, CSV, summary.json) are written under results/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import fields, replace

import numpy as np
import pandas as pd

import investigations as inv
import postprocess as pp
import validation as val
import visualization as viz
from blasius import F2_WALL_REF, solve_blasius
from config import DEFAULT, RESULTS, Config
from solver import solve

ENHANCED = dict(scheme="bdf2", stretching=1.0)     # best-practice variant


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        if f.name == "profile_stations":
            continue
        t = {"float": float, "int": int, "str": str}[f.type if isinstance(f.type, str) else f.type.__name__]
        p.add_argument(f"--{f.name}", type=t, default=None,
                       help=f"default: {getattr(DEFAULT, f.name)}")
    p.add_argument("--skip-investigations", action="store_true")
    p.add_argument("--outdir", default=str(RESULTS))
    return p.parse_args()


def export_field(res, path):
    g = res.grid
    X, Y = np.meshgrid(g.x, g.y, indexing="ij")
    pd.DataFrame({"x_m": X.ravel(), "y_m": Y.ravel(),
                  "u_mps": res.u.ravel(), "v_mps": res.v.ravel()}).to_csv(path, index=False,
                                                                           float_format="%.8g")


def export_streamwise(res, b, path):
    cur = pp.summary_curves(res)
    cfg = res.cfg
    x = cur["x"]
    df = pd.DataFrame(cur)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["delta99_blasius"] = b.delta99(x, cfg.U_inf, cfg.nu)
        df["delta_star_blasius"] = b.delta_star(x, cfg.U_inf, cfg.nu)
        df["theta_blasius"] = b.theta(x, cfg.U_inf, cfg.nu)
        df["cf_blasius"] = b.cf(x, cfg.U_inf, cfg.nu)
    df.to_csv(path, index=False, float_format="%.8g")


def run_case(cfg: Config, b, label: str, out):
    """Solve one configuration, validate it, and collect diagnostics."""
    res = solve(cfg)
    table = val.validate(res, b)
    checks = val.check_targets(table)
    cont = val.continuity_report(res)
    diag = {
        "label": label,
        "scheme": cfg.scheme, "stretching": cfg.stretching,
        "Nx": cfg.Nx, "Ny": cfg.Ny, "wall_time_s": res.wall_time,
        "u_min_mps": float(res.u.min()),
        "u_overshoot_over_Uinf": float(res.u.max() / cfg.U_inf - 1.0),
        "no_negative_u": bool(res.u.min() >= 0.0),
        "wall_u_max_abs": float(np.abs(res.u[:, 0]).max()),
        "wall_v_max_abs": float(np.abs(res.v[:, 0]).max()),
        "top_u_max_dev": float(np.abs(res.u[:, -1] - cfg.U_inf).max()),
        "domain_height": val.domain_height_check(cfg, b),
        "continuity": cont,
        "targets_met_all_stations": {k: bool(checks[k].all()) for k in val.TARGETS},
    }
    table.to_csv(out / "data" / f"validation_metrics_{label}.csv", index=False,
                 float_format="%.6g")
    return res, table, checks, diag


def main():
    args = parse_args()
    out = Path(args.outdir)
    for sub in ("velocity_profiles", "contours", "validation", "data"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    overrides = {k: v for k, v in vars(args).items()
                 if k in {f.name for f in fields(Config)} and v is not None}
    cfg = replace(DEFAULT, **overrides)

    print(f"Re_L = {cfg.Re_L:.4g}, grid {cfg.Nx} x {cfg.Ny}, scheme {cfg.scheme}")
    b = solve_blasius()
    print(f"[Phase 1] Blasius f''(0) = {b.fpp0:.9f} (reference {F2_WALL_REF:.9f})")

    summary = {"blasius": {"fpp0": b.fpp0, "eta99": b.eta99,
                           "delta_star_coeff": b.delta_star_coeff,
                           "theta_coeff": b.theta_coeff}}

    # ---- baseline + enhanced runs -----------------------------------------
    cases = {"baseline": cfg, "enhanced": replace(cfg, **ENHANCED)}
    results = {}
    for label, c in cases.items():
        res, table, checks, diag = run_case(c, b, label, out)
        results[label] = (res, table, checks)
        summary[label] = diag
        print(f"\n[{label}] {c.scheme}, beta={c.stretching}, {c.Nx}x{c.Ny}, "
              f"{res.wall_time:.2f} s, H/delta99(L) = {diag['domain_height']['H_over_delta99']:.2f}")
        print(table[["x_over_L", "rmse_pct", "emax_pct", "err_delta99_pct", "err_cf_pct",
                     "err_delta_star_pct", "err_theta_pct"]].round(3).to_string(index=False))
        print("targets met at all 5 stations:", diag["targets_met_all_stations"])

    res, table, _ = results["baseline"]
    export_field(res, out / "data" / "velocity_field_baseline.csv")
    export_streamwise(res, b, out / "data" / "streamwise_quantities_baseline.csv")
    export_streamwise(results["enhanced"][0], b, out / "data" / "streamwise_quantities_enhanced.csv")

    # ---- figures for the baseline ------------------------------------------
    viz.velocity_contours(res, b, out)
    viz.velocity_profiles(res, b, out)
    viz.similarity_profiles(res, b, out)
    viz.development_plots(res, b, out)
    viz.v_profile_plot(res, b, out)
    viz.error_vs_x_plot({"baseline (bdf1, uniform)": val.error_curves(res, b, 0.05),
                         "enhanced (bdf2, β=1)": val.error_curves(results["enhanced"][0], b, 0.05)},
                        out)
    viz.velocity_profiles(results["enhanced"][0], b, out / "enhanced")

    # ---- mesh independence -------------------------------------------------
    for label, c in cases.items():
        mtab, profiles, gtab = val.mesh_study(c, b)
        mtab.to_csv(out / "data" / f"mesh_study_{label}.csv", index=False, float_format="%.6g")
        gtab.to_csv(out / "data" / f"mesh_gci_{label}.csv", index=False, float_format="%.6g")
        d_cm, d_mf = val.nested_differences(c)
        summary[f"mesh_{label}"] = {"table": mtab.to_dict("records"), "gci": gtab.to_dict("records"),
                                    "max_diff_medium_minus_coarse": d_cm,
                                    "max_diff_fine_minus_medium": d_mf}
        print(f"nested-grid max|du/U|: medium-coarse {d_cm:.4g}, fine-medium {d_mf:.4g}")
        print(f"\n[mesh study: {label}]")
        print(mtab[["mesh", "Nx", "Ny", "rmse_mid_pct", "err_delta99_TE_pct", "err_cf_TE_pct",
                    "err_delta_star_TE_pct"]].round(4).to_string(index=False))
        print(gtab.round(4).to_string(index=False))
        if label == "baseline":
            viz.mesh_study_plots(mtab, profiles, b, c, out)
        else:
            viz.mesh_study_plots(mtab, profiles, b, c, out / "enhanced")

    # ---- extra checks ------------------------------------------------------
    if not args.skip_investigations:
        t = inv.reynolds_variation(cfg, b)
        t.to_csv(out / "data" / "investigation_reynolds.csv", index=False, float_format="%.6g")
        viz.reynolds_plot(t, out)
        print("\n[Investigation 1: Reynolds number]\n", t.round(4).to_string(index=False))

        t, curves = inv.wall_clustering(cfg, b)
        t.to_csv(out / "data" / "investigation_clustering.csv", index=False, float_format="%.6g")
        viz.clustering_plot(curves, out)
        print("\n[Investigation 2: wall clustering]\n", t.round(4).to_string(index=False))

        t, curves, order = inv.scheme_comparison(cfg, b)
        t.to_csv(out / "data" / "investigation_schemes.csv", index=False, float_format="%.6g")
        order.to_csv(out / "data" / "investigation_scheme_order.csv", index=False, float_format="%.6g")
        viz.error_vs_x_plot(curves, out, name="scheme_comparison.png")
        print("\n[Investigation 3: schemes]\n", t.round(4).to_string(index=False))
        print(order.round(3).to_string(index=False))

        t = inv.performance(cfg)
        t.to_csv(out / "data" / "investigation_performance.csv", index=False, float_format="%.6g")
        viz.performance_plot(t, out)
        print("\n[Investigation 4: performance]\n", t.round(4).to_string(index=False))

    (out / "data" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nAll outputs written to {out}")


if __name__ == "__main__":
    main()
