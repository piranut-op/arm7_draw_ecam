#!/usr/bin/env python3
"""Thesis figures for the Draw-ECAM result (SIMULATION) — greyscale, axes in mm.

Reads the saved authoritative run in results/ecam_drawing/:
  - ecam_wp_points.csv : actual EE trace + per-point letterform deviation
                         (columns: t_ns, x, y, z, deviation_m)  [pen-down samples]
  - ecam_ideal.csv     : ideal letterform polylines
                         (columns: letter_idx, stroke_idx, x, y, z)  [metres]

Produces (greyscale, axes in mm, numbered + captioned):
  fig1_ecam_overlay.png    ideal letters vs actual trace, Y-Z plane
  fig2_ecam_dev_hist.png   deviation histogram
  fig3_ecam_dev_cdf.png    deviation CDF
and prints the deviation accuracy table (median / mean / RMS / p95 / max, in mm).

Usage:
  python3 tools/plot_ecam_thesis.py
  python3 tools/plot_ecam_thesis.py --in results/ecam_drawing \
          --out results/thesis_figures/ecam_drawing
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_TAG = "SIMULATION — Gazebo, full orientation + writing posture (run: results/ecam_drawing)"


def load_trace(path):
    x, y, z, dev = [], [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            x.append(float(r["x"])); y.append(float(r["y"])); z.append(float(r["z"]))
            dev.append(float(r["deviation_m"]))
    return (np.array(x), np.array(y), np.array(z), np.array(dev))


def load_ideal(path):
    """Return list of polylines [(y_array, z_array), ...] grouped by (letter, stroke)."""
    strokes = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (int(r["letter_idx"]), int(r["stroke_idx"]))
            strokes.setdefault(key, []).append((float(r["y"]), float(r["z"])))
    return [ (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
             for _, pts in sorted(strokes.items()) ]


def caption(fig, text):
    fig.subplots_adjust(bottom=0.22)
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=8, wrap=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="results/ecam_drawing")
    ap.add_argument("--out", dest="outdir", default="results/thesis_figures/ecam_drawing")
    args = ap.parse_args()

    trace_csv = os.path.join(args.indir, "ecam_wp_points.csv")
    ideal_csv = os.path.join(args.indir, "ecam_ideal.csv")
    os.makedirs(args.outdir, exist_ok=True)

    x, y, z, dev = load_trace(trace_csv)
    ideal = load_ideal(ideal_csv)
    dev_mm = dev * 1e3
    n = len(dev_mm)

    # ----- stats table -----
    stats = dict(
        median=np.median(dev_mm), mean=np.mean(dev_mm),
        rms=np.sqrt(np.mean(dev_mm ** 2)),
        p95=np.percentile(dev_mm, 95), max=np.max(dev_mm))
    print("=" * 70)
    print(f"ECAM DRAWING ACCURACY  [{RUN_TAG}]")
    print("=" * 70)
    print(f"  pen-down samples : {n}")
    print(f"  letterform deviation [mm]:")
    print(f"    median={stats['median']:.4f}  mean={stats['mean']:.4f}  "
          f"rms={stats['rms']:.4f}  p95={stats['p95']:.4f}  max={stats['max']:.4f}")
    print("=" * 70)

    # plot in mm; X axis = -y (left->right), Y axis = z (height)
    Y = -y * 1e3
    Z = z * 1e3

    # ----- Fig 1: overlay (greyscale: ideal = thick light-grey, actual = black dots) -----
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, (iy, iz) in enumerate(ideal):
        ax.plot(-iy * 1e3, iz * 1e3, color="0.72", lw=6, solid_capstyle="round",
                zorder=1, label="ideal letterform" if k == 0 else None)
    ax.scatter(Y, Z, c="black", s=3, zorder=2, edgecolors="none",
               label="actual drawn trace")
    ax.set_xlabel("y [mm] (left → right)")
    ax.set_ylabel("z [mm]")
    ax.set_title("ECAM: ideal letterform vs actual drawn trace")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, color="0.9", lw=0.5)
    ax.legend(loc="upper right", framealpha=0.9)
    caption(fig, f"Figure 1. Drawn 'ECAM' trace (black) overlaid on the ideal letterform "
                 f"(grey), Y-Z drawing plane. n={n} pen-down samples. {RUN_TAG}.")
    f1 = os.path.join(args.outdir, "fig1_ecam_overlay.png")
    fig.savefig(f1, dpi=200); plt.close(fig)

    # ----- Fig 2: histogram -----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dev_mm, bins=50, color="0.55", edgecolor="black", lw=0.4)
    ax.axvline(stats["median"], color="black", ls="--", lw=1.2,
               label=f"median = {stats['median']:.3f} mm")
    ax.axvline(stats["p95"], color="0.25", ls=":", lw=1.4,
               label=f"p95 = {stats['p95']:.3f} mm")
    ax.set_xlabel("letterform deviation [mm]")
    ax.set_ylabel("count")
    ax.set_title("ECAM letterform deviation")
    ax.legend(); ax.grid(True, color="0.9", lw=0.5)
    caption(fig, f"Figure 2. Distribution of letterform deviation (pen-down, n={n}). "
                 f"{RUN_TAG}.")
    f2 = os.path.join(args.outdir, "fig2_ecam_dev_hist.png")
    fig.savefig(f2, dpi=200); plt.close(fig)

    # ----- Fig 3: CDF -----
    fig, ax = plt.subplots(figsize=(7, 4))
    s = np.sort(dev_mm)
    cdf = 100.0 * np.arange(1, n + 1) / n
    ax.plot(s, cdf, color="black", lw=2)
    ax.axvline(stats["median"], color="0.4", ls="--", lw=1,
               label=f"median = {stats['median']:.3f} mm")
    ax.axvline(stats["p95"], color="0.4", ls=":", lw=1.2,
               label=f"p95 = {stats['p95']:.3f} mm")
    ax.set_xlabel("letterform deviation [mm]")
    ax.set_ylabel("cumulative %")
    ax.set_ylim(0, 100)
    ax.set_title("ECAM letterform deviation CDF")
    ax.legend(loc="lower right"); ax.grid(True, color="0.9", lw=0.5)
    caption(fig, f"Figure 3. Cumulative distribution of letterform deviation (pen-down, "
                 f"n={n}). {RUN_TAG}.")
    f3 = os.path.join(args.outdir, "fig3_ecam_dev_cdf.png")
    fig.savefig(f3, dpi=200); plt.close(fig)

    for p in (f1, f2, f3):
        print("  wrote", p)


if __name__ == "__main__":
    main()
