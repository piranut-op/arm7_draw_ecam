#!/usr/bin/env python3
"""Measure how accurately the arm drew "ECAM" with the custom IK pipeline.

Reads a rosbag recorded by ecam_draw_ik.launch.py (topics /ee_target, /ee_pose,
/ecam_pen) plus the ideal letterform CSV written by ecam_path.py, and reports two
errors over the pen-DOWN (drawing) portion only:

  1. Letterform deviation — distance from each actual end-effector point to the
     ideal ECAM polyline. This is "how close is the drawing to the intended
     letters" (the thesis metric).
  2. Tracking error — actual /ee_pose vs commanded /ee_target at the same instant
     (how well the IK + control loop followed the commanded stream).

Outputs a stats table, an optional per-point CSV, and clean figures:
  ecam_overlay.png   — ideal letters vs actual trace in the drawing (Y-Z) plane
  ecam_deviation_hist.png, ecam_deviation_cdf.png

Run (ROS 2 sourced):
  python3 tools/verify_ecam_drawing.py --bag /tmp/ecam_run
"""

import argparse, math, os, sys
import numpy as np


# ── rosbag2 reading ──────────────────────────────────────────────────────────
def read_bag(bag_dir, storage_id):
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from rclpy.serialization import deserialize_message
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Bool

    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_dir, storage_id=storage_id),
                ConverterOptions('', ''))
    typemap = {'/ee_target': PoseStamped, '/ee_pose': PoseStamped, '/ecam_pen': Bool}
    out = {'/ee_target': [], '/ee_pose': [], '/ecam_pen': []}
    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in typemap:
            continue
        msg = deserialize_message(data, typemap[topic])
        if topic == '/ecam_pen':
            out[topic].append((t, bool(msg.data)))
        else:
            p = msg.pose.position
            out[topic].append((t, np.array([p.x, p.y, p.z], float)))
    return out


def detect_storage(bag_dir):
    """Read metadata.yaml to find the storage plugin (sqlite3 / mcap)."""
    meta = os.path.join(bag_dir, 'metadata.yaml')
    if os.path.exists(meta):
        txt = open(meta).read()
        for sid in ('mcap', 'sqlite3'):
            if sid in txt:
                return sid
    return 'sqlite3'


# ── geometry ─────────────────────────────────────────────────────────────────
def load_ideal(csv_path):
    """Return list of polylines (each Nx3), one per ideal stroke."""
    strokes = {}
    with open(csv_path) as f:
        next(f)  # header
        for line in f:
            li, si, x, y, z = line.strip().split(',')
            key = (int(li), int(si))
            strokes.setdefault(key, []).append([float(x), float(y), float(z)])
    return [np.array(v) for v in strokes.values()]


def point_to_segment(p, a, b):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-18:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def dist_to_polylines(p, polylines):
    best = math.inf
    for poly in polylines:
        for a, b in zip(poly[:-1], poly[1:]):
            d = point_to_segment(p, a, b)
            if d < best:
                best = d
        if len(poly) == 1:
            best = min(best, float(np.linalg.norm(p - poly[0])))
    return best


def pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float('nan')


def nearest_before(times, vals, t):
    """vals[i] valid from times[i]; return val active at time t (step-hold)."""
    import bisect
    i = bisect.bisect_right(times, t) - 1
    return vals[max(0, i)]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description='ECAM drawing accuracy from a rosbag.')
    ap.add_argument('--bag', required=True, help='rosbag2 directory')
    ap.add_argument('--ideal', default='/tmp/ecam_ideal.csv', help='ideal letterform CSV')
    ap.add_argument('--storage', default=None, help='sqlite3|mcap (auto-detected)')
    ap.add_argument('--plot', default='/tmp', help='directory for PNG plots')
    ap.add_argument('--plane', default='yz', choices=['yz', 'xy'],
                    help='overlay projection plane: yz (vertical writing, '
                         'default) | xy (ground writing, pen down)')
    ap.add_argument('--no-plot', action='store_true')
    ap.add_argument('--csv', default=None, help='write per-point CSV')
    args = ap.parse_args()

    if not os.path.isdir(args.bag):
        sys.exit(f'bag dir not found: {args.bag}')
    if not os.path.exists(args.ideal):
        sys.exit(f'ideal CSV not found: {args.ideal} (was ecam_path run?)')

    storage = args.storage or detect_storage(args.bag)
    print(f'Reading bag {args.bag} (storage={storage})...')
    data = read_bag(args.bag, storage)
    polylines = load_ideal(args.ideal)

    pose = data['/ee_pose']
    target = data['/ee_target']
    pen = data['/ecam_pen']
    if not pose or not pen:
        sys.exit('bag missing /ee_pose or /ecam_pen samples')

    pen_t = [t for t, _ in pen]
    pen_v = [v for _, v in pen]
    tgt_t = [t for t, _ in target]
    tgt_p = [p for _, p in target]

    dev, trk, draw_pts, draw_t = [], [], [], []
    for t, p in pose:
        if not nearest_before(pen_t, pen_v, t):
            continue  # pen up — skip travel moves
        dev.append(dist_to_polylines(p, polylines))
        draw_pts.append(p)
        draw_t.append(t)
        if tgt_t:
            tp = nearest_before(tgt_t, tgt_p, t)
            trk.append(float(np.linalg.norm(p - tp)))

    if not dev:
        sys.exit('no pen-down samples found (drawing may not have run)')
    dev = np.array(dev)
    trk = np.array(trk) if trk else np.array([])
    draw_pts = np.array(draw_pts)

    def block(name, a, scale=1e3, unit='mm'):
        a = a * scale
        print(f'  {name:<22} mean={np.mean(a):8.3f}  rms={np.sqrt(np.mean(a**2)):8.3f}  '
              f'median={np.median(a):8.3f}  p95={pct(a,95):8.3f}  max={np.max(a):8.3f}  [{unit}]')

    print('\n' + '=' * 78)
    print(f'ECAM DRAWING ACCURACY   (pen-down samples: {len(dev)})')
    print('=' * 78)
    print('  -- letterform deviation (actual trace vs ideal ECAM geometry) --')
    block('deviation', dev)
    if trk.size:
        print('  -- tracking error (actual /ee_pose vs commanded /ee_target) --')
        block('tracking', trk)
    print('=' * 78)

    if args.csv:
        import csv as _csv
        with open(args.csv, 'w', newline='') as f:
            w = _csv.writer(f)
            w.writerow(['t_ns', 'x', 'y', 'z', 'deviation_m'])
            for i in range(len(dev)):
                p = draw_pts[i]
                w.writerow([draw_t[i], p[0], p[1], p[2], dev[i]])
        print(f'  per-point CSV: {args.csv}')

    if not args.no_plot:
        make_plots(args.plot, polylines, draw_pts, dev, plane=args.plane)


def make_plots(out_dir, polylines, draw_pts, dev, *, plane='yz'):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:                            # noqa: BLE001
        print(f'  [plots skipped: matplotlib unavailable ({e})]')
        return
    os.makedirs(out_dir, exist_ok=True)

    def _save(fig, name):
        p = os.path.join(out_dir, name)
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        print(f'  plot written: {p}')

    dev_mm = dev * 1e3

    # 1) overlay: ideal letters vs actual trace, projected onto the drawing plane
    # so the word reads "ECAM" left→right, letters upright, matching the natural
    # RViz view (the third, ~constant axis is dropped):
    #   yz (vertical, pen -X): horizontal = -Y (left→right), vertical = +Z (up)
    #   xy (ground,  pen -Z): horizontal = +Y (left→right), vertical = -X (radial,
    #                          toward the base is up) — see find_ground_posture.py
    if plane == 'xy':
        h_idx, h_sgn, v_idx, v_sgn, v_label = 1, +1.0, 0, -1.0, 'x [m] (radial)'
    else:
        h_idx, h_sgn, v_idx, v_sgn, v_label = 1, -1.0, 2, +1.0, 'z [m]'
    poly_h = lambda a: h_sgn * a[:, h_idx]
    poly_v = lambda a: v_sgn * a[:, v_idx]
    fig, ax = plt.subplots(figsize=(11, 5))
    for poly in polylines:
        ax.plot(poly_h(poly), poly_v(poly), color='#1f9fd0', lw=3,
                solid_capstyle='round', label='_nolegend_')
    sc = ax.scatter(poly_h(draw_pts), poly_v(draw_pts), c=dev_mm, cmap='inferno',
                    s=8, zorder=3)
    fig.colorbar(sc, ax=ax, label='deviation [mm]')
    ax.plot([], [], color='#1f9fd0', lw=3, label='ideal ECAM')
    ax.scatter([], [], c='k', s=8, label='actual trace')
    ax.set(xlabel='y [m] (left→right)', ylabel=v_label,
           title=f'ECAM ({plane} plane): ideal letterform vs actual drawn trace')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=0.3); ax.legend(loc='upper right')
    _save(fig, 'ecam_overlay.png')

    # 2) deviation histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dev_mm, bins=50, color='#3b75af', alpha=0.85)
    ax.axvline(np.median(dev_mm), color='k', ls='--', lw=1, label=f'median={np.median(dev_mm):.2f} mm')
    ax.axvline(pct(dev_mm, 95), color='#c44', ls='--', lw=1, label=f'p95={pct(dev_mm,95):.2f} mm')
    ax.set(xlabel='letterform deviation [mm]', ylabel='count', title='ECAM deviation')
    ax.grid(alpha=0.3); ax.legend()
    _save(fig, 'ecam_deviation_hist.png')

    # 3) deviation CDF
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.sort(dev_mm); ys = np.arange(1, len(xs) + 1) / len(xs) * 100.0
    ax.plot(xs, ys, color='#3b75af', lw=2)
    ax.set(xlabel='letterform deviation [mm]', ylabel='cumulative %',
           title='ECAM deviation CDF', ylim=(0, 100))
    ax.grid(alpha=0.3)
    _save(fig, 'ecam_deviation_cdf.png')


if __name__ == '__main__':
    main()
