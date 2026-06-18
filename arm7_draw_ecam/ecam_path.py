#!/usr/bin/env python3
"""ECAM drawing path generator for the CUSTOM IK pipeline.

Ports the "ECAM" letter geometry from arm_moveit_config/src/text_trajectory.cpp
(which drives MoveIt) into a node that feeds the *custom* solver instead: it
publishes an interpolated /ee_target stream that ik_arm_final.py turns into
/joint_commands. Because the ideal path is now a real topic (/ee_target) and the
achieved pose is published by fk_arm_final.py (/ee_pose), drawing accuracy can be
measured directly (see tools/verify_ecam_drawing.py).

Letters are drawn on a flat X-plane (pen-down at x_down), with Y left/right and
Z up/down, anchored to the end-effector pose captured at startup — exactly like
the C++ demo. Orientation is held constant at the captured home orientation.

Publishes:
  /ee_target  (geometry_msgs/PoseStamped)  — commanded pose stream (drives IK)
  /ecam_pen   (std_msgs/Bool)              — True while drawing a stroke (pen down),
                                             False during pen-up travel moves
Writes:
  <ideal_csv> (default /tmp/ecam_ideal.csv) — ground-truth stroke polyline points
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped, Pose, Point
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String, ColorRGBA
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from arm7_draw_ecam.fk_arm_final import ArmKinematics, rot_to_quat

JOINT_NAMES = [f'joint_{i}' for i in range(1, 8)]
# Writing posture (rad) — same as the MoveIt demo's writing pose. Gives a
# configuration where the whole word is reachable while holding one orientation.
WRITING_POSTURE = [0.0, math.radians(30.0), 0.0, math.radians(-60.0), 0.0, 0.0, 0.0]

# Ground-writing posture (rad) — pen pointing straight DOWN onto a horizontal
# surface at ~base level (writes in FRONT of the base). DERIVED and VERIFIED
# offline by tools/find_ground_posture.py: among 0-word-box-failure candidates
# it has the best Yoshikawa manipulability and a comfortable, centred joint_6.
# Re-run that tool and paste its output here if the letter geometry OR the pen
# length (urdf/arm.xacro pen_length) changes — the tip height below is fixed by
# the joint angles, so a different pen would land the tip at a different z.
#   pen length 0.15 m; table top at z=0. BASELINE pen tip commanded at
#   (x=0.360, y=0.000, z=+0.002) m, pen straight down (-Z, 0 deg tilt) — a 2 mm
#   hover above the table. NOTE: the pen-tip z is TUNABLE LIVE from the ecam_gui
#   slider, which re-solves this posture by IK for the requested height (e.g. drop
#   it ~3 cm to z=-0.028 to press the pen down); this constant is just the
#   starting height the GUI loads. Re-run find_ground_posture.py only if the pen
#   length (urdf/arm.xacro) or letter geometry changes — and REBUILD FIRST, since
#   the tool reads the INSTALLED urdf via $(find arm7_draw_ecam).
#   manipulability w=0.0214 (front posture ~0.017);  joint_6=-0.001 rad (limit
#   [-0.489, +0.262]);  0/1122 word-box IK fails, max 0.006 mm.
#   joints (deg): j1 +0.2  j2 +44.3  j3 +0.5  j4 -64.4  j5 -0.5  j6 -0.1  j7 -71.4
GROUND_POSTURE = [0.004190, 0.772712, 0.009087, -1.123407, -0.009503,
                  -0.001177, -1.245515]

# Writing-plane rotations. The whole letter construction below is built in a
# CANONICAL frame — the world Y-Z plane, pen along -X, lift along +X — exactly
# as the validated vertical demo. build_word_samples then rotates that finished
# construction onto the real drawing plane as a single RIGID rotation about the
# anchor, applied AFTER the Y-mirror and the no-lean clamp so every in-plane
# semantic (lift along the plane normal, "don't press past the paper", mirror
# about the word's own centre) is preserved by the rotation. 'vertical' is the
# identity, so that path is bit-identical (the remap is skipped entirely).
# 'ground' (the 'tangential' layout chosen by find_ground_posture.py — the word
# sweeps SIDEWAYS at ~constant radius, far more reachable than a radial word)
# maps canonical (+X lift, -Y word-advance, +Z letter-up) onto (world +Z up,
# world -Y word-advance, world -X radial letter-up) so the pen points -Z and the
# letters lie flat on the world X-Y plane at the anchor height.
PLANE_ROT = {
    'vertical': np.eye(3),
    'ground': np.array([[0.0, 0.0, -1.0],
                        [0.0, 1.0,  0.0],
                        [1.0, 0.0,  0.0]]),
}

# Whether a plane's NATURAL viewer sees the canonical word mirrored. Viewed from
# the front the vertical word reads correctly (False); viewed from above, the
# ground plane has the opposite handedness so the same construction comes out
# left-right reversed -> reflect it once (True) to read "ECAM" properly. This is
# XOR'd with the user's mirror_y, so 'vertical' is bit-identical and the user can
# still flip either plane for a rear/under view.
PLANE_MIRROR = {'vertical': False, 'ground': True}


def _remap_to_plane(samples, ideal_rows, anchor, R):
    """Rigidly rotate the canonical (vertical-plane) construction about the
    anchor onto the real writing plane: P -> anchor + R @ (P - anchor). Applied
    as the LAST geometry step (after mirror/clamp), so the rotation carries all
    in-plane semantics across unchanged. R == I (vertical) never reaches here."""
    a = np.asarray(anchor, float)
    out = np.array([a + R @ (np.asarray(s, float) - a) for s in samples])
    rows = []
    for (li, si, x, y, z) in ideal_rows:
        p = a + R @ (np.array([x, y, z], float) - a)
        rows.append((li, si, float(p[0]), float(p[1]), float(p[2])))
    return out, rows


# ── ECAM letter geometry (faithful port of text_trajectory.cpp) ──────────────
def make_stroke(y0, z0, x_plane, pts):
    """pts: list of (dy, dz). Returns list of (x, y, z) with y = y0 - dy, z = z0 + dz."""
    return [(x_plane, y0 - dy, z0 + dz) for (dy, dz) in pts]


def letter_E(y0, z0, x_down, W, H):
    return [
        make_stroke(y0, z0, x_down, [(0, H), (0, 0), (W, 0)]),
        make_stroke(y0, z0, x_down, [(0, H), (W, H)]),
        make_stroke(y0, z0, x_down, [(0, H / 2), (W * 0.8, H / 2)]),
    ]


def letter_C(y0, z0, x_down, W, H):
    cy = y0 - W / 2.0
    cz = z0 + H / 2.0
    ry = W / 2.0
    rz = H / 2.0
    N = 14
    s = []
    for i in range(N + 1):
        t = i / N
        th = math.radians(30.0 + t * 300.0)
        s.append((x_down, cy - ry * math.cos(th), cz + rz * math.sin(th)))
    return [s]


def letter_A(y0, z0, x_down, W, H):
    return [
        make_stroke(y0, z0, x_down, [(0, 0), (W / 2.0, H), (W, 0)]),
        make_stroke(y0, z0, x_down, [(W * 0.2, H * 0.4), (W * 0.8, H * 0.4)]),
    ]


def letter_M(y0, z0, x_down, W, H):
    return [
        make_stroke(y0, z0, x_down, [(0, 0), (0, H), (W / 2.0, H * 0.4), (W, H), (W, 0)]),
    ]


def densify(points, ds):
    """Resample a polyline (list of 3-vectors) at ~ds spacing. Returns Nx3 array."""
    pts = [np.asarray(p, float) for p in points]
    if len(pts) == 1:
        return np.array([pts[0]])
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        n = max(1, int(math.ceil(L / ds)))
        for k in range(1, n + 1):
            out.append(a + seg * (k / n))
    return np.array(out)


# ── ECAM-LaSalle logo geometry (shield + ECAM + LaSalle + star) ──────────────
# Stroke-font port of photo_2026-06-11_10-25-33.jpg. Glyphs are built in a local
# (u right, v up) space and mapped to world via (x_down, y0 - u, z0 + v) — the
# same convention as make_stroke, so the whole logo follows the word pipeline
# (pen-up travel between strokes, mirror/clamp, ideal CSV) unchanged.
def _arc_uv(cu, cv, r, a0_deg, a1_deg, n=14):
    """Arc in glyph (u,v) space from a0 to a1 degrees (CCW positive)."""
    return [(cu + r * math.cos(math.radians(a0_deg + (a1_deg - a0_deg) * i / n)),
             cv + r * math.sin(math.radians(a0_deg + (a1_deg - a0_deg) * i / n)))
            for i in range(n + 1)]


def _glyph_L(ch):
    return [[(0.0, ch), (0.0, 0.0), (0.62 * ch, 0.0)]], 0.72 * ch


def _glyph_a(xh):
    bowl = _arc_uv(0.45 * xh, 0.5 * xh, 0.45 * xh, 90.0, 450.0, n=14)
    stem = [(0.9 * xh, xh), (0.9 * xh, 0.0)]
    return [bowl, stem], 1.1 * xh


def _glyph_S(ch):
    upper = _arc_uv(0.30 * ch, 0.72 * ch, 0.28 * ch, 30.0, 270.0, n=12)
    lower = _arc_uv(0.30 * ch, 0.22 * ch, 0.22 * ch, 90.0, -130.0, n=12)
    return [upper + lower[1:]], 0.74 * ch          # one continuous S stroke


def _glyph_l(ch):
    return [[(0.0, ch), (0.0, 0.0)]], 0.22 * ch


def _glyph_e(xh):
    r, cu, cv = 0.45 * xh, 0.45 * xh, 0.5 * xh
    bar = [(cu - r, cv + 0.02 * xh), (cu + r, cv + 0.02 * xh)]
    bowl = _arc_uv(cu, cv, r, 2.0, 305.0, n=14)
    return [bar + bowl[1:]], 1.05 * xh             # bar flows into the bowl


def lasalle_strokes(u0, v0, xh):
    """'LaSalle' as glyph strokes; baseline at v0, x-height xh. Returns
    (strokes_in_uv, total_advance)."""
    ch = 1.5 * xh                                  # cap/ascender height
    seq = [(_glyph_L, ch), (_glyph_a, xh), (_glyph_S, ch), (_glyph_a, xh),
           (_glyph_l, ch), (_glyph_l, ch), (_glyph_e, xh)]
    gap = 0.28 * xh
    strokes, u = [], u0
    for fn, size in seq:
        glyph, adv = fn(size)
        for s in glyph:
            strokes.append([(u + du, v0 + dv) for (du, dv) in s])
        u += adv + gap
    return strokes, u - gap - u0


def star_stroke(cu, cv, R):
    """Closed 5-pointed star outline, top point up."""
    pts = []
    for k in range(5):
        a_out = math.radians(90.0 + k * 72.0)
        a_in = math.radians(90.0 + 36.0 + k * 72.0)
        pts.append((cu + R * math.cos(a_out), cv + R * math.sin(a_out)))
        pts.append((cu + 0.382 * R * math.cos(a_in), cv + 0.382 * R * math.sin(a_in)))
    pts.append(pts[0])
    return pts


def logo_letters(y0, z0, x_down, W, H):
    """The ECAM-LaSalle badge as letter-groups for the word pipeline:
    [shield outline, E, C, A, M, LaSalle, star]. (y0, z0) is the TOP-LEFT of
    the shield (same convention as the word: y extends -Y/right, z up from the
    baseline z0 == bottom). W x H is the shield bounding box."""
    def uv(strokes):
        return [[(x_down, y0 - u, z0 + v) for (u, v) in s] for s in strokes]

    groups, names = [], []
    # shield: straight top, vertical sides, then converging to a bottom point
    shield = [(0.0, H), (W, H), (W, 0.30 * H), (0.5 * W, 0.0),
              (0.0, 0.30 * H), (0.0, H)]
    groups.append(uv([shield])); names.append('shield')
    # ECAM word centred, cap height 0.155H, baseline at 0.60H
    lw = 0.135 * W
    lgap = 0.05 * W
    lh = 0.155 * H
    word_w = 4.0 * lw + 3.0 * lgap
    u0 = (W - word_w) / 2.0
    vb = 0.60 * H
    # the letter_X helpers build world coords from a top-left y0/z0 directly
    for fn, nm, k in ((letter_E, 'E', 0), (letter_C, 'C', 1),
                      (letter_A, 'A', 2), (letter_M, 'M', 3)):
        groups.append(fn(y0 - (u0 + k * (lw + lgap)), z0 + vb, x_down, lw, lh))
        names.append(nm)
    # LaSalle centred under the word, x-height 0.105H, baseline at 0.385H
    xh = 0.105 * H
    _, adv = lasalle_strokes(0.0, 0.0, xh)         # measure first
    strokes, _ = lasalle_strokes((W - adv) / 2.0, 0.385 * H, xh)
    groups.append(uv(strokes)); names.append('LaSalle')
    # star centred near the bottom point
    groups.append(uv([star_stroke(0.5 * W, 0.245 * H, 0.10 * H)]))
    names.append('star')
    return groups, names


# ── shared path builders (module-level: reused by ecam_traj.py, the offline ──
# ── one-shot trajectory demo, so both demos draw the IDENTICAL path) ─────────
def shape_strokes(shape, hx, hy, hz, *, pen_lift, circle_radius=0.04,
                  square_side=0.06):
    """Geometry for the non-ECAM flat shapes, centred on the anchor (hx,hy,hz).

    circle/square live on the flat YZ writing plane (x == hx, the pen plane).
    Returns (letters, names, x_down, x_up): one shape == one continuous stroke.
    """
    x_down = hx
    x_up = x_down - pen_lift               # away-from-paper retract level
    cy, cz = hy, hz                        # centre on the anchor
    if shape == 'circle':
        R = circle_radius
        N = 72
        s = [(x_down, cy + R * math.cos(2.0 * math.pi * i / N),
              cz + R * math.sin(2.0 * math.pi * i / N)) for i in range(N + 1)]
        return [[s]], ['circle'], x_down, x_up
    # square (helix has its own builder — it's a 3D coil, not a flat stroke)
    a = square_side / 2.0
    corners = [(-a, -a), (a, -a), (a, a), (-a, a), (-a, -a)]   # closed loop
    s = [(x_down, cy + dy, cz + dz) for (dy, dz) in corners]
    return [[s]], ['square'], x_down, x_up


def build_helix_samples(home_pos, home_quat, *, radius=0.03, turns=2.0,
                        pitch=0.06, v_lin=0.02, rate=50.0, log=None):
    """Descending 3D helix, matching 7dof_ws dh_urdf_demo.profile_helix:
    a circle in the XY plane about the anchor with Z dropping by `pitch` per
    turn. Centred on (home_pos) — no writing posture is needed. Does NOT go
    through assemble_strokes (no forward-lean clamp / Y-mirror), which are
    ECAM-writing-plane specific and would distort the coil."""
    hx, hy, hz = [float(v) for v in home_pos]
    dq = home_quat
    ds = max(v_lin / rate, 1e-4)
    samples, quats, pen = [], [], []

    def add(seg_pts, is_pen):
        for p in densify(seg_pts, ds):
            samples.append(p); quats.append(dq); pen.append(is_pen)

    start = np.array([hx + radius, hy, hz])
    add([np.asarray(home_pos, float).copy(), start], False)   # travel to start (pen up)
    L = turns * math.hypot(2.0 * math.pi * radius, pitch)     # helix arc length
    N = max(8, int(math.ceil(L / ds)))
    coil = []
    for i in range(N + 1):
        t = i / N
        th = 2.0 * math.pi * turns * t
        coil.append((hx + radius * math.cos(th), hy + radius * math.sin(th),
                     hz - pitch * turns * t))
    add(coil, True)                                           # descend the coil (pen down)

    if log:
        log(f'helix plan ready: r={radius*1000:.0f} mm pitch={pitch*1000:.0f} mm/turn '
            f'{turns:g} turns -> {len(samples)} samples '
            f'(~{len(samples)/rate:.1f}s), centred on current pose')
    return np.array(samples), quats, pen


def assemble_strokes(letters, names, x_down, x_up, skip_first_travel, home_pos,
                     home_quat, *, v_lin=0.02, rate=50.0, dwell_s=0.0,
                     mirror_y=False, log=None, warn=None):
    """Turn a list of letters/shapes (each a list of strokes) into the
    /ee_target sample stream: pen-up travel between strokes, dwell at corners,
    optional Y-mirror, no-forward-lean clamp. Returns (samples Nx3, quats,
    pen, ideal_rows) where ideal_rows = (letter_idx, stroke_idx, x, y, z) of
    the pen-down strokes (ground truth for verify_ecam_drawing.py)."""
    hx = float(home_pos[0])
    ds = max(v_lin / rate, 1e-4)
    dq = home_quat                             # constant orientation while drawing
    samples, quats, pen = [], [], []
    cur = np.asarray(home_pos, float).copy()

    def add(seg_pts, is_pen, quat_seq=None):
        pts = densify(seg_pts, ds)
        qs = quat_seq if quat_seq is not None else [dq] * len(pts)
        for s, q in zip(pts, qs):
            samples.append(s); quats.append(q); pen.append(is_pen)

    n_dwell = int(max(0.0, dwell_s) * rate)

    def hold(pt, is_pen):
        # repeat the same target so the (open-loop) hardware can settle there
        for _ in range(n_dwell):
            samples.append(np.array(pt, float)); quats.append(dq); pen.append(is_pen)

    ideal_rows = []  # (letter_idx, stroke_idx, x, y, z) for the pen-down strokes
    first_stroke = True
    for li, L in enumerate(letters):
        for si, stroke in enumerate(L):
            start = np.array(stroke[0], float)
            # pen-up travel: lift, go above start, lower to start.
            # Skipped only for the very first stroke in start_at_pose mode
            # (we're already AT the start). When centering, the start is NOT
            # the anchor, so the travel is always needed.
            if not (skip_first_travel and first_stroke):
                add([cur, np.array([x_up, cur[1], cur[2]]),
                     np.array([x_up, start[1], start[2]]), start], False)
            first_stroke = False
            hold(start, True)                  # settle pen-down at the start corner
            add(stroke, True)                  # draw the stroke (pen down)
            hold(stroke[-1], True)             # settle at the end corner before lifting
            for (x, y, z) in stroke:
                ideal_rows.append((li, si, x, y, z))
            cur = np.array(stroke[-1], float)
        if log:
            log(f"queued letter '{names[li]}' ({len(L)} strokes)")

    # lift the pen at the end (pen up). Any return-home move is generated by the
    # caller (joint-space, like the prepose) rather than a Cartesian straight line.
    add([cur, np.array([x_up, cur[1], cur[2]])], False)

    # reflect the path in Y so the word reads left-to-right when drawn/viewed
    # from behind (un-mirrors glyphs and reverses direction). Reflect about the
    # word's OWN Y-centre (not the anchor) so the path stays inside the same,
    # already-reachable Y band — reflecting about the anchor would shove the word
    # to the far +Y side and the IK runs out of workspace.
    if mirror_y:
        ys = [float(s[1]) for s in samples]
        cy = 0.5 * (min(ys) + max(ys))
        samples = [np.array([s[0], 2.0 * cy - s[1], s[2]]) for s in samples]
        ideal_rows = [(li, si, x, 2.0 * cy - y, z)
                      for (li, si, x, y, z) in ideal_rows]

    # NO FORWARD LEAN guarantee: the drawing plane is x_down (== hx in center_y
    # mode, i.e. exactly the pen-tip X, no pen_depth press) and the only other X
    # value is x_up (a straight-back retract). Clamp every target X to never go
    # forward of the pen-tip plane (x < hx) regardless of params — so the marker
    # can never lean/press forward of the actual end-effector. Warn if it ever
    # would, so a bad config is visible instead of silently leaning.
    n_clamped = sum(1 for s in samples if s[0] < hx - 1e-9)
    if n_clamped:
        if warn:
            warn(f'{n_clamped} target(s) were forward of the EE plane (x<{hx:.4f}) '
                 f'— clamped to prevent leaning forward; check pen_depth/pen_lift signs')
        samples = [np.array([max(s[0], hx), s[1], s[2]]) for s in samples]
        ideal_rows = [(li, si, max(x, hx), y, z)
                      for (li, si, x, y, z) in ideal_rows]

    return np.array(samples), quats, pen, ideal_rows


def build_word_samples(home_pos, home_quat, *, shape='ecam', letter_w=0.025,
                       letter_h=0.05, gap=0.012, pen_depth=0.02, pen_lift=0.02,
                       v_lin=0.02, rate=50.0, dwell_s=0.0, mirror_y=False,
                       center_y=False, start_at_pose=False, circle_radius=0.04,
                       square_side=0.06, helix_radius=0.03, helix_turns=2.0,
                       helix_pitch=0.06, logo_w=0.10, logo_h=0.12,
                       plane='vertical', plane_rot=None, log=None, warn=None):
    """Build the complete target sample stream for a shape, anchored at
    (home_pos, home_quat). Pure geometry — the single source of truth shared by
    ecam_path (streaming demo) and ecam_traj (offline one-shot trajectory demo).

    `plane` ('vertical' default, or 'ground') selects the writing plane. The
    construction is always done in the canonical vertical frame and then rigidly
    rotated about the anchor by PLANE_ROT[plane] (or `plane_rot` if given, a 3x3
    used by the offline posture search to try candidate layouts). 'vertical' is
    the identity, so its output is bit-identical to before.

    Returns (samples Nx3, quats, pen, ideal_rows, x_down, x_up).
    ideal_rows is [] and x_down/x_up are None for the helix (a 3D coil has no
    flat letterform -> no ideal CSV / word markers)."""
    hx, hy, hz = [float(v) for v in home_pos]
    R = plane_rot if plane_rot is not None else PLANE_ROT.get(plane, np.eye(3))
    # effective mirror = user intent XOR the plane's natural reading flip (so the
    # word reads "ECAM" for that plane's natural viewer; vertical -> unchanged).
    eff_mirror = bool(mirror_y) ^ PLANE_MIRROR.get(plane, False)
    # Helix: 3D coil about the anchor (7dof_ws style), its own builder. It does
    # NOT go through assemble_strokes/the remap (vertical-plane only by design).
    if shape == 'helix':
        if not np.allclose(R, np.eye(3)) and warn:
            warn("shape:=helix ignores plane:=%s (3D coil, vertical only)" % plane)
        samples, quats, pen = build_helix_samples(
            home_pos, home_quat, radius=helix_radius, turns=helix_turns,
            pitch=helix_pitch, v_lin=v_lin, rate=rate, log=log)
        return samples, quats, pen, [], None, None
    # ECAM-LaSalle badge: always centred on y=0 / anchor height (like center_y)
    if shape == 'logo':
        x_down = hx
        x_up = x_down - pen_lift
        letters, names = logo_letters(logo_w / 2.0, hz - logo_h / 2.0,
                                      x_down, logo_w, logo_h)
        skip_first_travel = False
    # Other non-ECAM shapes: flat stroke on the writing plane via the shared path.
    elif shape != 'ecam':
        letters, names, x_down, x_up = shape_strokes(
            shape, hx, hy, hz, pen_lift=pen_lift,
            circle_radius=circle_radius, square_side=square_side)
        skip_first_travel = False
    else:
        W, H, GAP = letter_w, letter_h, gap
        total_w = 4.0 * W + 3.0 * GAP
        if center_y:
            # Word's Y extent is [y_left - total_w, y_left]; centre it on y=0 ->
            # y_left = total_w/2. X at the anchor (no lean); Z centred on anchor.
            x_down = hx
            y_left = total_w / 2.0
            z_base = hz - H / 2.0
            skip_first_travel = False
        elif start_at_pose:
            # First point of 'E' (top-left = (x_down, y_left, z_base + H)) must equal
            # the anchor (hx, hy, hz): no press offset, word extends -Y / -Z from here.
            x_down = hx
            y_left = hy
            z_base = hz - H
            skip_first_travel = True
        else:
            x_down = hx + pen_depth
            y_left = hy + total_w / 2.0
            z_base = hz - H / 2.0
            skip_first_travel = False
        x_up = x_down - pen_lift

        letters = [
            letter_E(y_left - 0 * (W + GAP), z_base, x_down, W, H),
            letter_C(y_left - 1 * (W + GAP), z_base, x_down, W, H),
            letter_A(y_left - 2 * (W + GAP), z_base, x_down, W, H),
            letter_M(y_left - 3 * (W + GAP), z_base, x_down, W, H),
        ]
        names = ['E', 'C', 'A', 'M']

    samples, quats, pen, ideal_rows = assemble_strokes(
        letters, names, x_down, x_up, skip_first_travel, home_pos, home_quat,
        v_lin=v_lin, rate=rate, dwell_s=dwell_s, mirror_y=eff_mirror,
        log=log, warn=warn)
    # Rotate the finished canonical construction onto the real writing plane.
    # Skipped (no float ops) for the vertical plane -> bit-identical output.
    if not np.allclose(R, np.eye(3)):
        samples, ideal_rows = _remap_to_plane(samples, ideal_rows, home_pos, R)
    if log:
        log(f'ECAM plan ready: {len(samples)} target samples '
            f'(~{len(samples)/rate:.1f}s)  x_down={x_down:.3f} x_up={x_up:.3f} '
            f'plane={plane}')
    return samples, quats, pen, ideal_rows, x_down, x_up


def word_marker_array(ideal_rows, frame, *, ns='ecam_word'):
    """One blue LINE_STRIP per pen-down stroke -> the full ideal ECAM word, static."""
    blue = ColorRGBA(r=0.1, g=0.4, b=1.0, a=1.0)
    ma = MarkerArray()
    # group consecutive rows by (letter_idx, stroke_idx)
    mid = 0
    cur_key = None
    m = None
    for (li, si, x, y, z) in ideal_rows:
        key = (li, si)
        if key != cur_key:
            if m is not None and len(m.points) >= 2:
                ma.markers.append(m); mid += 1
            m = Marker()
            m.header.frame_id = frame
            m.ns = ns
            m.id = mid
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.004
            m.color = blue
            m.pose.orientation.w = 1.0
            m.points = []
            cur_key = key
        m.points.append(Point(x=float(x), y=float(y), z=float(z)))
    if m is not None and len(m.points) >= 2:
        ma.markers.append(m)
    return ma


def write_ideal_csv(path, ideal_rows, log=None, warn=None):
    """Write the pen-down ground-truth polyline (verify_ecam_drawing.py input)."""
    try:
        with open(path, 'w') as f:
            f.write('letter_idx,stroke_idx,x,y,z\n')
            for r in ideal_rows:
                f.write(f'{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f}\n')
        if log:
            log(f'ideal letterform written: {path} ({len(ideal_rows)} pts)')
    except OSError as e:
        if warn:
            warn(f'could not write ideal csv: {e}')


class EcamPath(Node):
    def __init__(self):
        super().__init__('ecam_path')
        self.declare_parameter('max_lin_vel', 0.02)    # m/s — slower = cleaner letters
        self.declare_parameter('publish_rate', 50.0)   # Hz
        self.declare_parameter('frame_id', 'base_link')
        self.declare_parameter('letter_w', 0.025)
        self.declare_parameter('letter_h', 0.05)
        self.declare_parameter('gap', 0.012)
        self.declare_parameter('pen_depth', 0.02)      # +X from home = pen down
        self.declare_parameter('pen_lift', 0.02)       # how far to retract for travel
        self.declare_parameter('ideal_csv', '/tmp/ecam_ideal.csv')
        self.declare_parameter('loop', False)          # repeat forever
        self.declare_parameter('prepose', True)        # move to writing posture first
        self.declare_parameter('settle_time', 5.0)     # s to reach + settle the posture
        # max per-joint distance (rad) from the writing posture that still counts
        # as "posture reached" — gate before anchoring (prepose mode only)
        self.declare_parameter('posture_tol', 0.08)
        # the writing posture itself (rad). Default = the classic MoveIt-demo pose
        # (j2=+30°, j4=-60°: pen at x=+0.54 pointing +X). The hardware rig flips it
        # (j2=-30°, j4=+60°: pen at x=-0.54 pointing -X) to write at the FRONT.
        self.declare_parameter('posture_joints', list(WRITING_POSTURE))
        self.declare_parameter('controller_topic', '/arm_controller/joint_trajectory')
        self.declare_parameter('return_home', True)    # drive back to home (all-zeros) pose after
        self.declare_parameter('home_joint_vel', 0.6)  # rad/s for the joint-space return move
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('tip_link', 'ee')
        # Start drawing the first stroke EXACTLY at the current pen-tip pose: the
        # first point of 'E' == anchor (no pen_depth press, no centering offset, no
        # pen-up pre-travel). The word then extends right (-Y) and down (-Z) from
        # there. Use this so the arm does NOT lean/travel after the writing posture.
        self.declare_parameter('start_at_pose', False)
        # Reflect the whole letter path in Y about the anchor. Needed when drawing
        # is viewed/written from BEHIND (the word would otherwise come out mirrored
        # and right-to-left); this makes "ECAM" read properly left-to-right.
        self.declare_parameter('mirror_y', False)
        # Centre the whole word at y=0 (middle of "ECAM" on the base centerline) so
        # it sits in the safe, symmetric workspace. Overrides the Y placement of
        # start_at_pose; X stays at the anchor (no forward lean), Z centres on the
        # anchor height, and the first stroke gets a normal pen-up travel.
        self.declare_parameter('center_y', False)
        # Settle/dwell time (s) held at each stroke's START and END (pen down). On the
        # open-loop hardware the motors lag the command stream; without a dwell the
        # "move to next stroke" command arrives before the motor has reached the stroke
        # corner, so letters look unfinished. Holding the same target for dwell_s lets
        # the motor catch up before the path moves on. 0 = off (sim doesn't need it).
        self.declare_parameter('dwell_s', 0.0)
        # Which shape to draw. 'ecam' (default) draws the word; 'circle' / 'square'
        # draw a 2D outline on the YZ writing plane (constant X = pen plane); 'helix'
        # draws a true 3D coil (circle in YZ while advancing along the pen-lift/away-
        # from-paper axis). circle/square/helix are always centred on the anchor.
        self.declare_parameter('shape', 'ecam')
        self.declare_parameter('circle_radius', 0.04)   # m, for shape:=circle
        self.declare_parameter('square_side', 0.06)     # m, for shape:=square
        # helix defaults match 7dof_ws dh_urdf_demo.profile_helix (descending coil)
        self.declare_parameter('helix_radius', 0.03)    # m, for shape:=helix
        self.declare_parameter('helix_turns', 2.0)      # revolutions, for shape:=helix
        self.declare_parameter('helix_pitch', 0.06)     # m Z-drop per turn, helix
        self.declare_parameter('logo_w', 0.10)          # m, shield width, shape:=logo
        self.declare_parameter('logo_h', 0.12)          # m, shield height, shape:=logo

        self.v_lin = float(self.get_parameter('max_lin_vel').value)
        self.rate = float(self.get_parameter('publish_rate').value)
        self.frame = self.get_parameter('frame_id').value
        self.W = float(self.get_parameter('letter_w').value)
        self.H = float(self.get_parameter('letter_h').value)
        self.GAP = float(self.get_parameter('gap').value)
        self.pen_depth = float(self.get_parameter('pen_depth').value)
        self.pen_lift = float(self.get_parameter('pen_lift').value)
        self.ideal_csv = self.get_parameter('ideal_csv').value
        self.loop = bool(self.get_parameter('loop').value)
        self.prepose = bool(self.get_parameter('prepose').value)
        self.settle_time = float(self.get_parameter('settle_time').value)
        self.posture_tol = float(self.get_parameter('posture_tol').value)
        self.posture = [float(v) for v in self.get_parameter('posture_joints').value]
        if len(self.posture) != 7:
            self.get_logger().warn(
                f'posture_joints has {len(self.posture)} values, expected 7 — '
                'falling back to the default writing posture')
            self.posture = list(WRITING_POSTURE)
        self.controller_topic = self.get_parameter('controller_topic').value
        self.return_home = bool(self.get_parameter('return_home').value)
        self.home_joint_vel = float(self.get_parameter('home_joint_vel').value)
        self.base_link = self.get_parameter('base_link').value
        self.tip_link = self.get_parameter('tip_link').value
        self.start_at_pose = bool(self.get_parameter('start_at_pose').value)
        self.mirror_y = bool(self.get_parameter('mirror_y').value)
        self.center_y = bool(self.get_parameter('center_y').value)
        self.dwell_s = float(self.get_parameter('dwell_s').value)
        self.shape = str(self.get_parameter('shape').value).lower()
        self.circle_radius = float(self.get_parameter('circle_radius').value)
        self.square_side = float(self.get_parameter('square_side').value)
        self.helix_radius = float(self.get_parameter('helix_radius').value)
        self.helix_turns = float(self.get_parameter('helix_turns').value)
        self.helix_pitch = float(self.get_parameter('helix_pitch').value)
        self.logo_w = float(self.get_parameter('logo_w').value)
        self.logo_h = float(self.get_parameter('logo_h').value)
        if self.shape not in ('ecam', 'circle', 'square', 'helix', 'logo'):
            self.get_logger().warn(f"unknown shape '{self.shape}', falling back to 'ecam'")
            self.shape = 'ecam'

        self._home_quat = None
        self._home_pos = None
        self._last_pose = None      # latest /ee_pose
        self._samples = None        # Nx3 positions
        self._quats = None          # N quaternions (xyzw), per sample
        self._pen = None            # N bools
        self._idx = 0
        self._done = False
        self._anchored = False
        self._t_prepose = None      # time the prepose command was sent
        self._t_prepose_resend = None   # last settle-window re-send (DDS race heal)
        self._wait_start = None     # time we began waiting for the controller subscription
        self._kin = None            # ArmKinematics, for the home (all-zeros) EE pose
        self._q_now = None          # latest joint positions, ordered by JOINT_NAMES
        self._return = None         # return-home samples [(pos, quat), ...]
        self._ridx = 0

        self.create_subscription(PoseStamped, '/ee_pose', self._on_pose, 10)
        self.create_subscription(JointState, '/joint_states', self._on_js, 10)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self._on_urdf, latched)
        self._pub = self.create_publisher(PoseStamped, '/ee_target', 10)
        self._pen_pub = self.create_publisher(Bool, '/ecam_pen', 10)
        self._traj_pub = self.create_publisher(JointTrajectory, self.controller_topic, 10)
        # Latched publisher for the FULL ideal ECAM word (blue), so RViz shows the
        # complete word as a static reference the whole time (orange = live EE trail
        # from ee_trail_marker). Same topic as ee_trail_marker (/ee_markers) but a
        # distinct ns so they don't collide -> one MarkerArray display shows both.
        word_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self._word_pub = self.create_publisher(MarkerArray, '/ee_markers', word_qos)
        self._word_markers = None       # cached MarkerArray of the ideal word
        self.create_timer(1.0 / self.rate, self._tick)
        # re-publish the latched word at 1 Hz so late RViz joins always get it
        self.create_timer(1.0, self._publish_word_markers)

        if self.prepose:
            self.get_logger().info('ecam_path: commanding writing posture, then anchoring...')
        else:
            self.get_logger().info('ecam_path: prepose disabled, anchoring at current /ee_pose...')

    def _on_pose(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        self._last_pose = (np.array([p.x, p.y, p.z], float),
                           np.array([q.x, q.y, q.z, q.w], float))

    def _on_urdf(self, msg: String):
        if self._kin is not None:
            return
        try:
            self._kin = ArmKinematics.from_urdf(msg.data, self.base_link, self.tip_link)
        except Exception as e:                            # noqa: BLE001
            self.get_logger().warn(f'could not build kinematics for return-home: {e}')

    def _on_js(self, msg: JointState):
        idx = {n: i for i, n in enumerate(msg.name)}
        if all(n in idx for n in JOINT_NAMES):
            self._q_now = np.array([msg.position[idx[n]] for n in JOINT_NAMES], float)

    def _build_return(self):
        """Return-home as a JOINT-SPACE move (like the prepose), expressed as /ee_target
        poses via FK so it goes through IK without fighting it. Not a Cartesian straight line."""
        if self._q_now is None or self._kin is None:
            return False
        q0 = self._q_now.copy()
        q1 = np.zeros(self._kin.n)                      # home = all joints zero
        dur = max(float(np.max(np.abs(q1 - q0))) / max(self.home_joint_vel, 1e-3), 0.5)
        n = max(2, int(dur * self.rate))
        self._return = []
        for i in range(n + 1):
            s = i / n
            T = self._kin.fk((1.0 - s) * q0 + s * q1)
            self._return.append((T[:3, 3].copy(), rot_to_quat(T[:3, :3])))
        self.get_logger().info(f'returning home (joint-space move, ~{dur:.1f}s)...')
        return True

    def _send_prepose(self, duration=None):
        dur = self.settle_time if duration is None else max(float(duration), 1.0)
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        pt = JointTrajectoryPoint()
        pt.positions = list(self.posture)
        pt.time_from_start.sec = int(dur)
        pt.time_from_start.nanosec = int((dur % 1.0) * 1e9)
        traj.points = [pt]
        self._traj_pub.publish(traj)

    def _anchor_and_build(self):
        if self._last_pose is None:
            return False
        self._home_pos, self._home_quat = self._last_pose
        self._build()
        self._anchored = True
        return True

    def _fail_to_anchor(self):
        """Controller never connected — give up on the posture and draw where we are."""
        self._anchor_and_build()
        return

    def _build(self):
        """Thin wrapper around the shared module-level builders (see
        build_word_samples): pure geometry lives there so ecam_traj.py draws
        the IDENTICAL path; this method keeps the node-side effects (markers,
        ideal CSV)."""
        log = self.get_logger().info
        warn = self.get_logger().warn
        (self._samples, self._quats, self._pen,
         ideal_rows, _x_down, _x_up) = build_word_samples(
            self._home_pos, self._home_quat, shape=self.shape,
            letter_w=self.W, letter_h=self.H, gap=self.GAP,
            pen_depth=self.pen_depth, pen_lift=self.pen_lift,
            v_lin=self.v_lin, rate=self.rate, dwell_s=self.dwell_s,
            mirror_y=self.mirror_y, center_y=self.center_y,
            start_at_pose=self.start_at_pose,
            circle_radius=self.circle_radius, square_side=self.square_side,
            helix_radius=self.helix_radius, helix_turns=self.helix_turns,
            helix_pitch=self.helix_pitch, logo_w=self.logo_w,
            logo_h=self.logo_h, log=log, warn=warn)
        if not ideal_rows:
            return      # helix: 3D coil, no flat letterform -> no CSV/markers

        # build + publish the static blue ideal-word markers (pen-down strokes only)
        self._word_markers = word_marker_array(ideal_rows, self.frame)
        self._publish_word_markers()
        write_ideal_csv(self.ideal_csv, ideal_rows, log=log, warn=warn)

    def _publish_word_markers(self):
        if self._word_markers is None:
            return
        now = self.get_clock().now().to_msg()
        for m in self._word_markers.markers:
            m.header.stamp = now
        self._word_pub.publish(self._word_markers)

    def _tick(self):
        # Phase 1: move to writing posture, settle, then anchor the letters there.
        if not self._anchored:
            now = self.get_clock().now().nanoseconds * 1e-9
            if self.prepose:
                if self._t_prepose is None:
                    if self._last_pose is None:
                        return  # wait for first /ee_pose so feedback is live
                    # Wait until the controller is actually subscribed, otherwise the
                    # one-shot trajectory is silently dropped (ROS 2 discovery race) and
                    # the arm never moves to the writing posture -> it draws at spawn pose.
                    if self._traj_pub.get_subscription_count() < 1:
                        if self._wait_start is None:
                            self._wait_start = now
                        elif now - self._wait_start > 10.0:
                            self.get_logger().warn(
                                f'no subscriber on {self.controller_topic} after 10s — '
                                'is arm_controller running? anchoring at current pose instead.')
                            return self._fail_to_anchor()
                        return  # keep waiting for the controller
                    self._send_prepose()
                    self._t_prepose = now
                    self._t_prepose_resend = now
                    self.get_logger().info(
                        f'writing posture commanded; settling {self.settle_time:.1f}s...')
                    return
                if now - self._t_prepose < self.settle_time + 1.0:
                    # A matched subscription does NOT guarantee delivery: the
                    # reader side may not have registered our writer yet and a
                    # one-shot VOLATILE message is then dropped silently (seen
                    # intermittently as the arm drawing at the spawn pose).
                    # Re-send during the settle with the REMAINING duration so a
                    # lost prepose self-heals; the JTC just re-plans from its
                    # current command to the same posture, landing on schedule.
                    elapsed = now - self._t_prepose
                    if (elapsed < self.settle_time - 1.0
                            and now - self._t_prepose_resend >= 1.0):
                        self._send_prepose(duration=self.settle_time - elapsed)
                        self._t_prepose_resend = now
                    return  # still settling
                # NEVER anchor blindly: verify the posture was actually REACHED.
                # If the trajectory was lost, the controller wasn't active, or
                # the motor node is dead (/joint_states frozen), anchoring here
                # would place the word at the spawn pose -> unreachable targets,
                # BEST-EFFORT IK, and on real motors a flailing arm. Keep
                # re-sending and explain, instead of silently drawing garbage.
                if self._q_now is None or (
                        float(np.max(np.abs(self._q_now - np.array(self.posture))))
                        > self.posture_tol):
                    if now - self._t_prepose_resend >= 2.0:
                        self._send_prepose(duration=2.0)
                        self._t_prepose_resend = now
                        cur = (np.degrees(self._q_now).round(1).tolist()
                               if self._q_now is not None else 'no /joint_states')
                        self.get_logger().error(
                            'writing posture NOT reached after settle — holding off the '
                            f'drawing and re-commanding it. joints now: {cur} deg, want '
                            f'{[round(math.degrees(v),1) for v in self.posture]} deg. '
                            'Check: arm_controller active? motor node (damiao_hw_node / '
                            'relay) alive? /joint_states updating? STALE nodes from a '
                            'previous run still streaming /joint_commands and overriding '
                            'the posture trajectory? (ros2 topic info /joint_commands '
                            '--verbose; pkill -INT -f "ik_arm_final|ik_to_trajectory")')
                    return
            if not self._anchor_and_build():
                return
            return

        if self._done:
            return
        if self._idx >= len(self._samples):
            if self.loop and self._return is None:
                self._idx = 0
                self.get_logger().info('ECAM loop restart')
                return
            # Phase 3: drawing done -> return home as a joint-space move (via /ee_target).
            if self.return_home:
                if self._return is None and not self._build_return():
                    return                                  # waiting for joint_states/kin
                if self._ridx < len(self._return):
                    pos, q = self._return[self._ridx]
                    self._ridx += 1
                    self._publish(pos, q, False)
                    return
            if not self._done:
                self._done = True
                self.get_logger().info('ECAM drawing complete (returned home).')
            return

        self._publish(self._samples[self._idx], self._quats[self._idx], self._pen[self._idx])
        self._idx += 1

    def _publish(self, pos, q, pen):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame
        msg.pose = Pose()
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])
        self._pub.publish(msg)
        self._pen_pub.publish(Bool(data=bool(pen)))


def main():
    rclpy.init()
    node = EcamPath()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
