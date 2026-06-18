# ECAM Drawing — Deep Dive

A detailed walkthrough of the custom-IK ECAM drawing pipeline, grounded in the actual
source: `arm7_draw_ecam/ecam_path.py` and `arm7_draw_ecam/ik_arm_final.py`.

> **Companion files in this folder**
> - `posture_gui_architecture.md` / `.mmd` / `.dot` / `.png` — big-picture system architecture
> - `ecam_drawing_simple.dot` / `.png` — simplified 5-node ECAM chain
> - `ecam_path_state_machine.dot` / `.png` — the 3-phase state machine (this document's Phase model)

---

## The whole run has 3 phases

`ecam_path` is a state machine driven by a 50 Hz timer (`_tick`). It walks through three phases:

```
Phase 1: PREPOSE + ANCHOR   →   Phase 2: DRAW            →   Phase 3: RETURN HOME
"get into writing posture"      "stream the letter path"     "go back to all-zeros"
```

---

## Phase 1 — Prepose & anchor (the setup nobody thinks about)

Before drawing a single line, the arm must get into a known, reachable **writing posture**
(`ecam_path.py:39`):

```python
WRITING_POSTURE = [0, 30°, 0, -60°, 0, 0, 0]   # joint_2=+30°, joint_4=-60°
```

Why this matters: the orientation is held **fixed** while drawing. If you start from a bad
config, parts of "ECAM" fall outside the arm's reach (joint_6 has the tight `[-0.489, 0.262]`
limit) and the IK goes best-effort → ugly letters. The writing posture guarantees the whole word
is reachable.

The tricky bit (`_tick`, lines 514–543) — there's a **ROS 2 discovery race**:

```python
if self._traj_pub.get_subscription_count() < 1:   # controller not listening yet
    ... wait up to 10 s ...                        # else the one-shot command is DROPPED
self._send_prepose()                               # send writing posture as a JointTrajectory
```

If it blindly sent the posture command before the controller subscribed, the message vanishes
and the arm draws at its spawn pose. So it **waits for a subscriber first**.

Then **anchor** (`_anchor_and_build`, line 266): it captures the *current* `/ee_pose` as the
origin `(hx, hy, hz)` and the orientation quaternion `dq`. **Every letter coordinate is relative
to this anchor.** That's the bridge between "where the arm happens to be" and "where the letters
get drawn."

---

## Phase 2 — How the letters become a target stream

### Step A: letter geometry → polylines

Each letter is a list of strokes, each stroke a list of `(dy, dz)` offsets
(`letter_E/C/A/M`, lines 48–80). E.g. the letter E:

```python
[(0,H),(0,0),(W,0)]   # vertical spine + bottom bar
[(0,H),(W,H)]         # top bar
[(0,H/2),(W*0.8,H/2)] # middle bar
```

Drawn on a flat plane: **X = depth (into paper), Y = left/right, Z = up/down.** W=25 mm, H=50 mm.

### Step B: `_assemble` turns strokes into a sampled path (lines 380–475)

This is the heart of it. For each stroke it builds three kinds of motion:

1. **Pen-up travel** between strokes (line 412): lift in X → move above the next start → lower.
   Pen flag = `False`.
2. **Dwell/hold** at each corner (`hold`, line 397): repeats the *same* target `dwell_s × rate`
   times. Critical on **open-loop hardware** — the motors lag, so without holding, the "move to
   next corner" command arrives before the motor reached the current one → unfinished letters.
3. **Draw the stroke** (line 416), pen flag = `True`.

Then `densify` (line 83) resamples every line into points spaced `ds = v_lin / rate` apart — so
spacing controls speed (slower = cleaner).

Two geometric corrections also live here:
- **`mirror_y`** (line 433): reflects the word in Y so it reads left-to-right when viewed from
  behind the arm. Reflects about the word's *own* center (not the anchor) to stay in reachable
  workspace.
- **No-forward-lean clamp** (line 446): forces every target's X ≤ pen-tip plane, so the marker can
  never press *forward* of the EE. Warns if a bad config would.

Output: three parallel arrays — `_samples` (Nx3 positions), `_quats` (orientation, constant),
`_pen` (bool per sample). For the verified Gazebo run that's ~1923 samples.

### Step C: stream it (`_tick` line 566, `_publish` line 569)

Every 50 Hz tick publishes one sample as `/ee_target` (PoseStamped) + the matching `/ecam_pen`
(Bool). It also publishes the full ideal word as latched **blue markers** for RViz
(`_build_word_markers`).

---

## How `ik_arm_final` turns each target into joint angles

This is the thesis contribution. Two timescales:

### The solver (Damped Least Squares, `ik_update` line 39)

On each new target it computes the 6-D error and solves:

```python
e_pos = p_des − p_cur
e_rot = rotm2axang_vec(R_des · R_curᵀ)     # orientation error as axis-angle, base frame
dq    = Jᵀ (J Jᵀ + λ²I)⁻¹ e                # damped least squares
```

Three robustness layers on top — this is what makes it reach ~99.7% of poses instead of ~58%:

1. **Adaptive damping** (line 55): λ shrinks near the goal (`lam_eff`) → big stable steps far
   away, precise steps close in.
2. **Null-space centering** (line 60): `N = I − Jᵀ(JJᵀ+λ²I)⁻¹J` pushes joints toward mid-range
   *without* moving the EE (exploits the 7th DOF).
3. **Two-stage + random restarts** (`ik_attempt` line 83, `ik_solve` line 97): first a
   **position-only warm-up** (huge basin of attraction) lands near the target, *then* full 6-D
   converges orientation. If a seed fails, it restarts from `q_mid` then random seeds within joint
   limits — stops as soon as one converges.

### The ramp (200 Hz tick, line 207)

The solver runs once per target and gives `q_goal`. But the node **doesn't jump there** — it ramps:

```python
step = clip(q_goal − q_cmd, −dq_max, +dq_max)   # dq_max = 0.10 rad/tick
q_cmd += step                                    # → publish /joint_commands
```

This is why motion is smooth, and why **stopping is emergent**: when `ecam_path` stops sending new
targets, the IK keeps solving the same one, `q_goal − q_cmd → 0`, so commands stop changing and the
arm holds.

One hardware subtlety: `closed_loop=False` on real motors (line 214). It warm-starts from its *own
last command* (`q_cmd`), not `/joint_states` — because the bridge/motors sign-flip joints 3 & 5, so
feedback isn't a clean seed.

---

## Phase 3 — Return home (`_build_return` line 239)

After the last sample, it drives back to all-zeros — but **not** as a Cartesian straight line
(that could leave the workspace). Instead it's a **joint-space** move disguised as Cartesian
targets: interpolate joints `q0 → 0`, run **FK on each step**, and stream those poses as
`/ee_target`. So it still goes through the same IK (doesn't fight it) yet follows a safe
joint-space path, landing at ~zero (residual ≲0.02 rad).

---

## The full annotated picture

```
PHASE 1  ecam_path: wait for controller sub → send WRITING_POSTURE (JointTrajectory)
         → settle 5s → capture /ee_pose as anchor (hx,hy,hz, dq)
         → _build(): letters → strokes → densify → _samples/_quats/_pen

PHASE 2  ┌────────────────────────────────────────────────────────────────┐ 50 Hz
         │ /ee_target[i] + /ecam_pen[i]  ──►  ik_arm_final                  │
         │                                    • DLS solve (2-stage+restarts)│ solve once/target
         │                                    • ramp q_cmd by dq_max=0.10   │ 200 Hz
         │                                    ──► /joint_commands           │
         │   joint_commands_bridge ──/joint_states──► motors + RViz         │ 100 Hz
         │   fk_arm_final ──/ee_pose──► (feedback, trail, accuracy bag)     │
         └────────────────────────────────────────────────────────────────┘

PHASE 3  ecam_path: interpolate joints→0, FK each step, stream as /ee_target → home
```

**Three decoupled rates (50 → 200 → 100 Hz)** are exactly why there's transient *tracking lag*
during motion (~2.6 mm, controller chasing a moving target) but ~0 *steady-state* error once a
stroke settles.

---

## Verified result

Full orientation + writing posture, Gazebo: **1923/1923 targets converged**, letterform deviation
**median 0.017 mm / max 2.35 mm**. See `results/ecam_drawing/`.
