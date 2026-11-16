"""Expert trajectories -> window labels (structure h, per-phase spline latent), and the way back to a path.

Copied from dino-flow-parking: structure_flow/labels.py, the `squeeze` relabelling of
structure_flow/trim_labels.py and the frame helpers of sync_flow/windows.py. Everything takes the
config as an argument, so the coefficient count K is a parameter, not a module constant.

Latent [max_phases, K, 3]: log length at knot 0, curvature/scale, speed/max.
Paths are rear-axle poses in the frame of the rear axle at the window start.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline
from .config import DEFAULT


@dataclass
class Task:
    name: str
    pose: np.ndarray      # [T,3] world x, y, yaw rad (vehicle centre, CARLA frame)
    speed: np.ndarray     # [T] signed m/s
    steer: np.ndarray     # [T] CARLA command in [-1,1]
    reverse: np.ndarray   # [T] bool
    target: np.ndarray    # [3] world x, y, yaw rad

    def __len__(self):
        return len(self.speed)


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def to_ego(pose, points):
    """Express world [...,3] poses (x, y, yaw rad) in the frame of `pose`."""
    points = np.asarray(points, np.float64)
    dx, dy = points[..., 0]-pose[0], points[..., 1]-pose[1]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((c*dx+s*dy, -s*dx+c*dy, wrap(points[..., 2]-pose[2])), -1)


def to_world(pose, points):
    points = np.asarray(points, np.float64)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((pose[0]+c*points[..., 0]-s*points[..., 1],
                     pose[1]+s*points[..., 0]+c*points[..., 1],
                     wrap(points[..., 2]+pose[2])), -1)


def target_features(target_ego, cfg=DEFAULT):
    x, y, yaw = target_ego
    return np.array([x/cfg.coordinate_scale, y/cfg.coordinate_scale, np.sin(yaw), np.cos(yaw)], np.float32)


@lru_cache(maxsize=8)
def basis(coefficients, samples):
    knots = np.r_[np.zeros(4), np.linspace(0, 1, coefficients-2)[1:-1], np.ones(4)]
    return BSpline(knots, np.eye(coefficients), 3)((np.arange(samples)+.5)/samples)


@lru_cache(maxsize=8)
def fit_matrix(coefficients, samples):
    return np.linalg.pinv(basis(coefficients, samples))


def rear_axle(pose, cfg=DEFAULT):
    pose = np.array(pose, np.float64)
    pose[..., 0] -= cfg.rear_offset_m*np.cos(pose[..., 2])
    pose[..., 1] -= cfg.rear_offset_m*np.sin(pose[..., 2])
    return pose


def split_phases(task, start, end, cfg=DEFAULT):
    """[(gear, first, last)] over frames start..end, cut where the moving gear changes."""
    phases, first, gear = [], start, 0
    for frame in range(start, end+1):
        if abs(task.speed[frame]) < cfg.move_speed_mps:
            continue
        current = -1 if task.reverse[frame] else 1
        if gear and current != gear:
            phases.append((gear, first, frame))
            first = frame
        gear = current
    return phases+[(gear, first, end)] if gear else []


def fit_phase(pose, speed, gear, cfg=DEFAULT):
    """Spline coefficients [K,3] of one phase from its rear-axle poses and speeds."""
    yaw = np.unwrap(pose[:, 2])
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1))]
    total, result = arc[-1], np.zeros((cfg.coefficients, 3))
    result[0, 0] = np.log(max(total, cfg.min_length_m))
    if total < cfg.min_length_m:
        return result
    keep = np.r_[True, np.diff(arc) > 1e-6]
    at = (np.arange(cfg.samples)+.5)/cfg.samples*total
    heading = np.interp(at, arc[keep], yaw[keep])
    curvature = np.gradient(heading, at)*gear/cfg.curvature_scale
    matrix = fit_matrix(cfg.coefficients, cfg.samples)
    result[:, 1] = matrix@curvature
    result[:, 2] = np.clip(matrix@(np.interp(at, arc[keep], speed[keep])/cfg.max_speed_mps), 0, 1)
    return result


def window_label(task, frame, cfg=DEFAULT):
    """Label of the window at `frame`. Only the first two phases are kept."""
    end = min(frame+cfg.window_frames, len(task)-1)
    rear = rear_axle(task.pose, cfg)
    phases = split_phases(task, frame, end, cfg)[:cfg.max_phases]
    if not phases:      # stationary: the current gear with no length
        phases = [(-1 if task.reverse[frame] else 1, frame, frame)]
    h = (0 if phases[0][0] > 0 else cfg.max_phases)+len(phases)-1
    latent = np.zeros((cfg.max_phases, cfg.coefficients, 3))
    for index, (gear, first, last) in enumerate(phases):
        latent[index] = fit_phase(rear[first:last+1], np.abs(task.speed[first:last+1]), gear, cfg)
    return h, latent.astype(np.float32)


def inputs(task, frame, cfg=DEFAULT):
    """The numbers next to the BEV: target in the ego frame [4] and ego state [3]. They do not depend on K."""
    target = target_features(to_ego(task.pose[frame], task.target), cfg)
    ego = np.array([task.speed[frame]/cfg.max_speed_mps, task.steer[frame], float(task.reverse[frame])], np.float32)
    return target, ego


# The `squeeze` relabelling of the released model (stop depth -0.30 m): every expert path ends at one
# depth in the slot. See dino-flow-parking scripts/structure_flow/trim_labels.py for the reasons.

def _inward(task):
    return 1. if task.target[0] > 285.6 else -1.


def trimmed(task, stop_depth, cfg=DEFAULT):
    """The task cut where the car centre first reaches `stop_depth` in its last reverse run."""
    inward = _inward(task)
    depth = (task.pose[:, 0]-task.target[0])*inward
    forward = np.flatnonzero(~task.reverse)
    first = forward[-1]+1 if len(forward) else 0
    reached = np.flatnonzero(depth[first:] >= stop_depth)
    if not len(reached) or first+reached[0] >= len(task)-1:
        return task, len(task)-1
    cut = first+int(reached[0])
    rear = rear_axle(task.pose, cfg)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(rear[:, :2], axis=0), axis=1))]
    left_before, left_now = arc[-1]-arc, arc[cut]-arc[:cut+1]
    speed = task.speed[:cut+1].copy()
    run = slice(first, cut+1)
    tail = slice(first, None)
    speed[run] = np.sign(task.speed[run])*np.interp(left_now[run], left_before[tail][::-1], np.abs(task.speed[tail])[::-1])
    return replace(task, pose=task.pose[:cut+1], speed=speed, steer=task.steer[:cut+1], reverse=task.reverse[:cut+1]), cut


def squeezed(task, frame, stop_depth, keep=.7, least=.05):
    """The task with the path from `frame` ending at `stop_depth`, or None past that depth."""
    inward = _inward(task)
    depth = (task.pose[:, 0]-task.target[0])*inward
    forward = np.flatnonzero(~task.reverse)
    first = forward[-1]+1 if len(forward) else 0
    anchor = max(frame, first)
    shift = depth[-1]-stop_depth
    if anchor >= len(task)-1 or depth[anchor] >= stop_depth or shift <= 0:
        return None
    fix = first+int(np.flatnonzero(depth[first:] >= depth[-1]-keep)[0])
    room = depth[fix]-depth[anchor]
    if room-shift >= least:
        pose = task.pose.copy()
        pose[anchor:fix, 0] = pose[anchor, 0]+(room-shift)/room*(pose[anchor:fix, 0]-pose[anchor, 0])
        pose[fix:, 0] -= shift*inward
        return replace(task, pose=pose)
    ahead = np.flatnonzero(depth[anchor:] >= depth[anchor]+shift)
    if not len(ahead) or anchor+ahead[0] >= len(task)-1:
        return None
    match = anchor+int(ahead[0])
    moved = to_world(task.pose[anchor], to_ego(task.pose[match], task.pose[match:]))
    join = lambda values, tail: np.concatenate((values[:anchor], tail))
    return replace(task, pose=join(task.pose, moved), speed=join(task.speed, task.speed[match:]),
                   steer=join(task.steer, task.steer[match:]), reverse=join(task.reverse, task.reverse[match:]))


def reaching(task, frame, cfg=DEFAULT, near=-3., longest=112):
    """Window settings that reach the expert's stop from near the slot."""
    moving = np.flatnonzero(np.abs(task.speed) > .02)
    left = (moving[-1] if len(moving) else len(task)-1)-frame
    if not task.reverse[frame] or (task.pose[frame, 0]-task.target[0])*_inward(task) < near or left <= cfg.window_frames:
        return cfg
    return replace(cfg, horizon=-(-min(left, longest)//cfg.stride_frames))


def task_labels(task, cfg=DEFAULT, stop_depth=-.3):
    """All windows of one task, every stride_frames frames: (h [N], latent [N,P,K,3], target [N,4], ego [N,3], speed [N])."""
    short, cut = trimmed(task, stop_depth, cfg)
    hs, latents, targets, egos = [], [], [], []
    for frame in range(0, len(task), cfg.stride_frames):
        changed = squeezed(task, frame, stop_depth)
        h, latent = (window_label(changed, frame, reaching(changed, frame, cfg)) if changed
                     else window_label(short, min(frame, cut), cfg))
        target, ego = inputs(task, frame, cfg)
        hs.append(h); latents.append(latent); targets.append(target); egos.append(ego)
    return (np.array(hs), np.stack(latents), np.stack(targets), np.stack(egos),
            task.speed[::cfg.stride_frames].astype(np.float32))


def gears(h, cfg=DEFAULT):
    first = 1 if h < cfg.max_phases else -1
    return [first*(-1)**index for index in range(h % cfg.max_phases+1)]


def decode(latent, h, start_speed=None, cfg=DEFAULT):
    """Rear-axle path [N,5]: x, y, yaw, signed speed, phase. Known speeds are imposed:
    the current speed at the start and zero at a gear change."""
    grid = basis(cfg.coefficients, cfg.samples)
    state, rows, directions = np.zeros(3), [], gears(h, cfg)
    for index, gear in enumerate(directions):
        length = np.exp(latent[index, 0, 0])
        curvature = grid@latent[index, :, 1]*cfg.curvature_scale
        coefficients = latent[index, :, 2].astype(np.float64).copy()
        if index == 0 and start_speed is not None:
            coefficients[0] = abs(start_speed)/cfg.max_speed_mps
        if index > 0:
            coefficients[0] = 0.
        if index < len(directions)-1:
            coefficients[-1] = 0.
        speed = (grid@coefficients).clip(0, 1)*cfg.max_speed_mps
        step = length/cfg.samples
        for bend, value in zip(curvature, speed):
            turn = gear*bend*step
            middle = state[2]+turn/2
            state = state+np.array([gear*step*np.cos(middle), gear*step*np.sin(middle), turn])
            rows.append((*state, gear*value, index))
    return np.array(rows)


def is_stop(latent, h, cfg=DEFAULT):
    return sum(np.exp(latent[index, 0, 0]) for index in range(h % cfg.max_phases+1)) < cfg.stop_length_m
