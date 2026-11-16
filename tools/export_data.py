"""Maintainer tool: the release data of model-factory from the local cache that made pack_sq30b.

Usage: python tools/export_data.py PACK OUT --roots E2E_DATASET PARKINGLOT_1000

Writes to OUT:
  trajectories.npz   every task's expert frames (pose, speed, steer, reverse, target), in pack order,
                     with the train/validation split of the pack
  bev_0.npy ...      the pack's Fused BEV [N,267,16,16] float16, cut in parts under GitHub's 2 GB asset limit
  SHA256SUMS

Then checks that scripts/make_labels.py's code rebuilds the pack's labels.npz with K=12.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory.config import DEFAULT as cfg
from factory.labels import Task, task_labels

PART_WINDOWS = 13000      # 13000 x 136,704 bytes = 1.78 GB


def read_task(path):
    """sync_flow.windows.load_task: measurements/*.json or the sync_flow_arrays.npz cache it writes."""
    cache = path/'sync_flow_arrays.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            return z['pose'], z['speed'], z['steer'], z['reverse'], z['target']
    rows = [json.loads(name.read_text(encoding='utf-8')) for name in sorted((path/'measurements').glob('*.json'))]
    goal = json.loads((path/'parking_goal'/'0001.json').read_text(encoding='utf-8'))
    reverse = np.array([bool(r['Reverse']) for r in rows])
    speed = np.array([r['speed'] for r in rows], np.float64)/3.6*np.where(reverse, -1., 1.)
    pose = np.array([[r['x'], r['y'], np.deg2rad(r['yaw'])] for r in rows], np.float64)
    steer = np.array([r['Steer'] for r in rows], np.float64)
    return pose, speed, steer, reverse, np.array([goal['x'], goal['y'], np.deg2rad(goal['yaw'])], np.float64)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--roots', nargs='+', type=Path, required=True)
    parser.add_argument('--skip-bev', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Same order as structure_flow/train.py's pack(): the cached BEV rows follow it.
    paths = sorted(p.parent.parent for root in args.roots for p in root.glob('*/task*/fused_bev/summary.json'))
    held = {str(p) for p in json.loads((args.pack/'selection.json').read_text())['val']}
    with ThreadPoolExecutor(8) as pool:
        arrays = list(pool.map(read_task, paths))
    lengths = np.array([len(a[1]) for a in arrays])
    names = np.array([f'{p.parents[1].name}/{p.parent.name}/{p.name}' for p in paths])
    np.savez_compressed(args.out/'trajectories.npz', name=names, length=lengths,
                        pose=np.concatenate([a[0] for a in arrays]), speed=np.concatenate([a[1] for a in arrays]),
                        steer=np.concatenate([a[2] for a in arrays]), reverse=np.concatenate([a[3] for a in arrays]),
                        target=np.stack([a[4] for a in arrays]), val=np.array([str(p) in held for p in paths]))
    print(f'tasks {len(paths)}, frames {lengths.sum()}, validation tasks {sum(str(p) in held for p in paths)}', flush=True)

    # Rebuild the labels with K=12 and compare with the pack.
    tasks = [Task(str(n), *a) for n, a in zip(names, arrays)]
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda t: task_labels(t, cfg), tasks))
    with np.load(args.pack/'labels.npz') as z:
        old = {key: z[key] for key in z.files}
    h, latent, target, ego, speed = (np.concatenate([r[i] for r in rows]) for i in range(5))
    val = np.repeat([str(p) in held for p in paths], [len(r[0]) for r in rows])
    print(json.dumps(dict(windows=len(h), pack_windows=len(old['h']), h_equal=bool((h == old['h']).all()),
                          latent_max_diff=float(np.abs(latent-old['latent']).max()),
                          target_max_diff=float(np.abs(target-old['target']).max()),
                          ego_max_diff=float(np.abs(ego-old['ego']).max()),
                          speed_max_diff=float(np.abs(speed-old['speed']).max()),
                          val_equal=bool((val == old['val']).all()))), flush=True)

    files = ['trajectories.npz']
    if not args.skip_bev:
        bev = np.load(args.pack/'bev.npy', mmap_mode='r')
        assert len(bev) == len(h)
        for part, start in enumerate(range(0, len(bev), PART_WINDOWS)):
            name = f'bev_{part}.npy'
            out = np.lib.format.open_memmap(args.out/name, 'w+', bev.dtype, (min(PART_WINDOWS, len(bev)-start), *bev.shape[1:]))
            for at in range(0, len(out), 1000):
                out[at:at+1000] = bev[start+at:start+at+1000]
            out.flush()
            del out
            files.append(name)
            print(name, flush=True)
    (args.out/'SHA256SUMS').write_text(''.join(f'{sha256(args.out/name)}  {name}\n' for name in files))


if __name__ == '__main__':
    main()
