"""Window labels for one spline coefficient count K, from the expert trajectories.

Usage: python scripts/make_labels.py --coefficients K [--data data]

Writes data/labels_K{K}.npz: h, latent [N,2,K,3], target, ego, speed, val. The windows and their
order are those of the BEV parts, so they line up row by row. K=12 rebuilds the labels of the
released model exactly. Takes about a minute.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory.config import DEFAULT
from factory.data import DATA, labels_path
from factory.labels import Task, task_labels


def tasks(folder):
    with np.load(Path(folder)/'trajectories.npz') as z:
        arrays = {key: z[key] for key in z.files}
    ends = np.cumsum(arrays['length'])
    starts = ends-arrays['length']
    return [Task(str(name), arrays['pose'][a:b], arrays['speed'][a:b], arrays['steer'][a:b], arrays['reverse'][a:b], target)
            for name, a, b, target in zip(arrays['name'], starts, ends, arrays['target'])], arrays['val']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficients', '-K', type=int, required=True)
    parser.add_argument('--data', type=Path, default=DATA)
    args = parser.parse_args()
    if args.coefficients < 4:
        sys.exit('a cubic B-spline needs at least 4 coefficients')
    cfg = replace(DEFAULT, coefficients=args.coefficients)
    items, held = tasks(args.data)
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda task: task_labels(task, cfg), items))
    h, latent, target, ego, speed = (np.concatenate([r[i] for r in rows]) for i in range(5))
    val = np.repeat(held, [len(r[0]) for r in rows])
    out = labels_path(args.data, args.coefficients)
    np.savez(out, h=h, latent=latent, target=target, ego=ego, speed=speed, val=val)
    print(f'{out}: {len(h)} windows ({val.sum()} validation), latent {latent.shape[1:]}, '
          f'structures {np.bincount(h, minlength=4).tolist()}')


if __name__ == '__main__':
    main()
