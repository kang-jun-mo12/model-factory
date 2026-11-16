"""Validation path metrics of one or more checkpoints, in metres, so different K can be compared.

Usage: python scripts/evaluate.py models/K12_drop40_s1/best.pt [more.pt ...] [--steps N] [--windows 1024]

Labels are rebuilt for each checkpoint's own K (data/labels_K{K}.npz must exist). See
factory/evaluate.py for what each number means.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from factory.data import DATA, Data
from factory.evaluate import path_metrics
from factory.model import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+', type=Path)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--steps', type=int, help='Euler steps (default: the checkpoint config, 8)')
    parser.add_argument('--windows', type=int, default=1024)
    args = parser.parse_args()
    loaded = {}
    for path in args.checkpoints:
        model = load(path)
        K = model.cfg.coefficients
        if K not in loaded:
            loaded[K] = Data(args.data, K)
        data = loaded[K]
        result = path_metrics(model, data, np.flatnonzero(data.arrays['val']), args.windows, args.steps)
        print(json.dumps(dict(checkpoint=str(path), coefficients=K, **result)), flush=True)


if __name__ == '__main__':
    main()
