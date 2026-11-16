"""Train StructureFlow with a chosen spline coefficient count K.

Usage: python scripts/train.py --coefficients K [--seed 1] [--scene-drop 0.4] [--epochs 100]

Everything but K defaults to the released drop40_s1 run: seed 1, 100 epochs, batch 256, AdamW 3e-4
with OneCycle, and with probability 0.4 a training window gets an all-zero Fused BEV (scene drop).
The checkpoint with the best validation score (Flow loss + structure errors) is kept.

Writes models/K{K}_drop{40}_s{seed}/: best.pt, train.log, meta.json. That folder is what a pull
request adds. Needs data/ from scripts/download.py and data/labels_K{K}.npz from make_labels.py.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factory.config import DEFAULT
from factory.data import DATA, Data, labels_path
from factory.evaluate import losses, path_metrics
from factory.model import StructureFlow


def normalization(arrays, train, cfg):
    latent, h = arrays['latent'][train], arrays['h'][train]
    used = np.arange(cfg.max_phases)[None] <= (h % cfg.max_phases)[:, None]
    lengths = latent[..., 0, 0][used]
    rest = latent[used][:, :, 1:].reshape(-1, 2)
    return dict(mean=[float(lengths.mean()), *map(float, rest.mean(0))],
                std=[float(lengths.std()+1e-6), *map(float, rest.std(0)+1e-6)])


def commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coefficients', '-K', type=int, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--scene-drop', type=float, default=.4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, help='default models/K{K}_drop{..}_s{seed}')
    args = parser.parse_args()
    name = f'K{args.coefficients:02d}_drop{round(args.scene_drop*100)}_s{args.seed}'
    out = args.out or ROOT/'models'/name
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(out/'train.log', 'w', encoding='utf-8')

    def log(text):
        print(text, flush=True)
        log_file.write(text+'\n')
        log_file.flush()

    log(json.dumps({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    cfg = replace(DEFAULT, coefficients=args.coefficients, bev_attention=True, condition='bev')
    data = Data(args.data, args.coefficients)
    val = np.flatnonzero(data.arrays['val'])
    train = np.flatnonzero(~data.arrays['val'])
    counts = np.bincount(data.arrays['h'][train], minlength=2*cfg.max_phases).astype(np.float64)
    weights = np.where(counts > 0, 1/np.sqrt(np.maximum(counts, 1)), 0.)
    weights = torch.tensor(weights/weights[counts > 0].mean(), dtype=torch.float32, device='cuda')
    norm = normalization(data.arrays, train, cfg)
    model = StructureFlow(cfg, norm).cuda()
    log(f'K {cfg.coefficients}, train {len(train)}, val {len(val)}, structures {counts.astype(int).tolist()}, '
        f'parameters {sum(p.numel() for p in model.parameters())/1e6:.2f} M')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs*math.ceil(len(train)/args.batch)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=.05)
    best = math.inf
    for epoch in range(args.epochs):
        model.train()
        order, seen, ce_sum, flow_sum = rng.permutation(train), 0, 0., 0.
        for start in range(0, len(order), args.batch):
            part = order[start:start+args.batch]
            h, latent = data.truth(part)
            bev, target, ego = data.inputs(part)
            if args.scene_drop > 0:
                drop = torch.as_tensor(rng.random(len(part)) < args.scene_drop, device='cuda')
                bev = torch.where(drop[:, None, None, None], torch.zeros_like(bev), bev)
            context, logits = model.encode(bev, target, ego)
            ce = F.cross_entropy(logits, h, weight=weights)
            flow = model.loss(latent, context, h)
            optimizer.zero_grad(set_to_none=True)
            (ce+flow).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            seen += len(h)
            ce_sum += ce.item()*len(h)
            flow_sum += flow.item()*len(h)
        metrics = losses(model, data, val, weights)
        score = metrics['flow']+1-metrics['correct']  # val ce climbs from overconfidence alone, so rank by errors
        log(f"{epoch+1:3d} train ce {ce_sum/seen:.3f} flow {flow_sum/seen:.3f} | val ce {metrics['ce']:.3f} "
            f"acc {metrics['correct']:.3f} flow {metrics['flow']:.3f} other-BEV {metrics['flow_other_bev']:.3f} "
            f"| {time.time()-started:.0f} s")
        if score < best:
            best = score
            torch.save(dict(state=model.state_dict(), config=cfg.to_dict(), normalization=norm,
                            epoch=epoch+1, metrics=metrics), out/'best.pt')
    state = torch.load(out/'best.pt', map_location='cuda', weights_only=True)
    model.load_state_dict(state['state'])
    paths = path_metrics(model, data, val)
    log(f"best epoch {state['epoch']}: {json.dumps(state['metrics'])}")
    log(f'validation paths: {json.dumps(paths)}')
    sums = (args.data/'SHA256SUMS').read_text() if (args.data/'SHA256SUMS').exists() else ''
    meta = dict(name=name, coefficients=cfg.coefficients, seed=args.seed, scene_drop=args.scene_drop, epochs=args.epochs,
                batch=args.batch, lr=args.lr, best_epoch=state['epoch'], validation=state['metrics'], paths=paths,
                minutes=round((time.time()-started)/60, 1), code_commit=commit(),
                labels_sha256=hashlib.sha256(labels_path(args.data, cfg.coefficients).read_bytes()).hexdigest(),
                data_sha256=sums, gpu=torch.cuda.get_device_name(), torch=torch.__version__, python=platform.python_version())
    (out/'meta.json').write_text(json.dumps(meta, indent=1))
    log(f'wrote {out}')


if __name__ == '__main__':
    main()
