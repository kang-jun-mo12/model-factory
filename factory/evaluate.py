"""Validation metrics in metres, comparable across coefficient counts.

The Flow loss is not: K changes the size and normalisation of the latent. These are read off decoded
paths instead, against the label path of the same window.
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from .labels import basis, decode
from .model import Context

STEER_LIMIT_CURVATURE = .345     # 1/m at steer 0.8, the DDP controller's clamp


@torch.no_grad()
def losses(model, data, index, weights):
    """Structure cross entropy and accuracy, the Flow loss and the Flow loss with another window's BEV."""
    model.eval()
    generator = torch.Generator('cuda').manual_seed(0)
    partner = index[(np.arange(len(index))+len(index)//2) % len(index)]
    total = dict(ce=0., correct=0., flow=0., flow_other_bev=0.)
    for start in range(0, len(index), 256):
        part, other = index[start:start+256], partner[start:start+256]
        h, latent = data.truth(part)
        context, logits = model.encode(*data.inputs(part))
        noise = torch.randn(latent.shape, device='cuda', generator=generator)
        t = torch.rand(len(h), device='cuda', generator=generator)
        total['ce'] += F.cross_entropy(logits, h, weight=weights).item()*len(h)
        total['correct'] += (logits.argmax(-1) == h).sum().item()
        total['flow'] += model.loss(latent, context, h, noise, t).item()*len(h)
        total['flow_other_bev'] += model.loss(latent, model.encode(*data.inputs(part, other))[0], h, noise, t).item()*len(h)
    return {key: value/len(index) for key, value in total.items()}


@torch.no_grad()
def path_metrics(model, data, index, count=1024, steps=None, seed=0):
    """On `count` evenly spread validation windows:

    end_gap        one sample with the true structure vs the label path end (median, p95)
    best_of_8      the planner's 8 candidates (top-2 structures x 4) and the closest end to the label
    mean_of_8      the average end gap of those 8 candidates
    structure_hit  the true structure is among the top 2
    over_limit     share of candidates whose curvature passes the steer-0.8 limit somewhere
    ms_per_plan    one rollout of 8 candidates, one window
    """
    model.eval()
    cfg = model.cfg
    index = index[np.linspace(0, len(index)-1, min(count, len(index))).astype(int)]
    speed = data.arrays['speed'][index]
    grid = basis(cfg.coefficients, cfg.samples)
    generator = torch.Generator('cuda').manual_seed(seed)
    single, best, mean, hit, over = [], [], [], [], []
    k = cfg.candidates_per_structure
    for start in range(0, len(index), 128):
        part = index[start:start+128]
        h, latent = data.truth(part)
        context, logits = model.encode(*data.inputs(part))
        one = model.rollout(context, h, steps=steps, generator=generator).cpu().numpy()
        top = logits.topk(cfg.structures, -1).indices                                  # [B,2]
        hs = top.repeat_interleave(k, 1).reshape(-1)                                    # [B*8]
        repeat = lambda value: value.repeat_interleave(cfg.structures*k, 0)
        many = model.rollout(Context(repeat(context.queries), repeat(context.spatial), repeat(context.valid)),
                             hs, steps=steps, generator=generator).cpu().numpy()
        many = many.reshape(len(part), cfg.structures*k, *many.shape[1:])
        hs = hs.reshape(len(part), -1).cpu().numpy()
        for row, (label, structure, v) in enumerate(zip(latent.cpu().numpy(), h.tolist(), speed[start:start+128])):
            end = decode(label, structure, v, cfg)[-1, :2]
            single.append(np.linalg.norm(decode(one[row], structure, v, cfg)[-1, :2]-end))
            gaps = [np.linalg.norm(decode(c, s, v, cfg)[-1, :2]-end) for c, s in zip(many[row], hs[row])]
            best.append(min(gaps)); mean.append(np.mean(gaps))
            hit.append(structure in hs[row])
            for c, s in zip(many[row], hs[row]):
                phases = s % cfg.max_phases+1
                over.append(np.abs(grid@c[:phases, :, 1].T*cfg.curvature_scale).max() > STEER_LIMIT_CURVATURE)
    # Time: one plan, after a warm-up.
    one_context = model.encode(*data.inputs(index[:1]))[0]
    expanded = one_context.expand(cfg.structures*k)
    h8 = torch.arange(cfg.structures, device='cuda').repeat_interleave(k)
    for _ in range(3):
        model.rollout(expanded, h8, steps=steps)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(20):
        model.rollout(expanded, h8, steps=steps)
    torch.cuda.synchronize()
    ms = (time.perf_counter()-started)/20*1000
    single, best, mean = map(np.array, (single, best, mean))
    return dict(windows=len(index), euler_steps=steps or cfg.euler_steps,
                end_gap_median=float(np.median(single)), end_gap_p95=float(np.percentile(single, 95)),
                best_of_8_median=float(np.median(best)), best_of_8_p95=float(np.percentile(best, 95)),
                mean_of_8_median=float(np.median(mean)), structure_hit=float(np.mean(hit)),
                over_limit=float(np.mean(over)), ms_per_plan=ms)
