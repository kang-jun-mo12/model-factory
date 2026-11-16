"""StructureFlow: condition encoder, structure head and Flow Matching over per-phase spline latents.

Copied from dino-flow-parking (scripts/structure_flow/model.py with the blocks it imports from
k4_flow_solver and sync_flow), Fused BEV condition only. Parameter names are unchanged, so a
checkpoint trained here loads in dino-flow-parking's plan_server.py and benchmark.py as it is.
The U-Net pools with ceil_mode and upsamples to the skip size, so any coefficient count K works.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import DEFAULT


def time_embedding(t, dimension):
    frequency = torch.exp(torch.linspace(0, math.log(1000), dimension//2, device=t.device))
    angle = t[:, None]*frequency[None]*2*math.pi
    return torch.cat((angle.sin(), angle.cos()), dim=-1)


class TemporalBlock(nn.Module):
    def __init__(self, inputs, outputs, context):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, outputs, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, outputs)
        self.condition = nn.Linear(context, outputs*2)
        self.conv2 = nn.Conv1d(outputs, outputs, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, outputs)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, context):
        scale, shift = self.condition(context).chunk(2, dim=-1)
        value = self.norm1(self.conv1(x))*(1+scale[..., None])+shift[..., None]
        value = self.norm2(self.conv2(F.silu(value)))
        return F.silu(value+self.skip(x))


class BevTokens(nn.Module):
    """Fused BEV [267,16,16] -> 256 tokens of context_dim."""

    def __init__(self, cfg):
        super().__init__()
        d, g = cfg.context_dim, cfg.token_grid
        self.grid = g
        self.project = nn.Sequential(nn.Conv2d(cfg.bev_channels, d, 1), nn.GroupNorm(1, d),
                                     nn.SiLU(), nn.Conv2d(d, d, 3, padding=1))
        self.position = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)
        axis = (torch.arange(g)+.5)/g*2-1
        self.register_buffer('cells', torch.stack(torch.meshgrid(axis, axis, indexing='ij'), -1).reshape(-1, 2))

    def forward(self, bev):
        value = F.adaptive_avg_pool2d(self.project(bev), self.grid).flatten(2).transpose(1, 2)
        return self.norm(value+self.position(self.cells)[None])


@dataclass
class Context:
    queries: torch.Tensor
    spatial: torch.Tensor
    valid: torch.Tensor

    def expand(self, count):
        return Context(self.queries.expand(count, -1, -1), self.spatial.expand(count, -1, -1),
                       self.valid.expand(count, -1))


def latent_mask(h, cfg=DEFAULT, dtype=torch.float32):
    phase = torch.arange(cfg.max_phases, device=h.device)[None] < (h % cfg.max_phases+1)[:, None]
    mask = torch.zeros((len(h), cfg.max_phases, cfg.coefficients, 3), device=h.device, dtype=dtype)
    mask[:, :, 0, 0] = phase.to(dtype)
    mask[:, :, :, 1:] = phase[:, :, None, None].to(dtype)
    return mask, phase


class BevConditionEncoder(nn.Module):
    """24 learned queries read the BEV tokens and the task token, then mix in two transformer layers."""

    def __init__(self, cfg=DEFAULT):
        super().__init__()
        if cfg.condition != 'bev':
            raise ValueError('model-factory trains the Fused BEV condition only')
        d = cfg.context_dim
        self.bev = BevTokens(cfg)
        self.task = nn.Sequential(nn.Linear(cfg.target_dim+cfg.ego_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.queries = nn.Parameter(torch.randn(cfg.queries, d)*.02)
        self.cross = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        block = nn.TransformerEncoderLayer(d, cfg.heads, d*3, dropout=0., batch_first=True,
                                           activation='gelu', norm_first=True)
        self.mix = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, bev, target, ego):
        task = self.task(torch.cat((target, ego), -1))
        spatial = self.bev(bev)
        mask = torch.ones(spatial.shape[:2], device=task.device, dtype=torch.bool)
        keys = torch.cat((spatial, task[:, None]), 1)
        allowed = torch.cat((mask, torch.ones_like(mask[:, :1])), 1)
        query = self.queries[None]+task[:, None]
        out = query+self.cross(query, keys, keys, key_padding_mask=~allowed, need_weights=False)[0]
        return Context(self.norm(self.mix(out)), spatial, mask)


class StructureFlow(nn.Module):
    def __init__(self, cfg=DEFAULT, normalization=None):
        super().__init__()
        self.cfg = cfg
        d, w = cfg.context_dim, cfg.temporal_channels
        self.encoder = BevConditionEncoder(cfg)
        self.structure = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2*cfg.max_phases))
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.h_embedding = nn.Embedding(2*cfg.max_phases, d)
        self.phase_embedding = nn.Embedding(cfg.max_phases, d)
        self.summary = nn.Linear(5, d)
        self.phase_cross = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, d*2, dropout=0., batch_first=True, norm_first=True)
        self.phase_mix = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.down1 = TemporalBlock(6, w, d)
        self.down2 = TemporalBlock(w, w*2, d)
        self.middle = TemporalBlock(w*2, w*3, d)
        if cfg.bev_attention:
            self.bev_norm = nn.LayerNorm(w*3)
            self.bev_read = nn.MultiheadAttention(w*3, cfg.heads, kdim=d, vdim=d, batch_first=True)
        self.up2 = TemporalBlock(w*5, w*2, d)
        self.up1 = TemporalBlock(w*3, w, d)
        self.control_head = nn.Conv1d(w, 2, 1)
        self.length_head = nn.Sequential(nn.Linear(d+w*3, d), nn.SiLU(), nn.Linear(d, 1))
        for layer in (self.control_head, self.length_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        norm = normalization or dict(mean=[0., 0., 0.], std=[1., 1., 1.])
        self.register_buffer('latent_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('latent_std', torch.tensor(norm['std'], dtype=torch.float32))
        self.register_buffer('u', torch.linspace(0, 1, cfg.coefficients))

    def encode(self, bev, target, ego):
        context = self.encoder(bev, target, ego)
        return context, self.structure(context.queries.mean(1))

    def normalize(self, latent, mask):
        return (latent-self.latent_mean)/self.latent_std*mask

    def denormalize(self, y, mask):
        return (y*self.latent_std+self.latent_mean)*mask

    def forward(self, y, t, context, h):
        cfg = self.cfg
        batch, phases, knots, _ = y.shape
        mask, valid = latent_mask(h, cfg, y.dtype)
        y = y*mask
        index = torch.arange(phases, device=y.device)
        gear = torch.where(h < cfg.max_phases, 1., -1.)[:, None]*torch.where(index % 2 == 0, 1., -1.)[None]
        summary = torch.cat((y[:, :, 0, :1], y[:, :, :, 1:].mean(2), y[:, :, :, 1:].amax(2)), -1)
        cond = self.summary(summary)+self.phase_embedding(index)[None]
        cond = cond+self.h_embedding(h)[:, None]+self.time(time_embedding(t, cfg.context_dim))[:, None]
        cond = cond+self.phase_cross(cond, context.queries, context.queries, need_weights=False)[0]
        cond = self.phase_mix(cond, src_key_padding_mask=~valid)*valid[..., None]
        length = y[:, :, 0, 0, None].expand(-1, -1, knots)
        features = torch.stack((y[..., 1], y[..., 2], length, self.u[None, None].expand(batch, phases, -1),
                                gear[..., None].expand(-1, -1, knots),
                                index[None, :, None].expand(batch, -1, knots)/cfg.max_phases), 2)
        features = features.reshape(batch*phases, 6, knots)
        condition = cond.reshape(batch*phases, -1)
        a = self.down1(features, condition)
        b = self.down2(F.avg_pool1d(a, 2, ceil_mode=True), condition)
        c = self.middle(F.avg_pool1d(b, 2, ceil_mode=True), condition)
        if cfg.bev_attention:
            place = time_embedding(torch.linspace(0, 1, c.shape[-1], device=y.device), c.shape[1])
            tokens = context.spatial.repeat_interleave(phases, 0)
            hidden = ~context.valid.repeat_interleave(phases, 0)
            query = self.bev_norm(c.transpose(1, 2)+place[None])
            c = c+self.bev_read(query, tokens, tokens, key_padding_mask=hidden,
                                need_weights=False)[0].transpose(1, 2)
        d = self.up2(torch.cat((F.interpolate(c, size=b.shape[-1], mode='linear', align_corners=True), b), 1), condition)
        e = self.up1(torch.cat((F.interpolate(d, size=a.shape[-1], mode='linear', align_corners=True), a), 1), condition)
        control = self.control_head(e).transpose(1, 2).reshape(batch, phases, knots, 2)
        velocity = self.length_head(torch.cat((condition, c.mean(-1)), -1)).reshape(batch, phases)
        result = torch.zeros_like(y)
        result[..., 1:] = control
        result[:, :, 0, 0] = velocity
        return result*mask

    def loss(self, latent, context, h, noise=None, t=None):
        mask, _ = latent_mask(h, self.cfg, latent.dtype)
        data = self.normalize(latent, mask)
        noise = (torch.randn_like(data) if noise is None else noise)*mask
        t = torch.rand(len(h), device=h.device) if t is None else t
        mixed = (1-t[:, None, None, None])*noise+t[:, None, None, None]*data
        error = (self(mixed, t, context, h)-(data-noise))**2*mask
        return (error.sum((1, 2, 3))/mask.sum((1, 2, 3))).mean()

    @torch.no_grad()
    def rollout(self, context, h, steps=None, generator=None):
        steps = steps or self.cfg.euler_steps
        dtype = context.queries.dtype
        mask, _ = latent_mask(h, self.cfg, dtype)
        y = torch.randn(mask.shape, device=h.device, dtype=mask.dtype, generator=generator)*mask
        for k in range(steps):
            t = torch.full((len(h),), k/steps, device=h.device, dtype=dtype)
            y = (y+self(y, t, context, h)/steps)*mask
        return self.denormalize(y, mask)


def load(path, device='cuda'):
    """A checkpoint from train.py (or dino-flow-parking) -> eval-mode model with its own config."""
    from dataclasses import fields, replace
    state = torch.load(path, map_location=device, weights_only=True)
    known = {field.name for field in fields(DEFAULT)}
    values = {key: value for key, value in state['config'].items() if key in known}
    model = StructureFlow(replace(DEFAULT, **values), state['normalization']).to(device).eval()
    model.load_state_dict(state['state'])
    return model
