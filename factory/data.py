"""The training windows: Fused BEV parts (memory-mapped) and the labels of one coefficient count."""
from pathlib import Path
import numpy as np
import torch

DATA = Path(__file__).resolve().parents[1]/'data'


def labels_path(folder, coefficients):
    return Path(folder)/f'labels_K{coefficients:02d}.npz'


class Data:
    def __init__(self, folder, coefficients):
        folder = Path(folder)
        parts = sorted(folder.glob('bev_*.npy'), key=lambda p: int(p.stem.split('_')[1]))
        if not parts:
            raise SystemExit(f'no bev_*.npy in {folder}: run scripts/download.py first')
        self.parts = [np.load(p, mmap_mode='r') for p in parts]
        self.starts = np.cumsum([0]+[len(p) for p in self.parts])
        path = labels_path(folder, coefficients)
        if not path.exists():
            raise SystemExit(f'{path} is missing: run scripts/make_labels.py --coefficients {coefficients}')
        with np.load(path) as z:
            self.arrays = {key: z[key] for key in z.files}
        if len(self.arrays['h']) != self.starts[-1]:
            raise SystemExit(f"{path} has {len(self.arrays['h'])} windows, the BEV parts {self.starts[-1]}")

    def bev(self, index):
        """Rows `index` of the whole BEV, read part by part in sorted order."""
        order = np.argsort(index)
        rows = np.asarray(index)[order]
        part = np.searchsorted(self.starts, rows, side='right')-1
        out = np.empty((len(rows), *self.parts[0].shape[1:]), self.parts[0].dtype)
        for p in np.unique(part):
            keep = part == p
            out[order[keep]] = self.parts[p][rows[keep]-self.starts[p]]
        return out

    def inputs(self, index, scene_index=None):
        """Encoder inputs. `scene_index` swaps the scene and keeps the target and ego state."""
        source = index if scene_index is None else scene_index
        bev = torch.as_tensor(self.bev(source)).cuda().float()
        tensor = lambda key: torch.as_tensor(self.arrays[key][index]).cuda()
        return bev, tensor('target'), tensor('ego')

    def truth(self, index):
        return (torch.as_tensor(self.arrays['h'][index]).cuda().long(),
                torch.as_tensor(self.arrays['latent'][index]).cuda())
