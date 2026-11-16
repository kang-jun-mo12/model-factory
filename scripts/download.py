"""Download the training data (about 7 GB) from this repository's GitHub Release into data/.

Usage: python scripts/download.py [--data data] [--frontend]

Files: trajectories.npz (expert frames), bev_0..3.npy (cached Fused BEV), SHA256SUMS. Every file is
checked against SHA256SUMS; an interrupted download resumes. --frontend also fetches
ddp_frontend.pt (434 MB), the frozen DDP perception that made the BEV. Training does not need it;
closed-loop evaluation in CARLA does.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from factory.data import DATA

RELEASE = 'https://github.com/kang-jun-mo12/model-factory/releases/download/data-v1/'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(name, path):
    have = path.stat().st_size if path.exists() else 0
    request = urllib.request.Request(RELEASE+name, headers={'Range': f'bytes={have}-'} if have else {})
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError as error:
        if error.code == 416:       # already complete
            return
        raise
    total = have+int(response.headers.get('Content-Length', 0))
    mode = 'ab' if response.status == 206 else 'wb'
    done = have if mode == 'ab' else 0
    with open(path, mode) as f:
        while block := response.read(1 << 22):
            f.write(block)
            done += len(block)
            print(f'\r{name}: {done/1e9:.2f} / {total/1e9:.2f} GB', end='', flush=True)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--frontend', action='store_true')
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    sums = args.data/'SHA256SUMS'
    fetch('SHA256SUMS', sums) if not sums.exists() else None
    expected = dict(reversed(line.split()) for line in sums.read_text().splitlines() if line.strip())
    names = [n for n in expected if n != 'ddp_frontend.pt' or args.frontend]
    for name in names:
        path = args.data/name
        if path.exists() and sha256(path) == expected[name]:
            print(f'{name}: ok')
            continue
        fetch(name, path)
        if sha256(path) != expected[name]:
            path.unlink()
            sys.exit(f'{name}: checksum mismatch, removed; run again')
        print(f'{name}: ok')
    print(f'data in {args.data}. Next: python scripts/make_labels.py --coefficients K')


if __name__ == '__main__':
    main()
