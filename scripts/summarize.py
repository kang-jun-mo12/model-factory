"""Table of every model under models/ from its meta.json, as Markdown.

Usage: python scripts/summarize.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows = []
    for meta in sorted((ROOT/'models').glob('*/meta.json')):
        m = json.loads(meta.read_text(encoding='utf-8'))
        p = m.get('paths', {})
        rows.append((m['coefficients'], m['seed'], m['name'], m.get('best_epoch'), m['validation']['correct'],
                     m['validation']['flow'], p.get('end_gap_median'), p.get('end_gap_p95'), p.get('best_of_8_median'),
                     p.get('over_limit'), p.get('ms_per_plan')))
    print('| K | seed | model | best epoch | val structure acc | val flow loss | end gap median m | end gap p95 m | best-of-8 median m | over steer limit | ms/plan |')
    print('|---|---|---|---|---|---|---|---|---|---|---|')
    fmt = lambda v, f: '-' if v is None else format(v, f)
    for k, seed, name, epoch, acc, flow, med, p95, best, over, ms in sorted(rows):
        print(f'| {k} | {seed} | {name} | {epoch} | {fmt(acc, ".3f")} | {fmt(flow, ".3f")} | {fmt(med, ".3f")} | '
              f'{fmt(p95, ".3f")} | {fmt(best, ".3f")} | {fmt(over, ".3f")} | {fmt(ms, ".0f")} |')


if __name__ == '__main__':
    main()
