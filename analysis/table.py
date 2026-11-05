"""Main table from results/final_returns.csv: each rule's final return relative to bp, per task and pooled.

Per task: ratio of mean returns (rule / bp) with a 95 % percentile bootstrap interval (unpaired; seeds resampled with
replacement within each rule; 20,000 resamples; each interval uses its own generator seeded with --seed).
Pooled over tasks: every seed's return is divided by its task's mean bp return, and the ratio is
mean(normalised rule) / mean(normalised reference) over all seeds; seeds are resampled within each task. The pooled
ratios are also given with the lowest Hopper bp seed left out (from the ratios and from Hopper's normalising mean).

Usage: python analysis/table.py [--csv results/final_returns.csv] [--metric last100|window_100k|window_500k]
"""
import argparse
import csv
from collections import defaultdict

import numpy as np

TASKS = ('Hopper', 'Walker2d', 'HalfCheetah', 'Ant')
RULES = ('bpsc', 'fsign', 'sign', 'dfa', 'kp')
N_BOOT = 20_000


def load(fn: str, metric: str) -> dict:
    """{experiment: {task: {rule: {seed: value}}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    with open(fn, newline='') as f:
        for r in csv.DictReader(f):
            out[r['experiment']][r['task']].setdefault(r['rule'], {})[int(r['seed'])] = float(r[metric])
    return out


def values(cell: dict, drop: tuple = ()) -> np.ndarray:
    return np.array([v for s, v in sorted(cell.items()) if s not in drop])


def resample_sums(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    """Sum of x over N_BOOT resamples with replacement."""
    return x[rng.integers(0, len(x), (N_BOOT, len(x)))].sum(1)


def ratio_ci(a: np.ndarray, b: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    r = (resample_sums(rng, a) / len(a)) / (resample_sums(rng, b) / len(b))
    lo, hi = np.percentile(r, [2.5, 97.5])
    return a.mean() / b.mean(), lo, hi


def pooled_ci(data: dict, num: str, den: str, seed: int, drop: dict | None = None) -> tuple[float, float, float]:
    """drop: {(task, rule): seeds} left out, also from the bp mean that normalises the task."""
    drop = drop or {}
    cells = []
    for t in TASKS:
        cell = lambda r: values(data[t][r], drop.get((t, r), ()))
        m = cell('bp').mean()
        cells.append((cell(num) / m, cell(den) / m))
    rng = np.random.default_rng(seed)
    n_a, n_b = sum(len(x) for x, _ in cells), sum(len(y) for _, y in cells)
    a = sum(resample_sums(rng, x) for x, _ in cells) / n_a
    b = sum(resample_sums(rng, y) for _, y in cells) / n_b
    lo, hi = np.percentile(a / b, [2.5, 97.5])
    point = np.concatenate([x for x, _ in cells]).mean() / np.concatenate([y for _, y in cells]).mean()
    return point, lo, hi


def fmt(r: tuple[float, float, float]) -> str:
    return f'{r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]'


def main(args: argparse.Namespace) -> None:
    D = load(args.csv, args.metric)
    main_ = D['main']
    tasks = [t for t in TASKS if 'bp' in main_.get(t, {})]
    print(f'metric: {args.metric}; ratios of means to bp with 95 % bootstrap intervals; n = seeds\n')
    print(f"{'task':12s} {'bp mean (n)':>12s}  " + '  '.join(f'{r:>22s}' for r in RULES) + f"  {'fsign / bpsc':>18s}")
    for t in tasks:
        bp = values(main_[t]['bp'])
        cols = [f'{fmt(ratio_ci(values(main_[t][r]), bp, args.seed))} n={len(main_[t][r]):<2d}' if r in main_[t] else '-'
                for r in RULES]
        fb = (fmt(ratio_ci(values(main_[t]['fsign']), values(main_[t]['bpsc']), args.seed))
              if {'fsign', 'bpsc'} <= set(main_[t]) else '-')
        print(f'{t:12s} {bp.mean():7.0f} ({len(bp):2d})  ' + '  '.join(f'{c:>22s}' for c in cols) + f'  {fb:>18s}')
    if len(tasks) == len(TASKS):
        print('\npooled over the four tasks:')
        for num, den in (('fsign', 'bp'), ('fsign', 'bpsc'), ('bpsc', 'bp')):
            print(f'  {num} / {den}: {fmt(pooled_ci(main_, num, den, args.seed))}')

    if 'Hopper' in tasks and 'fsign' in main_['Hopper']:
        h = main_['Hopper']
        low = min(h['bp'], key=h['bp'].get)
        rest = values(h['bp'], (low,))
        print(f"\nHopper without the lowest bp seed (seed {low}: bp return {h['bp'][low]:.0f}; "
              f"the other bp seeds {rest.min():.0f} to {rest.max():.0f}):")
        print(f"  fsign (all seeds) / bp without seed {low}: {fmt(ratio_ci(values(h['fsign']), rest, args.seed))}")
        print(f"  fsign / bp, seed {low} dropped from both: {fmt(ratio_ci(values(h['fsign'], (low,)), rest, args.seed))}")
        if len(tasks) == len(TASKS):
            drop = {('Hopper', 'bp'): (low,)}
            print(f'  pooled over the four tasks with Hopper bp seed {low} left out:')
            for num, den in (('fsign', 'bp'), ('bpsc', 'bp')):
                print(f'    {num} / {den}: {fmt(pooled_ci(main_, num, den, args.seed, drop))}')

    fz = D.get('frozen_control', {})
    if fz:
        print('\nfrozen hidden layers (separate run at 2M steps; ratio to bp of the same run):')
        for t in [t for t in TASKS if {'bp', 'frozen'} <= set(fz.get(t, {}))]:
            print(f"  {t:12s} {fmt(ratio_ci(values(fz[t]['frozen']), values(fz[t]['bp']), args.seed))} "
                  f"n={len(fz[t]['frozen'])}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--csv', default='results/final_returns.csv')
    p.add_argument('--metric', default='last100', choices=['last100', 'window_100k', 'window_500k'])
    p.add_argument('--seed', type=int, default=0, help='seed of the bootstrap generators')
    main(p.parse_args())
