"""Learning curves from results/curves.csv: mean over seeds of the binned training return, with +- 1 standard error.

Usage: python analysis/plot_curves.py [--csv results/curves.csv] [--out results/figures/learning_curves.png]
"""
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TASKS = ('Hopper', 'Walker2d', 'HalfCheetah', 'Ant')
STYLE = dict(bp=('k', '-'), bpsc=('0.45', '--'), fsign=('C0', '-'), sign=('C2', '-'), dfa=('C3', '-'), kp=('C1', ':'))


def main(args: argparse.Namespace) -> None:
    cur = defaultdict(lambda: defaultdict(dict))   # (task, rule) -> bin_end -> seed -> mean return
    with open(args.csv, newline='') as f:
        for r in csv.DictReader(f):
            cur[(r['task'], r['rule'])][int(r['bin_end'])][int(r['seed'])] = float(r['mean_return'])
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, task in zip(axes, TASKS):
        for rule, (c, ls) in STYLE.items():
            bins = cur.get((task, rule))
            if not bins:
                continue
            x = np.array(sorted(bins))
            vals = [np.array(list(bins[b].values())) for b in x]
            m = np.array([v.mean() for v in vals])
            se = np.array([v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals])
            n = max(len(v) for v in vals)
            ax.plot(x / 1e6, m, color=c, ls=ls, lw=1.4, label=f'{rule} (n={n})')
            ax.fill_between(x / 1e6, m - se, m + se, color=c, alpha=0.15, lw=0)
        ax.set_title(task)
        ax.set_xlabel('environment steps (millions)')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('training return (100k-step bins)')
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print('saved', args.out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--csv', default='results/curves.csv')
    p.add_argument('--out', default='results/figures/learning_curves.png')
    main(p.parse_args())
