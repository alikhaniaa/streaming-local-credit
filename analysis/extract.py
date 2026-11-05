"""Reduce training logs written by slc/train.py to the small tables shipped in results/.

For every (task, rule, seed) listed in MANIFEST it writes one row of final_returns.csv:
  last100       mean return of the last 100 training episodes (the final return used in the README)
  window_100k   mean return of the episodes that end in (T - 100k, T]
  window_500k   mean return of the episodes that end in (T - 500k, T]
and binned learning curves to curves.csv (mean return of the episodes ending in each 100k-step bin).

Each manifest entry names the rules taken from one log, so no seed is counted twice; duplicates are an error.
Logs are looked up by name in the --log_dir folders, in order; missing logs are skipped with a message.

Usage: python analysis/extract.py --log_dir runs [--log_dir other_runs] [--out runs/summary]
"""
import argparse
import csv
import json
import os

import numpy as np

BASE, LOCAL = ('bp', 'bpsc', 'fsign'), ('dfa', 'sign', 'kp')
# (experiment, task, T, log name, rules taken, machine, code).
# machine: A = laptop CPU, B = desktop CPU (see README). code: 'current' = slc/train.py; 'legacy_init' = the same with
# --legacy_init; 'early_script' = an earlier single-file trainer (frozen control only).
MANIFEST = [
    ('main', 'Hopper', 2_000_000, 'Hopper-v5_r1', BASE, 'A', 'legacy_init'),
    ('main', 'Hopper', 2_000_000, 'Hopper-v5_r1s5', BASE, 'A', 'legacy_init'),
    ('main', 'Hopper', 2_000_000, 'Hopper-v5_f1a', LOCAL, 'B', 'current'),
    ('main', 'Hopper', 2_000_000, 'Hopper-v5_f1b', ('dfa', 'sign'), 'B', 'current'),
    ('main', 'Walker2d', 3_000_000, 'Walker2d-v5_r1', BASE, 'A', 'legacy_init'),
    ('main', 'Walker2d', 3_000_000, 'Walker2d-v5_r1s5', BASE, 'A', 'legacy_init'),
    ('main', 'Walker2d', 3_000_000, 'Walker2d-v5_f2a', LOCAL, 'B', 'current'),
    ('main', 'Walker2d', 3_000_000, 'Walker2d-v5_f2b', ('dfa', 'sign'), 'B', 'current'),
    ('main', 'Walker2d', 3_000_000, 'Walker2d-v5_p2d0', ('kp',), 'A', 'current'),
    ('main', 'HalfCheetah', 2_000_000, 'HalfCheetah-v5_r1', BASE, 'B', 'legacy_init'),
    ('main', 'HalfCheetah', 2_000_000, 'HalfCheetah-v5_r1s5', BASE, 'B', 'current'),
    ('main', 'HalfCheetah', 2_000_000, 'HalfCheetah-v5_f3a', LOCAL, 'B', 'current'),
    ('main', 'HalfCheetah', 2_000_000, 'HalfCheetah-v5_f3b', LOCAL, 'B', 'current'),
    ('main', 'Ant', 2_000_000, 'Ant-v5_r1', BASE, 'B', 'legacy_init'),
    ('main', 'Ant', 2_000_000, 'Ant-v5_s3', BASE, 'B', 'current'),
    ('main', 'Ant', 2_000_000, 'Ant-v5_s6', BASE, 'B', 'current'),
    ('main', 'Ant', 2_000_000, 'Ant-v5_f4a', LOCAL, 'B', 'current'),
    ('main', 'Ant', 2_000_000, 'Ant-v5_f4b', LOCAL, 'B', 'current'),
    # frozen hidden layers, from an earlier 3-seed run of bp, dfa, fa, sign and frozen at 2M steps on every task
    ('frozen_control', 'Hopper', 2_000_000, 'Hopper-v5_p1', ('bp', 'frozen'), 'B', 'early_script'),
    ('frozen_control', 'Walker2d', 2_000_000, 'Walker2d-v5_p1', ('bp', 'frozen'), 'B', 'early_script'),
    ('frozen_control', 'HalfCheetah', 2_000_000, 'HalfCheetah-v5_p1', ('bp', 'frozen'), 'B', 'early_script'),
    ('frozen_control', 'Ant', 2_000_000, 'Ant-v5_p1', ('bp', 'frozen'), 'B', 'early_script'),
]
BIN = 100_000


def find(name: str, dirs: list[str]) -> str | None:
    for d in dirs:
        fn = os.path.join(d, name + '.json')
        if os.path.exists(fn):
            return fn
    return None


def agent_rule_seed(log: dict, i: int) -> tuple[str, int]:
    # logs name agents 'rule' (slc.train) or 'actor:critic x lr' (the scripts that produced the reported runs)
    names = log.get('names') or [a[0] for a in log['agents']]
    return names[i].split(':')[0], int(log['agents'][i][-1])


def main(args: argparse.Namespace) -> None:
    rows, curves, seen = [], [], set()
    for exp, task, T, name, rules, machine, code in MANIFEST:
        fn = find(name, args.log_dir)
        if fn is None:
            print(f'skipped {name}: no log in {args.log_dir}')
            continue
        with open(fn) as f:
            log = json.load(f)
        for i in range(len(log['agents'])):
            rule, seed = agent_rule_seed(log, i)
            if rule not in rules:
                continue
            key = (exp, task, rule, seed)
            if key in seen:
                raise ValueError(f'duplicate seed {key} ({name})')
            seen.add(key)
            st, rt = np.asarray(log['steps'][i]), np.asarray(log['returns'][i])
            if st.max() < T - 20_000:
                raise ValueError(f'{name} {rule} seed {seed} only reached step {st.max()}')
            win = lambda W: rt[(st > T - W) & (st <= T)].mean()
            rows.append(dict(experiment=exp, task=task, rule=rule, seed=seed, log=name, machine=machine, code=code, T=T,
                             final_step=int(st.max()), episodes=len(rt), last100=rt[-100:].mean(),
                             window_100k=win(100_000), window_500k=win(500_000)))
            if exp == 'main':
                for b in range(BIN, T + 1, BIN):
                    m = (st > b - BIN) & (st <= b)
                    if m.any():
                        curves.append(dict(task=task, rule=rule, seed=seed, bin_end=b, mean_return=rt[m].mean(),
                                           episodes=int(m.sum())))
    os.makedirs(args.out, exist_ok=True)
    for fn, data in (('final_returns.csv', rows), ('curves.csv', curves)):
        if not data:
            continue
        with open(os.path.join(args.out, fn), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
            w.writeheader()
            # repr keeps every float exactly, so table.py on the CSV gives the same numbers as on the logs
            w.writerows({k: repr(float(v)) if isinstance(v, (float, np.floating)) else v for k, v in r.items()} for r in data)
        print(f'wrote {len(data)} rows to {os.path.join(args.out, fn)}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--log_dir', action='append', required=True, help='folder with <env>_<tag>.json logs (repeatable)')
    p.add_argument('--out', default='runs/summary', help='output folder (the shipped tables are in results/)')
    main(p.parse_args())
