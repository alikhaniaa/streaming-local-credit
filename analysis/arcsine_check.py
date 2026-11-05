"""Sign agreement between fsign and exact-backprop updates in the actor's last hidden layer, against the arcsine law.

For hidden unit i of layer 2, the exact error is w_i . g and the fsign error is s_i . g, where g is the error at the
actor outputs and w_i, s_i are column i of W3 and of its sign pattern S3. If g is zero-mean Gaussian with second
moment Sigma, the two errors have correlation rho_i = w_i' Sigma s_i / sqrt(w_i' Sigma w_i * s_i' Sigma s_i) and agree in
sign with probability 1/2 + arcsin(rho_i) / pi (Sheppard 1899). Because a synapse's update is its unit's error times
its input, this is also the fraction of the unit's synapses whose update signs agree (before LayerNorm).

For every fsign actor in the log: roll out the saved policy (sampled actions; env seeds 30000 + episode; at most 1000
steps per episode) until at least --min_states states are collected, estimate Sigma from the sampled g, and compare the
law with the measured agreement before LayerNorm (pre) and after the full LayerNorm backward (full), per unit.
The verdict rules and the two rival predictions were fixed for Humanoid before that run; on other tasks or settings the
verdict is only a label, not a test.

Usage: python analysis/arcsine_check.py runs/Humanoid-v5_humT5.json [--min_states 10000]
Needs the log's _params.npz. Writes <log>_arcsine.json next to the log.
"""
import argparse
import json

import gymnasium as gym
import numpy as np

SLOPE, EPS = 0.01, 1e-5
INIT_LAW, TREND = 0.630, (0.55, 0.62)   # rival predictions: the law at initialisation; a linear trend in action dim


def lrelu(x):
    return np.where(x > 0, x, SLOPE * x)


def ln_fwd(a):
    mu = a.mean()
    sig = np.sqrt(((a - mu) ** 2).mean() + EPS)
    return (a - mu) / sig, sig


def ln_bwd(g, n, sig):
    return (g - g.mean() - n * (g * n).mean()) / sig


def forward(p, x):
    """Same network as slc/network.py, in numpy; returns the outputs and the layer-2 LayerNorm quantities."""
    a1 = p['W1'] @ x + p['c1']
    n1, s1 = ln_fwd(a1)
    h1 = lrelu(n1)
    a2 = p['W2'] @ h1 + p['c2']
    n2, s2 = ln_fwd(a2)
    h2 = lrelu(n2)
    return p['W3'] @ h2 + p['c3'], n2, s2, np.where(n2 > 0, 1.0, SLOPE)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def arcsine_law(W3, S3, Sig):
    num = np.einsum('ki,kl,li->i', W3, Sig, S3)
    dw = np.einsum('ki,kl,li->i', W3, Sig, W3)
    ds = np.einsum('ki,kl,li->i', S3, Sig, S3)
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = num / np.sqrt(dw * ds)
    return 0.5 + np.arcsin(np.clip(rho, -1, 1)) / np.pi


def collect(env_name, pa, S3, obs_mean, obs_var, rng, min_states, entropy=0.01):
    """Output errors g (as in training: log-likelihood gradient of the sampled action plus the entropy term with a
    random sign of the TD error) and per-unit sign agreement before / after the LayerNorm backward, per state."""
    na = pa['W3'].shape[0] // 2
    W3 = pa['W3']
    env = gym.make(env_name)
    lo, hi = env.action_space.low, env.action_space.high
    G, agree_pre, agree_full, ep_len = [], [], [], []
    ep = 0
    while len(G) < min_states:
        o, _ = env.reset(seed=30_000 + ep)
        done, t = False, 0
        while not done:
            s = ((o - obs_mean) / np.sqrt(obs_var + 1e-8)).astype(np.float32)
            out, n2, s2, d2 = forward(pa, s)
            mu, pstd = out[:na], out[na:]
            std = softplus(pstd)
            a = mu + std * rng.standard_normal(na)
            sd = rng.choice([-1.0, 1.0])
            g = np.concatenate([(a - mu) / std ** 2, (((a - mu) ** 2 / std ** 3 - 1 / std) + entropy * sd / std) * sigmoid(pstd)])
            G.append(g)
            g_bp, g_fs = (W3.T @ g) * d2, (S3.T @ g) * d2
            agree_pre.append(np.sign(g_bp) == np.sign(g_fs))
            agree_full.append(np.sign(ln_bwd(g_bp, n2, s2)) == np.sign(ln_bwd(g_fs, n2, s2)))
            o, _, te, tr, _ = env.step(np.clip(a, lo, hi))
            t += 1
            done = te or tr or t >= 1000
        ep_len.append(t)
        ep += 1
    return np.array(G), np.array(agree_pre, float), np.array(agree_full, float), ep_len


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def dist_to_band(x, band):
    return max(band[0] - x, 0.0, x - band[1])


def main(args: argparse.Namespace) -> None:
    with open(args.log) as f:
        d = json.load(f)
    P = np.load(args.log.replace('.json', '_params.npz'))
    env_name = d['args']['env']
    idx = [i for i, n in enumerate(d['names']) if n.split(':')[0] == 'fsign']
    diag = [x for x in d['diag'] if 'ucos' in x]
    rows = []
    for j, i in enumerate(idx):
        pa = {k[3:]: P[k][i] for k in P.files if k.startswith('pa_')}
        S3 = P['fba_S3'][i]
        assert np.all(pa['W3'] * S3 >= 0), 'sign constraint violated'
        G, pre_u, full_u, ep_len = collect(env_name, pa, S3, P['obs_mean'][i], P['obs_var'][i],
                                           np.random.default_rng(1000 + j), args.min_states)
        law = arcsine_law(pa['W3'], S3, G.T @ G / len(G))
        pre, full = pre_u.mean(0), full_u.mean(0)
        h = len(G) // 2
        rows.append(dict(seed=int(d['agents'][i][-1]), states=len(G), episodes=len(ep_len), law=float(np.nanmean(law)),
                         pre=float(pre.mean()), full=float(full.mean()), r_pre=pearson(law, pre), r_full=pearson(law, full),
                         split_half_pre=pearson(pre_u[:h].mean(0), pre_u[h:].mean(0)),
                         logged_agreement=float(np.asarray(diag[-1]['ucos'])[i, 10]) if diag else float('nan'),
                         nonzero_W3=float((pa['W3'] != 0).mean())))
        print(json.dumps(rows[-1]), flush=True)
    M = {k: float(np.mean([r[k] for r in rows])) for k in ('law', 'pre', 'full', 'r_pre', 'r_full', 'split_half_pre', 'logged_agreement')}
    law_m = M['law']
    passed = M['r_pre'] >= 0.85 and abs(law_m - M['pre']) <= 0.04 and -0.01 <= M['full'] - law_m <= 0.06
    trend_fires = all(dist_to_band(v, TREND) <= 0.02 and abs(v - law_m) > 0.06 for v in (M['pre'], M['full'], M['logged_agreement']))
    kill = M['r_pre'] < 0.6 or abs(law_m - M['pre']) > 0.08 or trend_fires
    discriminating = abs(law_m - INIT_LAW) > 0.04 and dist_to_band(law_m, TREND) > 0.04
    verdict = 'Humanoid rule: ' + ('KILL' if kill else 'PASS (rival-discriminating)' if passed and discriminating else
                                   'PASS (per-unit only)' if passed else 'INCONCLUSIVE')
    out = dict(log=args.log, min_states=args.min_states, seeds=rows, mean=M, full_minus_law=M['full'] - law_m,
               abs_law_minus_pre=abs(law_m - M['pre']), trend_clause_fires=trend_fires, rival_discriminating=discriminating,
               verdict=verdict)
    with open(args.log.replace('.json', '_arcsine.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"law {law_m:.4f} | measured pre-LN {M['pre']:.4f} | full-LN {M['full']:.4f} | per-unit r (pre) {M['r_pre']:.3f} "
          f"(split-half {M['split_half_pre']:.3f}) | full - law {M['full'] - law_m:+.4f} | verdict {verdict}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('log', help='training log of a run with fsign agents (and --shadow for the logged agreement)')
    p.add_argument('--min_states', type=int, default=10_000)
    main(p.parse_args())
