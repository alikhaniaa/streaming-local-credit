"""Streaming actor-critic, stream AC(lambda) (Elsayed et al. 2024): batch size 1, no replay buffer, no target
networks, one update per environment step. The hidden layers of actor and critic learn with the rule given per agent
(see network.py).

An agent is a (rule, seed) pair with its own environment, weights, normalisers and traces. Agents are batched with
vmap only for speed, but the batch matters for exact reproduction: each agent's action noise comes from a split of one
key per step (seeded by --seed0) by its position in the batch, and XLA may round differently for other batch shapes.
A run reproduces bit for bit only with the same --rules, --seed0 and --n_seeds and the same JAX and MuJoCo versions.
All rules of one seed share the initial weights, the feedback matrices and the sign pattern S.

Output in --out: <env>_<tag>.json (per-agent episode returns and end steps, diagnostics) and, at the end,
<env>_<tag>_params.npz (final weights, feedback matrices and observation statistics).

Example: python -m slc.train --env Hopper-v5 --rules bp,bpsc,fsign --seed0 0 --n_seeds 5 --steps 2000000 --tag r1
"""
import argparse
import json
import os
import pickle
import time

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from slc.network import (KOLEN_POLLACK, RULES, SIGN_CONSTRAINED, RunningStat, backward, forward, init_feedback,
                         init_net, rule_ids, stream_step)


def make_step(na: int, hp: dict):
    """Returns jitted, vmapped (step, act). hp: gamma, lam, beta, entropy, kp_decay, shadow."""
    bp_id = jnp.int32(RULES.index('bp'))
    fkp_id = RULES.index('fkp')
    decay = hp['gamma'] * hp['lam']

    def act(pa, s, key):
        # the first of three subkeys, as in the code that produced the reported runs
        k0 = jax.random.split(key, 3)[0]
        out, _ = forward(pa, s)
        mu, std = out[:na], jax.nn.softplus(out[na:])
        return mu + std * jax.random.normal(k0, mu.shape)

    def shadow(e_sh, m_sh, G_bp, e, m, G, delta, reset):
        """Diagnostics only: run the optimiser state of an exact-backprop learner alongside the agent's own (same
        states, same TD errors) and compare the two normalised updates per layer: cosine and fraction of synapses
        whose update signs agree."""
        e_sh = jax.tree.map(lambda e_, g: decay * e_ + g, e_sh, G_bp)
        m_sh = jax.tree.map(lambda m_, e_: jnp.maximum(hp['beta'] * m_, jnp.abs(e_) * jnp.abs(delta)), m_sh, e_sh)
        e_new = jax.tree.map(lambda e_, g: decay * e_ + g, e, G)
        m_new = jax.tree.map(lambda m_, e_: jnp.maximum(hp['beta'] * m_, jnp.abs(e_) * jnp.abs(delta)), m, e_new)
        u_loc = jax.tree.map(lambda e_, m_: delta * e_ / (m_ + 1e-8), e_new, m_new)
        u_bp = jax.tree.map(lambda e_, m_: delta * e_ / (m_ + 1e-8), e_sh, m_sh)
        out = {}
        for k in ('W1', 'W2', 'W3'):
            u, v = u_loc[k], u_bp[k]
            out[k] = (jnp.sum(u * v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v) + 1e-12),
                      jnp.mean(jnp.sign(u) == jnp.sign(v)))
        e_sh = jax.tree.map(lambda e_: jnp.where(reset, 0.0, e_), e_sh)
        return e_sh, m_sh, out

    def project(p, fb, rule):
        # sign-constrained rules: a weight that would cross zero against its fixed sign stops at zero
        on = jnp.isin(rule, jnp.asarray(rule_ids(SIGN_CONSTRAINED)))
        clip = lambda W, S: jnp.where(on & (W * S < 0), 0.0, W)
        return dict(p, W2=clip(p['W2'], fb['S2']), W3=clip(p['W3'], fb['S3']))

    def kolen_pollack(p_new, p_old, fb, rule):
        # each feedback synapse K_ij gets the same change as its forward partner W_ij, and both decay by
        # (1 - kp_decay), so W - K shrinks by (1 - kp_decay) per step; at kp_decay = 0 it stays constant
        on = jnp.isin(rule, jnp.asarray(rule_ids(KOLEN_POLLACK)))
        fixed = rule == fkp_id
        q, f = dict(p_new), dict(fb)
        for k in ('W2', 'W3'):
            Kn = (fb['K' + k[1]] + (p_new[k] - p_old[k])) * (1.0 - hp['kp_decay'])
            Kn = jnp.where(fixed & (Kn * fb['S' + k[1]] < 0), 0.0, Kn)
            f['K' + k[1]] = jnp.where(on, Kn, fb['K' + k[1]])
            q[k] = jnp.where(on, p_new[k] * (1.0 - hp['kp_decay']), p_new[k])
        return q, f

    def step(st, s, a, r, s2, term, reset, s_next, key):
        pa, pc, rule = st['pa'], st['pc'], st['rule']
        out_c, cache_c = forward(pc, s)
        v2, _ = forward(pc, s2)
        delta = r + hp['gamma'] * v2[0] * (1.0 - term) - out_c[0]
        out_a, cache_a = forward(pa, s)
        mu, pstd = out_a[:na], out_a[na:]
        std = jax.nn.softplus(pstd)
        # d log pi(a) / d(head outputs), plus the entropy bonus with the sign of the TD error
        g_mu = (a - mu) / std ** 2
        g_std = ((a - mu) ** 2 / std ** 3 - 1.0 / std) + hp['entropy'] * jnp.sign(delta) / std
        g_out_a = jnp.concatenate([g_mu, g_std * jax.nn.sigmoid(pstd)])
        Ga, align_a = backward(pa, cache_a, g_out_a, st['fba'], rule)
        Gc, align_c = backward(pc, cache_c, jnp.ones(1), st['fbc'], rule)
        new = dict(st)
        if hp['shadow']:
            Ga_bp, _ = backward(pa, cache_a, g_out_a, st['fba'], bp_id)
            Gc_bp, _ = backward(pc, cache_c, jnp.ones(1), st['fbc'], bp_id)
            new['esa'], new['msa'], ua = shadow(st['esa'], st['msa'], Ga_bp, st['ea'], st['ma'], Ga, delta, reset)
            new['esc'], new['msc'], uc = shadow(st['esc'], st['msc'], Gc_bp, st['ec'], st['mc'], Gc, delta, reset)
            # [critic W1, W2, W3, actor W1, W2, W3] cosines, then the same six sign agreements
            ucos = jnp.stack([uc[k][0] for k in ('W1', 'W2', 'W3')] + [ua[k][0] for k in ('W1', 'W2', 'W3')]
                             + [uc[k][1] for k in ('W1', 'W2', 'W3')] + [ua[k][1] for k in ('W1', 'W2', 'W3')])
        else:
            ucos = jnp.zeros(12)
        pa_new, new['ea'], new['ma'] = stream_step(pa, st['ea'], st['ma'], Ga, delta, reset, st['lr_pi'], decay, hp['beta'])
        pc_new, new['ec'], new['mc'] = stream_step(pc, st['ec'], st['mc'], Gc, delta, reset, st['lr_v'], decay, hp['beta'])
        pa_new, pc_new = project(pa_new, st['fba'], rule), project(pc_new, st['fbc'], rule)
        new['pa'], new['fba'] = kolen_pollack(pa_new, pa, st['fba'], rule)
        new['pc'], new['fbc'] = kolen_pollack(pc_new, pc, st['fbc'], rule)
        a_next = act(new['pa'], s_next, key)
        return new, a_next, delta, jnp.concatenate([align_c, align_a]), ucos

    return jax.jit(jax.vmap(step)), jax.jit(jax.vmap(act))


def init_agents(agents: list[tuple[str, int]], d: int, na: int, H: int, sparsity: float, legacy_init: bool = False):
    """Actor and critic weights and feedback for each (rule, seed). The random streams depend only on the seed, so
    every rule of a seed starts from the same |W|, B, S and K; sign-constrained rules start inside S.
    legacy_init skips the draw of K, as the earlier revision that produced some of the reported runs did (README).
    This changes only the random signs that S gives to zero-initialised entries of W3."""
    if legacy_init and any(r in KOLEN_POLLACK for r, _ in agents):
        raise ValueError('legacy_init draws no Kolen-Pollack feedback; it cannot be used with kp or fkp')
    nets_a, nets_c, fbs_a, fbs_c = [], [], [], []
    for rule, seed in agents:
        rng = np.random.default_rng(1000 * seed + 7)
        nets_a.append(init_net(d, H, [na, na], sparsity, rng))
        nets_c.append(init_net(d, H, [1], sparsity, rng))
        rf = np.random.default_rng(1000 * seed + 99)
        fbs_a.append(init_feedback(H, 2 * na, rf))
        fbs_c.append(init_feedback(H, 1, rf))
        for net, fb in ((nets_a[-1], fbs_a[-1]), (nets_c[-1], fbs_c[-1])):
            rm = np.random.default_rng(1000 * seed + 555)
            for k in ('W2', 'W3'):
                W = net[k]
                # one fixed sign per synapse: the initial sign, random where the initial weight is zero
                S = np.where(W != 0, np.sign(W), np.where(rm.random(W.shape) < 0.5, 1.0, -1.0))
                S = np.ascontiguousarray(S, np.float32)
                fb['S' + k[1]] = S
                if rule in SIGN_CONSTRAINED:
                    net[k] = (np.abs(W) * S).astype(np.float32)
                if legacy_init:
                    fb['K' + k[1]] = np.zeros(W.shape, np.float32)   # unused by the non-Kolen-Pollack rules
                    continue
                # Kolen-Pollack feedback starts random (fkp: random magnitude with the sign of S)
                K = (rm.uniform(-1, 1, W.shape) * np.sqrt(1.0 / W.shape[1])).astype(np.float32)
                fb['K' + k[1]] = (np.abs(K) * S).astype(np.float32) if rule == 'fkp' else K
    return nets_a, nets_c, fbs_a, fbs_c


def save_json(obj, fn: str) -> None:
    with open(fn + '.tmp', 'w') as f:
        json.dump(obj, f)
    os.replace(fn + '.tmp', fn)


def main(args: argparse.Namespace) -> None:
    rules = args.rules.split(',')
    unknown = set(rules) - set(RULES)
    if unknown:
        raise SystemExit(f'unknown rules {sorted(unknown)}; choose from {RULES}')
    agents = [(r, s) for r in rules for s in range(args.seed0, args.seed0 + args.n_seeds)]
    A = len(agents)
    t_limit = gym.make(args.env).spec.max_episode_steps
    envs = [gym.make(args.env).unwrapped for _ in agents]
    lo, hi = envs[0].action_space.low, envs[0].action_space.high
    d, na, H = envs[0].observation_space.shape[0], envs[0].action_space.shape[0], args.hidden
    ep_ret, ep_len = np.zeros(A), np.zeros(A, np.int64)

    nets_a, nets_c, fbs_a, fbs_c = init_agents(agents, d, na, H, args.sparsity, args.legacy_init)
    stack = lambda L: jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *L)
    zeros = lambda t: jax.tree.map(jnp.zeros_like, t)
    pa, pc = stack(nets_a), stack(nets_c)
    st = dict(pa=pa, pc=pc, ea=zeros(pa), ma=zeros(pa), ec=zeros(pc), mc=zeros(pc), fba=stack(fbs_a), fbc=stack(fbs_c),
              rule=jnp.asarray([RULES.index(r) for r, _ in agents]),
              lr_pi=jnp.full(A, args.lr_policy, jnp.float32), lr_v=jnp.full(A, args.lr_value, jnp.float32))
    if args.shadow:
        st.update(esa=zeros(pa), msa=zeros(pa), esc=zeros(pc), msc=zeros(pc))
    hp = dict(gamma=args.gamma, lam=args.lam, beta=args.beta, entropy=args.entropy, kp_decay=args.kp_decay,
              shadow=args.shadow)
    vstep, vact = make_step(na, hp)
    obs_stat, rew_stat, rew_trace = RunningStat(A, (d,)), RunningStat(A, ()), np.zeros(A)

    def normalise(o, mask=None):
        obs_stat.update(o, mask)
        return ((o - obs_stat.mean) / np.sqrt(obs_stat.var() + 1e-8)).astype(np.float32)

    S = normalise(np.stack([e.reset(seed=agents[i][1])[0] for i, e in enumerate(envs)]))
    key = jax.random.PRNGKey(args.seed0 + 4242)
    key, k = jax.random.split(key)
    act = np.asarray(vact(st['pa'], jnp.asarray(S), jax.random.split(k, A)))
    log = dict(args=vars(args), agents=agents, names=[r for r, _ in agents],
               returns=[[] for _ in range(A)], steps=[[] for _ in range(A)], diag=[])
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f'{args.env}_{args.tag}.json')
    ckpt = fn.replace('.json', '_ckpt.pkl')
    start_step = 0
    if args.resume and os.path.exists(ckpt):
        # restore learner, normalisers and log; the environments restart at an episode boundary
        with open(ckpt, 'rb') as f:
            C = pickle.load(f)
        st = jax.tree.map(jnp.asarray, C['st'])
        obs_stat.mean, obs_stat.p, obs_stat.n = C['obs_mean'], C['obs_p'], C['obs_n']
        rew_stat.mean, rew_stat.p, rew_stat.n = C['rew_mean'], C['rew_p'], C['rew_n']
        key, start_step = jnp.asarray(C['key']), int(C['step'])
        if os.path.exists(fn):
            with open(fn) as f:
                old = json.load(f)
            for i in range(A):   # drop episodes logged after the checkpoint
                keep = [j for j, s_ in enumerate(old['steps'][i]) if s_ <= start_step]
                log['returns'][i] = [old['returns'][i][j] for j in keep]
                log['steps'][i] = [old['steps'][i][j] for j in keep]
            log['diag'] = [x for x in old['diag'] if x['step'] <= start_step]
        S = normalise(np.stack([e.reset(seed=agents[i][1] + 10_000 + start_step)[0] for i, e in enumerate(envs)]))
        key, k = jax.random.split(key)
        act = np.asarray(vact(st['pa'], jnp.asarray(S), jax.random.split(k, A)))
        print(f'resumed from step {start_step}', flush=True)

    align_sum, ucos_sum, n_diag = np.zeros((A, 4)), np.zeros((A, 12)), 0
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        O2, R = np.zeros((A, d)), np.zeros(A)
        term, trunc = np.zeros(A, bool), np.zeros(A, bool)
        act_c = np.clip(act, lo, hi)
        for i, e in enumerate(envs):
            o2, r, te, tr, _ = e.step(act_c[i])
            ep_ret[i] += r
            ep_len[i] += 1
            tr = tr or ep_len[i] >= t_limit
            O2[i], R[i], term[i], trunc[i] = o2, r, te, tr
            if te or tr:
                log['returns'][i].append(float(ep_ret[i]))
                log['steps'][i].append(step)
                ep_ret[i], ep_len[i] = 0.0, 0
        S2 = normalise(O2)
        # reward scaling by the running std of the discounted return
        rew_trace = rew_trace * args.gamma + R
        rew_stat.update(rew_trace)
        Rs = (R / np.sqrt(rew_stat.var() + 1e-8)).astype(np.float32)
        done = term | trunc
        rew_trace[done] = 0.0
        S_next = S2.copy()
        if done.any():
            O3 = O2.copy()
            for i in np.nonzero(done)[0]:
                O3[i] = envs[i].reset()[0]
            S_next[done] = normalise(O3, done)[done]
        key, k = jax.random.split(key)
        st, a_next, delta, align, ucos = vstep(st, jnp.asarray(S), jnp.asarray(act), jnp.asarray(Rs), jnp.asarray(S2),
                                               jnp.asarray(term, jnp.float32), jnp.asarray(done), jnp.asarray(S_next),
                                               jax.random.split(k, A))
        act, S = np.asarray(a_next), S_next
        if step % 100 == 0:
            align_sum += np.asarray(align)
            ucos_sum += np.asarray(ucos)
            n_diag += 1
        if step % args.log_every == 0 or step == args.steps:
            # align: cosine of hidden-layer errors to exact backprop [critic L1, L2, actor L1, L2];
            # ucos (with --shadow): update cosines and sign agreements against a shadow backprop learner
            log['diag'].append(dict(step=step, align=(align_sum / max(n_diag, 1)).round(4).tolist(),
                                    ucos=(ucos_sum / max(n_diag, 1)).round(4).tolist()))
            align_sum[:], ucos_sum[:], n_diag = 0, 0, 0
            recent = {}
            for i, r in enumerate(log['names']):
                if log['returns'][i]:
                    recent.setdefault(r, []).append(np.mean(log['returns'][i][-20:]))
            recent = {r: round(float(np.mean(v)), 1) for r, v in recent.items()}
            print(f'[{args.env}] step {step}  {(step - start_step) / (time.time() - t0):.0f} steps/s  '
                  f'last-20 return {recent}', flush=True)
            save_json(log, fn)
        if args.ckpt_every > 0 and step % args.ckpt_every == 0 and step < args.steps:
            C = dict(step=step, st=jax.tree.map(np.asarray, st), obs_mean=obs_stat.mean, obs_p=obs_stat.p,
                     obs_n=obs_stat.n, rew_mean=rew_stat.mean, rew_p=rew_stat.p, rew_n=rew_stat.n, key=np.asarray(key))
            with open(ckpt + '.tmp', 'wb') as f:
                pickle.dump(C, f, protocol=4)
            os.replace(ckpt + '.tmp', ckpt)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    np.savez_compressed(fn.replace('.json', '_params.npz'),
                        **{f'{n}_{k}': np.asarray(v) for n in ('pa', 'pc', 'fba', 'fbc') for k, v in st[n].items()},
                        obs_mean=obs_stat.mean, obs_var=obs_stat.var())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--env', default='Hopper-v5', help='Gymnasium MuJoCo task')
    p.add_argument('--rules', default='bp,bpsc,fsign', help=f'comma-separated, from {RULES}')
    p.add_argument('--seed0', type=int, default=0, help='first seed')
    p.add_argument('--n_seeds', type=int, default=3, help='seeds seed0 .. seed0 + n_seeds - 1 for every rule')
    p.add_argument('--steps', type=int, default=2_000_000, help='environment steps per agent')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--sparsity', type=float, default=0.95, help='initial weight sparsity (ERK)')
    p.add_argument('--lr_policy', type=float, default=1e-4)
    p.add_argument('--lr_value', type=float, default=2e-4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.8, help='trace decay lambda')
    p.add_argument('--beta', type=float, default=0.99995, help='decay of the running maximum in the optimiser')
    p.add_argument('--entropy', type=float, default=0.01)
    p.add_argument('--kp_decay', type=float, default=0.0, help='weight decay of W and K for kp / fkp')
    p.add_argument('--shadow', action='store_true', help='log update agreement with a shadow backprop learner')
    p.add_argument('--legacy_init', action='store_true',
                   help='initialise as the earlier revision did (no draw of K); needed to repeat some reported runs exactly')
    p.add_argument('--log_every', type=int, default=20_000)
    p.add_argument('--ckpt_every', type=int, default=100_000, help='checkpoint every N steps (0: never)')
    p.add_argument('--resume', action='store_true', help='resume from <out>/<env>_<tag>_ckpt.pkl if present')
    p.add_argument('--out', default='runs')
    p.add_argument('--tag', default='run')
    return p.parse_args(argv)


if __name__ == '__main__':
    main(parse_args())
