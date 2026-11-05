"""Checks of the hand-written backward pass (slc/network.py) and of one learner step (slc/train.py) against jax.grad.
Run from the repository root: python -m tests.test_backward (CPU, under a minute). Exits non-zero on any failure.

Reference for feedback that differs from the forward weights: y = W @ sg(x) + S @ x - sg(S @ x) has the forward value
W x, the gradient g x^T to W and the gradient S^T g to x, so jax.grad through a forward pass built from it is an
independent reference for "backprop through S instead of W". All feedback matrices below are drawn independently of
the forward weights, so a rule that read W by mistake would fail. A second guard rescales |W3| with its signs fixed:
the hidden-layer updates of rules that do not read W3 must not change.
"""
import sys

import jax
import jax.numpy as jnp
import numpy as np

from slc import train
from slc.network import LN_EPS, RULES, SIGN_CONSTRAINED, backward, forward, init_feedback, init_net, lrelu

FAILS: list[str] = []
sg = jax.lax.stop_gradient


def report(name: str, ok: bool, detail: str) -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name:64s} {detail}")
    if not ok:
        FAILS.append(name)


def check(name: str, a, b, tol: float = 1e-4) -> None:
    """Worst relative error over the leaves; NaN, or a sign flip on any non-negligible entry, fails."""
    a = a if isinstance(a, dict) else {'x': a}
    b = b if isinstance(b, dict) else {'x': b}
    worst, wk = 0.0, None
    for k in b:
        x, y = np.asarray(a[k], np.float64), np.asarray(b[k], np.float64)
        err = np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-6) if y.size else 0.0
        if not np.isfinite(err) or not np.all(np.isfinite(x)):
            err = np.inf
        big = np.abs(y) > 1e-3 * np.max(np.abs(y)) if y.size else np.zeros(0, bool)
        if y.size and np.any(np.sign(x[big]) != np.sign(y[big])):
            err = np.inf
        if err >= worst:
            worst, wk = err, k
    report(name, worst <= tol, f'worst rel err {worst:.1e} ({wk})')


def differs(name: str, a, b, keys, tol: float = 1e-3) -> None:
    """The opposite check: a rule that reads a weight must change when that weight changes."""
    d = max(float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k])))) / max(float(np.max(np.abs(np.asarray(b[k])))), 1e-6)
            for k in keys)
    report(name, d > tol, f'rel change {d:.1e}')


def ln_full(a):
    mu = a.mean()
    return (a - mu) / jnp.sqrt(((a - mu) ** 2).mean() + LN_EPS)


def ln_const(a):
    """LayerNorm with mean and std held constant: its gradient is the diagonal 1 / sigma."""
    mu = sg(a.mean())
    return (a - mu) / sg(jnp.sqrt(((a - a.mean()) ** 2).mean() + LN_EPS))


def through(W, S, v):
    """Forward W v, backward to v through S."""
    return W @ sg(v) + S @ v - sg(S @ v)


def forward_fb(p, x, S2, S3, ln):
    h1 = lrelu(ln(p['W1'] @ x + p['c1']))
    h2 = lrelu(ln(through(p['W2'], S2, h1) + p['c2']))
    return through(p['W3'], S3, h2) + p['c3']


def dfa_objective(p, x, g, B1, B2, ln):
    """jax.grad of this is DFA: layer l learns from <B_l g, h_l>, layers cut apart, readout exact."""
    h1 = lrelu(ln(p['W1'] @ x + p['c1']))
    h2 = lrelu(ln(p['W2'] @ sg(h1) + p['c2']))
    return g @ (p['W3'] @ sg(h2) + p['c3']) + (B2 @ g) @ h2 + (B1 @ g) @ h1


def small_net(seed: int, d: int = 11, H: int = 16, na: int = 3):
    rng = np.random.default_rng(seed)
    p = {k: jnp.asarray(v) for k, v in init_net(d, H, [na, na], 0.3, rng).items()}
    p = dict(p, c1=jnp.asarray(rng.normal(size=H), jnp.float32), c2=jnp.asarray(rng.normal(size=H), jnp.float32))
    fb = {k: jnp.asarray(v) for k, v in init_feedback(H, 2 * na, rng).items()}
    pm = lambda *s: jnp.asarray(np.where(rng.random(s) < 0.5, 1.0, -1.0), jnp.float32)
    nrm = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
    fb.update(S2=pm(H, H), S3=pm(2 * na, H), K2=nrm(H, H), K3=nrm(2 * na, H))
    x = jnp.asarray(rng.normal(size=d), jnp.float32)
    g = jnp.asarray(rng.normal(size=2 * na), jnp.float32)
    return p, fb, x, g, rng


def gradient_tests(seed: int) -> None:
    p, fb, x, g, rng = small_net(seed)
    G = lambda name, q=p: backward(q, forward(q, x)[1], g, fb, jnp.int32(RULES.index(name)))[0]
    ad = lambda F: jax.grad(F)(p)
    G_bp = ad(lambda q: forward(q, x)[0] @ g)
    check(f'[{seed}] bp == jax.grad', G('bp'), G_bp)
    check(f'[{seed}] bpsc == jax.grad', G('bpsc'), G_bp)
    for name, S2, S3 in (('fsign', fb['S2'], fb['S3']), ('kp', fb['K2'], fb['K3']), ('fkp', fb['K2'], fb['K3']),
                         ('sign', jnp.sign(p['W2']), jnp.sign(p['W3']))):
        check(f'[{seed}] {name} == jax.grad through its own feedback', G(name), ad(lambda q: forward_fb(q, x, S2, S3, ln_full) @ g))
    check(f'[{seed}] dfa == jax.grad of the DFA objective (diagonal LN)', G('dfa'),
          ad(lambda q: dfa_objective(q, x, g, fb['B1'], fb['B2'], ln_const)))
    zero_hidden = dict(G_bp, **{k: 0 * G_bp[k] for k in ('W1', 'c1', 'W2', 'c2')})
    check(f'[{seed}] frozen: hidden grads zero, readout exact', G('frozen'), zero_hidden)
    p3 = dict(p, W3=p['W3'] * jnp.asarray(rng.uniform(0.2, 3.0, p['W3'].shape), jnp.float32))
    hid = ('W1', 'c1', 'W2', 'c2')
    for name in ('fsign', 'dfa', 'kp', 'fkp'):
        check(f'[{seed}] {name} hidden grads blind to |W3|', {k: G(name, p3)[k] for k in hid}, {k: G(name)[k] for k in hid})
    differs(f'[{seed}] bp hidden grads do read |W3| (guard)', G('bp', p3), G('bp'), hid)


def learner_step_test(seed: int = 0, d: int = 11, H: int = 128, na: int = 3) -> None:
    """train.init_agents and the jit + vmap learner of train.make_step against single-agent maths over two normal
    steps and one episode-end step: traces, running maximum, update, sign projection, Kolen-Pollack feedback update.
    H = 128 as in the runs (at H = 16 the 95 % sparse W1 is all zero and LayerNorm degenerates)."""
    report('SIGN_CONSTRAINED == {fsign, bpsc, fkp}', set(SIGN_CONSTRAINED) == {'fsign', 'bpsc', 'fkp'}, str(SIGN_CONSTRAINED))
    hp = dict(gamma=0.99, lam=0.8, beta=0.99995, entropy=0.01, kp_decay=0.0, shadow=False)
    vstep, _ = train.make_step(na, hp)
    rng = np.random.default_rng(seed)
    rules = ['bp', 'bpsc', 'fsign', 'sign', 'dfa', 'kp', 'fkp']
    agents = [(r, 3) for r in rules]   # one seed: every rule must share init, feedback and sign pattern
    nets_a, nets_c, fbs_a, fbs_c = train.init_agents(agents, d, na, H, 0.95)
    for k in ('S2', 'S3', 'B1', 'B2'):
        same = all(np.array_equal(fbs_a[i][k], fbs_a[0][k]) and np.array_equal(fbs_c[i][k], fbs_c[0][k]) for i in range(len(rules)))
        report(f'init_agents: {k} identical for every rule of a seed', same, '')
    for i, r in enumerate(rules):
        inside = all(np.all(net[k] * fb['S' + k[1]] >= 0) for net, fb in ((nets_a[i], fbs_a[i]), (nets_c[i], fbs_c[i]))
                     for k in ('W2', 'W3'))
        same_w = all(np.array_equal(np.abs(nets_a[i][k]), np.abs(nets_a[0][k])) for k in ('W1', 'W2', 'W3'))
        report(f'init_agents: {r} shares |W|' + (' and starts inside S' if r in SIGN_CONSTRAINED else ''),
               same_w and (inside or r not in SIGN_CONSTRAINED), '')
    stack = lambda L: jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *L)
    zeros = lambda t: jax.tree.map(jnp.zeros_like, t)
    A = len(rules)
    pa, pc = stack(nets_a), stack(nets_c)
    st = dict(pa=pa, pc=pc, ea=zeros(pa), ma=zeros(pa), ec=zeros(pc), mc=zeros(pc), fba=stack(fbs_a), fbc=stack(fbs_c),
              rule=jnp.asarray([RULES.index(r) for r in rules]), lr_pi=jnp.full(A, 1e-2, jnp.float32),
              lr_v=jnp.full(A, 2e-2, jnp.float32))
    f32 = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
    decay = hp['gamma'] * hp['lam']
    # Each step is checked from the learner's own state before it. At the first step the running maximum makes every
    # update lr * sign(delta * e); where e is at rounding level its sign is decided by rounding, so weights are
    # compared only where |e| >= 1e-4 max|e| of the matrix.
    for t, reset in enumerate((False, False, True)):
        s, a = jnp.broadcast_to(f32(d), (A, d)), jnp.broadcast_to(f32(na), (A, na))
        s2, s_next = jnp.broadcast_to(f32(d), (A, d)), jnp.broadcast_to(f32(d), (A, d))
        r = jnp.broadcast_to(f32(1), (A,))
        prev = st
        st, *_ = vstep(st, s, a, r, s2, jnp.zeros(A), jnp.full(A, reset), s_next, jax.random.split(jax.random.PRNGKey(t), A))
        tag = f"[step {t}{' reset' if reset else ''}]"
        for i, rule in enumerate(rules):
            R = {k: jax.tree.map(lambda z: z[i], prev[k]) for k in ('pa', 'pc', 'ea', 'ma', 'ec', 'mc', 'fba', 'fbc')}
            rid = jnp.int32(RULES.index(rule))
            out_c, cc = forward(R['pc'], s[i])
            v2, _ = forward(R['pc'], s2[i])
            delta = r[i] + hp['gamma'] * v2[0] - out_c[0]
            out_a, ca = forward(R['pa'], s[i])
            logp = lambda o: jnp.sum(-0.5 * ((a[i] - o[:na]) / jax.nn.softplus(o[na:])) ** 2 - jnp.log(jax.nn.softplus(o[na:])))
            ent = lambda o: jnp.sum(jnp.log(jax.nn.softplus(o[na:])))
            g_a = jax.grad(lambda o: logp(o) + hp['entropy'] * jnp.sign(delta) * ent(o))(out_a)
            Ga, _ = backward(R['pa'], ca, g_a, R['fba'], rid)
            Gc, _ = backward(R['pc'], cc, jnp.ones(1), R['fbc'], rid)
            for net, e, m, fbk, Gx, lr in (('pa', 'ea', 'ma', 'fba', Ga, 1e-2), ('pc', 'ec', 'mc', 'fbc', Gc, 2e-2)):
                E = jax.tree.map(lambda e_, g_: decay * e_ + g_, R[e], Gx)
                M = jax.tree.map(lambda m_, e_: jnp.maximum(hp['beta'] * m_, jnp.abs(e_) * jnp.abs(delta)), R[m], E)
                P = jax.tree.map(lambda p_, e_, m_: p_ + lr * delta * e_ / (m_ + 1e-8), R[net], E, M)
                if rule in SIGN_CONSTRAINED:
                    P = dict(P, **{k: jnp.where(P[k] * R[fbk]['S' + k[1]] < 0, 0.0, P[k]) for k in ('W2', 'W3')})
                got = {k: jax.tree.map(lambda z: z[i], st[k]) for k in (net, e, m, fbk)}
                check(f'{tag} {rule} {m} (running max) == single-agent maths', got[m], M, tol=2e-4)
                if reset:
                    check(f'{tag} {rule} {e} reset to zero', got[e], jax.tree.map(jnp.zeros_like, E), tol=0.0)
                else:
                    check(f'{tag} {rule} {e} (trace) == single-agent maths', got[e], E, tol=2e-4)
                keep = {k: jnp.abs(E[k]) >= 1e-4 * jnp.max(jnp.abs(E[k])) for k in E}
                check(f'{tag} {rule} {net} (weights) == single-agent maths', {k: jnp.where(keep[k], got[net][k], 0.0) for k in E},
                      {k: jnp.where(keep[k], P[k], 0.0) for k in E}, tol=2e-4)
                if rule in ('kp', 'fkp'):
                    K = {'K' + k[1]: R[fbk]['K' + k[1]] + got[net][k] - R[net][k] for k in ('W2', 'W3')}
                    if rule == 'fkp':
                        K = {k: jnp.where(v * R[fbk]['S' + k[1]] < 0, 0.0, v) for k, v in K.items()}
                    check(f'{tag} {rule} feedback K follows the change of W', {k: got[fbk][k] for k in K}, K, tol=1e-5)
                else:
                    check(f'{tag} {rule} feedback K unchanged', {k: got[fbk][k] for k in ('K2', 'K3')},
                          {k: R[fbk][k] for k in ('K2', 'K3')}, tol=0.0)
    W_S = lambda i: min(float(jnp.min(st[n][k][i] * st['fb' + n[1]]['S' + k[1]][i])) for n in ('pa', 'pc') for k in ('W2', 'W3'))
    for i, rule in enumerate(rules):
        if rule in SIGN_CONSTRAINED:
            report(f'sign pattern kept after 3 steps: {rule}', W_S(i) >= 0, f'min W*S {W_S(i):.1e}')


if __name__ == '__main__':
    for s in range(3):
        gradient_tests(s)
    learner_step_test()
    print('ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)
