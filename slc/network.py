"""Networks, credit-assignment rules and the streaming optimiser used by train.py.

Actor and critic are MLPs with two hidden layers of the form Linear -> LayerNorm (no gain or bias) -> leaky ReLU,
followed by a linear readout. A rule decides how the error at the readout reaches the two hidden layers; the readout
weights always receive their exact local gradient.

Rules
  bp      exact backprop.
  bpsc    exact backprop on the sign-constrained network (the control for fsign: same constraint, exact feedback).
  fsign   fixed-sign variant of sign: W2 and W3 keep a fixed sign pattern S; errors travel back through S^T.
  sign    sign-concordant feedback: errors travel back through sign(W)^T of the current, unconstrained weights.
  dfa     direct feedback alignment: hidden layer l receives B_l g_out through a fixed random matrix B_l.
  kp      Kolen-Pollack update: feedback matrices K receive the same changes as W (see train.py); with kp_decay = 0,
          K - W keeps its random initial value.
  fkp     Kolen-Pollack on the sign-constrained network; K keeps the signs of S.
  frozen  hidden layers do not learn (random-feature baseline).

dfa divides each neuron's error by its own layer's LayerNorm std only (diagonal LayerNorm backward). All other rules
use the full LayerNorm backward, which needs the layer-wide mean of the errors.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

RULES = ['bp', 'dfa', 'sign', 'frozen', 'fsign', 'bpsc', 'kp', 'fkp']
SIGN_CONSTRAINED = ('fsign', 'bpsc', 'fkp')   # W2 and W3 are kept inside their sign pattern S
KOLEN_POLLACK = ('kp', 'fkp')                 # feedback matrices K are learned
SLOPE, LN_EPS = 0.01, 1e-5


def rule_ids(names: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(RULES.index(n) for n in names)


# Initialisation: sparse uniform weights with Erdos-Renyi-kernel layer sparsities, as in the stream-x reference code.
def erk_sparsities(shapes: list[tuple[int, int]], target_sparsity: float) -> list[float]:
    infos = [(fo, fi, fo * fi) for (fo, fi) in shapes]
    target_nz = (1 - target_sparsity) * sum(n for _, _, n in infos)
    dense, remaining, eps = set(), set(range(len(infos))), 0.0
    while remaining:
        nz_dense = sum(infos[i][2] for i in dense)
        sum_fans = sum(infos[i][0] + infos[i][1] for i in remaining)
        eps = (target_nz - nz_dense) / max(sum_fans, 1e-12)
        new = [i for i in remaining if eps * (infos[i][0] + infos[i][1]) / infos[i][2] >= 1.0]
        if not new:
            break
        dense.update(new)
        remaining.difference_update(new)
    return [1 - min(1.0 if i in dense else eps * (fo + fi) / n, 1.0) for i, (fo, fi, n) in enumerate(infos)]


def sparse_layer(fo: int, fi: int, sparsity: float, rng: np.random.Generator) -> np.ndarray:
    w = rng.uniform(-1, 1, (fo, fi)) * math.sqrt(1.0 / fi)
    nz = int(math.ceil(sparsity * fi))
    for r in range(fo):
        w[r, rng.permutation(fi)[:nz]] = 0
    return w.astype(np.float32)


def init_net(d_in: int, hidden: int, heads: list[int], sparsity: float, rng: np.random.Generator) -> dict:
    shapes = [(hidden, d_in), (hidden, hidden)] + [(o, hidden) for o in heads]
    sp = erk_sparsities(shapes, sparsity)
    ws = [sparse_layer(fo, fi, s, rng) for (fo, fi), s in zip(shapes, sp)]
    W3 = np.concatenate(ws[2:], 0)
    z = lambda n: np.zeros(n, np.float32)
    return dict(W1=ws[0], c1=z(hidden), W2=ws[1], c2=z(hidden), W3=W3, c3=z(W3.shape[0]))


def init_feedback(hidden: int, n_out: int, rng: np.random.Generator) -> dict:
    """Fixed random DFA matrices B1, B2. R2, R3 are drawn only to keep the random stream of later draws unchanged."""
    f = lambda *s: rng.standard_normal(s).astype(np.float32)
    return dict(B1=f(hidden, n_out), B2=f(hidden, n_out), R3=f(hidden, n_out) / math.sqrt(n_out),
                R2=f(hidden, hidden) / math.sqrt(hidden))


class RunningStat:
    """Per-agent running mean and variance (Welford), used for observation and reward scaling."""

    def __init__(self, n_agents: int, shape: tuple):
        self.mean = np.zeros((n_agents,) + shape)
        self.p = np.zeros((n_agents,) + shape)
        self.n = np.zeros(n_agents, np.int64)

    def update(self, x: np.ndarray, mask: np.ndarray | None = None) -> None:
        mask = np.ones(len(self.n), bool) if mask is None else mask
        first, rest = mask & (self.n == 0), mask & (self.n > 0)
        self.mean[first] = x[first]
        self.p[first] = 0
        self.n[first] = 1
        if rest.any():
            n = self.n[rest] + 1
            shp = (-1,) + (1,) * (x.ndim - 1)
            m_new = self.mean[rest] + (x[rest] - self.mean[rest]) / n.reshape(shp)
            self.p[rest] = self.p[rest] + (x[rest] - self.mean[rest]) * (x[rest] - m_new)
            self.mean[rest] = m_new
            self.n[rest] = n

    def var(self) -> np.ndarray:
        shp = (-1,) + (1,) * (self.mean.ndim - 1)
        v = self.p / np.maximum(self.n - 1, 1).reshape(shp)
        return np.where((self.n < 2).reshape(shp), 1.0, v)


# Single-agent maths; train.py vmaps it over agents.
def lrelu(x):
    return jnp.where(x > 0, x, SLOPE * x)


def dlrelu(x):
    return jnp.where(x > 0, 1.0, SLOPE)


def ln_fwd(a):
    mu = a.mean()
    sig = jnp.sqrt(((a - mu) ** 2).mean() + LN_EPS)
    return (a - mu) / sig, sig


def ln_bwd(gn, n, sig):
    return (gn - gn.mean() - n * (gn * n).mean()) / sig


def forward(p: dict, x):
    a1 = p['W1'] @ x + p['c1']
    n1, s1 = ln_fwd(a1)
    h1 = lrelu(n1)
    a2 = p['W2'] @ h1 + p['c2']
    n2, s2 = ln_fwd(a2)
    h2 = lrelu(n2)
    out = p['W3'] @ h2 + p['c3']
    return out, (x, n1, s1, h1, n2, s2, h2, dlrelu(n1), dlrelu(n2))


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return (a @ b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12)


def backward(p: dict, cache: tuple, g_out, fb: dict, rule):
    """Pseudo-gradients of F with respect to the parameters, given dF/d(out) = g_out, under rule index `rule`.
    fb holds the feedback matrices (B1, B2, S2, S3, K2, K3). Also returns the cosine between the rule's hidden-layer
    errors and the exact ones (layer 1, layer 2), which equals the cosine of the weight updates."""
    x, n1, s1, h1, n2, s2, h2, d1, d2 = cache
    gW3 = jnp.outer(g_out, h2)

    def chain(gh2, back1, diag):
        gn2 = gh2 * d2
        ga2 = gn2 / s2 if diag else ln_bwd(gn2, n2, s2)
        gh1 = back1(ga2)
        gn1 = gh1 * d1
        ga1 = gn1 / s1 if diag else ln_bwd(gn1, n1, s1)
        return ga1, ga2

    bp = chain(p['W3'].T @ g_out, lambda ga2: p['W2'].T @ ga2, False)
    dfa = chain(fb['B2'] @ g_out, lambda ga2: fb['B1'] @ g_out, True)
    sign = chain(jnp.sign(p['W3']).T @ g_out, lambda ga2: jnp.sign(p['W2']).T @ ga2, False)
    fsign = chain(fb['S3'].T @ g_out, lambda ga2: fb['S2'].T @ ga2, False)
    kp = chain(fb['K3'].T @ g_out, lambda ga2: fb['K2'].T @ ga2, False)
    zero = (jnp.zeros_like(bp[0]), jnp.zeros_like(bp[1]))
    per_rule = dict(bp=bp, dfa=dfa, sign=sign, frozen=zero, fsign=fsign, bpsc=bp, kp=kp, fkp=kp)
    ga1 = jnp.stack([per_rule[r][0] for r in RULES])[rule]
    ga2 = jnp.stack([per_rule[r][1] for r in RULES])[rule]
    G = dict(W1=jnp.outer(ga1, x), c1=ga1, W2=jnp.outer(ga2, h1), c2=ga2, W3=gW3, c3=g_out)
    return G, jnp.stack([cos(ga1, bp[0]), cos(ga2, bp[1])])


def stream_step(p, e, m, G, delta, reset, lr, decay, beta):
    """Streaming TD(lambda) step with per-synapse normalisation: trace e, running maximum m of |e * delta|, and the
    update lr * delta * e / m. Traces are cleared at episode ends."""
    e = jax.tree.map(lambda e_, g: decay * e_ + g, e, G)
    m = jax.tree.map(lambda m_, e_: jnp.maximum(beta * m_, jnp.abs(e_) * jnp.abs(delta)), m, e)
    p = jax.tree.map(lambda p_, e_, m_: p_ + lr * delta * e_ / (m_ + 1e-8), p, e, m)
    e = jax.tree.map(lambda e_: jnp.where(reset, 0.0, e_), e)
    return p, e, m
