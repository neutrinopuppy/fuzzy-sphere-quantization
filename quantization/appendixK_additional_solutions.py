"""Appendix K: every solution of the fuzzy sphere with lambda(z) = 1 + eta z^{-2} at even N = 2 .. 24, eta = 1e-2 and 1e-4,
listed exactly from the chain equations and then used as the initialization of the training (Fig. 4a2, b).

    python quantization/appendixK_additional_solutions.py [--workers 8]     # -> quantization/results/appendixK.json

Theta_N(z) = R^2 - z^2 + 2 hbar_N (lambda_0 z + lambda_{-2}/z)   (Eq. J3);  W W^dag = Theta_N(Z), W^dag W = Theta_N(-Z)  (Eq. J4).
The non-zero elements of W = X + iY visit the spectrum of Z in one connected sequence p_0 -> p_1 -> ... -> p_{N-1} (Eq. K3), with
    Theta_N(p_0) = 0,     p_{k+1} - p_k = 2 hbar_N (lambda_0 + lambda_{-2}/(p_k p_{k+1})),     Theta_N(-p_{N-1}) = 0     (Eq. K4)
and the pairing p_{N-1-k} = -p_k (App. J 1).  For a given hbar_N the first condition is a cubic for p_0, each connection a
quadratic for p_{k+1}, and the pairing closes the sequence at the middle connection p_{N/2-1} -> -p_{N/2-1}; the value of
hbar_N at which it closes is a root of  F(hbar_N) = -p_{N/2-1} - hbar_N (1 - eta/p_{N/2-1}^2).  Following every branch of
roots while scanning hbar_N lists all solutions: N/2 sequences (1, 3, ..., N-1 crossings of z = 0) with two solutions each
as eta -> 0, i.e. N solutions at every even N.
Initialization (Eq. K5):  hbar^(ini) = hbar_N,  Z^(ini) = diag(p sorted ascending),
    W^(ini) = sum_k w_k |p_{k+1}><p_k|,  w_k = sqrt(Theta_N(p_{k+1})),   X^(ini) = (W + W^dag)/2,  Y^(ini) = (W - W^dag)/(2i).
Training as Sec. III A: c_ac = 1, c_irr = 1e-18, Z diagonal, 250 training steps including the initial evaluation.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')  # allow work with multiplle threads since this code takes the longest to run (to find all solutions and not just one)
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzzy import InverseSquare, Problem, levenberg_marquardt, STEPS

ETAS, N_MAX = (1e-2, 1e-4), 24
OUT = Path(__file__).resolve().parent / 'results' / 'appendixK.json'
np.seterr(all='ignore')  # the scan probes p -> 0 on purpose so there might be divide-by-zero errors


def theta(z, h, eta):
    return 1 - z * z + 2 * h * (z + eta / z)


# ---------------------------------------------------------------- the connected sequence at a given hbar
def first_levels(h, eta):
    """Real roots of Theta_N(p) = 0, i.e. p^3 - 2 hbar p^2 - p - 2 hbar eta = 0, keyed by (sign, rank)."""
    r = np.roots([1, -2 * h, -1, -2 * h * eta]); return keyed(r[np.abs(r.imag) < 1e-12 * max(1, np.abs(r).max())].real)


def next_levels(p, h, eta):
    """Real roots q of the connection  q^2 - (p + 2 hbar) q - 2 hbar eta / p = 0  (Eq. K4), keyed by (sign, rank)."""
    b = p + 2 * h; disc = b * b + 8 * h * eta / p
    return keyed([(b - np.sqrt(disc)) / 2, (b + np.sqrt(disc)) / 2]) if disc >= 0 else []


def keyed(roots):
    neg = sorted(r for r in roots if r < 0); pos = sorted(r for r in roots if r > 0)
    return [((-1, i), r) for i, r in enumerate(neg)] + [((1, i), r) for i, r in enumerate(pos)]


def closure(p, h, eta):
    """Middle connection p_{N/2-1} -> -p_{N/2-1} of Eq. (K4):  F = -p - hbar (1 - eta/p^2) = 0."""
    return -p - h * (1 - eta / (p * p))


def branches(m, h, eta):
    """Every sequence p_0 .. p_{m-1} that Eq. (K4) allows at this hbar, keyed by the chosen roots."""
    out = {}; stack = [((k,), [p]) for k, p in first_levels(h, eta)]
    while stack:
        key, p = stack.pop()
        if len(p) == m: out[key] = p; continue
        stack += [(key + (k,), p + [q]) for k, q in next_levels(p[-1], h, eta)]
    return out


def follow(key, h, eta):
    """The sequence with this key at hbar = h, or None where the branch does not exist."""
    r = dict(first_levels(h, eta)); p = [r[key[0]]] if key[0] in r else None
    for k in key[1:]:
        if p is None: return None
        q = dict(next_levels(p[-1], h, eta)); p = p + [q[k]] if k in q else None
    return p


# ---------------------------------------------------------------- all roots of F(hbar) by scanning hbar
def system(x, signs, eta):
    """Eq. (K4) in log variables x = (log|p_0| .. log|p_{m-1}|, log hbar) with the pairing applied."""
    p = signs * np.exp(x[:-1]); h = np.exp(x[-1]); p = np.concatenate([p, -p[::-1]])
    links = [p[k + 1] - p[k] - 2 * h * (1 + eta / (p[k] * p[k + 1])) for k in range(p.size // 2)]
    return np.array(links + [theta(p[0], h, eta)])


def newton(p, h, eta, iters=50):
    """Polish a root of Eq. (K4) by Newton's method with a finite-difference Jacobian."""
    p = np.asarray(p, float); signs = np.sign(p); x = np.concatenate([np.log(np.abs(p)), [np.log(h)]])
    for _ in range(iters):
        f = system(x, signs, eta)
        if not np.all(np.isfinite(f)) or np.abs(f).max() < 1e-15: break
        J = np.column_stack([(system(x + 1e-7 * e, signs, eta) - f) / 1e-7 for e in np.eye(x.size)])
        try: x = x - np.linalg.solve(J, f)
        except np.linalg.LinAlgError: break
    return signs * np.exp(x[:-1]), float(np.exp(x[-1])), float(np.abs(system(x, signs, eta)).max())


def enumerate_solutions(n, eta, n_h=6000):
    """Every solution (p_0 .. p_{N-1}, hbar_N) of Eq. (K4).  Scan hbar on a log grid; along every branch a sign change of F
    between neighbouring grid points is bisected; a pair of branches that differ by the two roots of one quadratic (a fold)
    and carry F of opposite signs is a root where the branches merge, and Newton starts from their mean."""
    m = n // 2; hs = np.exp(np.linspace(np.log(0.02), np.log(3 / np.sqrt(eta)), n_h)); seeds = []
    evaluate = lambda h: {k: (closure(p[-1], h, eta), p) for k, p in branches(m, h, eta).items()}
    prev, cur = {}, evaluate(hs[0])
    for i in range(1, n_h + 1):
        nxt = evaluate(hs[i]) if i < n_h else {}
        for k, (F, p) in nxt.items(): # sign change along a branch
            if k in cur and np.isfinite(F) and np.isfinite(cur[k][0]) and cur[k][0] * F < 0:
                a, b, Fa = hs[i - 1], hs[i], cur[k][0]
                for _ in range(60):
                    c = np.sqrt(a * b); pc = follow(k, c, eta)
                    if pc is None: break
                    if Fa * closure(pc[-1], c, eta) <= 0: b = c
                    else: a, Fa = c, closure(pc[-1], c, eta)
                else: seeds.append((pc, c))
        for ka, (Fa, pa) in cur.items(): # fold: branches that appear or merge
            for j, (sg, rk) in enumerate(ka):
                kb = ka[:j] + ((sg, 1),) + ka[j + 1:]
                if rk == 0 and kb in cur and Fa * cur[kb][0] < 0 and not ((ka in nxt and kb in nxt) and (ka in prev and kb in prev)):
                    seeds.append(((np.asarray(pa) + np.asarray(cur[kb][1])) / 2, hs[i - 1]))
        prev, cur = cur, nxt
    sols = {}
    for p, h in seeds:
        p, h, res = newton(p, h, eta); seq = np.concatenate([p, -p[::-1]]); th = theta(seq, h, eta)
        if not (np.all(np.isfinite(seq)) and res < 1e-10 and len({round(abs(v), 10) for v in seq}) == m and np.all(th[1:] > 1e-13)):
            continue # not a root, degenerate levels, or a reducible sequence
        key = (round(h, 7),) + tuple(np.round(np.sort(np.abs(p)), 7))
        sols.setdefault(key, dict(N=n, eta=eta, hbar=h, sequence=seq.tolist(),
                                  crossings=int(np.sum(np.sign(seq[1:]) != np.sign(seq[:-1])))))
    return sorted(sols.values(), key=lambda s: s['hbar'])


# ---------------------------------------------------------------- initialization (Eq. K5) and training
def initial_triplet(seq, h, eta):
    n = seq.size; order = np.argsort(seq); v = seq[order]; pos = np.empty(n, int); pos[order] = np.arange(n)
    W = np.zeros((n, n), complex); th = theta(seq, h, eta)
    for k in range(n - 1): W[pos[k + 1], pos[k]] = np.sqrt(th[k + 1])      # w_k = sqrt(Theta_N(p_{k+1}))
    return (W + W.conj().T) / 2, (W - W.conj().T) / 2j, v


def run(sol):
    n, eta, h = sol['N'], sol['eta'], sol['hbar']; seq = np.array(sol['sequence'])
    pb = Problem(n, InverseSquare(eta)); x, y, v = initial_triplet(seq, h, eta)
    p, hist = levenberg_marquardt(pb, pb.encode(x, y, v, h), STEPS)
    x, y, v, h, _ = pb.decode(p)
    return dict(**sol, initial_L_ac=hist[0]['L_ac'], final_L_ac=hist[-1]['L_ac'], final_L_irr=hist[-1]['L_irr'],
                final_hbar=h, final_spectrum=sorted(v.tolist()), history={k: [r[k] for r in hist] for k in ('L_ac', 'L_irr', 'L_tot', 'hbar')})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nmax', type=int, default=N_MAX); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--eta', type=float, nargs='+', default=ETAS); a = ap.parse_args()
    cells = [(n, eta) for eta in a.eta for n in range(a.nmax, 1, -2)]
    with Pool(a.workers) as pool:
        sols = [s for cell in pool.starmap(enumerate_solutions, cells) for s in cell]
        runs = pool.map(run, sorted(sols, key=lambda s: -s['N']), chunksize=1)
    runs.sort(key=lambda r: (-r['eta'], r['N'], r['hbar']))
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(steps=STEPS, runs=runs)))
    for eta in a.eta:
        rs = [r for r in runs if r['eta'] == eta]
        print(f'eta = {eta:g}: {len(rs)} solutions of Eq. (K4) at even N = 2..{a.nmax}, '
              f'{sum(r["final_L_ac"] < 1e-18 for r in rs)} end the training with L_ac < 1e-18')
        for n in range(2, a.nmax + 1, 2):
            rn = [r for r in rs if r['N'] == n]
            print(f'  N={n:2d}: {len(rn):2d} solutions, crossings {sorted(r["crossings"] for r in rn)}, hbar '
                  + ' '.join(f'{r["hbar"]:.4f}' for r in rn) + '   final L_ac ' + ' '.join(f'{r["final_L_ac"]:.0e}' for r in rn))


if __name__ == '__main__':
    main()
