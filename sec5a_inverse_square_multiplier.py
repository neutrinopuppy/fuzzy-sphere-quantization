"""Section V A: fuzzy sphere with the non-polynomial Poisson multiplier lambda(z) = lambda_0 + lambda_{-2} z^{-2} (Eq. 45),
N = 2 .. 20, eta = lambda_{-2}/lambda_0 in {1, 1e-1, 1e-2, 1e-3, 1e-4}, R = lambda_0 = 1 (Fig. 4a1).

    python paper_code/sec5a_inverse_square_multiplier.py [--workers 8]     # -> results/sec5a.json

Loss (Eq. 17):  L_tot = c_ac L_ac + c_irr L_irr,  c_ac = 1, c_irr = 1e-18, residuals of Eq. (23) with
lambda(Z) Z = Z + eta Z^{-1}  and  S_lambda[A] = A + eta Z^{-1} A Z^{-1}  (Eq. 46).
Initialization: the conventional fuzzy sphere (Eq. 16) plus Gaussian noise of standard deviation 5% on every initial
matrix element (so that Z is not singular at odd N); 250 training steps including the initial evaluation.
Expected (Sec. V B): no solution at odd N (App. J 1); none for eta > R^2/(7 + 4 sqrt3) = 0.0718 (Eq. 47);
hbar_N >= (3 sqrt3 / 2) sqrt(eta) / lambda_0 (Eq. 48).
A result counts as a solution when L_ac < 1e-18 (Sec. V B).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzzy import InverseSquare, Problem, fuzzy_sphere, levenberg_marquardt, STEPS

N_MAX, ETAS, NOISE, SEED = 20, (1.0, 1e-1, 1e-2, 1e-3, 1e-4), 0.05, 0
OUT = Path(__file__).resolve().parent / 'results' / 'sec5a.json'


def run(args):
    n, eta, seed = args
    rng = np.random.default_rng(seed); pb = Problem(n, InverseSquare(eta)); x, y, v, h = fuzzy_sphere(n)
    scale = NOISE * 2 * h                                   # 5% of the level spacing 2 hbar_N of the fuzzy-sphere matrices
    for M in (x, y):
        g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); M += scale * (g + g.conj().T) / 2
    v = v + scale * rng.standard_normal(n)
    p, hist = levenberg_marquardt(pb, pb.encode(x, y, v, h), STEPS)
    x, y, v, h, _ = pb.decode(p)
    return dict(N=n, eta=eta, final_L_ac=hist[-1]['L_ac'], final_L_irr=hist[-1]['L_irr'], hbar=h,
                hbar_lower_bound=1.5 * np.sqrt(3) * np.sqrt(eta), spectrum=sorted(v.tolist()),
                history={k: [r[k] for r in hist] for k in ('L_ac', 'L_irr', 'L_tot', 'hbar')})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nmax', type=int, default=N_MAX); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    jobs = [(n, eta, SEED + 1000 * i + n) for n in range(a.nmax, 1, -1) for i, eta in enumerate(ETAS)]
    with Pool(a.workers) as pool: runs = pool.map(run, jobs, chunksize=1)
    runs.sort(key=lambda r: (-r['eta'], r['N']))
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(seed=SEED, noise=NOISE, steps=STEPS, runs=runs)))
    table = {(r['eta'], r['N']): r for r in runs}
    print('log10 of the final L_ac   (rows eta, columns N);  * marks L_ac < 1e-18')
    print('eta    ' + ''.join(f'{n:5d}' for n in range(2, a.nmax + 1)))
    for eta in ETAS:
        print(f'{eta:<7g}' + ''.join(f"{np.log10(max(table[(eta, n)]['final_L_ac'], 1e-300)):4.0f}"
                                     f"{'*' if table[(eta, n)]['final_L_ac'] < 1e-18 else ' '}" for n in range(2, a.nmax + 1)))


if __name__ == '__main__':
    main()
