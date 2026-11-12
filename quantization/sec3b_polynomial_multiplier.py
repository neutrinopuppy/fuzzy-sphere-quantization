"""Section III B: fuzzy sphere with a trainable polynomial Poisson multiplier, N = 2 .. 20, d_lambda = 1 .. 10 (Fig. 2b, c).

    python quantization/sec3b_polynomial_multiplier.py [--workers 8]     # -> quantization/results/sec3b.json

lambda(z) = sum_{alpha=0}^{d} lambda_alpha z^alpha,  lambda_0 = lambda_d = 1 fixed, lambda_1 .. lambda_{d-1} trainable,
lambda^(ini)(z) = 1 + z^d.   Loss (Eq. 22):  L_tot = c_ac L_ac + c_irr L_irr + c_pos L_pos,  c_ac = 1, c_irr = 1e-18,
c_pos = 1, residuals of Eq. (23), L_pos = (1/M) sum_m ReLU(-lambda(z_m)) on M = 100 uniformly spaced z_m in [-R, R].
Every (N, d_lambda) is an independent optimization initialized at the conventional fuzzy sphere (Eq. 16),
250 training steps including the initial evaluation.  Expected: solutions (small L_ac) only for N <= d_lambda.
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
from fuzzy import Polynomial, Problem, fuzzy_sphere, levenberg_marquardt, STEPS

N_MAX, D_MAX = 20, 10
OUT = Path(__file__).resolve().parent / 'results' / 'sec3b.json'


def run(args):
    n, d = args
    pb = Problem(n, Polynomial(d)); x, y, v, h = fuzzy_sphere(n)
    p, hist = levenberg_marquardt(pb, pb.encode(x, y, v, h, np.zeros(d - 1)), STEPS)
    x, y, v, h, c = pb.decode(p)
    return dict(N=n, d=d, final_L_ac=hist[-1]['L_ac'], final_L_irr=hist[-1]['L_irr'], final_L_pos=hist[-1]['L_pos'],
                hbar=h, coefficients=pb.mult.coefficients(c).tolist(), spectrum=sorted(v.tolist()),
                history={k: [r[k] for r in hist] for k in ('L_ac', 'L_irr', 'L_pos', 'L_tot', 'hbar')})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--nmax', type=int, default=N_MAX); ap.add_argument('--dmax', type=int, default=D_MAX)
    ap.add_argument('--workers', type=int, default=8); a = ap.parse_args()
    jobs = [(n, d) for n in range(a.nmax, 1, -1) for d in range(1, a.dmax + 1)]
    with Pool(a.workers) as pool: runs = pool.map(run, jobs, chunksize=1)
    runs.sort(key=lambda r: (r['N'], r['d']))
    OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(steps=STEPS, runs=runs)))
    table = {(r['N'], r['d']): r['final_L_ac'] for r in runs}
    print('log10 of the final L_ac   (rows N, columns d_lambda)')
    print('N\\d ' + ''.join(f'{d:6d}' for d in range(1, a.dmax + 1)))
    for n in range(2, a.nmax + 1):
        print(f'{n:3d} ' + ''.join(f'{np.log10(max(table[(n, d)], 1e-300)):6.0f}' for d in range(1, a.dmax + 1)))


if __name__ == '__main__':
    main()
