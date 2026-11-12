"""Section III A: fuzzy sphere with the constant Poisson multiplier lambda(z) = 1, R = 1, N = 2 .. 40.

    python paper_code/sec3a_constant_multiplier.py            # -> results/sec3a.json

Loss (Eq. 17):  L_tot = c_ac L_ac + c_irr L_irr,  c_ac = 1, c_irr = 1e-18, residuals of Eq. (19).
Expected answer (Eq. 16):  hbar_N = R / (lambda_0 sqrt(N^2 - 1)).

Initialization:
  N = 2:  hbar^(ini) = 1;  X, Y = symmetrized complex Gaussian matrices, Z = diagonal Gaussian,
          all three rescaled by one factor so that  Tr[X^2 + Y^2 + Z^2] = 1.
  N > 2:  hbar^(ini)_N = hbar^(fin)_{N-1},
          X^(ini)_N = [[X^(fin)_{N-1}, v_X], [v_X^dagger, 0]],  same for Y, with independent complex Gaussian
          v_X, v_Y of Euclidean norm N^{-1/2};   Z^(ini)_N = diag(Z^(fin)_{N-1}, 1).
250 training steps per dimension including the initial evaluation, fixed random seed.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fuzzy import Constant, Problem, levenberg_marquardt, STEPS

N_MAX, SEED = 40, 0
OUT = Path(__file__).resolve().parent / 'results' / 'sec3a.json'


def main(n_max=N_MAX, seed=SEED):
    rng = np.random.default_rng(seed); runs = []; prev = None
    for n in range(2, n_max + 1):
        pb = Problem(n, Constant())
        if prev is None:
            g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); x = (g + g.conj().T) / 2
            g = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)); y = (g + g.conj().T) / 2
            v = rng.standard_normal(n)
            s = np.sqrt(np.real(np.trace(x @ x + y @ y)) + v @ v)          # Tr[X^2 + Y^2 + Z^2] = 1
            x, y, v, h = x / s, y / s, v / s, 1.0
        else:
            x0, y0, v0, h = prev
            x = np.zeros((n, n), complex); y = np.zeros((n, n), complex); x[:-1, :-1] = x0; y[:-1, :-1] = y0
            for M in (x, y):
                u = rng.standard_normal(n - 1) + 1j * rng.standard_normal(n - 1); u *= n ** -0.5 / np.linalg.norm(u)
                M[:-1, -1] = u; M[-1, :-1] = u.conj()
            v = np.append(v0, 1.0)
        p, hist = levenberg_marquardt(pb, pb.encode(x, y, v, h), STEPS)
        x, y, v, h, _ = pb.decode(p); prev = (x, y, v, h)
        exact = 1 / np.sqrt(n * n - 1)
        runs.append(dict(N=n, hbar=h, hbar_exact=exact, relative_difference=abs(h / exact - 1),
                         final_L_ac=hist[-1]['L_ac'], final_L_irr=hist[-1]['L_irr'], spectrum=sorted(v.tolist()),
                         history={k: [r[k] for r in hist] for k in ('L_ac', 'L_irr', 'L_tot', 'hbar')}))
        print(f"N={n:2d}  hbar={h:.15f}  Eq.(16)={exact:.15f}  rel.diff={runs[-1]['relative_difference']:.1e}  "
              f"L_ac={hist[-1]['L_ac']:.1e}", flush=True)
        OUT.parent.mkdir(exist_ok=True); OUT.write_text(json.dumps(dict(seed=seed, steps=STEPS, runs=runs)))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else N_MAX)
