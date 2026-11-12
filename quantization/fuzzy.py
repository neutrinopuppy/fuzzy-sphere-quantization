"""Shared machinery of the numerical procedure of "A Numerical Approach to Quantization":
the residual matrices, the losses, the irreducibility criterion and the Levenberg-Marquardt minimizer.
The radius is R = 1 and lambda_0 = 1 throughout.

Casimir and commutators (Eqs. 10, 11, 15):
    X^2 + Y^2 + Z^2 - R^2 I = 0
    [X, Y] = i 2 hbar lambda(Z) Z
    [Y, Z] = i 2 hbar S_lambda[X],   [Z, X] = i 2 hbar S_lambda[Y]
Weyl map (Eq. 12, 13): S_lambda[A] = d/dt Lambda(Z + t A)|_{t=0},  Lambda' = lambda.  With Z = diag(v) this is
    (S_lambda[A])_jk = lbar(v_j, v_k) A_jk,   lbar(u, w) = (Lambda(u) - Lambda(w)) / (u - w),  lbar(u, u) = lambda(u)   (Eq. 29)
Residual matrices (Eq. 19 for lambda = 1, Eq. 23 in general):
    A_C  = X^2 + Y^2 + Z^2 - I
    A_XY = [X, Y]/(i 2 hbar) - lambda(Z) Z
    A_YZ = [Y, Z]/(i 2 hbar) - S_lambda[X]
    A_ZX = [Z, X]/(i 2 hbar) - S_lambda[Y]
Losses (Eqs. 17, 18, 20, 21, 22):
    L_ac  = ||A_C||^2 + ||A_XY||^2 + ||A_YZ||^2 + ||A_ZX||^2          (Hilbert-Schmidt norms)
    Gamma_ab = sum_{M in X,Y,Z} Tr([M, T_a]^dagger [M, T_b]),  T_a = generalized Gell-Mann basis, a = 1..N^2-1
    L_irr = Tr(Gamma^-1)             (+inf when Gamma is not positive definite, i.e. the triplet is reducible, App. B)
    L_pos = (1/M) sum_m ReLU(-lambda(z_m)),  z_m = M uniformly spaced points in [-R, R]   (Sec. III B only)
    L_tot = c_ac L_ac + c_irr L_irr + c_pos L_pos
Trainable parameters p = [coordinates of X in a Hermitian basis (N^2), same for Y (N^2), v (N), log hbar, coefficients].
"""
import numpy as np

STEPS = 250            # training steps per optimization, including the initial evaluation
C_AC, C_IRR, C_POS = 1.0, 1e-18, 1.0


# ------------------------------------------------------------------------------------------------ bases
def hermitian_basis(n):
    """Orthonormal basis B_k of Hermitian N x N matrices, Tr(B_k B_l) = delta_kl: e_ii, then (e_ij + e_ji)/sqrt2, i(e_ij - e_ji)/sqrt2."""
    B = np.zeros((n * n, n, n), complex); k = 0
    for i in range(n):
        B[k, i, i] = 1; k += 1
    for i in range(n):
        for j in range(i + 1, n):
            B[k, i, j] = B[k, j, i] = 1 / np.sqrt(2); k += 1
            B[k, i, j] = 1j / np.sqrt(2); B[k, j, i] = -1j / np.sqrt(2); k += 1
    return B


def gell_mann_basis(n):
    """Generalized Gell-Mann matrices T_a (Eq. 20): traceless, Hermitian, Tr(T_a T_b) = delta_ab, a = 1..N^2-1."""
    T = np.zeros((n * n - 1, n, n), complex)
    for k in range(1, n):                                   # diagonal ones: diag(1, .., 1, -k, 0, ..)/sqrt(k(k+1))
        d = np.zeros(n); d[:k] = 1; d[k] = -k; T[k - 1] = np.diag(d / np.sqrt(k * (k + 1)))
    T[n - 1:] = hermitian_basis(n)[n:]                      # off-diagonal ones
    return T


def fuzzy_sphere(n):
    """The conventional fuzzy sphere with lambda = 1, R = 1:  X = 2 hbar J_x, Y = 2 hbar J_y, Z = 2 hbar J_z in the
    basis where Z is diagonal (ascending), with  hbar_N = R / (lambda_0 sqrt(N^2 - 1))  (Eq. 16).  Returns X, Y, v, hbar."""
    j = (n - 1) / 2; m = np.arange(n) - j
    Jp = np.diag(np.sqrt((j - m[:-1]) * (j + m[:-1] + 1)), -1)      # J_+ |m> = sqrt((j-m)(j+m+1)) |m+1>
    h = 1 / np.sqrt(n * n - 1)
    return (h * (Jp + Jp.T)).astype(complex), h * (Jp - Jp.T) / 1j, 2 * h * m, h


# ------------------------------------------------------------------------------------------------ Poisson multipliers
class Constant:
    """lambda(z) = 1  (Sec. III A):  lambda(Z) Z = Z,  S_lambda[A] = A."""
    ncoef = 0

    def evaluate(self, v, c):
        n = v.size
        return dict(lamz=v.copy(), dlamz=np.ones(n), ker=np.ones((n, n)), dker_u=np.zeros((n, n)), dker_v=np.zeros((n, n)))


class Polynomial:
    """lambda(z) = sum_{alpha=0}^{d} lambda_alpha z^alpha with lambda_0 = lambda_d = 1 fixed and lambda_1..lambda_{d-1}
    trainable (Sec. III B).  lbar(u, w) = sum_alpha lambda_alpha (u^{alpha+1} - w^{alpha+1}) / ((alpha+1)(u - w))
    = sum_alpha lambda_alpha/(alpha+1) sum_{beta=0}^{alpha} u^beta w^{alpha-beta}   (Eq. 13 with Z diagonal)."""

    def __init__(self, d, m_pos=100):
        self.d = d; self.ncoef = d - 1; self.grid = np.linspace(-1, 1, m_pos)

    def coefficients(self, c):
        return np.concatenate([[1.0], c, [1.0]])            # lambda_0, ..., lambda_d

    def evaluate(self, v, c):
        a = self.coefficients(c); n = v.size; u, w = v[:, None], v[None, :]
        bank = np.zeros((self.d + 1, n, n)); dker_u = np.zeros((n, n)); dker_v = np.zeros((n, n))
        for k in range(self.d + 1):
            for b in range(k + 1):
                bank[k] += u ** b * w ** (k - b) / (k + 1)
                if b: dker_u += a[k] * b * u ** (b - 1) * w ** (k - b) / (k + 1)
                if k > b: dker_v += a[k] * (k - b) * u ** b * w ** (k - b - 1) / (k + 1)
        P = np.polynomial.polynomial
        lam, dlam = P.polyval(v, a), P.polyval(v, P.polyder(a))
        return dict(lamz=lam * v, dlamz=lam + v * dlam, ker=np.einsum('k,kij->ij', a, bank), dker_u=dker_u, dker_v=dker_v,
                    dlamz_dc=[v ** (k + 1) for k in range(1, self.d)], dker_dc=[bank[k] for k in range(1, self.d)])

    def positivity(self, c):
        """L_pos = (1/M) sum_m ReLU(-lambda(z_m)) and its gradient with respect to the trainable coefficients."""
        lam = np.polynomial.polynomial.polyval(self.grid, self.coefficients(c)); active = lam < 0
        L = np.mean(np.maximum(-lam, 0)); grad = np.array([-np.mean(active * self.grid ** k) for k in range(1, self.d)])
        return L, grad


class InverseSquare:
    """lambda(z) = lambda_0 + lambda_{-2} z^{-2} = 1 + eta z^{-2}  (Eq. 45, eta = lambda_{-2}/lambda_0 with lambda_0 = 1):
    lambda(Z) Z = Z + eta Z^{-1},  S_lambda[A] = A + eta Z^{-1} A Z^{-1}  (Eq. 46),  lbar(u, w) = 1 + eta/(u w)  (Eq. K1)."""
    ncoef = 0

    def __init__(self, eta):
        self.eta = eta

    def evaluate(self, v, c):
        inv = 1 / v; e = self.eta
        return dict(lamz=v + e * inv, dlamz=1 - e * inv ** 2, ker=1 + e * np.outer(inv, inv),
                    dker_u=-e * np.outer(inv ** 2, inv), dker_v=-e * np.outer(inv, inv ** 2))


# ------------------------------------------------------------------------------------------------ optimization problem
class Problem:
    def __init__(self, n, multiplier, c_ac=C_AC, c_irr=C_IRR, c_pos=C_POS):
        self.n, self.mult = n, multiplier; self.c_ac, self.c_irr, self.c_pos = c_ac, c_irr, c_pos
        self.B, self.T = hermitian_basis(n), gell_mann_basis(n)
        self.npar = 2 * n * n + n + 1 + multiplier.ncoef

    # mapping parameters <-> (X, Y, v, hbar, coefficients)
    def encode(self, x, y, v, h, c=()):
        q = self.n * self.n
        return np.concatenate([np.real(np.einsum('kij,ij->k', self.B.conj(), x)), np.real(np.einsum('kij,ij->k', self.B.conj(), y)),
                               np.asarray(v, float), [np.log(h)], np.asarray(c, float)])

    def decode(self, p):
        q, n = self.n * self.n, self.n
        x = np.tensordot(p[:q], self.B, axes=(0, 0)); y = np.tensordot(p[q:2 * q], self.B, axes=(0, 0))
        return x, y, p[2 * q:2 * q + n].copy(), float(np.exp(p[2 * q + n])), p[2 * q + n + 1:].copy()

    @staticmethod
    def pack(blocks):
        """Real vector r with r.r = sum of the squared Hilbert-Schmidt norms of the blocks."""
        return np.concatenate([np.concatenate([b.real.ravel(), b.imag.ravel()]) for b in blocks])

    # residuals r (Eq. 18: L_ac = r.r) and their Jacobian
    def residuals(self, p):
        x, y, v, h, c = self.decode(p); n = self.n; Z = np.diag(v).astype(complex); m = self.mult.evaluate(v, c)
        comm = np.array([(x @ y - y @ x) / (2j * h), (y @ Z - Z @ y) / (2j * h), (Z @ x - x @ Z) / (2j * h)])
        A = np.array([x @ x + y @ y + Z @ Z - np.eye(n),             # A_C
                      comm[0] - np.diag(m['lamz']),                  # A_XY = [X,Y]/(i2hbar) - lambda(Z) Z
                      comm[1] - m['ker'] * x,                        # A_YZ = [Y,Z]/(i2hbar) - S_lambda[X]
                      comm[2] - m['ker'] * y])                       # A_ZX = [Z,X]/(i2hbar) - S_lambda[Y]
        aux = dict(x=x, y=y, v=v, h=h, c=c, Z=Z, comm=comm, A=A, **m)
        return self.pack(A), aux

    def jacobian(self, aux):
        """d r / d p, column by column (blocks dA_C, dA_XY, dA_YZ, dA_ZX):
          X along B:   B X + X B,  [B,Y]/(i2h),  -lbar o B,         [Z,B]/(i2h)
          Y along B:   B Y + Y B,  [X,B]/(i2h),  [B,Z]/(i2h),       -lbar o B
          v_l (E=e_ll): 2 v_l E,  -(lambda(v) v)'_l E,  [Y,E]/(i2h) - dlbar o X,  [E,X]/(i2h) - dlbar o Y
          log hbar:    0,  -[X,Y]/(i2h),  -[Y,Z]/(i2h),  -[Z,X]/(i2h)
          lambda_k:    0,  -diag(v^{k+1}),  -(d lbar/d lambda_k) o X,  -(d lbar/d lambda_k) o Y"""
        n, q = self.n, self.n * self.n; x, y, Z, h, v, ker = (aux[k] for k in ('x', 'y', 'Z', 'h', 'v', 'ker'))
        D = np.zeros((self.npar, 4, n, n), complex); B = self.B
        D[:q, 0] = B @ x + x @ B;               D[q:2 * q, 0] = B @ y + y @ B
        D[:q, 1] = (B @ y - y @ B) / (2j * h);  D[q:2 * q, 1] = (x @ B - B @ x) / (2j * h)
        D[:q, 2] = -ker * B;                    D[q:2 * q, 2] = (B @ Z - Z @ B) / (2j * h)
        D[:q, 3] = (Z @ B - B @ Z) / (2j * h);  D[q:2 * q, 3] = -ker * B
        for l in range(n):
            E = np.zeros((n, n)); E[l, l] = 1
            dk = np.zeros((n, n)); dk[l, :] += aux['dker_u'][l, :]; dk[:, l] += aux['dker_v'][:, l]
            D[2 * q + l, 0] = 2 * v[l] * E
            D[2 * q + l, 1] = -aux['dlamz'][l] * E
            D[2 * q + l, 2] = (y @ E - E @ y) / (2j * h) - dk * x
            D[2 * q + l, 3] = (E @ x - x @ E) / (2j * h) - dk * y
        D[2 * q + n, 1:] = -aux['comm']
        for k in range(self.mult.ncoef):
            D[2 * q + n + 1 + k, 1] = -np.diag(aux['dlamz_dc'][k])
            D[2 * q + n + 1 + k, 2] = -aux['dker_dc'][k] * x
            D[2 * q + n + 1 + k, 3] = -aux['dker_dc'][k] * y
        return np.concatenate([D[:, i].real.reshape(self.npar, -1).T if part == 0 else D[:, i].imag.reshape(self.npar, -1).T
                               for i in range(4) for part in (0, 1)], axis=0)

    # irreducibility: Gamma (Eq. 20), L_irr = Tr(Gamma^-1) (Eq. 21) and its gradient
    def irreducibility(self, aux):
        """Returns (L_irr, grad L_irr).  L_irr = +inf (grad = nan) when Gamma is not positive definite.
        dL_irr = -Tr(Gamma^-1 dGamma Gamma^-1)  gives  grad_M L_irr = Hermitian part of 2 sum_a [T_a, K_a],
        K_a = sum_b (Gamma^-2)_ab [M, T_b];  the hbar and coefficient components vanish."""
        nt = self.n * self.n - 1; mats = [aux['x'], aux['y'], aux['Z']]
        C = [np.stack([(M @ t - t @ M).ravel() for t in self.T], axis=1) for M in mats]    # C_M[:, a] = vec([M, T_a])
        G = sum(np.real(cm.conj().T @ cm) for cm in C); G = (G + G.T) / 2
        try:
            Linv = np.linalg.inv(np.linalg.cholesky(G))                                   # Gamma positive definite
        except np.linalg.LinAlgError:
            return np.inf, np.full(self.npar, np.nan)
        Ginv = Linv.T @ Linv; L_irr = float(np.trace(Ginv)); inv2 = Ginv @ Ginv
        grad = np.zeros(self.npar); q = self.n * self.n
        for i, cm in enumerate(C):
            K = cm @ inv2.T; g = sum(2 * (t @ K[:, a].reshape(self.n, self.n) - K[:, a].reshape(self.n, self.n) @ t)
                                     for a, t in enumerate(self.T)); g = (g + g.conj().T) / 2
            if i < 2: grad[i * q:(i + 1) * q] = np.real(np.einsum('kij,ij->k', self.B.conj(), g))
            else: grad[2 * q:2 * q + self.n] = np.real(np.diag(g))
        return L_irr, grad

    # total loss
    def loss(self, p):
        """L_tot = c_ac L_ac + c_irr L_irr + c_pos L_pos; +inf for a non-finite point, hbar <= 0, singular Z with
        lambda ~ z^-2, or a reducible triplet (the irreducibility check of the search)."""
        out = dict(L_tot=np.inf, L_ac=np.inf, L_irr=np.inf, L_pos=0.0, r=None, aux=None, grad_extra=None)
        if not np.all(np.isfinite(p)): return out
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            r, aux = self.residuals(p)
        if not np.all(np.isfinite(r)): return out
        L_irr, g_irr = self.irreducibility(aux)
        if not np.isfinite(L_irr): return out
        L_pos, g_pos = 0.0, np.zeros(self.npar)
        if self.mult.ncoef and self.c_pos:
            L_pos, g = self.mult.positivity(aux['c']); g_pos[-self.mult.ncoef:] = g
        L_tot = self.c_ac * r @ r + self.c_irr * L_irr + self.c_pos * L_pos
        return dict(L_tot=L_tot, L_ac=float(r @ r), L_irr=L_irr, L_pos=L_pos, r=r, aux=aux,
                    grad_extra=self.c_irr * g_irr + self.c_pos * g_pos)


# ------------------------------------------------------------------------------------------------ Levenberg-Marquardt optimization
def levenberg_marquardt(problem, p0, steps=STEPS, s0=1e-3):
    """Minimize L_tot by Levenberg-Marquardt.  
    Training step 1 is the evaluation of the initial point; every later
    step solves  (2 c_ac J^T J + s I) delta = -(2 c_ac J^T r + grad(c_irr L_irr + c_pos L_pos)),  J = dr/dp,
    and accepts p + delta only if L_tot decreases and the point passes the irreducibility check (L_tot finite);
    s /= 10 after an accepted step, s *= 10 after a rejected one.  Returns (p, history)."""
    p = np.asarray(p0, float).copy(); cur = problem.loss(p)
    assert np.isfinite(cur['L_tot']), 'initial point is reducible or non-finite'
    s = s0; eye = np.eye(problem.npar); history = []
    def refresh(cur):
        J = problem.jacobian(cur['aux'])
        return 2 * problem.c_ac * J.T @ J, 2 * problem.c_ac * J.T @ cur['r'] + cur['grad_extra']
    H, g = refresh(cur)
    for step in range(1, steps + 1):
        history.append(dict(step=step, L_tot=cur['L_tot'], L_ac=cur['L_ac'], L_irr=cur['L_irr'], L_pos=cur['L_pos'], hbar=cur['aux']['h']))
        if step == steps: break
        try: delta = np.linalg.solve(H + s * eye, -g)
        except np.linalg.LinAlgError: delta = np.zeros(problem.npar)
        trial = problem.loss(p + delta)
        if trial['L_tot'] < cur['L_tot']:
            p, cur = p + delta, trial; H, g = refresh(cur); s = max(s / 10, 1e-18)
        else:
            s = min(s * 10, 1e10)
    return p, history
