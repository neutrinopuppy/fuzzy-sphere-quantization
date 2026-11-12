"""Fig. 3: fuzzy sphere axial spectra. Panels are saved as PNG to figures/fig_3/

    python figures/fig3.py

Data is read from the Sec. III B grid of the quantization scripts
    quantization/results/sec3b.json  sec3b_polynomial_multiplier.py --dmax 20      N = 2..20, d_lambda = 1..20

    Thus please run code for 3b to generate results for figures before running this code.

Notation:
    a_lambda   lambda(z) for N = 6, d_lambda = 6, 7, 8 (three rows, one lambda scale)
    a_H        H(z) for the same rows, with the spectrum of Z_6 and the sign changes of H
    b_d7, b_d8 H(z) for N = 8, d_lambda = 7 (unsolved) and 8 (solved), with the spectrum of Z_8
    c          the spectra of Z_N for N = d_lambda = 2..20 against the lambda = 1 spectra and the sign changes of H
    c_key      the key of (c)

    H(z) = z lambda''(z) + 3 lambda'(z)                                                          (Eq. 35)
    sign changes of H: real roots of H of odd multiplicity in (-R, R)
    lambda = 1 spectrum: N equally spaced values on [-R sqrt((N-1)/(N+1)), +R sqrt((N-1)/(N+1))]     (Eq. 16 ladder, R = 1)
    H(z) is drawn on a symmetric-log axis (linear within |H| < 0.1); gold fill H > 0, violet fill H < 0.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

P = np.polynomial.polynomial                               # coefficient arrays in ascending powers, as in the JSON
DATA = Path(__file__).resolve().parent.parent / 'quantization' / 'results' / 'sec3b.json'
OUT = Path(__file__).resolve().parent / 'fig_3'

DPI, R, LINTHRESH = 300, 1.0, 0.1
NS = list(range(2, 21))
INK, ZERO, NAVY, POS, NEG, LIME, REF = '#1f2938', '#6b7280', '#360aa6', '#e7bc58', '#6d1eb8', '#a4f76a', '#8b9099'
RAMP = LinearSegmentedColormap.from_list('paper_key', [                                 # the Fig. 2 key, N = 2..20
    '#74fbfd', '#6fedfd', '#6ae0fc', '#67d2fb', '#66c5fa', '#67b6f9', '#6aa7f9', '#6f9af8', '#768df8', '#807ff7',
    '#8972f7', '#9365f7', '#9f59f6', '#aa4ef7', '#b743f7', '#c33bf7', '#d036f7', '#de33f7', '#e932f7'])
LAM_HALF_WIDTHS = (0.02, 0.1, 0.2, 0.4, 0.5, 1.0, 2.0, 4.0)                           # lambda axis half-heights with round ticks
TICK_SCALE = 2.0                                                                        # multiplies every tick-number size
_dec = FuncFormatter(lambda v, pos: f'{v:g}')
_dec_no0 = FuncFormatter(lambda v, pos: '' if v == 0 else f'{v:g}')


def set_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans') if f in names), 'DejaVu Sans')
    plt.rcParams.update({'font.family': font, 'font.size': 9, 'mathtext.fontset': 'cm', 'axes.linewidth': 0.8,
                         'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 3, 'ytick.major.size': 3,
                         'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5})


def ramp(k): return RAMP(0.3 + 0.7 * np.clip((k - 2) / 18, 0, 1))                     # colour of the integer k (N or d_lambda)


# ---------------------------------------------------------------------------------------------------- the polynomials
def H_poly(coeffs):
    """H = z lambda'' + 3 lambda' (Eq. 35) as an ascending coefficient array."""
    a = np.asarray(coeffs, float); return P.polyadd(P.polymul([0.0, 1.0], P.polyder(a, 2)), 3.0 * P.polyder(a, 1))


def H_sign_changes(coeffs):
    """Real roots of H in (-R, R) of odd multiplicity."""
    H = H_poly(coeffs)
    if np.all(np.abs(H) < 1e-14): return np.array([])
    roots = P.polyroots(H); real = np.sort(roots[np.abs(roots.imag) < 1e-9].real); real = real[(real > -R) & (real < R)]
    out = []
    for r in real:
        if out and abs(r - out[-1][0]) < 1e-6: out[-1][1] += 1
        else: out.append([r, 1])
    return np.array([r for r, m in out if m % 2 == 1])


def exact_spectrum(N):
    e = R * np.sqrt((N - 1) / (N + 1)); return np.linspace(-e, e, N)


# ---------------------------------------------------------------------------------------------------- panels
def _axes_style(ax, xlabels):
    ax.set_xlim(-R, R); ax.set_xticks([-R, -R / 2, 0, R / 2, R]); ax.xaxis.set_major_formatter(_dec)
    ax.tick_params(length=3, width=0.8)
    if not xlabels: ax.tick_params(labelbottom=False)


def lam_half_width(cs):
    z = np.linspace(-R, R, 801)
    need = 1.15 * max(np.abs(P.polyval(z, np.asarray(c['coefficients'], float)) - 1).max() for c in cs)
    return next((w for w in LAM_HALF_WIDTHS if w >= need), LAM_HALF_WIDTHS[-1])


def panel_lambda(ax, case, h, xlabels):
    """lambda(z) in the d_lambda colour of the Fig. 2 key; y axis [1 - h, 1 + h]."""
    z = np.linspace(-R, R, 801)
    ax.plot(z, P.polyval(z, np.asarray(case['coefficients'], float)), color=ramp(case['d']), lw=1.8, zorder=3, solid_capstyle='round')
    ax.set_ylim(1 - h, 1 + h); ax.set_yticks([1 - h, 1 - h / 2, 1, 1 + h / 2, 1 + h])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f'{round(y, 4):g}'))
    _axes_style(ax, xlabels)


def panel_H(ax, case, xlabels):
    """H(z) on a symmetric-log axis: gold fill H > 0, violet fill H < 0, the spectrum of Z_N as vertical lines in the
    d_lambda colour, the sign changes of H as lime crosses."""
    z = np.linspace(-R, R, 1601); H = P.polyval(z, H_poly(case['coefficients']))
    v = np.asarray(case['spectrum'], float); zeros = H_sign_changes(case['coefficients'])
    hmax = max(10.0, 1.3 * np.abs(H).max())
    ax.fill_between(z, 0, H, where=H >= 0, interpolate=True, color=POS, alpha=0.45, lw=0, zorder=1)
    ax.fill_between(z, 0, H, where=H <= 0, interpolate=True, color=NEG, alpha=0.22, lw=0, zorder=1)
    for vj in v: ax.axvline(vj, color=ramp(case['d']), lw=1.6, zorder=2)
    ax.axhline(0, color=ZERO, lw=0.8, zorder=2)
    ax.plot(z, H, color=NAVY, lw=1.4, zorder=3)
    if zeros.size: ax.plot(zeros, np.zeros_like(zeros), 'X', ms=7.5, mfc=LIME, mec=INK, mew=0.7, zorder=5, clip_on=False)
    ax.set_yscale('symlog', linthresh=LINTHRESH, linscale=1 / np.log(10)); ax.set_ylim(-hmax, hmax)
    kmin, dec = int(np.round(np.log10(LINTHRESH))), int(np.floor(np.log10(hmax)))
    ax.set_yticks(sorted({0.0} | {s * 10.0 ** k for k in range(kmin, dec + 1) for s in (1, -1)}))
    ax.set_yticks([], minor=True); ax.yaxis.set_major_formatter(_dec_no0)
    _axes_style(ax, xlabels)


def scale_ticks(fig):
    for ax in fig.axes:
        for axis, side in ((ax.xaxis, 'x'), (ax.yaxis, 'y')):
            ticks = axis.get_major_ticks()
            if ticks: ax.tick_params(axis=side, which='both', labelsize=ticks[0].label1.get_fontsize() * TICK_SCALE)
    return fig


def rows(cases, N, degrees, col, name, height_scale=1.0):
    """The lambda(z) (col 'a') or H(z) (col 'b') panels of N for the given degrees, stacked, sharing the z axis; the axes
    keep the size of their half of the two-column Fig. 3 row."""
    W = 0.7 + 9.15 / 2.2 + 0.15; H = height_scale * (1.5 * len(degrees) + 0.6 + 0.3)
    bottom = max(0.06 * H, 0.2 + 0.15 * TICK_SCALE)
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(degrees), 1, hspace=0.12 + 0.18 * (TICK_SCALE - 1), left=0.7 / W, right=1 - 0.15 / W, top=0.955, bottom=bottom / H)
    h = lam_half_width([cases[(N, d)] for d in degrees])
    for i, d in enumerate(degrees):
        ax = fig.add_subplot(gs[i, 0]); last = i == len(degrees) - 1
        if col == 'a': panel_lambda(ax, cases[(N, d)], h, last)
        else: panel_H(ax, cases[(N, d)], last)
    save(scale_ticks(fig), name)


# ---------------------------------------------------------------------------------------------------- the spectral ladder
def ladder(cases, name):
    """Row N: the learned spectrum of Z_N at d_lambda = N (short thick rungs in the Fig. 2 colour of N) over the lambda = 1
    spectrum (long thin grey ticks); lime crosses at the sign changes of H."""
    W, H, left, bottom, right, top = 8.4, 3.8, 0.45, 0.3, 0.12, 0.12
    fig = plt.figure(figsize=(left + W + right, bottom + H + top))
    ax = fig.add_axes([left / fig.get_figwidth(), bottom / fig.get_figheight(), W / fig.get_figwidth(), H / fig.get_figheight()])
    for n in NS:
        c = cases[(n, n)]; v = np.asarray(c['spectrum'], float); zeros = H_sign_changes(c['coefficients'])
        ax.vlines(exact_spectrum(n), n - 0.45, n + 0.45, color=REF, lw=1.0, zorder=1)
        ax.vlines(v, n - 0.22, n + 0.22, color=ramp(n), lw=3.2, zorder=2)
        if zeros.size: ax.plot(zeros, np.full(zeros.size, n), 'X', ms=4.2, mfc=LIME, mec=INK, mew=0.45, zorder=3)
    ax.set_ylim(NS[0] - 0.7, NS[-1] + 0.7); ax.set_yticks(NS)
    ax.set_xlim(-R, R); ax.set_xticks([-R, -R / 2, 0, R / 2, R]); ax.xaxis.set_major_formatter(_dec)
    save(scale_ticks(fig), name)


def ladder_key(name):
    handles = [Line2D([], [], color=ramp(11), lw=3.2, label='learned spectrum $v_N^{(j)}$'),
               Line2D([], [], color=REF, lw=1.0, label='$\\lambda = 1$ spectrum'),
               Line2D([], [], ls='', marker='X', ms=5, mfc=LIME, mec=INK, mew=0.45, label='sign change of $H$')]
    fig = plt.figure(figsize=(6.2, 0.35))
    fig.legend(handles=handles, loc='center', ncol=3, frameon=False, fontsize=8.5, handlelength=1.6, columnspacing=1.4)
    save(fig, name, tight=True)


def save(fig, name, tight=False):
    fig.savefig(OUT / f'{name}.png', dpi=DPI, **(dict(bbox_inches='tight', pad_inches=0.03) if tight else {}))
    plt.close(fig); print('written', OUT / f'{name}.png')


def main():
    global TICK_SCALE
    set_style(); OUT.mkdir(exist_ok=True)
    cases = {(c['N'], c['d']): c for c in json.loads(DATA.read_text())['runs']}
    TICK_SCALE = 2.0
    rows(cases, 6, [6, 7, 8], 'a', 'a_lambda'); rows(cases, 6, [6, 7, 8], 'b', 'a_H')
    ladder(cases, 'c'); ladder_key('c_key')
    TICK_SCALE = 1.8 * 0.93
    rows(cases, 8, [7], 'b', 'b_d7', height_scale=0.9); rows(cases, 8, [8], 'b', 'b_d8', height_scale=0.9)


if __name__ == '__main__':
    main()
