"""Fig. 4: beyond polynomial fuzzy sphere, lambda(z) = lambda_0 (1 + eta z^-2), eta = lambda_-2/lambda_0, lambda_0 = R = 1. Panels are saved as PNG to figures/fig_4/

    python figures/fig4.py

Figures are drawn from the results of the data-generation scripts:
    quantization/results/sec5a.json      sec5a_inverse_square_multiplier.py    the fuzzy-sphere start, N = 2..20, eta = 1, 1e-1, 1e-2, 1e-3, 1e-4
    quantization/results/appendixK.json  appendixK_additional_solutions.py     the training started on every solution of Eq. (K4), even N, eta = 1e-2, 1e-4

    Thus please run code for 5a/appK to generate results for figures before running this code.

Notation:
    a1          final L_ac of the fuzzy-sphere start (Sec. V A), rows eta = 1, 1e-1, 1e-2, 1e-3, 1e-4, columns N = 2..20
    a2          lowest final L_ac over the fuzzy-sphere start and every App. K start, same frame
    colorbar    L_ac, 1e-5 (black) .. 1e-35 (white)
    b1, b2      hbar_N of every solution of Eq. (K4) at eta = 1e-2, 1e-4: dot colour = the final L_ac of the training started on it;
                bold dot = the original solution (one crossing of z = 0, the smaller hbar_N: the one the fuzzy-sphere start reaches);
                grey = hbar_N = 1/sqrt(N^2 - 1) (Eq. 16); red dotted = hbar_N >= (3 sqrt3 / 2) sqrt(eta) (Eq. 48)
    c1, c2, c3  running minimum of L_ac over the 250 training steps at eta = 1e-1, 1e-2, 1e-4: solid = the fuzzy-sphere start,
                every N (even N cyan .. magenta, odd N greens underneath); dashed = the best App. K start of that N (even N)
    key_N, key_lines, key_dots   the N key of (c), the line key of (c), the dot key of (b)

    L_ac = sum_C ||A_C||^2 (Eq. 18);  a run is a solution when its final L_ac < 1e-18 (Sec. V B).
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D

RESULTS = Path(__file__).resolve().parent.parent / 'quantization' / 'results'
OUT = Path(__file__).resolve().parent / 'fig_4'

DPI, FLOOR, STEPS, TICK = 600, 1e-35, 250, 7.5
NS = list(range(2, 21))
HEAT_ETAS, LOSS_ETAS, HBAR_ETAS = (1.0, 1e-1, 1e-2, 1e-3, 1e-4), (1e-1, 1e-2, 1e-4), (1e-2, 1e-4)
SIZE = {'a': (2.40, 1.12), 'b': (2.64, 1.35), 'c': (2.49, 1.01)}                     # axes sizes (in)
STOPS = [(0.0, (0, 0, 0)), (0.025, (0, 0, 46)), (0.05, (1, 0, 80)), (0.075, (13, 2, 105)), (0.1, (26, 3, 131)),
         (0.125, (44, 8, 153)), (0.15, (59, 13, 176)), (0.175, (77, 19, 191)), (0.2, (91, 24, 207)),
         (0.225, (104, 29, 191)), (0.25, (116, 33, 178)), (0.275, (128, 37, 165)), (0.3, (142, 42, 150)),
         (0.325, (149, 45, 131)), (0.35, (155, 47, 112)), (0.375, (165, 51, 93)), (0.4, (172, 55, 76)),
         (0.425, (180, 63, 63)), (0.45, (187, 72, 52)), (0.475, (195, 84, 53)), (0.5, (203, 96, 54)),
         (0.525, (213, 110, 60)), (0.55, (221, 124, 65)), (0.575, (224, 135, 69)), (0.6, (225, 144, 72)),
         (0.625, (227, 152, 75)), (0.65, (229, 162, 78)), (0.675, (231, 173, 82)), (0.7, (234, 183, 86)),
         (0.725, (237, 193, 90)), (0.75, (239, 205, 95)), (0.775, (242, 215, 99)), (0.8, (246, 225, 103)),
         (0.825, (248, 235, 108)), (0.85, (252, 245, 114)), (0.875, (253, 250, 136)), (0.9, (255, 255, 158)),
         (0.925, (255, 255, 194)), (0.95, (255, 255, 228)), (0.975, (255, 255, 243)), (1.0, (255, 255, 255))]   # the colour bar of Fig. 2(b), sampled at 41 points
CMAP = LinearSegmentedColormap.from_list('paper_loss', [(t, tuple(v / 255 for v in c)) for t, c in STOPS])
NORM = Normalize(5, 35)                                                               # of -log10 L_ac
EVEN = plt.colormaps['cool']
ODD = LinearSegmentedColormap.from_list('odd_greens', ['#2e6e57', '#6fae6a', '#d4e05a'])
DASH, INK, GREY, RED = (0, (3, 1.5)), (0.12, 0.16, 0.22), '0.55', '#c0392b'
BOLD_S, PLAIN_S = 38, 14                                                              # (b): dot area of the original / other solutions


def set_style():
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ('Helvetica', 'Helvetica Neue', 'Arial', 'Liberation Sans', 'DejaVu Sans') if f in names), 'DejaVu Sans')
    plt.rcParams.update({'font.family': font, 'mathtext.fontset': 'dejavusans', 'mathtext.default': 'it',
                         'axes.linewidth': 0.8, 'xtick.direction': 'in', 'ytick.direction': 'in',
                         'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.minor.size': 2, 'ytick.minor.size': 2,
                         'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.fancybox': False})


def colour(n):
    return EVEN((n - 2) / 18) if n % 2 == 0 else ODD((n - 3) / 16)


def hbar_min(eta): return 1.5 * np.sqrt(3) * np.sqrt(eta)                             # Eq. (48)


# ---------------------------------------------------------------------------------------------------- data
def same(a, b): return abs(np.log(a / b)) < 1e-3


def load():
    """sphere: {eta: {N: run}} of the fuzzy-sphere start;  extra: {eta: [runs]} of the App. K starts."""
    sphere, extra = {}, {}
    for r in json.loads((RESULTS / 'sec5a.json').read_text())['runs']: sphere.setdefault(r['eta'], {})[r['N']] = r
    for r in json.loads((RESULTS / 'appendixK.json').read_text())['runs']: extra.setdefault(r['eta'], []).append(r)
    return sphere, extra


def at(d, eta): return next((v for e, v in d.items() if same(e, eta)), {} if isinstance(next(iter(d.values())), dict) else [])


def best_extra(eta, extra):
    """{N: the App. K run with the lowest final L_ac at this eta}."""
    out = {}
    for r in at(extra, eta):
        if r['N'] not in out or r['final_L_ac'] < out[r['N']]['final_L_ac']: out[r['N']] = r
    return out


# ---------------------------------------------------------------------------------------------------- panels
def box_off(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.tick_params(top=False, right=False, which='both')


def heat(ax, sphere, extra, expanded):
    arr = np.full((len(HEAT_ETAS), len(NS)), np.nan)
    for i, eta in enumerate(HEAT_ETAS):
        best = {n: r['final_L_ac'] for n, r in at(sphere, eta).items()}
        if expanded:
            for n, r in best_extra(eta, extra).items(): best[n] = min(best.get(n, np.inf), r['final_L_ac'])
        for n, L in best.items():
            if n in NS: arr[i, NS.index(n)] = -np.log10(max(L, FLOOR))
    ax.imshow(arr, cmap=CMAP, norm=NORM, aspect='auto', interpolation='nearest', extent=(0.5, len(NS) + 0.5, len(HEAT_ETAS) + 0.5, 0.5))
    ax.set_xticks(range(1, len(NS) + 1)); ax.set_xticklabels([str(n) for n in NS], rotation=90)
    ax.set_yticks(range(1, len(HEAT_ETAS) + 1))
    ax.set_yticklabels(['1' if e == 1 else f'$10^{{{int(round(np.log10(e)))}}}$' for e in HEAT_ETAS])
    ax.set_xticks([], minor=True); ax.set_yticks([], minor=True); ax.tick_params(labelsize=TICK, length=2.5, width=0.7)


def colorbar(cax):
    cb = plt.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, ticks=[5, 10, 15, 20, 25, 30, 35]); cb.ax.invert_yaxis()
    cb.ax.set_yticklabels([f'$10^{{-{k}}}$' for k in (5, 10, 15, 20, 25, 30, 35)])
    cb.ax.tick_params(labelsize=TICK, length=2.5, width=0.7); cb.outline.set_linewidth(0.7)


def panel_hbar(ax, eta, extra):
    nn = np.linspace(NS[0], NS[-1], 200)
    ax.plot(nn, 1 / np.sqrt(nn ** 2 - 1), color=GREY, lw=0.9, zorder=1)                                   # Eq. (16)
    ax.axhline(hbar_min(eta), color=RED, lw=1.2, ls=(0, (1, 1.5)), zorder=1)                                # Eq. (48)
    runs = [r for r in at(extra, eta) if r['N'] in NS]
    original = {}                                                                                            # per N, where it exists
    for r in runs:
        if r['crossings'] == 1: original[r['N']] = min(original.get(r['N'], np.inf), r['hbar'])
    x, y, c = np.array([(r['N'], r['hbar'], -np.log10(max(r['final_L_ac'], FLOOR))) for r in runs]).T
    orig = np.array([r['hbar'] == original.get(r['N']) for r in runs])
    ax.scatter(x[~orig], y[~orig], s=PLAIN_S, c=c[~orig], cmap=CMAP, norm=NORM, edgecolors=INK, linewidths=0.35, zorder=3)
    ax.scatter(x[orig], y[orig], s=BOLD_S, c=c[orig], cmap=CMAP, norm=NORM, edgecolors=INK, linewidths=1.3, zorder=5)
    ax.set_yscale('log'); ax.set_ylim(1e-3, 1e2); ax.set_yticks([10.0 ** k for k in range(-3, 3)]); ax.set_yticks([], minor=True)
    ax.set_xlim(NS[0] - 0.6, NS[-1] + 0.6); ax.set_xticks(NS[::2])
    ax.tick_params(labelsize=TICK, length=2.5, width=0.7, pad=2); box_off(ax)


def curve(ax, n, hist, **kw):
    y = np.maximum(np.minimum.accumulate(np.array(hist, float)), FLOOR)                                     # running minimum of L_ac
    ax.plot(np.arange(1, y.size + 1), y, color=colour(n), **kw)


def panel_loss(ax, eta, sphere, extra):
    fs, other = at(sphere, eta), best_extra(eta, extra)
    for n in NS[1::2]:                                                                  # odd N, fuzzy-sphere start, underneath
        if n in fs: curve(ax, n, fs[n]['history']['L_ac'], lw=1.0, ls='-', zorder=2)
    for n in NS[::2]:                                                                   # even N, fuzzy-sphere start, solid
        if n in fs: curve(ax, n, fs[n]['history']['L_ac'], lw=1.0, ls='-', zorder=3)
    for n in NS[::2]:                                                                   # even N, best App. K start, dashed
        if n in other: curve(ax, n, other[n]['history']['L_ac'], lw=1.0, ls=DASH, zorder=4)
    ax.set_yscale('log'); ax.set_ylim(FLOOR, 1e3); ax.set_yticks([1e0, 1e-10, 1e-20, 1e-30]); ax.set_yticks([], minor=True)
    ax.set_xlim(1, STEPS); ax.set_xticks([50, 100, 150, 200, 250])
    ax.tick_params(labelsize=TICK, length=2.5, width=0.7, pad=2); box_off(ax)


def key(lax, handles, spacing):
    lax.set_axis_off()
    lax.legend(handles=handles, loc='center left', frameon=False, handlelength=2.0, borderaxespad=0, labelspacing=spacing, fontsize=TICK)


def key_N(lax):
    h = [Line2D([0], [0], color=colour(n), lw=1.6, label=f'N={n}') for n in NS[::2]] + [Line2D([0], [0], color='none', label=' ')]
    key(lax, h + [Line2D([0], [0], color=colour(n), lw=1.6, label=f'N={n}') for n in NS[1::2]], 0.3)


def key_lines(lax):
    key(lax, [Line2D([0], [0], color=INK, lw=1.0, label='original solution'),
              Line2D([0], [0], color=INK, lw=1.0, ls=DASH, label='expanded solutions')], 0.4)


def key_dots(lax):
    key(lax, [Line2D([0], [0], ls='', marker='o', ms=np.sqrt(BOLD_S), mfc=CMAP(0.85), mec=INK, mew=1.3, label='original solution'),
              Line2D([0], [0], ls='', marker='o', ms=np.sqrt(PLAIN_S), mfc=CMAP(0.85), mec=INK, mew=0.35, label='expanded solution')], 0.4)


# ---------------------------------------------------------------------------------------------------- output
def piece(w, h, draw, name, pad=(0.55, 0.35, 0.12, 0.12)):
    """One axes of exactly w x h inches with margins pad = (left, bottom, right, top); the PNG is cropped to the ink."""
    W, H = pad[0] + w + pad[2], pad[1] + h + pad[3]
    fig = plt.figure(figsize=(W, H), dpi=DPI); draw(fig.add_axes([pad[0] / W, pad[1] / H, w / W, h / H]))
    fig.savefig(OUT / f'{name}.png', dpi=DPI, bbox_inches='tight', pad_inches=0.03); plt.close(fig); print('written', OUT / f'{name}.png')


def main():
    set_style(); OUT.mkdir(exist_ok=True); sphere, extra = load()
    piece(*SIZE['a'], lambda ax: heat(ax, sphere, extra, False), 'a1', pad=(0.55, 0.65, 0.1, 0.1))
    piece(*SIZE['a'], lambda ax: heat(ax, sphere, extra, True), 'a2', pad=(0.55, 0.65, 0.1, 0.1))
    piece(0.12, SIZE['a'][1], colorbar, 'colorbar', pad=(0.05, 0.1, 0.6, 0.1))
    for i, eta in enumerate(HBAR_ETAS, 1): piece(*SIZE['b'], lambda ax, e=eta: panel_hbar(ax, e, extra), f'b{i}')
    for i, eta in enumerate(LOSS_ETAS, 1): piece(*SIZE['c'], lambda ax, e=eta: panel_loss(ax, e, sphere, extra), f'c{i}')
    piece(0.7, 2.6, key_N, 'key_N', pad=(0.02,) * 4)
    piece(0.9, 0.4, key_lines, 'key_lines', pad=(0.02,) * 4)
    piece(0.9, 0.4, key_dots, 'key_dots', pad=(0.02,) * 4)


if __name__ == '__main__':
    main()
