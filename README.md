# Code for "A Numerical Approach to Quantization"

## Code for generating results (quantization folder)

| file | paper | what it does |
|---|---|---|
| `quantization/fuzzy.py` | Secs. II B, III A, App. B | common functions for the residual matrices (Eqs. 19, 23), L_ac (18), Γ̂ and L_irr (20, 21), L_pos (Sec. III B), the Levenberg–Marquardt step, the conventional fuzzy sphere (Eq. 16) |
| `quantization/sec3a_constant_multiplier.py` | Sec. III A, Fig. 2a | λ = 1, N = 2 … 40 sequentially from a random N = 2 start |
| `quantization/sec3b_polynomial_multiplier.py` | Sec. III B, Fig. 2b–c | trainable λ(z) = 1 + λ₁z + … + z^d, N = 2 … 20, d = 1 … 10 |
| `quantization/sec5a_inverse_square_multiplier.py` | Sec. V A, Fig. 4a1 | λ(z) = 1 + η z⁻², N = 2 … 20, η ∈ {1, 10⁻¹, …, 10⁻⁴}|
| `quantization/appendixK_additional_solutions.py` | App. K, Fig. 4a2, 4b | lists every solution of Eq. (K4) at even N = 2 … 24 for η = 10⁻², 10⁻⁴, initializes on each (Eq. K5) and trains |

```
python quantization/sec3a_constant_multiplier.py                      # ~1 h to N = 40          -> quantization/results/sec3a.json
python quantization/sec3b_polynomial_multiplier.py --workers 8        # 190 optimizations        -> quantization/results/sec3b.json
python quantization/sec5a_inverse_square_multiplier.py --workers 8    # 95 optimizations         -> quantization/results/sec5a.json
python quantization/appendixK_additional_solutions.py --workers 8     # 182 solutions + runs     -> quantization/results/appendixK.json
python figures/fig2.py                                   # and fig3.py, fig4.py: the panels of Figs. 2-4 from quantization/results/
```

Every script prints a summary and writes one JSON with, per optimization, the final losses, ħ_N, the spectrum of Ẑ_N and the loss history over the 250 training steps. R = 1 and λ₀ = 1 throughout.

## Code for making figures (figures folder)

```
python quantization/sec3b_polynomial_multiplier.py --dmax 20     # Figs. 2(b, c2) and 3(c) use degrees up to 20 (the text stops at 10)
python figures/fig2.py      # fig_2/a1 a2 b c1 c2 key                      <- quantization/results/sec3a.json, sec3b.json
python figures/fig3.py      # fig_3/a_lambda a_H b_d7 b_d8 c c_key           <- quantization/results/sec3b.json
python figures/fig4.py      # fig_4/a1 a2 colorbar b1 b2 c1 c2 c3 key_N key_lines key_dots   <- quantization/results/sec5a.json, appendixK.json
```

| figure | panels |
|---|---|
| Fig. 2 | (a1) ħ_N vs predicted value in Eq. (16), (a2) relative difference to 50 digits, (b) final L_ac on the (N, d_λ) grid, (c1–2) running minimum of L_ac for d_λ = 7, 14 |
| Fig. 3 | (a) λ(z) and H(z) = zλ'' + 3λ' for N = 6, d_λ = 6, 7, 8 with the spectrum of Ẑ₆ and the sign changes of H; (b) H(z) for N = 8, d_λ = 7, 8; (c) spectra of Ẑ_N at N = d_λ against the λ = 1 spectra |
| Fig. 4 | (a1) final L_ac of the fuzzy-sphere solution, (a2) lowest over all solutions; (b1–2) ħ_N of every solution at η = 10⁻², 10⁻⁴, dot colour = its final L_ac, bold = the original solution; (c1–3) running minimum of L_ac at η = 10⁻¹, 10⁻², 10⁻⁴, solid = fuzzy-sphere start, dashed = best general-solution start (App. K)|

Each script reads `quantization/results/` (the JSON files written by the four `quantization` scripts) and writes the panels of its figure as PNGs to `figures/fig_N/` (ex. `figures/fig_2/` contains panels for fig. 2).