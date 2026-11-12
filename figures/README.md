# Figures

One script per figure of the paper. Each reads only `quantization/results/` (the JSON files written by the four `quantization` scripts)
and writes the panels of its figure as separate PNGs to `figures/fig_2/`, `figures/fig_3/`, `figures/fig_4/`: tick numbers and data only; axis titles,
panel tags, boxes and arrows are added by hand on reassembly. Nothing is recomputed. Fig. 1 is a drawing; no code produced it.

```
python quantization/sec3b_polynomial_multiplier.py --dmax 20     # Figs. 2(b, c2) and 3(c) use degrees up to 20 (the text stops at 10)
python figures/fig2.py      # fig_2/a1 a2 b c1 c2 key                      <- quantization/results/sec3a.json, sec3b.json
python figures/fig3.py      # fig_3/a_lambda a_H b_d7 b_d8 c c_key           <- quantization/results/sec3b.json
python figures/fig4.py      # fig_4/a1 a2 colorbar b1 b2 c1 c2 c3 key_N key_lines key_dots   <- quantization/results/sec5a.json, appendixK.json
```

| figure | panels |
|---|---|
| Fig. 2 | (a1) ħ_N against Eq. (16), (a2) relative difference to 50 digits, (b) final L_ac on the (N, d_λ) grid, (c1–2) running minimum of L_ac for d_λ = 7, 14, the N key |
| Fig. 3 | (a) λ(z) and H(z) = zλ'' + 3λ' for N = 6, d_λ = 6, 7, 8 with the spectrum of Ẑ₆ and the sign changes of H; (b) H(z) for N = 8, d_λ = 7, 8; (c) spectra of Ẑ_N at N = d_λ against the λ = 1 spectra, with the key |
| Fig. 4 | (a1) final L_ac of the fuzzy-sphere start, (a2) lowest over the fuzzy-sphere and App. K starts, colour bar; (b1–2) ħ_N of every solution at η = 10⁻², 10⁻⁴, dot colour = its final L_ac, bold = the original solution; (c1–3) running minimum of L_ac at η = 10⁻¹, 10⁻², 10⁻⁴, solid = fuzzy-sphere start, dashed = best App. K start; keys |

Where the paper's Fig. 4 drew on runs that are not part of the procedures in `quantization`, these panels differ from it:

- (a2) the η = 10⁻³ row has only the fuzzy-sphere start (App. K runs η = 10⁻² and 10⁻⁴); `appendixK_additional_solutions.py --eta 0.01 0.001 0.0001` fills it.
- (b2) the compact solutions at η = 10⁻⁴ (ħ_N ≈ 25) end the 250 steps at L_ac ≈ 10⁻¹² under c_irr = 10⁻¹⁸ and are coloured so.
- (c1) has no dashed curves: there are no App. K runs at η = 10⁻¹.
