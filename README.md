# Code for "A Numerical Approach to Quantization"

| file | paper | what it does |
|---|---|---|
| `fuzzy.py` | Secs. II B, III A, App. B | common functions for the residual matrices (Eqs. 19, 23), L_ac (18), Γ̂ and L_irr (20, 21), L_pos (Sec. III B), the Levenberg–Marquardt step, the conventional fuzzy sphere (Eq. 16) |
| `sec3a_constant_multiplier.py` | Sec. III A, Fig. 2a | λ = 1, N = 2 … 40 sequentially from a random N = 2 start |
| `sec3b_polynomial_multiplier.py` | Sec. III B, Fig. 2b–c | trainable λ(z) = 1 + λ₁z + … + z^d, N = 2 … 20, d = 1 … 10 |
| `sec5a_inverse_square_multiplier.py` | Sec. V A, Fig. 4a1 | λ(z) = 1 + η z⁻², N = 2 … 20, η ∈ {1, 10⁻¹, …, 10⁻⁴}|
| `appendixK_additional_solutions.py` | App. K, Fig. 4a2, 4b | lists every solution of Eq. (K4) at even N = 2 … 24 for η = 10⁻², 10⁻⁴, initializes on each (Eq. K5) and trains |

```
python paper_code/sec3a_constant_multiplier.py                      # ~1 h to N = 40          -> results/sec3a.json
python paper_code/sec3b_polynomial_multiplier.py --workers 8        # 190 optimizations        -> results/sec3b.json
python paper_code/sec5a_inverse_square_multiplier.py --workers 8    # 95 optimizations         -> results/sec5a.json
python paper_code/appendixK_additional_solutions.py --workers 8     # 182 solutions + runs     -> results/appendixK.json
```

Every script prints a summary and writes one JSON with, per optimization, the final losses, ħ_N, the spectrum of Ẑ_N and the loss history over the 250 training steps. R = 1 and λ₀ = 1 throughout.