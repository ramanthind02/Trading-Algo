# In-Sample Permutation Summary

**Feature:** signal  |  **Type:** signed_signal  |  **Objective:** `t_stat`

## Vector shuffle

| Tested | Passed |
|--------|--------|
| 24 | 24 |

## Scope

Each row is one **expanded** param combo from the same grid as in-sample EDA. For ``feature_type=CONTINUOUS``, each combo is **quantile-binned** per ticker (``binning_params.bin_counts[0]``) then mapped to ±1/0 from ``strategy`` (LONG: lowest bin long; SHORT: highest bin short; LONG_SHORT: both tails). Signed-signal nodes use native discrete output.

## Per-combo results

| param_combo (readable) | observed_metric | p-value | pass | alpha | n_reps |
|------------------------|-----------------|--------|------|-------|--------|
| combo_f37206894e7941df3997ad288c717258 | 6.1106 | 0.0198 | ✓ | 0.1 | 100 |
| combo_8920a7443e13420b8124b7dbc4f0014a | 5.7179 | 0.0099 | ✓ | 0.1 | 100 |
| combo_e0258dc8e1955a7eb6c0c0464f324dbe | 5.7170 | 0.0099 | ✓ | 0.1 | 100 |
| combo_a464b2110cdb881ac27521ca58496fe8 | 6.4986 | 0.0099 | ✓ | 0.1 | 100 |
| combo_ca9aaf58248600b8e1c40e059def06c5 | 6.6454 | 0.0099 | ✓ | 0.1 | 100 |
| combo_356d9889afc2b703944f448da0349e6d | 6.3986 | 0.0099 | ✓ | 0.1 | 100 |
| combo_7c8be145fffbae098ae0c2ae6b07790b | 5.7644 | 0.0198 | ✓ | 0.1 | 100 |
| combo_2e4513ef705be3b44a21ed079eee659a | 5.5171 | 0.0099 | ✓ | 0.1 | 100 |
| combo_4182a425d128573befa3d71ce8e1d8da | 5.5430 | 0.0099 | ✓ | 0.1 | 100 |
| combo_51e9e520dd67b846d4dd9f2f59ebea3d | 6.7564 | 0.0099 | ✓ | 0.1 | 100 |
| combo_55d9773aaf21a383a21a00a1eee1a49e | 6.8994 | 0.0099 | ✓ | 0.1 | 100 |
| combo_76660b50c6714fad01e0fb8153d9ac17 | 5.3318 | 0.0099 | ✓ | 0.1 | 100 |
| combo_c8474ff577cc64ca87e72ec7a8114ce5 | 5.5618 | 0.0495 | ✓ | 0.1 | 100 |
| combo_0e1ea575bcec52ca46763eddcaf7adb8 | 5.9029 | 0.0099 | ✓ | 0.1 | 100 |
| combo_0d0f13fddd5c867f4b8e2814c09c47fb | 5.1582 | 0.0198 | ✓ | 0.1 | 100 |
| combo_3446fed5f94c45671fc33730752abac4 | 6.6173 | 0.0099 | ✓ | 0.1 | 100 |
| combo_40619daa8dd8a2ee5d46ef13235da315 | 6.3981 | 0.0099 | ✓ | 0.1 | 100 |
| combo_a1acf9b0fe191b08e572a609407dd95f | 5.4656 | 0.0099 | ✓ | 0.1 | 100 |
| combo_b77dde11a40092f6e9061a9d69ef3834 | 5.5905 | 0.0297 | ✓ | 0.1 | 100 |
| combo_86769f85cb70d4f1b37abd5f350d4c26 | 6.0392 | 0.0099 | ✓ | 0.1 | 100 |
| combo_eb551db30d30ad7bb196bf80a79336de | 5.4431 | 0.0099 | ✓ | 0.1 | 100 |
| combo_49b41c5fc8304e3281d9603cb43f8770 | 6.5775 | 0.0099 | ✓ | 0.1 | 100 |
| combo_77e4840386b39c5358a480a6887defbd | 5.4195 | 0.0099 | ✓ | 0.1 | 100 |
| combo_47e353f5ef7176bbed725fa3a243adad | 4.8127 | 0.0099 | ✓ | 0.1 | 100 |

Power BI: `permutation_vector_shuffle.csv` in `C:\Users\raman\Documents\repos\Trading-Algo\feature_research\in_sample\results\powerbi`
Flat summary: `permutation_summary.csv` (columns include `param_combo` = stable key, `param_combo_label` = readable).
