# In-Sample Permutation Summary

**Feature:** signal  |  **Type:** signed_signal  |  **Objective:** `t_stat`

## Vector shuffle

| Tested | Passed |
|--------|--------|
| 120 | 102 |

## Scope

Each row is one **expanded** param combo from the same grid as in-sample EDA. For ``feature_type=CONTINUOUS``, each combo is **quantile-binned** per ticker (``binning_params.bin_counts[0]``) then mapped to ±1/0 from ``strategy`` (LONG: lowest bin long; SHORT: highest bin short; LONG_SHORT: both tails). Signed-signal nodes use native discrete output.

## Per-combo results

| param_combo (readable) | observed_metric | p-value | pass | alpha | n_reps |
|------------------------|-----------------|--------|------|-------|--------|
| complex_spec · id_48f3ecce2acd | 3.1769 | 0.1782 | ✗ | 0.1 | 100 |
| complex_spec · id_44b27d51b67b | 3.9806 | 0.0495 | ✓ | 0.1 | 100 |
| complex_spec · id_8ed632f4937d | 4.4287 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_d15bf75e34b2 | 4.2062 | 0.1485 | ✗ | 0.1 | 100 |
| complex_spec · id_906c5671e50e | 4.0888 | 0.1683 | ✗ | 0.1 | 100 |
| complex_spec · id_9af841cf813f | 4.1877 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_e5ac0b574384 | 5.1968 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_4a007935bd6e | 5.1004 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_518c55f473e0 | 4.7352 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_46cc7940f8a8 | 4.6358 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_6c617090dc68 | 2.8266 | 0.2970 | ✗ | 0.1 | 100 |
| complex_spec · id_dadff57ba265 | 3.2552 | 0.2277 | ✗ | 0.1 | 100 |
| complex_spec · id_a5850d082bb4 | 3.7184 | 0.1683 | ✗ | 0.1 | 100 |
| complex_spec · id_d4939a64067f | 4.7421 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_f876471fdabd | 5.0599 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_f17058c6e8af | 3.5296 | 0.0792 | ✓ | 0.1 | 100 |
| complex_spec · id_4ab5fa7d4bea | 4.0226 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_8478c9efff53 | 4.4071 | 0.0792 | ✓ | 0.1 | 100 |
| complex_spec · id_69a02866eb56 | 4.8378 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_cdd7ae8a7a98 | 5.0535 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_e683910e1694 | 4.5227 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_4846aa5ad5b9 | 4.4450 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_47aa08dcad63 | 5.1515 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_e2f63c3322b0 | 4.9824 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_3854d69ad3d7 | 5.7146 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_817981d001fe | 4.2745 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_e5d36d4dabd3 | 5.1940 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_55d6a3eb1784 | 4.7581 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_158ec2182781 | 5.1838 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_8f0f9390ebbf | 5.4521 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_784546ac40dc | 3.9519 | 0.0495 | ✓ | 0.1 | 100 |
| complex_spec · id_012713bbe3a5 | 4.3189 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_4ff845e3975e | 4.6539 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_bdc06742b5c6 | 4.8766 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_f10f698534b1 | 4.8207 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_fbe5e809d7fb | 4.9759 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_59b64d70930c | 4.8830 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_8dd97a4008b3 | 4.8719 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_5eaace5e354c | 4.5727 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_174aa39d92d5 | 4.2262 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_ef9760d47627 | 3.6333 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_6c981c00eb0f | 4.0586 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_499f461b3b84 | 4.1751 | 0.0891 | ✓ | 0.1 | 100 |
| complex_spec · id_9139889c47e5 | 5.0429 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_095eb3c8a2e5 | 5.0770 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_700e265e3039 | 5.0069 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_aa9516ff3cc7 | 4.3902 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_d8d6789957ab | 4.1910 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_bc73684804d3 | 5.1281 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_70fffcf80105 | 5.0218 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_09e8d65f2c26 | 3.8825 | 0.0792 | ✓ | 0.1 | 100 |
| complex_spec · id_036ce92fa5af | 4.1715 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_97dbf910db48 | 4.1181 | 0.0891 | ✓ | 0.1 | 100 |
| complex_spec · id_44329e2c0f0e | 3.7601 | 0.2079 | ✗ | 0.1 | 100 |
| complex_spec · id_0e19ac34d87d | 5.0808 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_8a80ee5ea01f | 3.6855 | 0.1089 | ✓ | 0.1 | 100 |
| complex_spec · id_12c6cdc5d833 | 5.2888 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_533dcc6e8323 | 4.8189 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_78d5d3e2c90b | 4.9782 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_06ddf984f3c9 | 5.3356 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_d076f029fbd3 | 4.6982 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_89581093938c | 3.8317 | 0.1089 | ✓ | 0.1 | 100 |
| complex_spec · id_c68b39b4ee24 | 3.9706 | 0.1089 | ✓ | 0.1 | 100 |
| complex_spec · id_feaab59d0599 | 4.3668 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_3130ca0f8b9f | 4.4562 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_d634264040fd | 3.4205 | 0.1683 | ✗ | 0.1 | 100 |
| complex_spec · id_8cb03d3bb445 | 3.9456 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_ebf29a980987 | 4.3042 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_0b42a2de5868 | 4.3452 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_719f2f8ae992 | 4.2739 | 0.0792 | ✓ | 0.1 | 100 |
| complex_spec · id_d69393994e58 | 4.7100 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_d8b6b407ee78 | 5.1139 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_195feb685059 | 5.2920 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_718aaa502f42 | 4.7509 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_2368866d0f24 | 4.7935 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_dfa34c3a02ca | 4.6845 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_e186b736ad6b | 5.1029 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_ec8a9742cb2e | 5.5750 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_041fd0a7ce1b | 5.3609 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_fd09e1fd6bcd | 5.3650 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_b463e377c335 | 4.1722 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_525e69de2e4f | 4.3298 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_1360e31661d5 | 4.7044 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_1755aba0ed47 | 4.6198 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_9a62c56dbf6a | 4.7773 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_f74bff7ff292 | 5.1508 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_923558653219 | 4.9068 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_7a225dda1f1e | 4.4503 | 0.0495 | ✓ | 0.1 | 100 |
| complex_spec · id_dbc858b3a59e | 5.1332 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_a465fc7f7ea1 | 5.3503 | 0.0198 | ✓ | 0.1 | 100 |
| complex_spec · id_2a39836bae4d | 4.6094 | 0.0099 | ✓ | 0.1 | 100 |
| complex_spec · id_8456f5d6dbe7 | 4.4210 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_d1ac8493d0a9 | 4.4347 | 0.0495 | ✓ | 0.1 | 100 |
| complex_spec · id_79be289810b0 | 4.3865 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_f37510507138 | 4.3676 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_c3ca09ba1163 | 4.0613 | 0.0495 | ✓ | 0.1 | 100 |
| complex_spec · id_ac9090aba546 | 4.0819 | 0.0693 | ✓ | 0.1 | 100 |
| complex_spec · id_da9b000befcf | 4.7386 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_fb8f67d226d1 | 4.6610 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_c3a450002ab3 | 4.6151 | 0.0594 | ✓ | 0.1 | 100 |
| complex_spec · id_f1d6feb52e65 | 2.9409 | 0.1782 | ✗ | 0.1 | 100 |
| complex_spec · id_0104ede8e925 | 3.5156 | 0.0792 | ✓ | 0.1 | 100 |
| complex_spec · id_acf3017170b9 | 3.5536 | 0.1683 | ✗ | 0.1 | 100 |
| complex_spec · id_05d048e2357d | 3.9146 | 0.1485 | ✗ | 0.1 | 100 |
| complex_spec · id_50a401016b69 | 4.3656 | 0.1089 | ✓ | 0.1 | 100 |
| complex_spec · id_38faa11851cd | 4.1181 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_c2917fdd5191 | 4.3164 | 0.0297 | ✓ | 0.1 | 100 |
| complex_spec · id_e22baef3d421 | 4.4668 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_b1ea3660273c | 4.7102 | 0.0495 | ✓ | 0.1 | 100 |
| complex_spec · id_685d2724b6e2 | 4.8710 | 0.0396 | ✓ | 0.1 | 100 |
| complex_spec · id_424c5daba0a2 | 2.7303 | 0.2673 | ✗ | 0.1 | 100 |
| complex_spec · id_298814928493 | 3.0272 | 0.2475 | ✗ | 0.1 | 100 |
| complex_spec · id_582620ce4c25 | 3.4000 | 0.1980 | ✗ | 0.1 | 100 |
| complex_spec · id_e20acb321733 | 4.0239 | 0.0990 | ✓ | 0.1 | 100 |
| complex_spec · id_bbfd5c3aecc3 | 3.9513 | 0.1782 | ✗ | 0.1 | 100 |
| complex_spec · id_5377affa8241 | 3.2736 | 0.1485 | ✗ | 0.1 | 100 |
| complex_spec · id_c04724e14e4c | 3.7741 | 0.0792 | ✓ | 0.1 | 100 |
| complex_spec · id_0ffe202f5a08 | 3.8928 | 0.1188 | ✗ | 0.1 | 100 |
| complex_spec · id_9264ee8e0db3 | 3.7997 | 0.1881 | ✗ | 0.1 | 100 |
| complex_spec · id_0124c43d3030 | 4.6496 | 0.0693 | ✓ | 0.1 | 100 |

Power BI: `permutation_vector_shuffle.csv` in `C:\Users\raman\Documents\repos\Trading-Algo\feature_research\in_sample\results\powerbi`
Flat summary: `permutation_summary.csv` (columns include `param_combo` = stable key, `param_combo_label` = readable).
