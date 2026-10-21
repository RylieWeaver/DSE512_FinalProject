# Doubling-time baseline results

The fixed assembly-level split has 301 train, 25 validation, and 74 test examples. Mean and temperature regression are fitted on train only.

MSE and standardized RMSE use the training-standardized `log_dob_h` target, matching the sequence regression trainer. RMSE in log hours equals standardized RMSE times the training target standard deviation. R² is computed against the observed mean of each scored cohort and split. Each assembly is scored once.

| Cohort | Split | Method | N | MSE (std. log h) | RMSE (std. log h) | RMSE (log h) | R² |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | val | train_mean | 25 | 0.6676 | 0.8171 | 1.2114 | -0.0002 |
| all | val | temperature_linear | 25 | 0.6861 | 0.8283 | 1.2281 | -0.0280 |
| grodon_matched | val | train_mean | 25 | 0.6676 | 0.8171 | 1.2114 | -0.0002 |
| grodon_matched | val | temperature_linear | 25 | 0.6861 | 0.8283 | 1.2281 | -0.0280 |
| grodon_matched | val | grodon | 25 | 0.5043 | 0.7102 | 1.0529 | 0.2444 |
| all | test | train_mean | 74 | 0.8885 | 0.9426 | 1.3976 | -0.0614 |
| all | test | temperature_linear | 74 | 0.7945 | 0.8913 | 1.3215 | 0.0509 |
| grodon_matched | test | train_mean | 72 | 0.9046 | 0.9511 | 1.4102 | -0.0619 |
| grodon_matched | test | temperature_linear | 72 | 0.8143 | 0.9024 | 1.3380 | 0.0441 |
| grodon_matched | test | grodon | 72 | 0.7650 | 0.8746 | 1.2968 | 0.1021 |

The train mean predicts one constant log doubling time. Temperature regression fits an intercept and slope using only training rows. gRodon uses genome-wide annotated coding genes and ribosomal-protein labels with `mode=full`, the `madin` training set, and no temperature correction.

gRodon coverage (valid predictions): train 285/301, val 25/25, test 72/74.
The `grodon_matched` cohort compares all methods on those same assemblies. Review `summary.json` and the gRodon status CSV for failures.
On validation, gRodon MSE was 0.5043 versus 0.6861 for temperature regression. On the covered test assemblies, the scores were 0.7650 and 0.8143, respectively. gRodon uses additional genome-wide annotation, so this comparison describes predictive performance with different input information.
gRodon predictions above five hours: train 95, val 4, test 23. Its codon-usage signal can saturate for slow growers, so errors on those rows need careful interpretation.
Clarification: Are the `log_dob_h` labels natural-log hours, and do they represent a doubling-time quantity comparable to gRodon's minimum doubling time estimate?

For the paper comparison, I recommend we use model predictions from the validation-selected checkpoint, with each unique assembly counted once. The Frontier distributed sampler can repeat validation and test rows.
