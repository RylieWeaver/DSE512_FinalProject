# Doubling-time baseline results

The fixed assembly-level split has 301 train, 25 validation, and 74 test examples. Mean and temperature regression are fitted on train only.

MSE uses the training-standardized `log_dob_h` target, matching the sequence regression trainer. Each assembly is scored once.

| Cohort | Split | Method | N | MSE | RMSE (log hours) |
| --- | --- | --- | ---: | ---: | ---: |
| all | val | train_mean | 25 | 0.6676 | 1.2114 |
| all | val | temperature_linear | 25 | 0.6861 | 1.2281 |
| grodon_matched | val | train_mean | 25 | 0.6676 | 1.2114 |
| grodon_matched | val | temperature_linear | 25 | 0.6861 | 1.2281 |
| grodon_matched | val | grodon | 25 | 0.5043 | 1.0529 |
| all | test | train_mean | 74 | 0.8885 | 1.3976 |
| all | test | temperature_linear | 74 | 0.7945 | 1.3215 |
| grodon_matched | test | train_mean | 72 | 0.9046 | 1.4102 |
| grodon_matched | test | temperature_linear | 72 | 0.8143 | 1.3380 |
| grodon_matched | test | grodon | 72 | 0.7669 | 1.2984 |

The train mean predicts one constant log doubling time. Temperature regression fits an intercept and slope using only training rows. gRodon uses genome-wide annotated coding genes and ribosomal-protein labels with `mode=full`, the `madin` training set, and no temperature correction.

gRodon coverage (valid predictions): train 285/301, val 25/25, test 72/74.
The `grodon_matched` cohort compares all methods on those same assemblies. Review `summary.json` and the gRodon status CSV for failures.
On validation, gRodon MSE was 0.5043 versus 0.6861 for temperature regression. On the covered test assemblies, the scores were 0.7669 and 0.8143, respectively. gRodon uses additional genome-wide annotation, so this comparison describes predictive performance with different input information.
gRodon predictions above five hours: train 95, val 4, test 23. Its codon-usage signal can saturate for slow growers, so errors on those rows need careful interpretation.
Clarification:
Is the `log_dob_h` labels natural-log hours and represent a doubling-time quantity comparable to gRodon's minimum doubling time estimate.

For the paper comparison, I recommend we use model predictions from the validation-selected checkpoint, with each unique assembly counted once. The Frontier distributed sampler can repeat validation and test rows.
