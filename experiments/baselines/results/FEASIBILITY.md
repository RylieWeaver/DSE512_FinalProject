# gRodon input feasibility (2026-10-01)

An NCBI Datasets CLI 18.38.0 metadata audit of the 400 assembly IDs found:

| Split | Total | NCBI annotation available | No annotation |
| --- | ---: | ---: | ---: |
| Train | 301 | 285 | 16 |
| Validation | 25 | 25 | 0 |
| Test | 74 | 72 | 2 |
| **Total** | **400** | **382** | **18** |

See `grodon_feasibility.csv` for the status of each assembly. The first
unannotated assembly, `GCA_000243155.1`, is suppressed in NCBI and its data
package contains no CDS FASTA. The final download manifest is in the ignored
`../cds/` directory. It records 382 successful CDS downloads and 18 assemblies
without NCBI annotation. The CDS files total about 1.3 GiB on D:.

gRodon 2.7.3 returned a finite prediction for every downloaded annotated CDS:
285/301 training, 25/25 validation, and 72/74 test assemblies. The 18 missing
predictions are exactly the assemblies without NCBI annotation. Other annotated
CDS sources could cover them in a future analysis. The final three-method
comparison uses the `grodon_matched` cohort produced by
`evaluate_baselines.py` and reports its sample size.

The raw labels correspond to doubling times above five hours for 102/301
training, 10/25 validation, and 18/74 test assemblies. gRodon's authors warn
that codon-usage predictions of slow growers can saturate around this range.
Keep the overall comparison, and describe this limitation when interpreting
the gRodon result.
