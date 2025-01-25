# Sampling Results (mean ± std across trials)

CV values in %, Batch Time in ms

| Configuration                | Individual CV   | Relationship CV   | Family CV    | Batch Time (s)   |
|:-----------------------------|:----------------|:------------------|:-------------|:-----------------|
| baseline                     | 89.73 ± 0.00    | 78.12 ± 0.00      | 77.45 ± 0.00 | 0.0 ± 0.0        |
| sampler_random               | 46.36 ± 0.38    | 63.03 ± 0.27      | 70.86 ± 0.58 | 5.3 ± 0.4        |
| sampler_random_limited       | 46.31 ± 0.32    | 62.93 ± 0.40      | 71.35 ± 0.84 | 5.5 ± 0.3        |
| sampler_relationship         | 54.55 ± 0.60    | 47.94 ± 0.51      | 78.71 ± 1.18 | 164.1 ± 13.3     |
| sampler_relationship_limited | 55.42 ± 0.81    | 48.33 ± 1.15      | 79.56 ± 0.23 | 25.3 ± 2.6       |
| sampler_difficulty           | 49.19 ± 0.40    | 62.48 ± 0.35      | 75.30 ± 0.42 | 164.7 ± 18.5     |
| sampler_difficulty_limited   | 49.32 ± 0.31    | 62.28 ± 0.56      | 74.78 ± 1.03 | 27.4 ± 3.4       |
| sampler_balanced             | 45.13 ± 0.57    | 54.59 ± 0.25      | 69.03 ± 1.54 | 180.3 ± 16.6     |
| sampler_balanced_limited     | 46.52 ± 0.71    | 53.82 ± 0.34      | 70.18 ± 1.28 | 27.4 ± 3.2       |
