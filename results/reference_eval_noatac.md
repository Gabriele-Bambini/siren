# Reference-class evaluation (corrected protocol)

- ATAC features available: False; backbone = GBM dist
- bootstrap: pigeonhole (element x gene), B=2000; deltas vs -distance

## E1 chr19 (pre-registered)

| scorer | AUROC | AP | prec@2% | enrich | dAP vs -dist [95% CI] |
|---|---|---|---|---|---|
| -distance | 0.8229 | 0.3477 | 0.4222 | 16.45x | ref |
| H3K27ac RPM | 0.672 | 0.0562 | 0.0889 | 3.46x | -0.2915 [-0.5151, -0.0923] |
| DHS RPM | 0.655 | 0.0476 | 0.0571 | 2.23x | -0.3001 [-0.5269, -0.1041] |
| GBM dist | 0.8174 | 0.3117 | 0.3833 | 14.94x | -0.036 [-0.1148, 0.0554] |
| GBM dist (monotone, diagnostic) | 0.8222 | 0.3024 | 0.4284 | 16.69x | -0.0453 [-0.0976, -0.0029] |

## E2 leave-one-chromosome-out

| scorer | AUROC | AP | prec@2% | enrich | dAP vs -dist [95% CI] |
|---|---|---|---|---|---|
| -distance | 0.8534 | 0.4089 | 0.6135 | 11.64x | ref |
| H3K27ac RPM | 0.7232 | 0.1132 | 0.1304 | 2.47x | -0.2957 [-0.3776, -0.2124] |
| DHS RPM | 0.7072 | 0.1104 | 0.1498 | 2.84x | -0.2985 [-0.3793, -0.2119] |
| GBM dist | 0.8127 | 0.3437 | 0.5362 | 10.17x | -0.0652 [-0.0949, -0.0334] |
| GBM dist (monotone, diagnostic) | 0.8399 | 0.3707 | 0.6086 | 11.54x | -0.0382 [-0.0601, -0.0133] |

## E2 per screen (LOCO predictions)

- **Gasperini et al., 2019** (n=5299, prevalence 0.0759): -distance AP 0.4594 / prec@2% 0.6952; GBM dist AP 0.4053 / prec@2% 0.6571 (dAP -0.0542 [-0.0888, -0.0205])
- **Fulco et al., 2019** (n=3501, prevalence 0.0234): -distance AP 0.2847 / prec@2% 0.3; GBM dist AP 0.2447 / prec@2% 0.2905 (dAP -0.04 [-0.0949, 0.0278])
- **Schraivogel et al., 2020** (n=1306, prevalence 0.0191): -distance AP 0.515 / prec@2% 0.4231; GBM dist AP 0.396 / prec@2% 0.375 (dAP -0.119 [-0.2948, 0.0404])

## E3 K562 held-out screens

- **all** (n=1918, prevalence 0.0892): -distance AP 0.3295 / prec@2% 0.6053; H3K27ac RPM AP 0.22 / prec@2% 0.2895; DHS RPM AP 0.1974 / prec@2% 0.2368; GBM dist AP 0.2997 / prec@2% 0.5556; GBM dist (monotone, diagnostic) AP 0.3077 / prec@2% 0.5578
- **pair_disjoint** (n=1918, prevalence 0.0892): -distance AP 0.3295 / prec@2% 0.6053; H3K27ac RPM AP 0.22 / prec@2% 0.2895; DHS RPM AP 0.1974 / prec@2% 0.2368; GBM dist AP 0.2997 / prec@2% 0.5556; GBM dist (monotone, diagnostic) AP 0.3077 / prec@2% 0.5578
- **elem_disjoint** (n=1491, prevalence 0.0657): -distance AP 0.2395 / prec@2% 0.4828; H3K27ac RPM AP 0.168 / prec@2% 0.1379; DHS RPM AP 0.1493 / prec@2% 0.1724; GBM dist AP 0.2047 / prec@2% 0.4061; GBM dist (monotone, diagnostic) AP 0.2157 / prec@2% 0.4269
- **elem_gene_disjoint** (n=558, prevalence 0.1111): -distance AP 0.2642 / prec@2% 0.6364; H3K27ac RPM AP 0.4173 / prec@2% 0.7273; DHS RPM AP 0.3607 / prec@2% 0.6364; GBM dist AP 0.2414 / prec@2% 0.6364; GBM dist (monotone, diagnostic) AP 0.2421 / prec@2% 0.6364

## E4 conformal

q90 = 0.0679 (5-fold chromosome OOF residuals on train)
- chr19: marginal 0.9027; significant pairs 0.569; by strength quintile [0.9535, 0.9501, 0.9183, 0.9224, 0.7652]
- K562_heldout_screens: marginal 0.8905; significant pairs 0.1345; by strength quintile [0.9508, 0.9476, 0.949, 0.9312, 0.6711]
