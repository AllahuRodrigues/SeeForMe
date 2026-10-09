# Reproducibility check

- CampusBench v2 test: 63 metrics compared against `results/campus_v2/summary_test_zeroshot.json`, all identical
  command: `scripts/run_campus.py --split test --repeats 1 --policies oracle,pcacs,pcacs_learned:rl --tag repro`
- EgoBlind holdout (10% unseen split, COCO caches): 100 metrics compared against `results/egoblind/summary_holdout_final.json`, all identical
  command: `scripts/run_egoblind.py --split holdout --policies pcacs,egoblind_learned:il@models/egoblind_only --tag repro`
