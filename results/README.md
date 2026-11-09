# Results

The evaluation files behind every number and figure in the README, one file per training run, all on H100 NVL GPUs.
`runs.csv` lists them: method, label, GPUs, seed and file. `python tools/make_figures.py` turns them into the figures and
tables.

Each file has one row per evaluated checkpoint (every 100 iterations): the iteration, the training time in hours
(`train_wall_h`, compilation excluded), the normalized evaluation return and the number of failed clips (both averaged
over 4 replicas of all 40 clips), and the names of the clips that failed in any replica.

- `xmimic_*`: XMimic runs, scored by `python -m xmimic.evaluate <run> --reps 4`. XMimic compiles before its clock starts.
  The runs of `xmimic_1gpu`, `xmimic_al_mps`, `xmimic_awl_mps`, `xmimic_awlp` and `instanttrack` train for at least 6
  minutes and come from one job on one 4-GPU node. Those of `xmimic_fp1`, `xmimic_fp` and `xmimic_alp` train 3500
  iterations, as every run of the first release did. The first release's runs of the other layouts (4 GPUs: 0.91
  minutes to the target) are in the history of this repository, at commit `571715f`.
- `blinkmimic`, `instanttrack`: the baselines, scored by their own `tools/eval_batch.py --reps 4`. They compile inside
  their training loop; `seed*.csv` have that one-time cost removed by `tools/exclude_compile.py` from the files in `raw/`
  (their unchanged evaluation files and their train logs).
