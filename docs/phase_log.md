# Phase log

What was built in each phase, and why. Kept short so it can be read before the viva.

## Phase 0 — Setup

**Built**
- Folder structure from the project spec; `src/` sub-packages are empty until their phase.
- `configs/config.yaml`: seeds, paths, the 10 SAP modules, the directed dependency edges
  (each with a one-line business reason), look-back, horizons, target switch, split ratios.
- `requirements.txt`: every top-level package pinned; CPU PyTorch index.
- `src/utils.py`: `load_config`, `get_path`, `set_seed` (one place for all seeding).
- `src/check_env.py`: checks Python 3.11, that installed versions equal the pins, and runs
  a tiny forward pass through GATConv, DCRNN, A3T-GCN and STConv.
- `run.py` (works on Windows). An optional `Makefile` that calls it is provided separately.
- Tests: config sanity, seed reproducibility, environment.

**Decisions**
- *CPU-only PyTorch.* 10 nodes × ~1,000 days is a tiny problem; a GPU adds install pain and
  nondeterminism for no real speed-up.
- *torch 2.6.0 + PyG 2.6.1 + PyG Temporal 0.56.2.* **[Corrected in Phase 4: the released
  0.56.2 does still need them.]** PyG Temporal's last release (Jul 2025)
  no longer needs `torch_scatter` / `torch_sparse`, the C++ extensions that are hard to
  install on Windows. PyG ≥ 2.3 also works without them. PyG publishes wheels for torch 2.6.
  `decorator==4.4.2` is pinned because PyG Temporal requires exactly that version.
- *Edge direction = direction of impact.* `[SD, FI]` means an SD problem (e.g. billing not
  posting) can cause FI incidents. BASIS has an edge into every module because all modules
  run on it.
- *`run.py` instead of only a Makefile.* `make` is not installed on Windows by default.
- *Environment tests fail rather than skip* when torch is missing, so a broken install
  cannot look green.

**Verified so far**
- Config, seed and runner tests pass, and ruff is clean, in the build sandbox (Python 3.13,
  no torch). The torch / PyG / PyG Temporal checks have not been run yet: the build sandbox
  cannot download packages. They must pass on the target machine before Phase 0 is done.

## Phase 1 — Data

**Built**
- `src/data/schema.py`: incident schema (8 columns), daily signals, calendar, validators that
  return errors / warnings / missing-value counts. Calendar: month-end close = last 3 working
  days + first 2 working days of the next month; quarter-end = the close of Mar/Jun/Sep/Dec.
- `src/graph/sap_graph.py`: `SAPGraph` with adjacency (row = source), PyG `edge_index`,
  optional co-occurrence edge weights computed on the training period only, and a plot.
- `src/data/generate_synthetic.py` + `text_templates.py`: the synthetic generator.
- `src/data/load_real.py`: maps a CSV export to the schema through `real_data` in the config
  (column names, priority labels, assignment groups -> modules); skips cleanly if absent.
- Tests: `test_schema.py`, `test_graph.py`, `test_generator.py`, `test_load_real.py`.

**How the generator works** (one line each, for the viva)
- Each module-day count is a sum of independent Poisson parts: *base* (base rate x weekday /
  weekend x month-end and quarter-end close x hidden stress), *change* (a transport raises the
  rate on the next 1-3 days) and *propagation* (incidents in module j raise the rate of module i
  on the next 1-3 days with strength W[j, i]). This is a discrete-time multivariate Hawkes
  process; the spectral radius of W (0.22) is below 1, so cascades die out.
- Because the parts are separate Poisson draws, every incident has a known cause, and turning
  propagation off leaves everything else identical (a clean test of the mechanism).
- Weekend reporting also damps change and propagation incidents (fewer users report problems).
- Job failures and IDoc errors are noisy readings of the hidden stress state plus that day's
  job / interface incidents. They are only weakly related to the next day's incidents
  (correlation 0.08 and 0.16 in `results/data_summary.json`), so expect a small gain at most
  from auxiliary signals.
- The true W differs from the hand-built graph on purpose (QM -> MM has no effect; SD -> WM
  and MM -> PP are real but missing). A real expert graph is never perfect either.
- Ticket text: 8 categories x 5 templates, module vocabularies, prefixes, 10% vague tickets,
  and the same object/job named in the description and the resolution.

**Generated data** (from `results/data_summary.json`, seed 42, SYNTHETIC)
- 8,724 incidents over 1,096 days (2023-01-01 to 2025-12-31), 7.96 per day; 17.6% P1/P2.
- Causes: 62.9% base, 4.7% change, 32.4% propagation.
- Many zero days, as in real ticket data: from 26% of days (FI) to 80% (QM).
- Close-day / other-weekday ratio: FI 1.54, CO 1.42, SD 1.64; other modules 0.87-1.17.

**Decisions**
- *Categories 5 -> 8.* With only 5 categories, a random top-5 retrieval would already share
  the category most of the time, making the recommender's hit@5 meaningless. Phase 7 will also
  report a random-retrieval baseline.
- *Public holidays are not modelled* (1 January is treated as a working day). Documented as a
  limitation.
- *No public stand-in dataset yet.* Public IT-incident logs have no SAP modules, so mapping them
  to modules would be artificial; the adapter is ready for a real export instead.

**Verified**
- 48 tests pass and ruff is clean in the build sandbox (Python 3.13, pandas 3.0, numpy 2.5).
- Smoke check with pandas 2.3.3 / numpy 2.2.6 (close to the pinned 2.2.3 / 2.2.5): the generator
  produces an identical dataset (same 8,724 incidents and texts), validation passes, the
  propagation-off comparison, calendar, leakage-free co-occurrence weights and real-data
  adapter all behave as in the tests.
- Not yet run on the pinned Python 3.11 environment (install pending on the target machine).

## Phase 2 — Dataset and features

**Built**
- `src/data/build_dataset.py`: `build_daily` (daily tensor), `chronological_split`,
  `rolling_origin_splits`, `Scaler`, `make_windows`, `prepare` (scaled windows for one split),
  `tabular_features` (flat lag features for LightGBM), `load_source` (synthetic | real).
- `tests/test_build_dataset.py`: 15 tests (shapes, aggregation, window alignment, splits,
  leakage, causality, tabular features, ablation switches, save / load).

**Shapes** (from `results/dataset_summary.json`, SYNTHETIC)
- Daily tensor [1096, 10, 16]; look-back 28; forecast days 1..7.
- Train 2023-01-01..2025-02-05: X [733, 28, 10, 16], Y [733, 7, 10]
- Val 2025-02-06..2025-07-20: 159 windows; test 2025-07-21..2025-12-31: 158 windows.
- Rolling-origin: 4 folds with 60-day validation and 60-day test windows, the last ending
  2025-12-31.
- P1/P2 on a module-day (classification positives): 12.8% train, 10.5% val, 12.6% test.
  The validation period is a little quieter than train and test (0.69 vs 0.82 / 0.78
  incidents per module-day), which is normal variation and worth remembering when comparing
  validation and test scores.

**Decisions**
- *Targets are per-day counts at t+h* (not the total over the next h days). This matches
  the multi-step setup of STGCN / DCRNN / Graph WaveNet papers, so results are comparable
  in form. Models predict all 7 days at once; metrics are reported at h = 1, 3, 7.
- *Classification*: at least one P1/P2 on day t+h; the headline metric uses h = 1
  ("next 24 hours").
- *Calendar of the target days is an input* (`Xf`). It is known in advance (month-end is
  on the calendar), so it is not leakage, and it lets models anticipate close-day peaks.
- *Boundary rule*: a sample is in a split only if all 7 target days are inside it. This drops
  6 samples per boundary but guarantees no target is shared between train / val / test.
- *Scaling*: count-like features get log1p then z-score (counts are skewed), pooled over
  modules so the absolute level stays visible; calendar features are left as 0/1 and sin/cos.
  Targets are not scaled (Poisson loss needs raw counts).
- *Rolling means use min_periods=1*, so in the first 27 days they average fewer days. Those
  days only appear as input history in the earliest training windows.
- *Lags live in `tabular_features`*: a deep model sees the whole 28-day window, so lag
  columns would be redundant there; LightGBM needs them explicitly. Lags reaching before the
  first day are NaN, never wrapped around.
- *Real data never falls back silently to synthetic*: `data.source: real` without an export
  raises an error, so results cannot be mislabelled.

**How leakage is ruled out** (viva answer)
1. Features on day d only use days <= d (test: adding incidents on day d leaves all earlier
   feature rows identical).
2. Splits are by time with the boundary rule above (test: target-day sets do not overlap).
3. The scaler is fit on training days only (test: multiplying every validation/test incident
   by 4 and changing all later signals leaves the scaler and every training window
   bit-for-bit identical).
4. LightGBM features use the origin day and earlier, plus the target day's calendar (test:
   adding incidents after the origin changes only the target column).

## Phase 3 — Baselines

**Built**
- `src/metrics.py`: MAE, RMSE, Poisson deviance, precision@k and recall@k (tie-aware),
  PR-AUC, ROC-AUC, F1-optimal threshold (chosen on validation). Shared by all later models.
- `src/models/baselines.py`: naive, seasonal naive, historical mean (per module and weekday,
  training days only), LightGBM (one global model per forecast day, module as a categorical
  feature, Poisson objective for counts / binary for P1/P2, early stopping on validation).
- `src/train.py`: runs everything for both targets; saves `results/baselines.json`,
  `results/baselines_{regression,classification}.csv`, `results/baselines_table.md`,
  `results/predictions/baselines_*.npz`, and LightGBM gain importance when it runs.
- Tests: `test_metrics.py` (4), `test_baselines.py` (8).

**Results so far** (test split, synthetic data, `results/baselines.json`)
- Historical mean is the strongest simple baseline: MAE@1 0.720 vs 0.854 (naive) and
  0.842 (seasonal naive); Poisson deviance 1.10 vs 2.45 / 2.39.
- The naive baselines are worse than the mean because daily counts are mostly Poisson noise
  around a slowly varying level: copying one noisy day forward copies the noise.
- P1/P2 ranking: historical mean PR-AUC@1 0.214 (random ≈ 0.12), R@3 0.497 (random ≈ 0.30).
- LightGBM: not run at the time; see the Phase 3 addendum under Phase 4 for its results.

**Decisions**
- *Precision@3 in regression mode* counts a hit when the module had any incident. A fixed
  ranking of the busiest modules (what the historical mean almost is) already gets 0.686,
  so P@3 must be read against that number. The P1/P2 version (classification) is the more
  telling one for a support team.
- *Recall on P1/P2 is reported as R@3*: the share of P1/P2 module-days caught when the team
  looks at 3 modules per day. It needs no threshold and matches how precision@3 is defined.
  A threshold-based recall (F1-optimal threshold from validation) is kept in the JSON, but
  for weak scores it degenerates to "flag everything" (recall 1.0, precision = base rate),
  which is why it is not the headline number.
- *Poisson deviance floors predictions at 0.05*, because a point forecast of 0 for a day
  with incidents has infinite deviance. This mainly penalises the naive forecasts.
- *LightGBM uses the direct strategy*: one model per forecast day h (7 models per target),
  rather than feeding its own predictions back in. Its number of trees is chosen by early
  stopping on validation, so its validation scores are optimistic; compare on test.
- *`src/metrics.py` is a separate module* (not in the original layout) because every phase
  from here on needs the same metrics; `src/evaluate.py` (Phase 6) will use it.

**Verified**
- 75 tests pass, ruff clean (sandbox, pandas 3.0). Simple baselines were run for real.
- LightGBM could not be installed here. Its code path (features, categorical module column,
  early stopping, reshaping predictions, importance) was exercised with a scikit-learn
  stand-in that mimics LightGBM's interface, kept outside the project; those numbers are not
  LightGBM results and are not reported. The real LightGBM run happens on the target machine.

## Phase 3 addendum — LightGBM run (Python 3.11, pinned versions)

- LightGBM 4.6.0, test split: MAE@1 0.704 (historical mean 0.720), Poisson deviance@1 1.051
  (1.099), P1/P2 PR-AUC@1 0.261 (0.214), ROC-AUC@1 0.724 (0.699). P@3@1 0.673 is slightly
  below the historical mean's 0.686.
- Early stopping chose 267-373 trees for counts and 79-240 for P1/P2 (`results/baselines.json`).
- Gain importance at h = 1 (`results/lightgbm_importance_*.csv`): module identity 38% (counts)
  / 24% (P1/P2), target-day weekend flag 22% / 17%, 14- and 28-day means next; change events
  in the last 3 days 4% (counts), job failures 5% (P1/P2). Yesterday's count alone matters
  little: daily counts are noisy, the level is what is predictable.

## Phase 4 — Temporal deep models (no graph)

**Built**
- `src/models/temporal.py`: `TemporalModel` with an LSTM or a Transformer encoder. Each module's
  28-day window is encoded separately with shared weights plus a learned module embedding.
  A `ForecastHead` combines the encoding with the calendar of each target day and a
  horizon embedding, so one forward pass predicts days 1-7.
- `src/models/trainer.py`: one training loop for every neural model (also used in Phase 5).
  Poisson NLL on a log-rate output (counts) or BCE with positives weighted by
  negatives/positives (P1/P2); Adam + weight decay, learning rate halved on plateaus,
  gradient clipping at 1.0, early stopping (patience 15) restoring the best weights. The
  output bias starts at log(mean count) or logit(positive rate).
- `src/train.py`: now runs any of baselines / lstm / transformer per target and seed; each
  run is saved separately (`results/runs/`, `results/predictions/`, checkpoints in
  `results/checkpoints/`) and `results/comparison.md` (mean ± std over seeds) is rebuilt from
  all saved runs.
- `src/metrics.py`: `score()` so baselines and neural models are scored by the same code.
- Tests: `test_temporal.py` (output shapes incl. 14-day look-back; changing one module's input
  leaves every other module's forecast unchanged - the proof that these models have no
  graph; objectives; a short end-to-end training run).

**Results** (test split, SYNTHETIC, mean ± std over seeds 42/43/44, `results/comparison.md`)
- Counts: LSTM MAE@1 0.702 ± 0.004, Transformer 0.709 ± 0.006, LightGBM 0.704,
  historical mean 0.720. RMSE / Poisson deviance: LightGBM best (0.924 / 1.051), LSTM
  0.932 / 1.064, Transformer 0.942 / 1.083.
- P1/P2: PR-AUC@1 LightGBM 0.261, Transformer 0.251 ± 0.002, LSTM 0.232 ± 0.012,
  historical mean 0.214.
- Training: best validation epoch 6-14 (stopped at 21-29); LSTM 40k parameters, ~40-55 s per
  run; Transformer 76k parameters, ~95-130 s per run (2 CPU threads).

**Honest interpretation** (viva)
- The temporal deep models do not beat LightGBM, so by our rule they are not an improvement.
- Every learned model beats the simple baselines by a small margin (MAE about 2-3% better
  than the historical mean): most of a module's daily count is Poisson noise around a level
  set by the module, the weekday, the close calendar and a slowly drifting hidden stress
  state. That level is what all models learn; the noise cannot be predicted.
- No model so far sees other modules, yet about a third of synthetic incidents are ripples
  from upstream modules. That is the information the graph models in Phase 5 add. If they
  also fail to beat LightGBM, that is a valid finding to report, with the reason.
- The neural models overfit quickly (best epoch 6-14) because there are only 733 training
  windows; this is a known limit of deep models on short daily series.

**Environment** (where these numbers come from)
- Python 3.11.17 on Linux (the Linux sandbox of the target machine), every pinned version
  installed except the PyG Temporal stack. torch 2.6.0 came from PyPI (its CUDA build, run on
  CPU) because PyTorch's CPU index was not reachable there; same version and Python API.
- Synthetic data regenerated there with the pinned pandas 2.2.3 / numpy 2.2.5 is
  byte-for-byte identical to the build-sandbox version (pandas 3.0 / numpy 2.5).
- 85 tests pass; the 2 failures are the PyG Temporal environment checks. ruff 0.11.8 clean.

**Fixes made in this phase**
- `requirements.txt`: the published torch-geometric-temporal 0.56.2 still declares
  torch-scatter and torch-sparse and imports torch_sparse when the package loads (through
  EvolveGCN-O). The Phase 0 note that it no longer needed them was wrong: I had read the
  GitHub setup.py, not the released package. Both are now pinned and come as prebuilt wheels
  from PyG's index (verified to exist for torch 2.6 CPU, Python 3.11, Windows and Linux).
- `src/check_env.py`: reads the PyG Temporal version from package metadata (its own
  `__version__` string still says 0.54.0).

## Phase 5 — Spatio-temporal graph models

**Built**
- `src/models/gat_transformer.py` (main model): per-day GAT over the SAP graph -> Transformer
  over each module's daily embeddings -> shared ForecastHead. Graph modes `hand_built`,
  `adaptive` (learned node-embedding adjacency, Graph WaveNet style), `both` (default, used
  in the results), `none` (same network without the spatial layer, for the Phase 6
  with/without-graph ablation). The spatial branch is residual, so the model can learn to
  ignore the graph.
- `src/models/stgnn.py`: STGCN-style (gated temporal convolutions around a PyG ChebConv,
  undirected graph) and DCRNN-style (GRU with forward/backward diffusion convolution on the
  directed graph; encoder + direct multi-horizon head instead of seq2seq with scheduled
  sampling). Both are re-implementations because PyTorch Geometric Temporal could not be
  installed (below). The spec allows "the closest available equivalents".
- `src/models/trainer.py`: resumable training. Full state saved every epoch (weights,
  optimiser, scheduler, early-stopping counters, both random-number streams); a time budget
  pauses a run and the same command resumes it. A test proves a run paused after every
  epoch ends bit-for-bit identical to an uninterrupted one.
- `src/train.py`: models `stgcn`, `dcrnn`, `gat_transformer` (+ ablation variants
  `gat_transformer_hand_built`, `_adaptive`, `_none`), `--time-budget`, `--resume-dir`;
  every run saves GAT attention per edge and the learned adjacency for Phase 6.
- Tests: `test_gat_transformer.py` (8), `test_stgnn.py` (7), resume test in
  `test_temporal.py`. Full suite: 101 pass in the Python 3.11 environment; the 2 failures are
  the PyG Temporal environment checks.

**Results** (test split, SYNTHETIC, mean ± std over seeds 42/43/44, `results/comparison.md`)
- Counts, MAE@1: LSTM 0.702 ± 0.004, LightGBM 0.704, Transformer 0.709 ± 0.006,
  DCRNN 0.709 ± 0.002, GAT-Transformer 0.713 ± 0.007, STGCN 0.716 ± 0.001,
  historical mean 0.720. Poisson deviance@1: LightGBM 1.051 best; graph models 1.083-1.108.
- P1/P2, PR-AUC@1: LightGBM 0.261, Transformer 0.251 ± 0.002, DCRNN 0.250 ± 0.009,
  GAT-Transformer 0.239 ± 0.009, LSTM 0.232 ± 0.012, STGCN 0.230 ± 0.005.
- Training: best validation epoch 4-13 for every graph model; 84k (GAT-Transformer),
  139k (STGCN), 90k (DCRNN) parameters; 80-133 s per run on 2 CPU threads.

**Honest interpretation** (viva)
- Adding the graph did not improve the forecasts on this data. DCRNN, which keeps edge
  direction, does best of the three graph models and matches the Transformer; STGCN, which
  needs an undirected graph, does worst.
- Why might a third of incidents being ripples not help? Hypotheses to test in Phase 6, not
  conclusions yet: (1) a module's own rolling level already absorbs the AVERAGE ripple
  inflow, and the day-to-day deviation of a few upstream counts is small next to Poisson
  noise, so even a perfect model gains little; (2) with 733 training windows the extra
  parameters overfit (best epoch 4-13); (3) the hand-built graph is partly wrong by design
  (one useless edge, two missing), which the learned adjacency may or may not fix.
- Phase 6 will measure (1) directly with an "oracle" forecast that knows the generator's
  true rates, and check (3) by comparing attention / learned adjacency with the true matrix.

**PyTorch Geometric Temporal: what happened**
- The released torch-geometric-temporal 0.56.2 imports torch_sparse when it loads. With
  approval, torch-scatter 2.1.2 was compiled from its PyPI source in the Linux sandbox and
  works. torch-sparse 0.6.18 could not be built there: compiling its `hgt_sample` extension
  restarted the sandbox twice (also with debug info off and a memory cap), so it was stopped.
  The build ran outside the project folder; the sandbox and all results survived.
- On the Windows machine, `requirements.txt` installs torch-scatter / torch-sparse as
  prebuilt wheels from PyG's index, so the library itself should work there (not verified).

**Bugs caught by tests in this phase**
- Attention extraction: PyG appends all self-loops after all edges of a batched graph;
  attention must be split into edges and self-loops before averaging over graphs.
- STGCN normalisation: reference implementations apply LayerNorm over [nodes, channels]
  jointly, which shares statistics between modules - information crossing between modules
  outside the graph. Caught by the "no edges -> no cross-module influence" test; fixed by
  normalising each module separately (a documented deviation from the reference).

## Phase 6 — Evaluation, ablations, explainability

**Built**
- `src/evaluate.py` -> `results/summary.md`, `results/evaluation.json`, figures
  `horizon_mae.png`, `rolling_origin.png`, `explain_adjacency.png`, `lightgbm_importance.png`.
- Generator now also saves the true expected count of every module-day
  (`data/synthetic/ground_truth_rates.csv`); the data itself is byte-for-byte unchanged.
- `train.py`: `--lookback`, `--no-aux`, `--split foldK`; `prepare(drop_groups=...)`.
- 32 new runs (graph ablations x both targets x 3 seeds; look-back 14 and no-aux on counts
  x 3 seeds; LSTM and GAT-Transformer on 4 rolling-origin folds). `tests/test_evaluate.py`.
  Full suite: 103 pass (2 failures = PyG Temporal environment checks).

**1. Oracle ceiling - how much is predictable at all** (test split, counts)
- A forecast from the generator's TRUE rates (it even knows the hidden stress state) has
  MAE@1 0.659 and Poisson deviance@1 0.979. The best real models: LSTM 0.702, LightGBM 0.704.
- Only 38.1% of the day-to-day variance of the counts is predictable even in principle;
  the rest is Poisson noise.
- The same oracle WITHOUT ripple knowledge (ripple part replaced by its training average
  per module and weekday) has MAE@1 0.679. So perfect knowledge of every ripple is worth
  0.020 MAE on this data, although ripples are 33% of expected incidents: the AVERAGE ripple
  inflow is part of a module's normal level, which every model learns anyway; only the
  day-to-day deviation is extra information, and it is small.
- The best models are 0.023 above even the ripple-blind oracle. Most of the remaining gap is
  NOT about ripples (hidden stress state, exact calendar effects, finite training data).
- This answers Phase 5's question: a graph model could gain at most about 0.02 MAE here,
  which is the same size as the differences between seeds. Not finding a gain is
  consistent with the data, not a sign of a broken model.

**2. Ablations of the main model** (paired with the main model, same seed)
- No graph vs main model: MAE@1 -0.004 ± 0.009, PR-AUC@1 +0.013 ± 0.008. Removing the
  graph does not hurt; for P1/P2 it is slightly better.
- Hand-built graph only / learned adjacency only: within ± 0.007 MAE and ± 0.004 PR-AUC of the
  main model - no difference beyond seed noise.
- Look-back 14 vs 28 days: MAE@1 -0.004 ± 0.005 (no loss with half the history).
- Without auxiliary signals: MAE@1 +0.000 ± 0.009 (no effect, as the weak correlations in
  Phase 1 predicted).

**3. Rolling-origin stability** (test MAE@1, counts, 4 windows of 60 days, seed 42)
- LightGBM 0.684 ± 0.026, LSTM 0.690 ± 0.023, GAT-Transformer 0.692 ± 0.021,
  historical mean 0.705 ± 0.013. The ranking is stable: LightGBM is best or tied-best (fold 2: 0.690 vs LSTM 0.690) in 3 of 4 windows,
  every learned model beats the historical mean except in the last window (Nov-Dec), where
  all are within 0.01 of each other.

**4. Explainability vs the true propagation matrix** - an honest negative result
- Learned adjacency (main model, 6 runs): Spearman with true W -0.23; separating true edges
  from non-edges AUC 0.33 (0.5 = random); within-target rank correlation -0.29. The two
  hidden real edges rank 12th (MM -> PP) and 61st (SD -> WM) of 90.
- Learned adjacency only (6 runs): Spearman -0.09, AUC 0.44, within-target -0.18.
- GAT attention (12 runs): within-target rank correlation with true strengths -0.13. The
  useless edge QM -> MM gets 0.155 attention, as much as the real edges into MM (0.144).
- Conclusion: the models did NOT learn the propagation mechanism. That fits section 1:
  ripple variation carries so little predictive signal that the training loss gives the
  models almost no reason to discover it. Raw attention is also confounded by
  normalisation (a target with one incoming edge must put its weight there - e.g. BASIS is
  the only source of SD, PP, WM, QM, PM, HCM), which is why within-target comparisons are
  reported too.

**5. LightGBM importance** (gain share, h = 1): module identity 0.38, target-day weekend
  0.22, 28-day mean 0.085 for counts; change events in the last 3 days 0.041.

**What this means for the project (viva)**
- The pipeline works and is honest: leakage-free splits, baselines first, 3 seeds, paired
  ablations, an oracle ceiling and a ground-truth check of explanations.
- On this synthetic data the graph adds nothing measurable, and the reason is quantified:
  the extra information in the graph is worth at most about 0.02 MAE.
- What would change it: stronger or more bursty propagation (major outages that cascade),
  aggregation to coarser targets (e.g. incidents over the next 7 days, where noise
  averages out), or real data with stronger cross-module coupling. These are future work,
  not changes made to the data to make the graph look good.

## Phase 7 — Resolution recommender

**Built**
- `src/recommender/embedders.py`: `minilm` (sentence-transformers all-MiniLM-L6-v2, from
  `models/all-MiniLM-L6-v2` if present, else the Hub), `minilm_onnx` (the same model as
  ChromaDB's ONNX build, from `models/onnx.tar.gz`, accepted only if its SHA-256 equals the one
  pinned inside chromadb), `tfidf` (keyword baseline). An unavailable embedder raises an
  error and is reported as "not available" - never silently replaced.
- `src/recommender/build_index.py`: embeds "module | category | description" of every
  TRAINING-period ticket into a ChromaDB collection (cosine) with its resolution notes.
  Index folder: `$CHROMA_DIR` if set, else `results/chroma` (git-ignored).
- `src/recommender/recommend.py`: `Recommender.similar(description, module)` for a new
  ticket; `Recommender.for_module(module, before)` for a module flagged by the forecast (its
  most frequent recent category, then the most similar past tickets of that kind);
  `evaluate()` -> `results/recommender.json`, `results/recommender.md`.
- `run.py index` builds the index and runs the evaluation. Tests: `test_recommender.py` (5).
- `requirements.txt`: `posthog==5.4.0`, because chromadb 0.6.3 calls the old posthog API and
  otherwise prints "Failed to send telemetry" on every call.

**Evaluation** (SYNTHETIC; index 6,305 tickets up to 2025-02-06; 1,287 held-out test-period
tickets; query = "module | description", since a new ticket's category is unknown)

| retriever | hit@1 | hit@5 | precision@5 | MRR |
|---|---|---|---|---|
| TF-IDF | 0.908 | 0.943 | 0.888 | 0.920 |
| MiniLM (sentence-transformers / ONNX) | not run: model not reachable from the sandbox | | | |
| random past ticket (expected) | 0.158 | 0.546 | 0.158 | - |
| random past ticket of the same module | 0.183 | 0.588 | 0.183 | - |

- hit@5 by category ranges from 0.875 (output/print) to 0.982 (posting errors); 83% of the
  retrieved tickets come from the same module.

**Honest interpretation** (viva)
- Retrieval works far above both random baselines, but the numbers are high mainly because
  the synthetic ticket text comes from templates with category-specific words ("IDoc",
  "authorisation", "job"): keyword matching is enough. The misses are mostly the 10%
  deliberately vague tickets. Real tickets (free text, typos, mixed languages) would be
  much harder, which is where a semantic model like MiniLM should help - not measured here.
- hit@5 with 8 categories is a lenient metric (random already gets 0.55), which is why
  precision@5, hit@1 and the random baselines are reported next to it.
- ChromaDB's HNSW search is approximate: rebuilding the index can move a metric in the third
  decimal (MRR 0.921 vs 0.920 in two builds).
- Optional LLM summary of the fixes (`llm_summary`): not implemented.

**Blocked**: MiniLM could not be evaluated - Hugging Face and Chroma's model server are not
reachable from the Linux sandbox. Providing `models/onnx.tar.gz` (Chroma's archive) or
`models/all-MiniLM-L6-v2/` unblocks it with no code change.

## Phase 8 — API, dashboard, Docker

**Built**
- `src/api/service.py`: loads the served model's two checkpoints (counts and P1/P2) with the
  scaler saved at training time; builds the input window for any date (future calendar
  days computed, as they are known in advance); forecast, explain (GAT attention into the
  module + learned adjacency), history (test period only).
- `src/api/main.py`: FastAPI with Pydantic schemas; models and index load on first use;
  bad input -> 422 with a clear message, missing artifacts -> 503 "run `python run.py ...`".
- `src/dashboard/app.py`: Streamlit; synthetic banner, risk-coloured module graph, top-3 with
  influences and past fixes, forecast vs actual, free-text ticket search, model comparison.
- `Dockerfile`, `.dockerignore`, `docker-compose.yml` (api + dashboard, health-checked).
- Tests: `test_api.py` (15, TestClient), `test_dashboard.py` (3, Streamlit AppTest with the
  dashboard's HTTP calls routed into the real API, including "API down").

**Verified** (Linux sandbox, Python 3.11, real checkpoints and index)
- Full suite 126 pass (2 failures = PyG Temporal environment checks); ruff clean.
- The exact commands from docker-compose.yml were run: uvicorn API and headless Streamlit
  both answered over HTTP (health, forecast, 422 on a bad date, recommend; dashboard page
  200). Not run: building the image and `docker compose up` (no Docker available).

**Decisions** (viva)
- *Risk score naming*: the P1/P2 model was trained with positive-class weighting, so its
  outputs rank well but are inflated (e.g. 0.75 for FI where the base rate is ~0.12). The
  field is called `p1p2_risk` with a note, not "probability". Calibration (e.g. isotonic on
  the validation split) is future work.
- *Served model*: GAT-Transformer, because `/explain` needs attention, although LightGBM
  scored best (Phase 6). Documented, configurable.
- *Explanations carry a caveat* pointing to Phase 6: attention did not match the true
  propagation, so it must not be presented as causes.
- *Artifacts are mounted, not baked into the image*: the image stays small and always serves
  what was last trained.
- *Recommender embedder for the API*: TF-IDF (the evaluated one); MiniLM switchable once its
  model file is available.
- A robustness bug found on the way: after `st.stop()` the dashboard code fell through when
  imported outside Streamlit; explicit returns added.

## Phase 9 — Quality: end-to-end script, tests, lint, README

**Built**
- `python run.py all`: data -> all models -> ablations and rolling-origin folds -> evaluate
  -> recommender index (14 steps; the 32 Phase 6 runs are now part of the pipeline).
- Config inheritance: `load_config` honours `$SAP_CONFIG`, and a config can start with
  `base: other.yaml`. `configs/quick.yaml` (400 days, 1 seed, 2 epochs) writes to
  `data_quick/` and `results_quick/` (git-ignored) so it cannot overwrite real results.
- `tests/conftest.py`: clears `SAP_CONFIG` for the test run; a `data_ready` fixture skips
  (with the stage to run) the tests that need generated data in a fresh clone.
- README rewritten: problem, approach, Mermaid architecture diagram, how to run, a table of
  what was and was not verified, results copied from `results/`, limitations.

**Removed**
- PyTorch Geometric Temporal and its add-ons (torch-scatter, torch-sparse, decorator, cython,
  the PyG wheel index) from `requirements.txt`. No code has used them since Phase 5, and they
  caused every install problem and the only 2 failing tests. `check-env` now smoke-tests the
  PyG layers the models use (GATConv, ChebConv). To restore the library for a comparison:
  add `torch-geometric-temporal==0.56.2`, `torch-scatter==2.1.2`, `torch-sparse==0.6.18`,
  `decorator==4.4.2` and `--find-links https://data.pyg.org/whl/torch-2.6.0+cpu.html`.

**Bug found by the clean-clone test**
- `.gitignore` had `models/` (meant for downloaded model files). Unanchored, it also matched
  `src/models/`, so git would not have committed any model code and a clone would fail with
  `No module named 'src.models'`. Changed to `/models/`; a test now guards it.

**Verified** (Linux sandbox, Python 3.11)
- `python run.py check-env`: OK (all pins match).
- Full suite: 133 passed, 0 failed, 0 skipped (data, checkpoints and index present).
- Fresh `git clone` of a snapshot (216 tracked files), no data: 113 passed, 20 skipped.
- `ruff check .`: no findings.
- `SAP_CONFIG=configs/quick.yaml python run.py all` in the fresh clone, as ONE command:
  exit 0, 14 steps, 163 s. Its numbers are a smoke test only and are not reported anywhere.
- The Mermaid diagram in the README renders (checked with the Mermaid CLI).

**Not verified**
- `python run.py all` with the full config as one command (sandbox limit of 3 minutes per
  command). The results in `results/` were produced by the same commands run one at a time.
- Windows install; `docker compose up --build`; MiniLM.

**Known gap, documented in the README**
- Some figure titles in `src/evaluate.py` and the header of `results/recommender.md` say
  "SYNTHETIC DATA" unconditionally. Correct today; must be made conditional before a
  real-data run.
