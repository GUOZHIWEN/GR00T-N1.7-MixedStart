# MoS Experiment Ledger (G0 → Phase 1)

Every arm below was pre-registered before execution; verdicts archived with primary metrics (`compare_metrics.json` / `phase1_metrics.json` / training `log.txt`). This is the condensed English ledger; the full Chinese archive lives in the main repo (`GR00T_N1后训练与前沿物理AI模型综述.md` §13.1–13.78).

## Master table

| Arm / round | Single variable vs previous | Deploy accuracy (mse_eff, same protocol) | Verdict | Mechanism |
|---|---|---|---|---|
| **db7** (G5 champion) | role anchor + A2/db7 data recipe + β~U(0,1) + direct_beta | synthetic eval **0.0214** (β=0); 2.24× vs zero-shot base | **reference arm** | role anchor resolves anchor-flow conflict; data-convention fixes are the convergence prerequisite |
| db8/db9 functional anchor | +L2 functional anchor (mix / pure_z0 source) | 0.0295 / 0.0301 (38–41% worse) | **worse, terminated** | Goodhart: proxy loss −43/−53% while deploy regresses — field-side shortcut |
| db10 diff anchor | +differential anchor (smoke only) | — (no long run) | archived | only worth it after λ cosine-anneal kills point anchor |
| **no_mem arms** (ablate/droid) | remove memory encoder (pure Gaussian source) | synthetic **0.0193 (beats db7)**; DROID **0.0157 = 0.0157 tie** | **controlled rejection, two domains** | zerosrc +156% "dependence" is a co-adaptation artifact; component value only counts via train-time ablation |
| **integrated_droid** | db7 recipe ported to real DROID (300 ep) | DROID eval **0.0157** (β=0) | **real-domain reference** | β̂ interior solution confirmed on real data (0.31) but not an accuracy advantage; bottom-swap (real history > zero source) |
| db11 wave probe | +oscillate segments + λ=0.3 constant | 0.0346 (db11-eval) | **F2 triggered** | frequency channel opened via text (0.0003), phase inverted by h contamination (−0.552) |
| db12 debias | +anchor β-gate (β≤0.5) + β=0 debias 0.05 | 0.0358 (within ±10% tolerance) | **mechanism closed** | inversion removed (−0.55→+0.02) but initial phase per-segment random: VL-only encoder structurally exempt from state-dependent quantities |
| **Phase 1-L1** | no training (stress evaluation) | accuracy tie extends to stress domain | P1 pass (amended) / **P2 fail** | routed is the only compounding arm (+45% over 200 steps); latency is a hard constraint (stride ratio 3.85×) |
| **Phase 1 strong-cond** | same suite on db11 eval set | accuracy cost **directional** 10–16% (four strides consistent, one significant, multi-seed CI); repeat gap **stride-dependent** (vanishes at 8/16, recovers 1.6–1.8× at 24/32, CI excludes 1) | **third rejection of accuracy value (CI-amended); determinism narrows stride-dependently** | stronger conditioning → field absorbs start-point randomness |
| **db13** rescue arm | +β≡0 same-train + L=32 bottleneck readout | teleport-head 0.1210 vs 0.0186 (**6.5× worse**) | **pre-registered gate FAIL** | segment-mid low-rank content suffices (0.90×), segment-head high-rank collapses; three collinear factors (β≡0 / bottleneck / η=0) |
| **db14** attribution | db13 + η=0.04 restored | teleport-head 0.104 (5.6× worse) | **bottleneck main cause confirmed** | η=0 consistent secondary factor (14–29% head recovery); repeat returns to η floor (0.019) |
| **db15/db16** constant start | z₀ replaced by single learnable 40×132 table | db15 (β≡0): 1.65× accuracy loss, repeat 0.0235; db16 (β~U): teleport-head 0.170, **9.2× — worst arm ever** | **FAIL in both regimes** | β=0 accuracy = source gap × deployment-path training density (joint property); per-condition anchor is the prerequisite — encoder survives the last cut |
| **mos_best / mos_best_no_mem** (release pair) | merged 17D base (300 DROID + 100 synthetic), same recipe | crown A: ×1.03 CI [0.89, 1.23] vs domain champion (directional pass); B: 2.7× synth-17D degradation; C: repeat **3.82× [3.58, 4.05]** + accuracy **1.18× [1.02, 1.37]** | **release pair, CI-endorsed** | determinism asset survives the 17D merge; B2 interaction hypothesis rejected (1.07× [0.86, 1.28]) — channel stripping hits both arms equally |
| **multi-seed CI rounds** | 3 seeds × 10 traj, trajectory-cluster bootstrap, champion + release pairs | repeat 2.26× [1.98, 2.57] (synth) / 3.47× [3.11, 3.86] (DROID); accuracy ratio synth [0.88, 1.25] contains 1, DROID [1.06, 1.38] | **statistical backbone of all three findings** | single-seed ratios no longer enter conclusion sentences (methodology asset ⑤) |

## Three findings: evidence map

### Finding 1 — train-time ablation discipline

| Evidence | Result |
|---|---|
| Inference-time zerosrc ablation (synthetic) | +156% degradation → "necessity" appearance |
| Train-time `ablate_no_mem` (synthetic) | 0.0193 ≤ 0.0214 — **no necessity, slight win** (teleport heads 2.3×) |
| Train-time `droid_no_mem` (real DROID) | 0.0157 = 0.0157 exact tie (per-traj 5:5) |
| Strong-conditioning stress (db12 vs ablate_no_mem) | no_mem wins 9–17% at every stride |

### Finding 2 — deterministic-start law

repeat‖Δ‖ (lower = more deterministic), three replications, weak-conditioning deployment domain:

| Round | pure z₀ | routed | no_mem | zerosrc | pure history |
|---|---|---|---|---|---|
| ① synthetic open-loop (ds=4, traj 0-2) | **0.0222** | 0.0620 | — | 0.0171 (trivial) | 0.1073 |
| ② DROID open-loop (ds=4, n=10) | **0.0143** | 0.0251 | 0.0444 | 0.0190 | 0.0382 |
| ③ DROID stress sim (stride 16, n=10) | **0.0146** | 0.0247 | 0.0461 | — | — |

Segment-boundary jump (round ③): pure z₀ **0.557** < no_mem 0.589 < routed 0.718. Same-session accuracy: tied with no_mem in ②③ → determinism is orthogonal to accuracy. **Multi-seed CI (3 seeds × 10 traj, trajectory-cluster bootstrap)**: repeat ratio **2.26× [1.98, 2.57]** (synthetic 22D) / **3.47× [3.11, 3.86]** (DROID), champion pairs; release pair **3.82× [3.58, 4.05]** (DROID) / **3.73× [3.08, 4.57]** (synthetic 17D). Strong-conditioning boundary (CI-amended): accuracy cost directional 10–16% (four strides consistent, one significant); repeat ratio stride-dependent — ~1.0× at stride 8/16, 1.6–1.8× (CI excludes 1) at stride 24/32; M4 inversion backed at low/mid strides.

### Finding 3 — learnability boundary (wave probe)

| Arm | eval osc-segment z₀: freq err / shape_corr | Verdict |
|---|---|---|
| db11 (role anchor λ=0.3) | **0.0003** / **−0.552** (inverted) | frequency learned via text; phase contaminated by h |
| db12 (+β-gate +debias) | **0.0004** / **+0.017** | inversion removed as designed; phase random |

Per-segment signs: train 0.75 Hz three segments sc = +0.88/+0.86/−0.90; eval two segments −0.567/+0.600 — frequency perfect, phase coin-flip. Root causes (each with controlled evidence): conditional-mean collapse of MSE anchors → role-anchor h-phase contamination (equilibrium z₀* = 1.5E[y|c] − 0.5h; β-gate cuts the coefficient 0.25→0.036) → VL-only encoder structurally cannot see state-side phase.

## Lessons worth stealing

1. **Data side > algorithm side** for accuracy: the 2.24× total gain over the zero-shot base decomposes to data-convention fixes (sin/cos dyaw, gripper inheritance, dead dims) first, algorithmic additions second.
2. **Goodhart gate**: any proxy objective that bypasses ground-truth y must pass a small-data Goodhart check before scaling (functional anchor lost −43% on proxy while losing 38–41% on deployment).
3. **Config propagation discipline**: new flags must be verified in the *run's* `config.json` and the *running* interpreter — grep of source files is not proof (two full 200-step smokes were voided by stale environments).
4. **β̂ router**: learnable gates converge to interior solutions that reflect data properties, not accuracy optima — deployment optimum sat at the endpoint β=0 in every domain tested.
5. **Latency is a hard constraint**: mse(32)/mse(8) = 3.85× (DROID) / 5.78× (wave-bearing db11); replan ≤1.07 s at 15 Hz, ≤0.53 s preferred.

## Open items (priority order)

1. **P3/P4 real-robot final trial** — can the determinism asset convert to a success-rate gap on repeatability-sensitive tasks (T5 wave specialist)? Requires a DROID-compatible platform. **The only remaining evidence gap** — all registered statistical items are closed.
2. ~~ds=8/16 denoising-budget control for repeat‖Δ‖~~ **DONE (2026-10-20)**: gap holds across ds∈{4,8,16} in both protocols — the determinism asset is intrinsic to the source architecture, not a few-step integration artifact.
3. ~~Multi-seed CI for core ratios, the strong-conditioning suite, and the appendix-A crown checks~~ **DONE (2026-10-21)**: 3 seeds × 10 traj trajectory-cluster bootstrap; repeat 2.26×/3.47× (champion pairs) and 3.82× [3.58, 4.05] (release pair); strong-conditioning accuracy cost downgraded to directional, repeat stride-dependent; crown A directional pass / C CI-endorsed / B2 interaction rejected.
4. Cross-architecture replication of the conditioning-strength law (π0 / OpenVLA-OFT).
5. M5 threshold calibration (metric units); M3 perturbation probe final form = real-robot push test.
