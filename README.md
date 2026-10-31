# MoS: Mixed-Start Flow Matching for Action Generation

[![中文](https://img.shields.io/badge/README-中文-blue.svg)](README_CN.md)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/Mixed-Start_Flow_Matching_for_Action_Generation_v1.pdf)
[![ModelScope](https://img.shields.io/badge/ModelScope-Weights-624ad3.svg)](https://www.modelscope.cn/models/joyogzw/GR00T-N1.7-MixedStart)

MoS separates two properties of a flow-matching action source: **conditional content** and **stochasticity**. Learnable queries read a frozen VLM, and an MLP maps their features to a semantic start `z₀`. Training uses `s = βh + (1 − β)z₀ + ηε`, with `β ~ U(0, 0.3)`, `η = 0.04`, and a mixed-source consistency loss. At deployment, `β = 0` removes history mixing, while `η = 0` additionally removes source sampling.

## 📣 News

- **2026-10-31**: Released the `mos_best` and `mos_best_no_mem` weight pair, together with the accompanying code, datasets, and paper.

## 🧩 Contributions

1. **Conditional semantic start encoder.** Sixty-four queries pass through the frozen VLM; an MLP produces an action-chunk-shaped `z₀`. The semantic start enters only the flow source, not as an additional DiT token.
2. **Mixed training and semantic deployment.** A strict past-only shift/1 history window and a mixed-source consistency loss avoid the supervision conflict caused by direct action-target regression.
3. **Controlled source evaluation.** Multiple training seeds, matched-noise controls, zero-noise deployment, and fixed-noise controls jointly evaluate action accuracy and repeat divergence.

## 🏗️ Model architecture

![MoS mixed-start architecture](release/assets/mos_architecture_mixed_start.png)

The frozen Cosmos-Reason2-2B (Qwen3-VL) backbone processes images, instructions, and learnable queries. An MLP maps query features to the semantic start `z₀`. Training uses `s = βh + (1 − β)z₀ + ηε` with mixed-source consistency supervision; Action DiT learns the velocity field from the source to the target action chunk. Deployment uses `β = 0` by default, while `η = 0` additionally gives a deterministic semantic source. See the [paper](paper/Mixed-Start_Flow_Matching_for_Action_Generation_v1.pdf) for the mathematical definition and the [model card](release/README.md) for release defaults.

## 📦 Models and resources

| Resource | Location |
|---|---|
| Semantic-start weights | `release/mos_best` |
| Gaussian-source control weights | `release/mos_best_no_mem` |
| Model repository | [ModelScope models](https://www.modelscope.cn/models/joyogzw/GR00T-N1.7-MixedStart) |
| Dataset repository | [ModelScope datasets](https://www.modelscope.cn/datasets/joyogzw/GR00T-N1.7-MixedStart-Datasets) |
| Source and reproduction | [GitHub](https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart) · [Guide](docs/REPRODUCE.md) |

Both release models are fine-tuned from a **pretrained Action DiT**. The paper's multi-seed main results use **from-scratch DiT initialization**; these evidence lines must not be interchanged.

## 📊 Official release-pair results

**Table 1 | Open-loop release-pair results; lower is better.** Ratios are `mean(Gaussian_i / Semantic_i)`; values above 1 favor the semantic arm.

| Domain | Metric ↓ | `mos_best` | `mos_best_no_mem` | Gaussian/semantic paired ratio [95% CI] |
|---|---|---:|---:|---:|
| Synthetic held-out | Effective action MSE | **0.0471** | 0.0560 | **1.33 [1.04, 1.60]** |
| Synthetic held-out | Repeat divergence | **0.0220** | 0.0644 | **3.00 [2.71, 3.31]** |
| DROID replay | Effective action MSE | 0.0174 | **0.0165** | 0.97 [0.81, 1.14], not significant |
| DROID replay | Repeat divergence | **0.0206** | 0.0432 | **2.10 [1.90, 2.27]** |

The release models use 100 synthetic and 300 DROID episodes, 17D state/action, 5,000 steps, and training seed 2. All reported results are open-loop replay. Repeat divergence is not closed-loop success. No significant DROID accuracy improvement is observed for the release pair, and no real-robot deployment benefit is claimed.

## 📈 From-scratch DiT trajectory comparison

All results below use **zero-initialized DiT** and must be interpreted separately from the released pretrained-DiT fine-tunes above.

**Table 2 | Accuracy and run-to-run repeatability over three training seeds.**

| Metric | Synthetic | DROID replay |
|---|---:|---:|
| Repeat divergence paired ratio | **2.80 [2.66, 2.95]** | **2.04 [1.95, 2.15]** |
| Effective action MSE paired ratio | **1.12 [1.06, 1.18]** | **1.20 [1.10, 1.30]** |

Brackets contain 95% cluster-bootstrap confidence intervals. Ratios are Gaussian/semantic; values above 1 favor the semantic source.

![Open-loop held-out accuracy versus training steps](assets/fig_train_efficiency.png)

**Open-loop held-out accuracy versus training steps.** Independent pairs from two training seeds show that the semantic source reaches usable accuracy at earlier checkpoints. Threshold-crossing steps are the first hits on a discrete checkpoint grid, and the curves are not strictly monotonic. These are open-loop deployment metrics, not training loss, and do not establish wall-clock acceleration.

### Action trajectory comparison

The following panels compare action predictions from the **semantic-start** and **Gaussian-source** models on identical held-out trajectories, with ground truth and both model outputs sharing each action-dimension axis. Both models use **from-scratch Action DiT initialization, training seed 2, and the 5,000-step checkpoint**. These are open-loop replay results, not results from the released pretrained-DiT fine-tunes.

**Synthetic held-out｜Clear-gap example, trajectory 8**

![A clear-gap synthetic trajectory example favoring the semantic arm](assets/db28_action_trajectories_clear_gap_synth.png)

Effective action MSE is **0.0242** for the semantic arm and **0.0307** for the Gaussian arm. This explicitly labeled example uses the largest Gaussian/semantic MSE ratio among the same 10 evaluation-seed-0 trajectories, with the same fixed action dimensions. It illustrates a clear-gap case, **not the typical gain**; overall conclusions remain based on multi-seed paired statistics.

**DROID replay｜Representative trajectory 9**

![DROID trajectories for the from-scratch semantic and Gaussian arms](assets/db28_action_trajectories_droid.png)

Effective action MSE is **0.0172** for the semantic arm and **0.0198** for the Gaussian arm in this example. Fixed dimensions and declared trajectory-selection rules make every panel reproducible; individual trajectories do not replace paired statistics. See [the reproduction guide](docs/REPRODUCE.md#101-readme-trajectory-figures) for paired inference and figure generation.

## 🚀 Quickstart

### 📥 Release training and evaluation

```bash
git clone https://github.com/GUOZHIWEN/GR00T-N1.7-MixedStart.git
cd GR00T-N1.7-MixedStart
pip install uv && uv sync
uv pip install modelscope

# Download pinned data, verify SHA256, and extract safely
uv run python scripts/prepare_mos_data.py --output-dir data

# 200-step smoke training for both release arms
BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B \
  bash scripts/train_mos_release.sh both smoke 2

# Full release training and held-out evaluation
BASE_MODEL_PATH=/path/to/GR00T-N1.7-3B \
  bash scripts/train_mos_release.sh both final 2
bash scripts/eval_mos_release.sh all 2

# Export and validate the two loadable model directories
uv run python scripts/export_mos_release.py --run-suffix _v2 --output-dir release
uv run python scripts/validate_mos_release.py --release-dir release --write-sums
```

### 🤖 Inference only

```bash
modelscope download --model joyogzw/GR00T-N1.7-MixedStart --local_dir release
uv run python examples/mos_inference.py \
  --model-path release/mos_best --sample-npz examples/real_observation.npz \
  --task "indicate all the blocks one after another" \
  --device cuda --denoising-steps 4 --fixed-beta 0 --fixed-noise-std 0 \
  --history-replan-stride 16 --output runs/mos_inference_action.npz
```

The included `examples/real_observation.npz` is a real decoded **synthetic evaluation** observation, not a random template or real-robot capture. Its sidecar JSON records provenance. Trajectory NPZ files contain action outputs, not model inputs. See [the inference contract](docs/INFERENCE.md) for NPZ keys, shapes, dtypes, action unnormalization, history buffering, and episode reset semantics.

## 🗂️ Repository layout

```text
MoS/
├── gr00t/                  # Complete source package with MoS modifications
├── scripts/                # Data, training, evaluation, aggregation, export, validation
├── examples/               # Minimal single-observation inference
├── tests/                  # Core unit tests
├── docs/                   # Current reproduction and release documentation
│   └── history/            # Historical experiment ledger only
├── paper/                  # English paper PDF only
└── release/                # Canonical model card and assets; weights are downloaded
```

## ⚠️ Statistics and limitations

Paired ratios are `mean(reference_i / candidate_i)`, not ratios of aggregate means. Confidence intervals use whole-trajectory cluster bootstrap. Synthetic results aggregate four evaluation seeds; DROID uses evaluation seed 0. Step-to-threshold efficiency is not wall-clock acceleration.

Evidence is limited to one VLA family, DROID-like embodiment, two camera views, and open-loop replay. Cross-embodiment transfer, closed-loop success, and real-robot benefits require separate validation.

## ✒️ Citation

```bibtex
@misc{mos2026mixedstart,
  title  = {Mixed-Start Flow Matching for Action Generation},
  author = {Guo, Zhiwen},
  year   = {2026},
  note   = {arXiv technical report}
}
```

## 📬 Contact

Zhiwen Guo · [zhiwenguo99@gmail.com](mailto:zhiwenguo99@gmail.com)

## ⚖️ License and attribution

Code is released under [Apache-2.0](LICENSE). See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for upstream and third-party notices. DROID retains its original license. The former experiment ledger is archived under `docs/history/` and is not the current reproduction entry point.
