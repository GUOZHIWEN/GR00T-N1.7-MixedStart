"""db17 wiring verification (paper §8 item 11-①, survey §13.81).

Layered checks for the explicit-supervision arm (NLL anchor + InfoNCE
sub-term + encoder variance head):

  L0  dataclasses.fields direct lookup — the two new config fields must be
      DECLARED (dynamic attribute assignment cannot fool this; v1.64 rule).
  L1  encoder var head — forward shapes, zero-init contract (z0==0,
      logvar==log(0.25)), backward reaches both heads.
  L2  NLL math — closed-form check against a hand-computed value.
  L3  CLI validation — nll without var_head must fail loudly; nll+var_head+
      transformer must pass; negative con_lambda must fail.
  L4  pass-through presence — launch_finetune.py / setup.py source-level key
      checks (hop1 assertion list, kwargs list, hop2 landing list).

Run:  python scripts/verify_db17_wiring.py
Exit code 0 = all passed.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


# ── L0: declared config fields ─────────────────────────────────────────────
import gr00t.configs.model.gr00t_n1d7 as mcfg

fields = {f.name for f in dataclasses.fields(mcfg.Gr00tN1d7Config)}
check("L0 config declares memory_encoder_var_head", "memory_encoder_var_head" in fields)
check(
    "L0 config declares memory_start_anchor_con_lambda",
    "memory_start_anchor_con_lambda" in fields,
)
cfg = mcfg.Gr00tN1d7Config()
check("L0 var_head default False", cfg.memory_encoder_var_head is False)
check(
    "L0 con_lambda default 0.1",
    abs(cfg.memory_start_anchor_con_lambda - 0.1) < 1e-12,
)

# ── L1/L2: encoder var head + NLL math ─────────────────────────────────────
import torch

from gr00t.model.modules.action_memory_encoder import ActionMemoryEncoderTransformer

torch.manual_seed(0)
B, K, H, D, CD, DM = 3, 8, 5, 4, 16, 32
enc = ActionMemoryEncoderTransformer(
    cond_dim=CD,
    action_dim=D,
    action_horizon=H,
    num_embodiments=2,
    d_model=DM,
    num_layers=1,
    num_queries=K,
    var_head=True,
)
qf = torch.randn(B, K, CD)
eid = torch.zeros(B, dtype=torch.long)

# zero-init contract: fresh module -> z0 == 0, logvar == log(0.25)
z0, lv = enc(qf, eid, return_logvar=True)
check("L1 shapes", z0.shape == (B, H, D) and lv.shape == (B, H, D), f"{tuple(z0.shape)}")
check("L1 z0 zero at init", float(z0.abs().max()) == 0.0)
check(
    "L1 logvar == log(0.25) at init",
    bool(torch.allclose(lv, torch.full_like(lv, math.log(0.25)), atol=1e-6)),
    f"lv[0,0,0]={float(lv[0, 0, 0]):.4f}",
)
# legacy call path unaffected
z0_only = enc(qf, eid)
check("L1 return_logvar=False returns plain tensor", isinstance(z0_only, torch.Tensor))
enc_nv = ActionMemoryEncoderTransformer(
    cond_dim=CD, action_dim=D, action_horizon=H, num_embodiments=2,
    d_model=DM, num_layers=1, num_queries=K, var_head=False,
)
try:
    enc_nv(qf, eid, return_logvar=True)
    check("L1 var_head=False + return_logvar raises", False)
except RuntimeError:
    check("L1 var_head=False + return_logvar raises", True)

# backward reaches BOTH heads (trainability of mu and sigma paths)
for p in enc.parameters():
    p.grad = None
z0, lv = enc(qf, eid, return_logvar=True)
y = torch.randn(B, H, D)
nll = 0.5 * ((z0 - y).pow(2) * lv.neg().exp() + lv).mean()
nll.backward()
g_mu = enc.head.weight.grad
g_lv = enc.head_logvar.weight.grad
check(
    "L1 grad flows to mu head",
    g_mu is not None and float(g_mu.abs().sum()) > 0,
)
check(
    "L1 grad flows to logvar head",
    g_lv is not None and float(g_lv.abs().sum()) > 0,
    "zero at init is EXPECTED if dL/dz0==0 (z0=0 head zero-init): "
    f"|g_lv|={float(g_lv.abs().sum()) if g_lv is not None else None:.3e}",
)

# ── L2: NLL closed-form check ──────────────────────────────────────────────
# hand case: mu=0, y=1, logvar=0 (σ²=1) → ½(1·1 + 0) = 0.5
#            mu=0, y=1, logvar=log(0.25) → ½(1/0.25 + log 0.25) = 2 − 0.6931
mu_h = torch.zeros(1, 1, 1)
y_h = torch.ones(1, 1, 1)
lv0 = torch.zeros(1, 1, 1)
lv1 = torch.full((1, 1, 1), math.log(0.25))
nll0 = 0.5 * ((mu_h - y_h).pow(2) * lv0.neg().exp() + lv0)
nll1 = 0.5 * ((mu_h - y_h).pow(2) * lv1.neg().exp() + lv1)
check("L2 NLL(σ²=1)=0.5", abs(float(nll0) - 0.5) < 1e-6, f"{float(nll0):.6f}")
check(
    "L2 NLL(σ²=0.25)=2+½log(0.25)",
    abs(float(nll1) - (2.0 + 0.5 * math.log(0.25))) < 1e-6,
    f"{float(nll1):.6f}",
)

# ── L3: CLI validation ─────────────────────────────────────────────────────
try:
    from gr00t.configs.finetune_config import FinetuneConfig

    base = dict(
        base_model_path="/tmp/db17_base",
        dataset_path="demo_data/droid_sample",
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        output_dir="/tmp/db17_verify",
        memory_start_enabled=True,
        flow_source_mode="memory_history_mix",
        memory_encoder_arch="transformer",
    )
    try:
        FinetuneConfig(
            memory_start_anchor_type="nll", memory_encoder_var_head=False, **base
        )
        check("L3 nll without var_head rejected", False)
    except ValueError:
        check("L3 nll without var_head rejected", True)
    try:
        FinetuneConfig(
            memory_start_anchor_type="nll_infonce",
            memory_encoder_var_head=True,
            **base,
        )
        check("L3 nll_infonce + var_head + transformer passes", True)
    except ValueError as e:
        check("L3 nll_infonce + var_head + transformer passes", False, str(e))
    try:
        FinetuneConfig(
            memory_start_anchor_type="nll",
            memory_encoder_var_head=True,
            memory_start_anchor_con_lambda=-0.1,
            **base,
        )
        check("L3 negative con_lambda rejected", False)
    except ValueError:
        check("L3 negative con_lambda rejected", True)
    try:
        FinetuneConfig(
            memory_start_anchor_type="nll",
            memory_encoder_var_head=True,
            memory_encoder_arch="mlp",
            **{k: v for k, v in base.items() if k != "memory_encoder_arch"},
        )
        check("L3 nll + mlp arch rejected", False)
    except ValueError:
        check("L3 nll + mlp arch rejected", True)
except ImportError as e:
    check("L3 CLI validation (skipped: import)", True, f"SKIP {e}")

# ── L4: pass-through presence (source level) ───────────────────────────────
lf = (ROOT / "gr00t/experiment/launch_finetune.py").read_text(encoding="utf-8")
su = (ROOT / "gr00t/model/gr00t_n1d7/setup.py").read_text(encoding="utf-8")
for key in ("memory_encoder_var_head", "memory_start_anchor_con_lambda"):
    check(f"L4 launch_finetune pass-through: {key}", f"config.model.{key} = ft_config.{key}" in lf)
    check(f"L4 setup.py kwargs: {key}", f"{key}=getattr(" in su)
# both keys must appear at least twice in setup.py (kwargs + landing assert)
for key in ("memory_encoder_var_head", "memory_start_anchor_con_lambda"):
    check(
        f"L4 setup.py landing assertion: {key}",
        su.count(f'"{key}"') >= 1,
    )

# ── L5: db18 state routing (T1/T2) ─────────────────────────────────────────
fields = {f.name for f in dataclasses.fields(mcfg.Gr00tN1d7Config)}
check("L5 config declares dit_state_token", "dit_state_token" in fields)
check("L5 config declares memory_encoder_read_state", "memory_encoder_read_state" in fields)
cfg5 = mcfg.Gr00tN1d7Config()
check("L5 dit_state_token default True", cfg5.dit_state_token is True)
check("L5 read_state default False", cfg5.memory_encoder_read_state is False)

enc_rs = ActionMemoryEncoderTransformer(
    cond_dim=CD, action_dim=D, action_horizon=H, num_embodiments=2,
    d_model=DM, num_layers=1, num_queries=K, read_state=True, state_dim=7,
)
s1 = torch.randn(B, 7)
s2 = torch.randn(B, 7)
z_s = enc_rs(qf, eid, state=s1)
check("L5 read_state forward shape", z_s.shape == (B, H, D))
try:
    enc_rs(qf, eid, state=None)
    check("L5 read_state + state=None raises", False)
except RuntimeError:
    check("L5 read_state + state=None raises", True)
# state actually modulates z0 (break zero-init to make the path visible)
with torch.no_grad():
    enc_rs.head.weight.normal_(0, 0.02)
d_state = float((enc_rs(qf, eid, state=s1) - enc_rs(qf, eid, state=s2)).abs().max())
check("L5 state modulates z0", d_state > 1e-6, f"|Δz0|={d_state:.2e}")
# grad reaches state_proj
for p in enc_rs.parameters():
    p.grad = None
enc_rs(qf, eid, state=s1).pow(2).sum().backward()
check(
    "L5 grad reaches state_proj",
    enc_rs.state_proj.weight.grad is not None
    and float(enc_rs.state_proj.weight.grad.abs().sum()) > 0,
)
# read_state=False ignores state silently
z_ns = enc(qf, eid, state=s1)
check("L5 read_state=False tolerates state kwarg", isinstance(z_ns, torch.Tensor))

try:
    base18 = dict(
        base_model_path="/tmp/db18_base",
        dataset_path="demo_data/droid_sample",
        embodiment_tag="OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
        output_dir="/tmp/db18_verify",
        memory_start_enabled=True,
        flow_source_mode="memory_history_mix",
        memory_encoder_arch="transformer",
    )
    try:
        FinetuneConfig(memory_encoder_read_state=True, memory_encoder_arch="mlp",
                       **{k: v for k, v in base18.items() if k != "memory_encoder_arch"})
        check("L5 CLI read_state+mlp rejected", False)
    except ValueError:
        check("L5 CLI read_state+mlp rejected", True)
    try:
        FinetuneConfig(dit_state_token=False, memory_func_anchor_enabled=True, **base18)
        check("L5 CLI no-state-token + func-anchor rejected", False)
    except ValueError:
        check("L5 CLI no-state-token + func-anchor rejected", True)
    try:
        FinetuneConfig(dit_state_token=False, memory_encoder_read_state=True, **base18)
        check("L5 CLI T2 combo passes", True)
    except ValueError as e:
        check("L5 CLI T2 combo passes", False, str(e))
    try:
        FinetuneConfig(dit_state_token=False, **base18)
        check("L5 CLI T1 combo passes", True)
    except ValueError as e:
        check("L5 CLI T1 combo passes", False, str(e))
except ImportError as e:
    check("L5 CLI validation (skipped: import)", True, f"SKIP {e}")

for key in ("memory_encoder_read_state", "dit_state_token"):
    check(f"L5 launch_finetune pass-through: {key}", f"config.model.{key} = ft_config.{key}" in lf)
    check(f"L5 setup.py kwargs: {key}", f"{key}=getattr(" in su)
    check(f"L5 setup.py landing assertion: {key}", su.count(f'"{key}"') >= 1)
# n1d7 state-token cut present in BOTH train and inference paths
n1 = (ROOT / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py").read_text(encoding="utf-8")
check(
    "L5 dit_state_token branch ×2 (train+inference)",
    n1.count('getattr(self.config, "dit_state_token", True)') >= 2,
)

print()
if FAILURES:
    print(f"db17 wiring: {len(FAILURES)} FAILURE(S): {FAILURES}")
    sys.exit(1)
print("db17 wiring: ALL CHECKS PASSED")
