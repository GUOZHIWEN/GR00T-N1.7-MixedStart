"""Isolated tests for MoS deployment overrides; no model weights are loaded."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "gr00t/policy/gr00t_policy.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "_configure_mos_deployment")
namespace = {"Any": object}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
configure = namespace["_configure_mos_deployment"]


class DeploymentOptionsTest(unittest.TestCase):
    def model(self, shared=False):
        head_config = SimpleNamespace(
            confidence_routing_enabled=True, memory_start_mix_beta=0.2,
            memory_start_noise_std=0.04,
        )
        config = head_config if shared else SimpleNamespace(
            confidence_routing_enabled=True, memory_start_mix_beta=0.2,
            memory_start_noise_std=0.04,
        )
        return SimpleNamespace(config=config, action_head=SimpleNamespace(
            config=head_config, num_inference_timesteps=10,
        ))

    def test_explicit_controls_reach_model_and_shared_head_config(self):
        model = self.model(shared=True)
        configure(model, denoising_steps=4, fixed_beta=0.0, fixed_noise_std=0.0)
        self.assertEqual(model.action_head.num_inference_timesteps, 4)
        self.assertEqual(model.config.memory_start_mix_beta, 0.0)
        self.assertEqual(model.config.memory_start_noise_std, 0.0)
        self.assertFalse(model.config.confidence_routing_enabled)

    def test_distinct_configs_both_receive_overrides(self):
        model = self.model()
        configure(model, fixed_beta=0.25, fixed_noise_std=0.01)
        for config in (model.config, model.action_head.config):
            self.assertEqual(config.memory_start_mix_beta, 0.25)
            self.assertEqual(config.memory_start_noise_std, 0.01)

    def test_none_keeps_checkpoint_values(self):
        model = self.model()
        configure(model)
        self.assertEqual(model.action_head.num_inference_timesteps, 10)
        self.assertEqual(model.config.memory_start_mix_beta, 0.2)
        self.assertEqual(model.config.memory_start_noise_std, 0.04)
        self.assertTrue(model.config.confidence_routing_enabled)

    def test_invalid_values_fail_fast(self):
        for kwargs in ({"denoising_steps": 0}, {"fixed_beta": -0.1},
                       {"fixed_beta": 1.1}, {"fixed_noise_std": -0.01}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                configure(self.model(), **kwargs)


class LoaderWiringTest(unittest.TestCase):
    def test_policy_constructor_applies_helper_after_pretrained_load(self):
        text = SOURCE.read_text(encoding="utf-8")
        load = text.index("model = AutoModel.from_pretrained(model_dir)")
        apply = text.index("_configure_mos_deployment(\n            model,", load)
        safeguard = text.index("mem_enc = getattr", apply)
        self.assertLess(load, apply)
        self.assertLess(apply, safeguard)
        for argument in ("denoising_steps=denoising_steps", "fixed_beta=fixed_beta",
                         "fixed_noise_std=fixed_noise_std"):
            self.assertIn(argument, text[apply:safeguard])


if __name__ == "__main__":
    unittest.main()
