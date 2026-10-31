# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for scripts/verify_synthetic_gt_actions.py numeric checks."""

import numpy as np

from scripts.verify_synthetic_gt_actions import check_episode


def _states_full(T: int = 120) -> np.ndarray:
    """Synthetic (T+1, 17) states: smooth xyz with a switch jump, Rz(yaw)
    rotation signal (rot2/rot5 structurally zero), gripper, joints."""
    t = np.linspace(0.0, 1.0, T + 1)
    xyz = np.stack(
        [
            0.4 * np.sin(2 * np.pi * t),
            0.3 * np.cos(2 * np.pi * t),
            0.1 + 0.05 * np.sin(4 * np.pi * t),  # z must vary (not dead)
        ],
        axis=1,
    )
    xyz[T // 2 :, 0] += 0.2  # switch discontinuity -> diff guard stays large
    yaw = 2 * np.pi * t
    yaw[T // 2 :] += 1.0
    cos_t, sin_t = np.cos(yaw), np.sin(yaw)
    zeros = np.zeros(T + 1)
    rot6d = np.stack([cos_t, -sin_t, zeros, sin_t, cos_t, zeros], axis=1)
    grip = (t > 0.5).astype(np.float64)
    joints = np.tanh(xyz @ (0.3 * np.ones((3, 7))))
    return np.concatenate([xyz, rot6d, grip[:, None], joints], axis=1).astype(np.float32)


class TestCheckEpisode:
    def test_rz_structural_zeros_pass(self):
        """rot2/rot5 are z-components of Rz rows: identically 0 by design —
        they must be reported as structural, not dead."""
        s = _states_full()
        c = check_episode(s[:-1], s[1:])  # absolute next-pose contract
        assert c["pass"]
        assert c["structural_zero_dims"] == ["rot2", "rot5"]
        assert c["dead_dims"] == []

    def test_unexpected_dead_dim_fails(self):
        s = _states_full()
        s[:, 9] = 0.0  # gripper never moves -> genuinely dead channel
        c = check_episode(s[:-1], s[1:])
        assert not c["pass"]
        assert c["dead_dims"] == ["grip"]

    def test_diff_convention_fails(self):
        """Regression (2026-10-16 root cause): storing per-step state diffs
        must trip the diff guard (≈0) and the row-norm check (≈1)."""
        s = _states_full()
        c = check_episode(s[:-1], np.diff(s, axis=0))
        assert not c["pass"]
        assert c["diff_guard_max"] < 1e-6
        assert c["rot6d_row_norm_max_dev"] > 0.5
