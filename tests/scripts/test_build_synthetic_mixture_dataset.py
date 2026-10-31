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

"""Unit tests for scripts/build_synthetic_mixture_dataset.py (pure-numpy
generation logic; parquet/video serialization is exercised on the server)."""

import numpy as np

from scripts.build_synthetic_mixture_dataset import (
    BLOCK_ZONE,
    EVAL_POOL,
    TRAIN_POOL,
    ZONES,
    SegmentSpec,
    generate_trajectory,
    sample_schedule,
)


def _gen(seed=0, n_segments=4, steps_per_segment=50, teleport_prob=0.5):
    rng = np.random.default_rng(seed)
    schedule = sample_schedule(rng, TRAIN_POOL, n_segments)
    return schedule, generate_trajectory(
        schedule, steps_per_segment, rng, teleport_prob=teleport_prob
    )


class TestSchedule:
    def test_no_consecutive_repeat(self):
        rng = np.random.default_rng(0)
        schedule = sample_schedule(rng, TRAIN_POOL, 8)
        for a, b in zip(schedule, schedule[1:]):
            assert (a.kind, a.color) != (b.kind, b.color)

    def test_pools_disjoint(self):
        """Eval combos must never appear in the train pool (generalisation)."""
        assert set(EVAL_POOL).isdisjoint(set(TRAIN_POOL))
        # yellow is fully held out; grasp-blue is held out.
        assert all(c != "yellow" for _, c in TRAIN_POOL)
        assert ("grasp", "blue") not in TRAIN_POOL
        assert ("point", "blue") in TRAIN_POOL


class TestTrajectory:
    def test_shapes(self):
        _, ep = _gen()
        T = 4 * 50
        assert ep["states"].shape == (T, 17)
        assert ep["actions"].shape == (T, 17)
        assert ep["task_ids"].shape == (T,)
        assert ep["segment_starts"] == [0, 50, 100, 150]
        assert len(ep["segment_meta"]) == 4

    def test_action_absolute_next_pose_consistency(self):
        """GR00T RELATIVE contract: action[t] = pose[t+1] (absolute) for ALL
        channels; the pipeline itself computes relative actions against the
        current state (StateActionProcessor._convert_to_relative_action)."""
        _, ep = _gen()
        s, a = ep["states"], ep["actions"]
        nxt = s[1:]  # (T-1, 17); a[T-1] targets the terminal state not in s
        assert np.allclose(a[:-1, :9], nxt[:, :9], atol=1e-6)
        assert np.allclose(a[:-1, 9], nxt[:, 9], atol=1e-6)  # gripper absolute
        assert np.allclose(a[:-1, 10:], nxt[:, 10:], atol=1e-6)

    def test_actions_are_not_state_diffs(self):
        """Regression guard (2026-10-16 root cause): storing per-step state
        diffs corrupts GR00T's RELATIVE pipeline — delta-rot6d noise gets
        row-normalized into random rotation matrices, turning rot6d training
        targets into uniform sphere noise (std = 1/sqrt(3))."""
        _, ep = _gen()
        s, a = ep["states"], ep["actions"]
        assert not np.allclose(a[:-1, :3], np.diff(s[:, :3], axis=0), atol=1e-3)
        # rot6d rows stay a valid rotation (Rz(yaw)): unit + orthogonal
        row1, row2 = a[:, 3:6], a[:, 6:9]
        assert np.allclose(np.linalg.norm(row1, axis=-1), 1.0, atol=1e-5)
        assert np.allclose(np.linalg.norm(row2, axis=-1), 1.0, atol=1e-5)
        assert np.allclose(np.sum(row1 * row2, axis=-1), 0.0, atol=1e-5)
        # and the yaw signal is non-degenerate across segments (no dead dims)
        _, ep_t = _gen(teleport_prob=1.0)
        assert ep_t["actions"][:, 3:9].std(axis=0).max() > 0.05

    def test_task_ids_constant_within_segment(self):
        schedule, ep = _gen()
        task_ids = ep["task_ids"]
        for m in ep["segment_meta"]:
            seg_ids = task_ids[m["start"] : m["end"]]
            assert len(np.unique(seg_ids)) == 1
            assert seg_ids[0] == m["task_index"]
        # consecutive segments differ (no-repeat schedule)
        for m0, m1 in zip(ep["segment_meta"], ep["segment_meta"][1:]):
            assert m0["task_index"] != m1["task_index"]

    def test_gripper_semantics(self):
        _, ep = _gen()
        for m in ep["segment_meta"]:
            g = ep["states"][m["start"] : m["end"], 9]
            if m["kind"] == "grasp":
                assert g[-1] > 0.95  # closed by the dwell phase
            else:
                assert np.all(g < 0.05)  # stays open

    def test_dwell_tail_stationary(self):
        """Last 40% of each segment dwells at the target (Prop. 1 region)."""
        _, ep = _gen()
        for m in ep["segment_meta"]:
            tail = ep["states"][m["end"] - 5 : m["end"], :3]
            # sensor noise std=0.002/coord -> frame spread ~0.01
            assert np.allclose(tail, tail[0], atol=0.01)
            zone = np.array(ZONES[BLOCK_ZONE[m["color"]]])
            assert np.allclose(tail[0][:2], zone, atol=0.05)

    def test_dwell_tail_rot6d_stationary(self):
        """Yaw dwells too: rot6d is constant over the last steps of a segment
        (yaw noise std=0.002 rad -> rot6d spread ~0.01)."""
        _, ep = _gen()
        for m in ep["segment_meta"]:
            tail = ep["states"][m["end"] - 5 : m["end"], 3:9]
            assert np.allclose(tail, tail[0], atol=0.01)

    def test_yaw_tracks_approach_direction(self):
        """Directional yaw: at the dwell the heading matches the segment
        start->target direction (shortest-arc interpolation). Teleported
        segments restart from a random position, so they are skipped."""
        _, ep = _gen()
        prev_xy = np.array([0.0, 0.0])  # home
        for m in ep["segment_meta"]:
            zone = np.array(ZONES[BLOCK_ZONE[m["color"]]])
            if not m["teleported"]:
                d = zone - prev_xy
                tail_rot = ep["states"][m["end"] - 1, 3:9]
                yaw_end = float(np.arctan2(tail_rot[3], tail_rot[0]))
                if np.linalg.norm(d) > 1e-6:
                    expected = float(np.arctan2(d[1], d[0]))
                    err = abs((yaw_end - expected + np.pi) % (2 * np.pi) - np.pi)
                    assert err < 0.05  # ~noise scale
            prev_xy = zone

    def test_within_segment_smoothness_vs_switch(self):
        """Velocity is small mid-dwell; the teleport jump dominates switches."""
        _, ep = _gen(teleport_prob=1.0)
        # actions are absolute poses (GR00T RELATIVE contract), so smoothness
        # is measured on per-step STATE displacement
        v = np.diff(ep["states"][:, :3], axis=0)  # (T-1, 3)
        metas = ep["segment_meta"]
        dwell_speed, head_speed = [], []
        for m in metas:
            # diff row k = state[k+1]-state[k]; dwell rows shift down by one
            dwell_speed.append(np.linalg.norm(v[m["end"] - 6 : m["end"] - 2], axis=-1).mean())
            if m["segment_index"] > 0:
                # the teleport jump appears at diff row start-1
                head_speed.append(np.linalg.norm(v[m["start"] - 1 : m["start"] + 9], axis=-1).mean())
        assert np.mean(head_speed) > 3 * np.mean(dwell_speed)

    def test_determinism_with_seed(self):
        _, ep1 = _gen(seed=42)
        _, ep2 = _gen(seed=42)
        assert np.allclose(ep1["states"], ep2["states"])
        _, ep3 = _gen(seed=43)
        assert not np.allclose(ep1["states"], ep3["states"])


class TestEpisodeMetaRecord:
    """Dual text port: episodes.jsonl carries ``high_level_task`` only when
    the builder is asked for it (sub-instruction training)."""

    def _ep(self):
        _, ep = _gen()
        return ep

    def test_high_level_task_attached_when_provided(self):
        from scripts.build_synthetic_mixture_dataset import _episode_meta_record

        ep = self._ep()
        meta = _episode_meta_record(3, ep, len(ep["task_ids"]), "clean the table")
        assert meta["high_level_task"] == "clean the table"
        assert meta["episode_index"] == 3
        assert meta["length"] == len(ep["task_ids"])
        assert meta["tasks"] == ep["task_texts"]

    def test_high_level_task_omitted_by_default(self):
        from scripts.build_synthetic_mixture_dataset import _episode_meta_record

        ep = self._ep()
        meta = _episode_meta_record(0, ep, len(ep["task_ids"]), None)
        assert "high_level_task" not in meta
        # legacy fields intact
        assert set(meta) == {"episode_index", "tasks", "length"}

    def test_high_level_task_pool_diverse_and_unique(self):
        """Pool mode: ≥8 distinct high-level tasks so the episode-level text
        carries information beyond the per-segment sub-instructions."""
        from scripts.build_synthetic_mixture_dataset import HIGH_LEVEL_TASK_POOL

        assert len(HIGH_LEVEL_TASK_POOL) >= 8
        assert len(set(HIGH_LEVEL_TASK_POOL)) == len(HIGH_LEVEL_TASK_POOL)
        assert all(isinstance(t, str) and t.strip() for t in HIGH_LEVEL_TASK_POOL)

    def test_high_level_task_pool_sampling_deterministic_per_seed(self):
        import numpy as np

        from scripts.build_synthetic_mixture_dataset import HIGH_LEVEL_TASK_POOL

        rng_a = np.random.default_rng(7 + 12345)
        a = [str(rng_a.choice(HIGH_LEVEL_TASK_POOL)) for _ in range(3)]
        rng_b = np.random.default_rng(7 + 12345)
        b = [str(rng_b.choice(HIGH_LEVEL_TASK_POOL)) for _ in range(3)]
        assert a == b  # same seeded stream -> same sequence
        rng_c = np.random.default_rng(7 + 12345)
        many = {str(rng_c.choice(HIGH_LEVEL_TASK_POOL)) for _ in range(50)}
        assert len(many) >= 4  # stream is not degenerate
