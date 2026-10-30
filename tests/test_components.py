"""Scientific invariants, split isolation, and interrupted generation recovery."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ablation_core import batch_plan
from analyze_component_dataset import paired_statistics, summarize_pairs
from component_core import (ComponentAccumulator, geometry_strata, loss_db,
                            select_receivers, validate_components)
from dataset_selection import freeze_selection
from generate_dataset import map_path, read_map, run_map
from terrain_mesh import GridProjector, triangulate
from ablation_core import write_json, write_npz
from dataset_resume import FP32_COMPONENT_COMMIT, resume_manifest
import copy
import json


class ComponentTests(unittest.TestCase):
    def test_closure_preserves_weak_components_and_detects_omitted_family(self):
        families = np.asarray([[1e-30, 0], [2e-31, 0], [3e-32, 0], [4e-33, 0], [5e-34, 0]])
        gain = np.concatenate((families.sum(0, keepdims=True), families))
        validate_components(gain)
        broken = gain.copy()
        broken[1] = 0
        with self.assertRaises(ValueError):
            validate_components(broken)
        self.assertTrue(np.isposinf(loss_db(np.asarray([0.])))[0])
        self.assertTrue(np.isnan(loss_db(np.asarray([np.nan])))[0])

    def test_zero_batches_are_included_and_hit_sums_exceed_uint32(self):
        accumulator = ComponentAccumulator(2)
        zero = np.zeros((6, 2))
        hits = np.zeros((5, 2), np.uint64)
        accumulator.add(zero, hits, hits, 3)
        gain = zero.copy()
        gain[0] = gain[1] = [8., 4.]
        hits[0] = 3000000000
        accumulator.add(gain, hits, hits, 1)
        accumulator.add(gain, hits, hits, 1)
        np.testing.assert_allclose(accumulator.mean()[0], [3.2, 1.6])
        self.assertEqual(accumulator.hits[0, 0], 6000000000)

    def test_los_boundary_uses_shared_edges_and_rx_sampling_has_no_duplicates(self):
        _, faces, centers, _ = triangulate(np.zeros((4, 4)), 1.)
        visible = centers[:, 0] < 1.5
        region, distance_bin, _ = geometry_strata(visible, centers, [0, 0, 3], faces)
        self.assertTrue((region == 1).any())
        self.assertTrue((region == 0).any())
        self.assertTrue((region == 2).any())
        first = select_receivers(region, distance_bin, 12, 7)
        np.testing.assert_array_equal(first, select_receivers(region, distance_bin, 12, 7))
        self.assertEqual(len(set(first)), 12)

    def test_seed_error_uses_pooled_sse_and_counts_zero_support(self):
        a, b = np.asarray([1., 1., 0.]), np.asarray([.1, 0., 1.])
        stats = paired_statistics(a, b, np.ones(3, bool), np.ones(3), np.ones(3), np.ones(3, bool))
        self.assertEqual(stats["both_positive_count"], 1)
        self.assertEqual(stats["a_positive_b_zero_count"], 1)
        self.assertEqual(stats["a_zero_b_positive_count"], 1)
        pooled = summarize_pairs([
            {"comparison": "seed", "partition": "fit", "region": -1, "component": "G_all", **stats},
            {"comparison": "seed", "partition": "fit", "region": -1, "component": "G_all",
             **paired_statistics(np.ones(9), np.ones(9), np.ones(9, bool), np.ones(9), np.ones(9), np.ones(9, bool))}])[0]
        self.assertAlmostEqual(pooled["pooled_log_power_rmse_db"], np.sqrt(10))

    def test_4b_ray_schedule_uses_global_disjoint_offsets(self):
        schedule = batch_plan(4000000000, 1000000, 0)
        self.assertEqual(len(schedule), 4000)
        self.assertEqual(schedule[-1]["offset"], 3999000000)
        self.assertEqual(sum(b["samples"] for b in schedule), 4000000000)


class FakeData:
    def __init__(self, split, count, first_id):
        self.split, self.grid = split, (4, 4)
        self.metadata = {"resolution_m_per_px": 1., "simulation": {"tx_height_m_agl": 3., "rx_height_m_agl": 1.}}
        self.rows, self.h = [], []
        r, c = np.indices(self.grid)
        for i in range(count):
            terrain = str(first_id+i)
            self.h.append((r*.1+c*.2)*(1+i%13))
            for tx in range(2 if i%4 == 0 else 1):
                self.rows.append({"sample_id": terrain+"_"+str(tx), "terrain_id": terrain,
                    "hm_row": str(i), "tx_row": str((i+tx)%4), "tx_col": str((2*i+tx)%4)})
        self.h = np.asarray(self.h)

    def array(self, key):
        if key != "hm":
            raise AssertionError("Selection must not read labels or masks")
        return self.h


class SelectionTests(unittest.TestCase):
    def test_528_terrain_isolation_and_deterministic_geometry_only_selection(self):
        train, val = FakeData("train", 500, 10000), FakeData("val", 90, 20000)
        selection = freeze_selection(train, val, 17)
        self.assertEqual(selection, freeze_selection(train, val, 17))
        rows = selection["samples"]
        self.assertEqual(len({r["terrain_id"] for r in rows}), 528)
        self.assertEqual([sum(r["partition"] == k for r in rows) for k in ("fit", "internal_val", "external_val")], [400, 64, 64])
        self.assertEqual([sum(r[k] for r in rows) for k in ("pilot", "single_mechanism_control", "representative", "hard")], [16, 4, 32, 8])
        self.assertTrue(all(not r["pilot"] or r["partition"] == "fit" for r in rows))
        self.assertTrue(all(not r["hard"] or r["representative"] for r in rows))
        self.assertTrue(all(r["parent_id"] is None for r in rows))


class ResumeTests(unittest.TestCase):
    def test_legacy_fp32_checkpoint_resumes_only_the_remaining_rays(self):
        class Backend:
            offsets = []

            def solve_components(self, scene, measurement, samples, seed, depth, total_samples, sample_offset):
                self.offsets.append(sample_offset)
                gain, hits = np.zeros((6, 8)), np.zeros((5, 8), np.uint64)
                gain[0] = gain[1] = 4.
                hits[0] = 1
                return gain, hits, hits
        _, _, centers, _ = triangulate(np.zeros((3, 3)), 1.)
        projector = GridProjector(centers, (3, 3), 1.)
        plan = {"samples_per_tx": 6, "rays_per_batch": 2, "max_depth": 3,
                "checkpoint_every_batches": 1, "log_every_batches": 100}
        sample = {"partition": "fit", "sample_id": "legacy"}
        backend = Backend()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            final = map_path(root, sample, 0)
            weighted, hits = np.zeros((6, 8)), np.zeros((5, 8), np.uint64)
            weighted[0] = weighted[1] = 8.
            hits[0] = 2
            write_npz(final.with_name("seed0.progress.npz"), weighted_gain=weighted,
                hit_count=hits, positive_hit_count=hits, samples=4, completed_batches=2, elapsed_seconds=1.)
            run_map(backend, None, None, projector, 8, plan, sample, 0, root, float32_samples=4)
            self.assertEqual(backend.offsets, [4])
            np.testing.assert_allclose(read_map(final, 6, 8)[0], 16/6)
            with np.load(final) as saved:
                self.assertEqual(int(saved["float32_accumulation_samples"]), 4)
                self.assertEqual(int(saved["float64_accumulation_samples"]), 2)

    def test_resume_includes_zero_chunk_and_never_repeats_completed_chunks(self):
        class Backend:
            fail = True
            offsets = []

            def solve_components(self, scene, measurement, samples, seed, depth, total_samples, sample_offset):
                self.offsets.append(sample_offset)
                if self.fail and sample_offset == 4:
                    raise RuntimeError("simulated interruption")
                gain, hits = np.zeros((6, 8)), np.zeros((5, 8), np.uint64)
                if sample_offset > 0:
                    gain[0] = gain[1] = sample_offset
                    hits[0] = 1
                return gain, hits, hits
        _, _, centers, _ = triangulate(np.zeros((3, 3)), 1.)
        projector = GridProjector(centers, (3, 3), 1.)
        plan = {"samples_per_tx": 6, "rays_per_batch": 2, "max_depth": 3,
                "checkpoint_every_batches": 1, "log_every_batches": 100}
        sample = {"partition": "fit", "sample_id": "example"}
        backend = Backend()
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            arguments = (backend, None, None, projector, 8, plan, sample, 0, root)
            with self.assertRaises(RuntimeError):
                run_map(*arguments)
            backend.fail = False
            run_map(*arguments)
            np.testing.assert_allclose(read_map(map_path(root, sample, 0), 6, 8)[0], 2.)
            self.assertEqual(backend.offsets, [0, 2, 4, 4])
            self.assertFalse(map_path(root, sample, 0).with_name("seed0.progress.npz").exists())
            run_map(*arguments)
            self.assertEqual(backend.offsets, [0, 2, 4, 4])


class UpgradeTests(unittest.TestCase):
    def previous(self):
        return {"experiment": {"samples_per_tx": 6, "rays_per_batch": 2, "seeds": [0, 1],
                    "data_files": ["unchanged"], "max_depth": 100},
                "code": {"commit": FP32_COMPONENT_COMMIT, "dirty": False},
                "runtime": {"sionna-rt": "1.1.0"}}

    def test_upgrade_requires_opt_in_and_keeps_original_arrays_and_manifest(self):
        previous = self.previous()
        plan = {**previous["experiment"], "native_gain_accumulation_dtype": "float64"}
        version = {"commit": "tested-numeric-fix", "dirty": False}
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            write_json(root / "manifest.json", previous)
            path = root / "maps/fit/legacy/seed0.progress.npz"
            write_npz(path, samples=4, completed_batches=2, weighted_gain=np.ones((6, 8)))
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                resume_manifest(root, plan, version, previous["runtime"])
            result = resume_manifest(root, plan, version, previous["runtime"], True)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(json.loads((root / "manifest.before_fp64.json").read_text()), previous)
            self.assertEqual(result["accumulation_upgrade"]["float32_samples_by_map"], {"maps/fit/legacy/seed0.npz": 4})
            self.assertEqual(resume_manifest(root, plan, version, previous["runtime"]), result)

    def test_upgrade_rejects_changed_settings_unknown_versions_and_runtime(self):
        for change in ("samples_per_tx", "rays_per_batch", "seeds", "data_files", "max_depth", "code", "runtime"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                previous = self.previous()
                plan = {**copy.deepcopy(previous["experiment"]), "native_gain_accumulation_dtype": "float64"}
                runtime = dict(previous["runtime"])
                if change == "code":
                    previous["code"]["commit"] = "unsupported-source"
                elif change == "runtime":
                    runtime["sionna-rt"] = "changed"
                elif change in ("seeds", "data_files"):
                    plan[change] = ["changed"]
                else:
                    plan[change] += 1
                root = Path(temporary)
                write_json(root / "manifest.json", previous)
                with self.assertRaises(ValueError):
                    resume_manifest(root, plan, {"commit": "fix", "dirty": False}, runtime, True)
                self.assertFalse((root / "manifest.before_fp64.json").exists())
                self.assertEqual(json.loads((root / "manifest.json").read_text()), previous)


if __name__ == "__main__":
    unittest.main()
