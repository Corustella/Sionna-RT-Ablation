import argparse
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ablation_core import CONFIGS, GainAccumulator, batch_plan, compare_gains, label_diagnostic, switches, write_json, write_npz
from analyze_ablation import analyze, main as analyze_main, pooled_statistics
from lunar_data import LunarData
from run_ablation import make_plan, parser, run_case
from terrain_mesh import GridProjector, triangulate


def make_data(root):
    metadata = {"grid": [4, 4], "resolution_m_per_px": 1.0,
                "bands_hz": {"58": 5775000000.0},
                "simulation": {"engine": "Sionna RT 1.1.0", "max_depth": 100,
                               "samples_per_tx": 4000000000,
                               "propagation_mechanisms": {"diffraction": False}}}
    write_json(root / "metadata.json", metadata)
    rows = [dict(sample_id="a0", terrain_id="a", hm_row=1, tx_row=1, tx_col=2),
            dict(sample_id="b0", terrain_id="b", hm_row=0, tx_row=2, tx_col=1),
            dict(sample_id="a1", terrain_id="a", hm_row=1, tx_row=1, tx_col=1)]
    with (root / "val_index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    np.save(root / "val_hm.npy", np.stack((np.zeros((4, 4)), np.ones((4, 4)) * 7)))
    np.save(root / "val_pl58.npy", np.stack([np.ones((4, 4)) * x for x in (110, 120, 130)]))
    masks = np.ones((3, 16), np.uint8)
    masks[0, 5] = 0
    np.save(root / "val_mask58.npy", np.packbits(masks, axis=1))
    return metadata


class ScientificStatisticsTests(unittest.TestCase):
    def test_switches_remove_only_requested_mechanism(self):
        full = switches("full")
        for case, flag in (("no_specular", "specular_reflection"), ("no_diffuse", "diffuse_reflection"), ("no_refraction", "refraction")):
            self.assertEqual([k for k in full if switches(case)[k] != full[k]], [flag])
        self.assertNotIn("diffraction", full)

    def test_zero_hit_batches_have_weight_and_batch_sizes_matter(self):
        mean = GainAccumulator((2,))
        mean.add(np.array([0., 2.]), 3)
        mean.add(np.array([4., 0.]), 1)
        np.testing.assert_allclose(mean.mean(), [1., 1.5])
        with self.assertRaises(ValueError):
            mean.add(np.array([np.nan, 1]), 1)

    def test_paired_batch_seeds_and_remainder(self):
        schedule = batch_plan(10, 4, 7)
        self.assertEqual(schedule, batch_plan(10, 4, 7))
        self.assertEqual([b["samples"] for b in schedule], [4, 4, 2])
        self.assertEqual([b["offset"] for b in schedule], [0, 4, 8])
        self.assertNotEqual(schedule, batch_plan(10, 4, 8))
        self.assertEqual(len({b['seed'] for b in schedule}), 3)

    def test_zero_coverage_and_negative_effect_are_not_discarded(self):
        out = compare_gains(np.array([4., 4., 0., 0.]), np.array([0., 8., 1., 0.]), np.ones(4, bool))
        self.assertEqual(out["lost_to_zero_count"], 1)
        self.assertEqual(out["gained_from_zero_count"], 1)
        self.assertEqual(out["both_positive_count"], 1)
        self.assertEqual(out["both_zero_count"], 1)
        self.assertAlmostEqual(out["relative_gain_removed"], -1 / 8)
        self.assertAlmostEqual(out["delta_pathloss_db_mean_on_both_positive"], -10 * np.log10(2))

    def test_weighted_pooling_is_not_mean_fraction(self):
        a = compare_gains(np.array([1.]), np.array([0.]), np.array([True]))
        b = compare_gains(np.array([9.]), np.array([9.]), np.array([True]))
        self.assertAlmostEqual(pooled_statistics([a, b])["relative_gain_removed"], .1)
        weighted = compare_gains(np.array([1., 9.]), np.array([0., 9.]), np.ones(2, bool), np.array([2., 1.]))
        self.assertAlmostEqual(weighted["relative_gain_removed"], 2 / 11)

    def test_label_score_exposes_missing_support(self):
        gain = np.array([.01, 0., np.nan, 1.])
        out = label_diagnostic(gain, np.array([20., 30., 40., 145.]),
                               np.array([1, 1, 1, 0], bool), np.array([1, 1, 0, 1], bool))
        self.assertEqual(out["positive_overlap_count"], 1)
        self.assertEqual(out["valid_but_zero_count"], 1)
        self.assertEqual(out["valid_outside_interpolation_footprint_count"], 1)
        self.assertEqual(out["rmse_db_on_positive_overlap"], 0)


class GeometryAndDataTests(unittest.TestCase):
    def test_up_normals_area_and_linear_grid_projection(self):
        rows, cols = np.indices((5, 6))
        height = 2 * cols + 3 * rows
        for diagonal in ("nw_se", "ne_sw"):
            vertices, faces, centers, areas = triangulate(height, 1., diagonal)
            triangles = vertices[faces]
            normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            self.assertTrue((normals[:, 2] > 0).all())
            self.assertAlmostEqual(areas.sum(), 20 * np.sqrt(14))
            projector = GridProjector(centers, height.shape, 1.)
            values = 2 * centers[:, 0] + 3 * centers[:, 1] + 5
            actual = projector.apply(values)
            np.testing.assert_allclose(actual[projector.footprint], (height + 5)[projector.footprint])
            self.assertTrue(np.isnan(actual[~projector.footprint]).all())
            zero = projector.apply(np.zeros(len(faces)))
            self.assertTrue((zero[projector.footprint] == 0).all())

    def test_index_rows_are_not_heightmap_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root)
            data = LunarData(root, "val")
            a0, a1 = data.sample(0, ["58"]), data.sample(2, ["58"])
            np.testing.assert_array_equal(a0["height_m"], a1["height_m"])
            self.assertEqual(a0["height_m"][0, 0], 7)
            self.assertEqual(a0["target_58"][0, 0], 110)
            self.assertEqual(a1["target_58"][0, 0], 130)
            self.assertFalse(a0["mask_58"][1, 1])
            selected = data.select(None, 2, 1, 4)
            self.assertEqual(len({data.rows[i]["terrain_id"] for i in selected}), 2)
            data.close()

    def test_plan_defaults_to_58_and_requires_two_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_data(root)
            args = parser().parse_args(["--data-root", directory, "--out", str(root / "out"),
                                       "--samples-per-tx", "100", "--sample-ids", "a0"])
            plan = make_plan(args, LunarData(root, "val"))
            self.assertEqual(plan["bands"], ["58"])
            self.assertEqual(plan["count"]["case_runs"], 12)
            self.assertEqual(plan["count"]["total_launched_rays"], 1200)
            self.assertFalse((root / "out").exists())
            args.seeds = [0]
            with self.assertRaises(ValueError):
                make_plan(args, LunarData(root, "val"))


class AnalysisIntegrationTests(unittest.TestCase):
    def test_interrupted_batches_resume_without_repeating_completed_work(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, faces, centers, _ = triangulate(np.zeros((4, 4)), 1.)
            projector = GridProjector(centers, (4, 4), 1.)
            schedule = batch_plan(10, 4, 0)
            plan = {"batches": {"0": schedule}, "max_depth": 3, "samples_per_tx": 10}
            calls = []
            sampling_calls = []
            class Backend:
                fail = True
                def solve(self, scene, measurement, samples, seed, depth, flags, **sampling):
                    calls.append(seed)
                    sampling_calls.append(sampling)
                    if self.fail and len(calls) == 2:
                        raise RuntimeError("simulated interruption")
                    return np.full(len(faces), seed % 23, np.float64)
            backend = Backend()
            sample = {"sample_id": "sample"}
            arguments = (backend, None, None, projector, len(faces), plan, "58", sample, 0, "full", root)
            with self.assertRaises(RuntimeError):
                run_case(*arguments)
            backend.fail = False
            run_case(*arguments)
            self.assertEqual(calls, [schedule[0]["seed"], schedule[1]["seed"], schedule[1]["seed"], schedule[2]["seed"]])
            self.assertEqual(sampling_calls, [
                {"total_samples": 10, "sample_offset": offset} for offset in (0, 4, 4, 8)])
            expected = sum((b["seed"] % 23) * b["samples"] for b in schedule) / 10
            with np.load(root / "maps" / "sample" / "58" / "seed0" / "full.npz") as result:
                np.testing.assert_allclose(result["cell_gain"], expected)
            run_case(*arguments)
            self.assertEqual(len(calls), 4)

    def test_complete_analysis_averages_gain_before_db(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {"samples": [{"sample_id": "a", "terrain_id": "t"}], "bands": ["58"],
                    "seeds": [0, 1], "samples_per_tx": 100, "scope": "test"}
            write_json(root / "manifest.json", {"experiment": plan})
            write_npz(root / "inputs" / "a.npz", cell_areas=np.ones(2),
                      target_58=np.full((2, 2), -10 * np.log10(3)),
                      mask_58=np.ones((2, 2), bool), interpolation_footprint=np.ones((2, 2), bool))
            for seed, full in ((0, 2.), (1, 4.)):
                for case in CONFIGS:
                    factor = 1 if case == "full" else .5
                    write_npz(root / "maps" / "a" / "58" / f"seed{seed}" / f"{case}.npz",
                              cell_gain=np.full(2, full * factor), grid_gain=np.full((2, 2), full * factor), samples=100)
            _, comparisons, labels, closure = analyze(root)
            label = next(r for r in labels if r["seed"] == "mean" and r["configuration"] == "full")
            self.assertAlmostEqual(label["rmse_db_on_positive_overlap"], 0)
            self.assertTrue(all(abs(r["signed_relative_full_minus_los_minus_nonlos"]) < 1e-10 for r in closure))
            self.assertEqual(len(comparisons), 30)
            analyze_main(argparse.Namespace(run=str(root), out=None))
            summary = json.loads((root / "analysis" / "summary.json").read_text())
            self.assertEqual(len(summary["contribution_summary"]), 10)
            missing = root / "maps" / "a" / "58" / "seed1" / "no_diffuse.npz"
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
