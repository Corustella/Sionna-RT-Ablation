"""Missing-data, scoring-denominator and mean-domain contracts."""

import unittest

import numpy as np

from export_aligned_labels import check_geometry, combine, filled, pooled, statistics


class AlignedLabelTests(unittest.TestCase):
    def test_missing_seed_does_not_halve_an_observed_gain(self):
        gains = np.asarray([[[.01, 0., np.nan]], [[0., .001, np.nan]]])
        pl, count = combine(gains, "gain")
        np.testing.assert_allclose(pl[0, :2], [20., 30.])
        np.testing.assert_array_equal(count, [[1, 1, 0]])
        self.assertTrue(np.isnan(pl[0, 2]))
        self.assertEqual(gains[1, 0, 0], 0.)

    def test_domain_is_explicit_and_log_mean_is_not_power_mean(self):
        gains = np.asarray([[[1.]], [[.01]]])
        power, _ = combine(gains, "gain")
        db, _ = combine(gains, "db")
        self.assertAlmostEqual(float(power[0, 0]), -10 * np.log10(.505))
        self.assertEqual(float(db[0, 0]), 10.)
        with self.assertRaises(ValueError):
            combine(gains, "guess")

    def test_real_145_is_resolved_and_prediction_missing_is_still_scored(self):
        pl = np.asarray([[145., np.nan, 100.]])
        target = np.asarray([[145., 125., 0.]])
        mask = np.asarray([[True, True, False]])
        label, resolved = filled(pl)
        self.assertEqual(label.dtype, np.float16)
        np.testing.assert_array_equal(resolved, [[True, False, True]])
        result = statistics(pl, target, mask)
        self.assertEqual(result["official_valid_count"], 2)
        self.assertEqual(result["resolved_official_valid_count"], 1)
        self.assertEqual(result["masked_sse_db"], 400.)
        self.assertAlmostEqual(result["masked_rmse_db"], np.sqrt(200))

    def test_pool_uses_pixel_sse_not_mean_of_map_rmse(self):
        rows = []
        for pl, target in ((np.asarray([[10.]]), np.zeros((1, 1))),
                           (np.zeros((1, 9)), np.zeros((1, 9)))):
            rows.append({"partition": "fit", "variant": "test",
                         **statistics(pl, target, np.ones(pl.shape, bool))})
        result = pooled(rows)[0]
        self.assertEqual(result["official_valid_count"], 10)
        self.assertAlmostEqual(result["masked_rmse_db"], np.sqrt(10))

    def test_saved_measurement_must_be_exact_translated_copy(self):
        vertices = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
        faces = np.asarray([[0, 1, 2]], np.uint32)
        geometry = {"terrain_vertices": vertices, "measurement_vertices": vertices + [0, 0, 1],
            "terrain_faces": faces, "measurement_faces": faces, "height_m": np.zeros((2, 2)),
            "tx_rc": [0, 0], "tx_xyz": np.asarray([0., 0., 3.], np.float32)}
        metadata = {"resolution_m_per_px": 1., "simulation": {"tx_height_m_agl": 3., "rx_height_m_agl": 1.}}
        self.assertTrue(check_geometry(geometry, metadata)["measurement_is_translated_copy"])
        geometry["measurement_vertices"][0, 2] = 1.1
        with self.assertRaises(ValueError):
            check_geometry(geometry, metadata)


if __name__ == "__main__":
    unittest.main()
