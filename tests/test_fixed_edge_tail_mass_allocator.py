import unittest

import numpy as np

from scripts.allocate_fixed_edge_tail_mass import (
    EXPERTS,
    FIXED_EDGES,
    LAYERS,
    RADIUS,
    TOTAL_REMOVED,
    allocate_middle_dp,
    build_manifest,
    uniform_counts,
)


class FixedEdgeTailMassAllocatorTests(unittest.TestCase):
    def test_exact_budget_fixed_edges_and_middle_bounds(self):
        curve = np.tile(np.linspace(0.0, 1.0, EXPERTS + 1), (LAYERS, 1))
        counts = allocate_middle_dp(curve, TOTAL_REMOVED, uniform_counts(TOTAL_REMOVED))
        self.assertEqual(sum(counts.values()), TOTAL_REMOVED)
        for layer, count in FIXED_EDGES.items():
            self.assertEqual(counts[layer], count)
        uniform = uniform_counts(TOTAL_REMOVED)
        for layer in range(4, 44):
            self.assertLessEqual(abs(counts[layer] - uniform[layer]), RADIUS)
            self.assertGreaterEqual(EXPERTS - counts[layer], 10)

    def test_dp_uses_tail_mass_cost_on_middle_layers(self):
        # Make layer 4 cheap and layer 5 expensive; exact budget should shift
        # removals toward layer 4 within the bound while edges remain frozen.
        curve = np.zeros((LAYERS, EXPERTS + 1), dtype=np.float64)
        for layer in range(LAYERS):
            slope = 0.001 if layer == 4 else (100.0 if layer == 43 else 1.0)
            curve[layer] = np.arange(EXPERTS + 1) * slope
        counts = allocate_middle_dp(curve, TOTAL_REMOVED, uniform_counts(TOTAL_REMOVED))
        self.assertGreater(counts[4], uniform_counts(TOTAL_REMOVED)[4])
        self.assertLess(counts[43], uniform_counts(TOTAL_REMOVED)[43])
        for layer, count in FIXED_EDGES.items():
            self.assertEqual(counts[layer], count)

    def test_manifest_preserves_prefix_original_expert_ids(self):
        order = np.tile(np.arange(EXPERTS, dtype=np.int16), (LAYERS, 1))
        curve = np.tile(np.linspace(0.0, 1.0, EXPERTS + 1), (LAYERS, 1))
        # build_manifest only hashes these input paths, so use existing files
        # from the repo fixtures by passing a temp directory is unnecessary;
        # validate the IDs directly via a temporary test file below.
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp:
            curves_path = Path(temp) / "curves.npz"
            uniform_path = Path(temp) / "uniform.json"
            curves_path.write_bytes(b"curves")
            uniform_path.write_bytes(b"uniform")
            manifest = build_manifest(curves_path, uniform_path, curve, order)
        counts = manifest["per_layer_removal_counts"]
        removed = manifest["removed_expert_ids_original"]
        for layer in range(LAYERS):
            self.assertEqual(removed[str(layer)], list(range(int(counts[str(layer)]))))
        self.assertEqual(manifest["total_removed"], TOTAL_REMOVED)


if __name__ == "__main__":
    unittest.main()
