import unittest

from scripts.compare_layerwise_reuse_lcb import exact_mcnemar, paired_metrics


class LayerwiseReuseComparisonTests(unittest.TestCase):
    def test_exact_mcnemar_for_balanced_discordances(self):
        self.assertEqual(exact_mcnemar(11, 11), 1.0)
        self.assertEqual(exact_mcnemar(0, 0), 1.0)

    def test_pairs_by_problem_and_reports_bootstrap_intervals(self):
        candidate = {
            "a": (1, "contest-1"),
            "b": (0, "contest-1"),
            "c": (1, "contest-2"),
            "d": (0, "contest-2"),
        }
        control = {
            "a": (0, "contest-1"),
            "b": (1, "contest-1"),
            "c": (1, "contest-2"),
            "d": (0, "contest-2"),
        }
        metrics = paired_metrics(candidate, control, seed=42, n_bootstrap=200)
        self.assertEqual(metrics["paired_delta_solved"], 0)
        self.assertEqual(metrics["candidate_only_passes"], 1)
        self.assertEqual(metrics["control_only_passes"], 1)
        self.assertEqual(metrics["exact_mcnemar_p"], 1.0)
        self.assertEqual(metrics["contest_clusters"], 2)
        self.assertEqual(len(metrics["task_bootstrap_95_ci"]), 2)
        self.assertEqual(len(metrics["contest_cluster_bootstrap_95_ci"]), 2)


if __name__ == "__main__":
    unittest.main()
