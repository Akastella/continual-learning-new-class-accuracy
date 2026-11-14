"""Scientific metric checks, including future-class exclusion and label identity."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('r15_analysis', ROOT/'analysis/analyze_r15_normal_budget.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Metrics(unittest.TestCase):
    def test_candidates_and_ground_truth(self):
        seed, task = 3101, 6
        order = np.random.default_rng(seed+40000).permutation(100)
        original = np.repeat(np.arange(100), 100)
        mapped = np.argsort(order)[original]
        logits = np.zeros((10000, 100), dtype=np.float32)
        logits[np.arange(10000), mapped] = 1
        logits[:, 99] = 100  # Future class must be excluded at t=6.
        current = mapped//10 == task
        logits[current, 0] = 2  # Old/current competition defeats all-seen only.
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'raw.npz'
            np.savez(path, logits=logits, indices=np.arange(10000), original_labels=original,
                     mapped_labels=mapped, class_order=order)
            r = a.endpoint(path, seed, task)
            self.assertEqual(r['current'], {'correct': 0, 'n': 1000})
            self.assertEqual(r['current_masked'], {'correct': 1000, 'n': 1000})
            self.assertEqual(r['old'], {'correct': 6000, 'n': 6000})
            mapped[0] = (mapped[0]+1)%100
            np.savez(path, logits=logits, indices=np.arange(10000), original_labels=original,
                     mapped_labels=mapped, class_order=order)
            with self.assertRaisesRegex(ValueError, 'remapping'):
                a.endpoint(path, seed, task)

    def test_bootstrap_fixed_matrix(self):
        r = a.stats([1, 2, 3, 4, 5])
        self.assertEqual(r['mean'], 3)
        self.assertEqual(r['bootstrap_resamples'], 3125)
        self.assertAlmostEqual(r['sample_sd'], np.sqrt(2.5))
        self.assertEqual(a.stats([2]*5)['approximate_bootstrap_percentile_95'], [2, 2])
        with self.assertRaises(ValueError):
            a.stats([1, 2, 3, 4])


if __name__ == '__main__':
    unittest.main()
