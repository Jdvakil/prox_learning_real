import json
from pathlib import Path
import tempfile
import unittest

from prox_real.pc_act_wandb import flatten_metrics, read_history


class WandbMetricsTests(unittest.TestCase):
    def test_preserves_validation_and_ablation_meanings(self):
        x=flatten_metrics(dict(epoch=100,train_loss=2.,train_l1=.2,train_kl=.18,elapsed_s=30.,
            validation=dict(normalized_l1=.1,live_rad=.004,clear_rad=.005,shuffled_rad=.006,near_delta_rad=.00001)))
        self.assertEqual(x['validation/arm_mae_rad'],.004)
        self.assertEqual(x['ablation/clear_arm_mae_rad'],.005)
        self.assertEqual(x['ablation/synthetic_near_action_delta_rad'],.00001)
        self.assertEqual(x['epoch'],100)

    def test_missing_validation_is_not_logged_as_zero(self):
        x=flatten_metrics(dict(epoch=1,train_loss=.3))
        self.assertFalse(any(k.startswith('validation/') for k in x))

    def test_rejects_duplicate_epoch_history(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)
            (p/'metrics.jsonl').write_text('\n'.join(json.dumps(dict(epoch=n)) for n in [0,1,1]))
            with self.assertRaises(ValueError):read_history(p)


if __name__=='__main__':unittest.main()
