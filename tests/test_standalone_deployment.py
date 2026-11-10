import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prox_real import eval_act_real as ev
from prox_real.export_model import export


class StandaloneTests(unittest.TestCase):
    def test_default_paths_and_model_check_need_no_external_checkout(self):
        self.assertEqual(ev.DEFAULT_RUN,ev.ROOT/'models/act_test_20')
        args=ev.parse_args(['--check-model'])
        self.assertTrue(args.check_model)
        self.assertFalse(hasattr(args,'act_repo'))
        self.assertTrue((ev.ROOT/'act_model/backbone.py').is_file())

    def test_export_embeds_all_metadata_and_portable_statistics(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'training';source.mkdir()
            cfg=dict(camera_names=['exo_camera_1','wrist_camera'],nheads=8,state_dim=9,action_dim=8)
            (source/'training_config.json').write_text(json.dumps(dict(policy_config=cfg)))
            weights=b'checkpoint fixture'
            (source/'policy_best.ckpt').write_bytes(weights)
            np.savez(source/'dataset_stats.npz',qpos_mean=np.zeros(9,np.float32),
                     qpos_std=np.ones(9,np.float32),action_mean=np.zeros(8,np.float32),
                     action_std=np.ones(8,np.float32))
            dest=root/'deployment'
            export(source,dest,image_height=240,image_width=320)
            meta=json.loads((dest/'image_metadata.json').read_text())
            self.assertEqual(meta,dict(camera_names=cfg['camera_names'],image_h=240,image_w=320))
            self.assertEqual(ev.read_training_config(dest)['camera_names'],cfg['camera_names'])
            with np.load(dest/'dataset_stats.npz',allow_pickle=False) as stats:
                np.testing.assert_array_equal(stats['action_std'],np.ones(8))
            manifest=json.loads((dest/'manifest.json').read_text())
            self.assertEqual(manifest['policy_best.ckpt']['sha256'],hashlib.sha256(weights).hexdigest())
            self.assertFalse((dest/'dataset_stats.pkl').exists())


if __name__=='__main__':unittest.main()
