import unittest
import numpy as np
import torch

from prox_real.train_pc_act import history_at,losses,Samples
from act_model.sensor_order import HYBRID_SKIN_SENSOR_ORDER


class PCTrainingTests(unittest.TestCase):
    def test_causal_window_never_reads_future(self):
        x=np.repeat(np.arange(12,dtype=np.float32)[:,None,None,None],40*8*8,axis=1).reshape(12,40,8,8)
        h=history_at(x,2)
        self.assertEqual(h.shape,(40,32,8,8))
        np.testing.assert_array_equal(h[0,:,0,0],np.repeat([0,0,0,0,0,0,1,2],4))
        self.assertEqual(h.max(),2)

    def test_sensor_order_matches_raw_hdf5_contract(self):
        self.assertEqual(len(HYBRID_SKIN_SENSOR_ORDER),40)
        self.assertEqual(len(set(HYBRID_SKIN_SENSOR_ORDER)),40)
        self.assertLess(HYBRID_SKIN_SENSOR_ORDER.index('link5_back_sensor_0'),
                        HYBRID_SKIN_SENSOR_ORDER.index('link5_front_sensor_0'))

    def test_masked_loss_has_no_padded_gradient(self):
        pred=torch.ones(1,4,8,requires_grad=True);target=torch.zeros_like(pred)
        pad=torch.tensor([[False,False,True,True]])
        total,l1,kl=losses(pred,target,pad)
        total.backward()
        self.assertEqual(float(l1),.5)
        self.assertEqual(float(kl),0)
        self.assertTrue(torch.all(pred.grad[:,2:]==0))

    def test_real_loader_shift_and_history_alignment(self):
        e=dict(q=np.zeros((120,9),np.float32),actions=np.repeat(np.arange(120)[:,None],8,axis=1).astype(np.float32),
               skin=np.full((120,40,8,8),4.,np.float32),rgb=np.zeros((120,2,4,4,3),np.uint8))
        stats=dict(qpos_mean=np.zeros(9),qpos_std=np.ones(9),action_mean=np.zeros(8),action_std=np.ones(8))
        ds=Samples([e],[0],stats,50,validation=True)
        rgb,q,a,pad,skin=ds[1]  # fixed observation 38, real ACT action starts at 37
        self.assertEqual(float(a[0,0]),37)
        self.assertEqual(skin.shape,(40,32,8,8))
        self.assertFalse(pad.any())


if __name__=='__main__':unittest.main()
