from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch

import h5py
import numpy as np

import eval_pc_act_real as pc
from test_eval_act_real import FakeTracker


class HistoryTests(unittest.TestCase):
    def test_history_matches_training_causal_repeat(self):
        history=pc.CausalSkinHistory()
        for t in range(3):h=history.append(np.full((40,8,8),t,dtype=np.float32))
        np.testing.assert_array_equal(h[0,:,0,0],np.repeat([0,0,0,0,0,0,1,2],4))
        self.assertEqual(h.shape,(40,32,8,8))

    def test_model_and_sensor_check_need_no_camera_serials(self):
        import eval_act_real as act
        args=act.parse_args(['--sensor-check'],pc.configure)
        self.assertTrue(args.sensor_check)
        self.assertTrue(args.no_reset)
        self.assertEqual(args.query_every,1)


class GuardTests(unittest.TestCase):
    def context(self,temp,background=4.,mode='live',stale=False):
        path=Path(temp)/'baseline.npz'
        np.savez(path,minimum_depth_m=np.full((40,8,8),background,np.float32),sensor_names=np.asarray(pc.NAMES))
        args=SimpleNamespace(guard_baseline=path,guard_drop_m=.01,guard_min_zones=2,stop_distance_m=.08,
            sensor_timeout=.75,sensor_warmup=.1,proximity_mode=mode)
        depth=np.full((40,128),4000,np.uint16)
        lookup={pc.sim_name(link,ident):(link,ident) for link,ident in pc.base.SENSORS}
        def snapshot():return {lookup[name]:(depth[i].copy(),time.monotonic()-(1 if stale else 0),2) for i,name in enumerate(pc.NAMES)}
        listener=SimpleNamespace(snapshot=snapshot,missing=lambda:[])
        return pc.LiveSkinContext(args,listener),depth

    def test_guard_does_not_trigger_on_known_near_background(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp,background=.02)
            depth[:,:64]=18
            x=c.sample()
            self.assertFalse(x['novel'].any())
            self.assertIsNone(c.trigger)

    def test_novel_intrusion_aborts_even_with_policy_proximity_cleared(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp,mode='clear')
            first=c.sample()
            np.testing.assert_array_equal(c.policy_input(first),np.full((40,32,8,8),4.))
            depth[3,:2]=40
            with self.assertRaises(pc.ProximityAbort):c.control_guard()
            self.assertTrue(c.result_fields()['guard_triggered'])
            self.assertEqual(c.trigger['sensor'],pc.NAMES[3])
            self.assertIsNone(c.result_fields()['learned_avoidance_demonstrated'])

    def test_stale_sensor_is_a_fault_not_avoidance(self):
        with tempfile.TemporaryDirectory() as temp:
            c,_=self.context(temp,stale=True)
            with self.assertRaises(pc.ProximityAbort):c.control_guard()
            self.assertEqual(c.trigger['reason'],'stale_sensor')
            self.assertTrue(c.result_fields()['sensor_fault'])
            self.assertFalse(c.result_fields()['guard_triggered'])

    def test_individual_close_pixel_is_logged_but_requires_two_zones_to_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp)
            depth[0,0]=40
            x=c.sample()
            self.assertEqual(int(x['close'].sum()),1)
            self.assertIsNone(c.trigger)

    def test_records_raw_sensor_data_and_guard_event(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp)
            x=c.sample();fields=c.record_fields(x)
            self.assertEqual(fields['proximity_depth_mm'].shape,(40,2,8,8))
            depth[0,:2]=30
            with self.assertRaises(pc.ProximityAbort):c.control_guard()
            out=Path(temp)/'output';out.mkdir()
            with h5py.File(out/'rollout.h5','w'):pass
            c.save_events(out,time.monotonic()-1)
            events=json.loads((out/'sensor_events.json').read_text())
            self.assertEqual(events[0]['reason'],'novel_proximity')
            with h5py.File(out/'rollout.h5') as f:
                self.assertEqual(f['guard/trigger_depth_mm'].shape,(40,2,8,8))


class RealLoopTests(unittest.TestCase):
    def run_fake(self,temp,trigger=False):
        args=pc.act.parse_args(['--wrist-serial','w','--exo-serial','e','--duration','.22',
                               '--success-hold','.07','--proximity-mode','clear'],pc.configure)
        args.goal=list(pc.act.START)
        baseline=Path(temp)/'baseline.npz'
        np.savez(baseline,minimum_depth_m=np.full((40,8,8),4.,np.float32),sensor_names=np.asarray(pc.NAMES))
        args.guard_baseline=baseline
        context=pc.LiveSkinContext(args,SimpleNamespace(missing=lambda:[]))
        calls=[0]
        def snapshot():
            calls[0]+=1
            raw=np.full((40,128),4000,np.uint16)
            raw[0,0]=150  # recorded as real sensor data even when policy mode is clear
            if trigger and calls[0]>20:raw[1,:2]=30
            return {key:(raw[i],time.monotonic(),calls[0]) for i,key in enumerate(context.keys)}
        context.listener.snapshot=snapshot
        robot=SimpleNamespace(state=SimpleNamespace(q=np.asarray(pc.act.START).copy()),recover_from_errors=lambda:None)
        camera_names=['exo_camera_1','wrist_camera']
        def latest():return ((np.zeros((4,4,3),np.uint8),None,1,0.,time.monotonic()),1)
        cams=[SimpleNamespace(name=n,latest=latest,close=lambda:None) for n in camera_names]
        def predict(q,rgb,skin):
            self.assertEqual(skin.shape,(40,32,8,8))
            np.testing.assert_array_equal(skin,np.full_like(skin,4.))
            return np.tile(np.r_[pc.act.START,0],(50,1))
        policy=SimpleNamespace(camera_names=camera_names,query_every=1,
            prepare_images=lambda x:np.stack([x[n] for n in camera_names]),predict=predict)
        out=Path(temp)/'rollout';out.mkdir()
        with patch.dict(sys.modules,{'franky':SimpleNamespace(Robot=lambda h:robot,JointMotion=lambda q:q,
                JointImpedanceTracker=FakeTracker)}),patch.object(pc.base,'open_cameras',return_value=cams),\
                patch.object(pc.base,'read_robot_state',side_effect=lambda r:dict(q=r.state.q.copy(),dq=np.zeros(7))):
            if trigger:
                with self.assertRaises((pc.ProximityAbort,RuntimeError)):
                    pc.act.live(args,policy,out,None,context=context)
            else:pc.act.live(args,policy,out,None,context=context)
        return out

    def test_clear_policy_still_logs_live_proximity(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.run_fake(temp)
            result=json.loads((out/'result.json').read_text())
            self.assertTrue(result['success'])
            self.assertFalse(result['guard_triggered'])
            with h5py.File(out/'rollout.h5') as f:
                self.assertEqual(f['proximity_depth_mm'][0,0,0,0,0],150)
                self.assertTrue(np.all(f['proximity_policy_history_m'][:]==4.))
                self.assertLess(abs(f['proximity_t'][0]),.05)

    def test_servo_guard_abort_is_not_policy_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out=self.run_fake(temp,trigger=True)
            result=json.loads((out/'result.json').read_text())
            self.assertFalse(result['success'])
            self.assertTrue(result['guard_triggered'])
            self.assertEqual(result['stop_event']['reason'],'novel_proximity')
            self.assertIsNone(result['learned_avoidance_demonstrated'])


if __name__=='__main__':unittest.main()
