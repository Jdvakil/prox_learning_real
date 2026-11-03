import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import eval_act_real as ev


class InferenceContractTests(unittest.TestCase):
    def test_arm_state_padding(self):
        np.testing.assert_array_equal(ev.state_vector(np.arange(7), 9), np.r_[np.arange(7), 0, 0])
        np.testing.assert_array_equal(ev.state_vector(np.arange(7), 7), np.arange(7))
        for dimension in (8, 14):
            with self.assertRaises(ValueError):ev.state_vector(np.arange(7), dimension)

    def test_chunk_execution_and_requery(self):
        player = ev.ChunkPlayer(3)
        for tick in range(9):
            if player.needs_query(tick):
                player.add(tick, np.repeat(np.arange(tick, tick+5)[:, None], 7, axis=1))
            np.testing.assert_array_equal(player.action(tick), np.full(7,tick))

    def test_temporal_aggregation_preserves_zeros_and_favors_recent(self):
        player=ev.ChunkPlayer(1, temporal=True, decay=1)
        player.add(0,np.zeros((3,7)))
        player.add(1,np.ones((3,7)))
        expected=1/(1+np.exp(-1))
        np.testing.assert_allclose(player.action(1),expected)
        with self.assertRaises(RuntimeError):player.action(5)

    def test_prefetch_keeps_chunk_time_origin(self):
        player=ev.ChunkPlayer(50)
        player.add(0,np.zeros((50,7)))
        self.assertEqual(player.next_query_tick(8),42)
        player.add(42,np.repeat(np.arange(42,92)[:,None],7,axis=1))
        # A result requested at tick 42 and received at 47 starts at 42,
        # rather than replaying old predictions from the start of that chunk.
        np.testing.assert_array_equal(player.action(47),np.full(7,47))

    def test_target_checks(self):
        q=np.asarray(ev.START);lower=q-.1;upper=q+.1
        np.testing.assert_array_equal(ev.checked_target(np.r_[q,0],q,lower,upper,.12),q)
        for action in (np.full(7,np.nan),q+.2,np.zeros(14)):
            with self.assertRaises(ValueError):ev.checked_target(action,q,lower,upper,.12)
        with self.assertRaises(ValueError):ev.checked_target(q+.09,q,lower,upper,.05)

    def test_saved_training_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'training_config.json').write_text(json.dumps(dict(arguments={'task_name':'test'},
                policy_config={'camera_names':['left','right']})))
            self.assertEqual(ev.read_training_config(root)['camera_names'],['left','right'])

    def test_reference_has_bounded_velocity_and_acceleration(self):
        q=np.asarray(ev.START);filter=ev.SmoothReference(q,.15,.5);old_v=filter.dq.copy()
        target=q+.05
        for i in range(500):
            position,velocity=filter.step(target,.01)
            self.assertLessEqual(abs(velocity).max(),.15+1e-12)
            self.assertLessEqual(abs(velocity-old_v).max()/.01,.5+1e-12)
            old_v=velocity
        np.testing.assert_allclose(position,target,atol=1e-5)
        with self.assertRaises(RuntimeError):filter.step(target,.2)


class FakeTracker:
    def __init__(self,robot,**kwargs):self.robot=robot
    def __enter__(self):return self
    def __exit__(self,*args):
        self.robot.state.q=np.asarray(ev.GOAL)+.3
        time.sleep(.05)
    def tick(self):time.sleep(.002);return .002
    def set_target(self,q,dq):self.robot.state.q=q.copy()


class ServoTests(unittest.TestCase):
    def make_servo(self,watchdog=.05):
        q=np.asarray(ev.START)
        args=SimpleNamespace(max_velocity=.15,max_acceleration=.5,max_error=.15,
                             max_target_delta=.12,watchdog=watchdog)
        robot=SimpleNamespace(state=SimpleNamespace(q=q.copy()))
        return ev.RobotServo(robot,q,args,(q-.1,q+.1),None)

    def test_watchdog_stops_control(self):
        servo=self.make_servo()
        with patch.dict(sys.modules,{'franky':SimpleNamespace(JointImpedanceTracker=FakeTracker)}):
            servo.start();self.assertTrue(servo.ready.wait(1));servo.join(1)
        self.assertTrue(servo.done.is_set())
        self.assertIn('watchdog',str(servo.error))

    def test_commands_are_logged_before_shutdown(self):
        servo=self.make_servo(watchdog=1)
        with patch.dict(sys.modules,{'franky':SimpleNamespace(JointImpedanceTracker=FakeTracker)}):
            servo.start();self.assertTrue(servo.ready.wait(1))
            servo.publish(np.asarray(ev.START)+.02,np.asarray(ev.START),time.monotonic())
            time.sleep(.03);servo.stop()
        self.assertIsNone(servo.error)
        self.assertTrue(servo.commands)
        self.assertLess(abs(servo.commands[-1][1]-np.asarray(ev.START)).max(),.02)
        self.assertGreater(abs(servo.robot.state.q-np.asarray(ev.GOAL)).max(),.2)

    def test_artifact_records_camera_order_and_commands(self):
        records=[dict(t=0.,q=np.asarray(ev.START),rgb=np.zeros((2,4,4,3),np.uint8))]
        commands=[(10.,np.asarray(ev.START),np.zeros(7))]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            ev.save_results(out,records,[(0,np.zeros((3,8)))],commands,
                            dict(camera_names=['exo','wrist'],monotonic_t0=10,success=False))
            with h5py.File(out/'rollout.h5') as f:
                self.assertEqual(f['rgb'].shape,(1,2,4,4,3))
                self.assertEqual(f['queries/actions'].shape,(1,3,8))
                np.testing.assert_array_equal(f['control/t'][:],[0])
                self.assertEqual(json.loads(f.attrs['camera_names_json']),['exo','wrist'])


class LiveWorkflowTests(unittest.TestCase):
    def run_fake_live(self,output,stale=False,slow=False):
        args=ev.parse_args(['--wrist-serial','w','--exo-serial','e','--no-reset',
                            '--duration','.22','--success-hold','.07'])
        args.goal=list(ev.START)
        if slow:args.duration=.55
        robot=SimpleNamespace(state=SimpleNamespace(q=np.asarray(ev.START).copy()),
                              recover_from_errors=lambda:None)
        names=['exo_camera_1','wrist_camera']
        cams=[]
        for name in names:
            def latest():
                return ((np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint16),
                         1,0.,time.monotonic()-(1 if stale else 0)),1)
            cams.append(SimpleNamespace(name=name,latest=latest,close=lambda:None))
        def predict(q,rgb):
            if slow:time.sleep(.1)
            return np.tile(np.r_[ev.START,0],(5 if slow else 50,1))
        policy=SimpleNamespace(camera_names=names,query_every=5 if slow else 50,
             prepare_images=lambda images:np.stack([images[n] for n in names]),
             predict=predict)
        def state(robot):return dict(q=robot.state.q.copy(),dq=np.zeros(7))
        with patch.dict(sys.modules,{'franky':SimpleNamespace(
            Robot=lambda host:robot,JointMotion=lambda q:q,JointImpedanceTracker=FakeTracker
        )}),patch.object(ev.base,'open_cameras',return_value=cams),\
             patch.object(ev.base,'read_robot_state',side_effect=state):
            ev.live(args,policy,output,None)

    def test_live_workflow_stops_logging_before_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            self.run_fake_live(output)
            result=json.loads((output/'result.json').read_text())
            self.assertTrue(result['success'])
            with h5py.File(output/'rollout.h5') as f:
                np.testing.assert_allclose(f['q'][-1],ev.START)
                self.assertEqual(f['rgb'].shape,(4,2,4,4,3))

    def test_stale_camera_aborts_and_saves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            with self.assertRaisesRegex(RuntimeError,'stale'):
                self.run_fake_live(output,stale=True)
            result=json.loads((output/'result.json').read_text())
            self.assertFalse(result['success'])
            self.assertIn('stale',result['error'])

    def test_slow_inference_does_not_block_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)
            self.run_fake_live(output,slow=True)
            result=json.loads((output/'result.json').read_text())
            self.assertTrue(result['success'])
            self.assertGreater(result['queries'],1)
            with h5py.File(output/'rollout.h5') as f:
                self.assertEqual(len(f['t']),9)
                self.assertLess(np.diff(f['t'][:]).max(),.09)


if __name__=='__main__':unittest.main()
