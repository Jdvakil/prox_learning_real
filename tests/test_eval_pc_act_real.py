from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import sys
import io
from contextlib import redirect_stdout
from unittest.mock import patch

import h5py
import numpy as np

from prox_real import eval_pc_act_real as pc
from test_eval_act_real import FakeTracker


class HistoryTests(unittest.TestCase):
    def test_history_matches_training_causal_repeat(self):
        history=pc.CausalSkinHistory()
        for t in range(3):h=history.append(np.full((40,8,8),t,dtype=np.float32))
        np.testing.assert_array_equal(h[0,:,0,0],np.repeat([0,0,0,0,0,0,1,2],4))
        self.assertEqual(h.shape,(40,32,8,8))

    def test_model_and_sensor_check_need_no_camera_serials(self):
        from prox_real import eval_act_real as act
        args=act.parse_args(['--sensor-check'],pc.configure)
        self.assertTrue(args.sensor_check)
        self.assertTrue(args.no_reset)
        self.assertEqual(args.query_every,1)


class GuardTests(unittest.TestCase):
    def context(self,temp,background=4.,mode='live',stale=False,disable_guard=False):
        path=Path(temp)/'baseline.npz'
        np.savez(path,minimum_depth_m=np.full((40,8,8),background,np.float32),sensor_names=np.asarray(pc.NAMES))
        args=SimpleNamespace(guard_baseline=path,guard_drop_m=.01,guard_min_zones=2,stop_distance_m=.08,
            sensor_timeout=.75,sensor_warmup=.1,proximity_mode=mode,no_proximity_guard=disable_guard)
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

    def test_disabled_distance_guard_preserves_live_model_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp,disable_guard=True)
            depth[0,:4]=32
            x=c.sample();c.control_guard()
            self.assertEqual(x['guard_sensors'],[pc.NAMES[0]])
            self.assertFalse(c.result_fields()['proximity_guard_enabled'])
            self.assertIsNone(c.trigger)
            self.assertAlmostEqual(float(c.policy_input(x)[0,-1,0,0]),.032)

    def test_disabling_distance_guard_does_not_disable_stale_sensor_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            c,_=self.context(temp,stale=True,disable_guard=True)
            with self.assertRaises(pc.ProximityAbort):c.control_guard()
            self.assertEqual(c.trigger['reason'],'stale_sensor')

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

    def test_diagnostic_reports_all_triggering_sensors_without_latching_abort(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp)
            depth[0,:2]=40;depth[17,:3]=50
            x=c.read(diagnostic=True)
            self.assertEqual(x['guard_sensors'],[pc.NAMES[0],pc.NAMES[17]])
            self.assertIsNone(c.trigger)
            rows=pc.diagnostic_rows(x,2)
            self.assertEqual(len(rows),40)
            self.assertEqual(sum(row['guard_would_trigger'] for row in rows),2)
            # Rollout mode still aborts on the same readings.
            with self.assertRaises(pc.ProximityAbort):c.control_guard()

    def test_diagnostic_shows_missing_and_stale_sensors_instead_of_exiting(self):
        with tempfile.TemporaryDirectory() as temp:
            c,_=self.context(temp)
            original=c.listener.snapshot
            def snapshot():
                values=original();values.pop(c.keys[3])
                raw,arrived,count=values[c.keys[8]]
                values[c.keys[8]]=(raw,arrived-2,count)
                return values
            c.listener.snapshot=snapshot
            rows=pc.diagnostic_rows(c.read(diagnostic=True),2)
            self.assertEqual(len(rows),40)
            self.assertEqual(rows[3]['status'],'MISSING')
            self.assertEqual(rows[8]['status'],'STALE')
            self.assertEqual(sum(row['status']=='LIVE' for row in rows),38)

    def test_sensor_check_continues_after_first_sensor_and_saves_all_40(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp)
            c.args.output=Path(temp)/'check';c.args.duration=.025;c.args.sensor_check_interval=.01
            original=c.listener.snapshot;calls=[0];clock=[1000.]
            def snapshot():
                depth[:]=4000
                depth[0 if calls[0]==0 else 9,:2]=40
                calls[0]+=1
                return original()
            c.listener.snapshot=snapshot
            with patch.object(pc,'LiveSkinContext',return_value=c),\
                 patch.object(pc.time,'monotonic',side_effect=lambda:clock[0]),\
                 patch.object(pc.time,'sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)),\
                 redirect_stdout(io.StringIO()) as output:
                pc.sensor_check(c.args)
            result=json.loads((c.args.output/'sensor_check.json').read_text())
            self.assertEqual(result['sensor_count'],40)
            self.assertEqual(result['sensors_with_packets'],40)
            self.assertEqual(result['snapshots'],3)
            self.assertTrue(result['sensors'][0]['guard_would_trigger'])
            self.assertTrue(result['sensors'][9]['guard_would_trigger'])
            self.assertIn(pc.NAMES[-1],output.getvalue())
            self.assertEqual(len((c.args.output/'snapshots.jsonl').read_text().splitlines()),3)

    def test_preflight_guard_is_reported_cleanly_before_robot_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            c,depth=self.context(temp,mode='clear')
            depth[pc.NAMES.index('link2_sensor_4'),:4]=32
            args=pc.act.parse_args(['--wrist-serial','w','--exo-serial','e','--proximity-mode','clear'],pc.configure)
            args.guard_baseline=c.args.guard_baseline;c.args=args
            policy=SimpleNamespace(camera_names=['exo_camera_1','wrist_camera'],query_every=1)
            out=Path(temp)/'rollout';out.mkdir()
            with patch.dict(sys.modules,{'franky':SimpleNamespace(Robot=unittest.mock.Mock(),JointMotion=lambda q:q)}),\
                 patch.object(pc.base,'open_cameras') as cameras,redirect_stdout(io.StringIO()) as captured:
                robot=sys.modules['franky'].Robot
                code=pc.guarded_rollout(args,policy,out,None,c)
                robot.assert_not_called();cameras.assert_not_called()
            self.assertEqual(code,2)
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['outcome'],'blocked_at_sensor_preflight')
            self.assertFalse(result['robot_connected'])
            self.assertEqual(result['termination_phase'],'sensor_preflight')
            self.assertEqual(result['steps'],0)
            self.assertTrue(result['guard_triggered'])
            self.assertFalse(result['success'])
            self.assertEqual(result['stop_event']['zones'],4)
            self.assertAlmostEqual(result['stop_event']['distance_m'],.032)
            self.assertIn('32.0 mm',captured.getvalue())
            self.assertNotIn('Traceback',captured.getvalue())
            self.assertTrue((out/'sensor_events.json').exists())

    def test_unrelated_runtime_bug_is_not_misreported_as_guard_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            c,_=self.context(temp)
            c.trigger=dict(reason='novel_proximity',sensor=pc.NAMES[0])
            with patch.object(pc.act,'live',side_effect=RuntimeError('unrelated model failure')):
                with self.assertRaisesRegex(RuntimeError,'unrelated model failure'):
                    pc.guarded_rollout(None,None,Path(temp),None,c)

    def test_wrapped_controller_guard_stop_keeps_the_failure_result(self):
        with tempfile.TemporaryDirectory() as temp:
            c,_=self.context(temp)
            c.trigger=dict(reason='novel_proximity',sensor=pc.NAMES[0],distance_m=.032,zones=4,threshold_m=.08)
            out=Path(temp)
            (out/'result.json').write_text(json.dumps(dict(success=False,robot_connected=True,termination_phase='rollout')))
            error=RuntimeError('robot control stopped: '+json.dumps(c.trigger))
            with patch.object(pc.act,'live',side_effect=error),redirect_stdout(io.StringIO()):
                code=pc.guarded_rollout(None,None,out,None,c)
            self.assertEqual(code,2)
            self.assertEqual(json.loads((out/'result.json').read_text())['outcome'],'proximity_guard_stop')


class ReturnToStartTests(unittest.TestCase):
    def fixture(self,temp,completed=True):
        out=Path(temp)
        result=dict(success=True,error='',termination_phase='completed' if completed else 'sensor_preflight')
        (out/'result.json').write_text(json.dumps(result))
        return SimpleNamespace(start=list(pc.act.START)),out

    def test_completed_run_returns_to_a_after_path_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out=self.fixture(temp)
            with patch.object(pc.os,'isatty',return_value=True),patch('builtins.input',return_value=''),\
                 patch.object(pc,'reset_to_start',return_value=.001) as reset,redirect_stdout(io.StringIO()):
                self.assertEqual(pc.reset_after_run(args,out,0),0)
                reset.assert_called_once_with(args)
            record=json.loads((out/'reset.json').read_text())
            self.assertEqual(record['status'],'complete')
            self.assertTrue(json.loads((out/'result.json').read_text())['success'])

    def test_guard_abort_never_automatically_resets(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out=self.fixture(temp,completed=False)
            with patch.object(pc,'reset_to_start') as reset,patch('builtins.input') as prompt,redirect_stdout(io.StringIO()):
                self.assertEqual(pc.reset_after_run(args,out,2),2)
                reset.assert_not_called();prompt.assert_not_called()
            self.assertEqual(json.loads((out/'reset.json').read_text())['status'],'skipped')

    def test_redirected_stdin_never_implies_clear_path(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out=self.fixture(temp)
            with patch.object(pc.os,'isatty',return_value=False),patch.object(pc,'reset_to_start') as reset,\
                 patch('builtins.input') as prompt,redirect_stdout(io.StringIO()):
                pc.reset_after_run(args,out,0)
                reset.assert_not_called();prompt.assert_not_called()

    def test_operator_can_skip_return_to_start(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out=self.fixture(temp)
            with patch.object(pc.os,'isatty',return_value=True),patch('builtins.input',return_value='skip'),\
                 patch.object(pc,'reset_to_start') as reset,redirect_stdout(io.StringIO()):
                pc.reset_after_run(args,out,0)
                reset.assert_not_called()
            self.assertEqual(json.loads((out/'reset.json').read_text())['status'],'skipped')

    def test_interrupted_return_is_not_reported_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            args,out=self.fixture(temp)
            with patch.object(pc.os,'isatty',return_value=True),patch('builtins.input',return_value=''),\
                 patch.object(pc,'reset_to_start',side_effect=KeyboardInterrupt),redirect_stdout(io.StringIO()):
                self.assertEqual(pc.reset_after_run(args,out,0),130)
            self.assertEqual(json.loads((out/'reset.json').read_text())['status'],'interrupted')


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
