"""Landing geometry and diagnosis guards; no simulation integration."""
import unittest
from pathlib import Path
import numpy as np


class LandingAuditTests(unittest.TestCase):
    def test_clearance_is_world_mesh_minimum_and_translates(self):
        try:
            import mujoco
        except ImportError:
            self.skipTest('MuJoCo not installed')
        from movement_lab.landing_diagnostics import foot_clearances
        p=Path('results/01b/gpu-gate/replay_model.mjb')
        if not p.exists(): self.skipTest('local compiled geometry not available')
        m=mujoco.MjModel.from_binary_path(str(p));d=mujoco.MjData(m)
        with np.load('results/01b/gpu-gate/zero/initial.npz') as ini:d.qpos[:]=ini['qpos']
        mujoco.mj_forward(m,d);before=foot_clearances(m,d)
        self.assertAlmostEqual(before['left_foot_collision'],.0078176373,places=9)
        shift=.001-min(before.values());d.qpos[2]+=shift
        mujoco.mj_forward(m,d);after=foot_clearances(m,d)
        self.assertAlmostEqual(min(after.values()),.001,places=10)
        for n in before:self.assertAlmostEqual(after[n]-before[n],shift,places=10)

    def test_diagnostic_runner_has_only_zero_plan(self):
        # Guard against inadvertently scheduling the inherited departure library.
        import ast
        tree=ast.parse(Path('movement_lab/landing01b.py').read_text())
        plans=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='plans' for t in n.targets)]
        self.assertEqual(len(plans),1)
        self.assertEqual(ast.literal_eval(plans[0].value),['zero'])
