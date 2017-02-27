""" Testing for Problem.check_partial_derivatives and check_total_derivatives."""

import unittest
from six import iteritems, StringIO, PY3
from six.moves import cStringIO as StringIO

import numpy as np

from openmdao.api import Group, ExplicitComponent, IndepVarComp, Problem
from openmdao.devtools.testutil import assert_rel_error


class TestProblemCheckPartials(unittest.TestCase):
    def test_incorrect_jacobian(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partial_derivs(self, inputs, outputs, partials):
                """Intentionally incorrect derivative."""
                J = partials
                J['y', 'x1'] = np.array([4.0])
                J['y', 'x2'] = np.array([40])

        prob = Problem()
        prob.model = Group()

        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run_model()

        string_stream = StringIO()

        data = prob.check_partial_derivatives(out_stream=string_stream)

        lines = string_stream.getvalue().split("\n")

        y_wrt_x1_line = lines.index("  comp: 'y' wrt 'x1'")

        self.assertTrue(lines[y_wrt_x1_line+6].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertTrue(lines[y_wrt_x1_line+7].endswith('*'),
                        msg='Error flag expected in output but not displayed')
        self.assertFalse(lines[y_wrt_x1_line+8].endswith('*'),
                        msg='Error flag not expected in output but displayed')

    def test_missing_entry(self):
        class MyComp(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('x1', 3.0)
                self.add_input('x2', 5.0)

                self.add_output('y', 5.5)

            def compute(self, inputs, outputs):
                """ Doesn't do much. """
                outputs['y'] = 3.0*inputs['x1'] + 4.0*inputs['x2']

            def compute_partial_derivs(self, inputs, outputs, partials):
                """Intentionally left out derivative."""
                J = partials
                J['y', 'x1'] = np.array([3.0])

        prob = Problem()
        prob.model = Group()
        prob.model.add_subsystem('p1', IndepVarComp('x1', 3.0))
        prob.model.add_subsystem('p2', IndepVarComp('x2', 5.0))
        prob.model.add_subsystem('comp', MyComp())

        prob.model.connect('p1.x1', 'comp.x1')
        prob.model.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run_model()

        stream = StringIO()

        data = prob.check_partial_derivatives(out_stream=stream)

        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x1']['J_fwd'] - 3.), 0.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x1']['J_rev'] - 3.), 0.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x1']['J_fd'] - 3.), 0.,
                               delta=1e-6)

        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x2']['J_fwd']), 0.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x2']['J_rev']), 0.)
        self.assertAlmostEqual(np.linalg.norm(data['comp']['y', 'x2']['J_fd'] - 4.), 0.,
                               delta=1e-6)

        self.fail('Finish implementing')

    def test_nested_fd_units(self):
        class UnitCompBase(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

            def initialize_partials(self):
                self.approx_partials(of='*', wrt='*')

            def compute(self, inputs, outputs):
                outputs['flow:T'] = inputs['T']
                outputs['flow:P'] = inputs['P']

        p = Problem()
        model = p.model = Group()
        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        units = model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        p.setup()
        data = p.check_partial_derivatives(out_stream=None)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                forward = partial['J_fwd']
                reverse = partial['J_rev']
                fd = partial['J_fd']
                self.assertAlmostEqual(np.linalg.norm(forward - reverse), 0.)
                self.assertAlmostEqual(np.linalg.norm(forward - fd), 0., delta=1e-6)

    def test_units(self):
        class UnitCompBase(ExplicitComponent):
            def initialize_variables(self):
                self.add_input('T', val=284., units="degR", desc="Temperature")
                self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")

                self.add_output('flow:T', val=284., units="degR", desc="Temperature")
                self.add_output('flow:P', val=1., units='lbf/inch**2', desc="Pressure")

            def compute_partial_derivs(self, inputs, outputs, partials):
                partials['flow:T', 'T'] = 1.
                partials['flow:P', 'P'] = 1.

            def compute(self, inputs, outputs):
                outputs['flow:T'] = inputs['T']
                outputs['flow:P'] = inputs['P']

        p = Problem()
        model = p.model = Group()
        indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])

        indep.add_output('T', val=100., units='degK')
        indep.add_output('P', val=1., units='bar')

        units = model.add_subsystem('units', UnitCompBase(), promotes=['*'])

        p.setup()
        data = p.check_partial_derivatives(out_stream=None)

        for comp_name, comp in iteritems(data):
            for partial_name, partial in iteritems(comp):
                forward = partial['J_fwd']
                reverse = partial['J_rev']
                fd = partial['J_fd']
                self.assertAlmostEqual(np.linalg.norm(forward - reverse), 0.)
                self.assertAlmostEqual(np.linalg.norm(forward - fd), 0., delta=1e-6)

if __name__ == "__main__":
    unittest.main()