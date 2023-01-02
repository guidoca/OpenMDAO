""" Unit tests for PygmoDriver."""

import unittest
import os

import numpy as np

import openmdao.api as om
import pygmo as pg
from openmdao.core.constants import INF_BOUND

from openmdao.test_suite.components.branin import Branin
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.components.paraboloid_distributed import DistParab
from openmdao.test_suite.components.sellar_feature import SellarMDA

from openmdao.utils.general_utils import run_driver
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.mpi import MPI
try:
    from parameterized import parameterized
except ImportError:
    from openmdao.utils.assert_utils import SkipParameterized as parameterized
 

extra_prints = True  # enable printing results
  
class TestDifferentialEvolution(unittest.TestCase):
 

    def test_basic_with_assert(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Branin(), promotes_inputs=[('x0', 'xI'), ('x1', 'xC')])

        model.add_design_var('xI', lower=-5.0, upper=10.0)
        model.add_design_var('xC', lower=0.0, upper=15.0)
        model.add_objective('comp.f')

        # prob.driver = om.PygmoDriver(uda=pg.pso(gen=100,seed=12345))
        prob.driver = om.PygmoDriver(uda=pg.de(gen=300,seed=12345),pop_size=100)
        # prob.driver = om.PygmoDriver(uda=pg.compass_search(max_fevals=1000,stop_range=0.00001) )

        prob.setup()
        prob.run_driver()

        # Optimal solution (actual optimum, not the optimal with integer inputs as found by SimpleGA)
        assert_near_equal(prob['comp.f'], 0.397887, 1e-4)

    def test_rastrigin(self):

        ORDER = 6  # dimension of problem
        span = 5   # upper and lower limits

        class RastriginComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.zeros(ORDER))
                self.add_output('y', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                # nth dimensional Rastrigin function, array input and scalar output
                # global minimum at f(0,0,0...) = 0
                n = len(x)
                s = 10 * n
                for i in range(n):
                    if np.abs(x[i]) < 1e-200:  # avoid underflow runtime warnings from squaring tiny numbers
                        x[i] = 0.0
                    s += x[i] * x[i] - 10 * np.cos(2 * np.pi * x[i])

                outputs['y'] = s

        prob = om.Problem()

        prob.model.add_subsystem('rastrigin', RastriginComp(), promotes_inputs=['x'])
        prob.model.add_design_var('x',
                                  lower=-span * np.ones(ORDER),
                                  upper=span * np.ones(ORDER))
        prob.model.add_objective('rastrigin.y')
        # prob.model.set_input_defaults('x', np.zeros(ORDER)+0.1)
        # prob.driver = om.PygmoDriver()
        prob.driver = om.PygmoDriver(uda=pg.de(gen=1500,seed=12345))
        # prob.driver = om.PygmoDriver(uda=pg.compass_search(max_fevals=1000,stop_range=0.00001) )
        # prob.driver.options['uda'].gen = 400
        # prob.driver.options['Pc'] = 0.5
        # prob.driver.options['F'] = 0.5

        prob.setup()
        prob.run_driver()

        assert_near_equal(prob['x'], np.zeros(ORDER), 5e-5)
        assert_near_equal(prob['rastrigin.y'], 0.0, 1e-5)

    def test_rosenbrock(self):
        ORDER = 6  # dimension of problem
        span = 2   # upper and lower limits

        class RosenbrockComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.zeros(ORDER))
                self.add_output('y', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                # nth dimensional Rosenbrock function, array input and scalar output
                # global minimum at f(1,1,1...) = 0
                n = len(x)
                assert (n > 1)
                s = 0
                for i in range(n - 1):
                    s += 100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2

                outputs['y'] = s

        prob = om.Problem()

        prob.model.add_subsystem('rosenbrock', RosenbrockComp(), promotes_inputs=['x'])
        prob.model.add_design_var('x',
                                  lower=-span * np.ones(ORDER),
                                  upper=span * np.ones(ORDER))
        prob.model.add_objective('rosenbrock.y')

        prob.driver = om.PygmoDriver(uda=pg.de(gen=500))
        # prob.driver.options['max_gen'] = 800

        prob.setup()
        prob.run_driver()

        # show results
        if extra_prints:
            print('rosenbrock.y', prob['rosenbrock.y'])
            print('x', prob['x'])
            print('objective function calls', prob.driver.iter_count, '\n')

        assert_near_equal(prob['rosenbrock.y'], 0.0, 1e-5)
        assert_near_equal(prob['x'], np.ones(ORDER), 1e-3)

    def test_simple_test_func(self):
        class MyComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', np.zeros((2, )))

                self.add_output('a', 0.0)
                self.add_output('b', 0.0)
                self.add_output('c', 0.0)
                self.add_output('d', 0.0)

            def compute(self, inputs, outputs):
                x = inputs['x']

                outputs['a'] = (2.0*x[0] - 3.0*x[1])**2
                outputs['b'] = 18.0 - 32.0*x[0] + 12.0*x[0]**2 + 48.0*x[1] - 36.0*x[0]*x[1] + 27.0*x[1]**2
                outputs['c'] = (x[0] + x[1] + 1.0)**2
                outputs['d'] = 19.0 - 14.0*x[0] + 3.0*x[0]**2 - 14.0*x[1] + 6.0*x[0]*x[1] + 3.0*x[1]**2

        prob = om.Problem()

        prob.model.add_subsystem('comp', MyComp(), promotes_inputs=['x'])
        prob.model.add_subsystem('obj', om.ExecComp('f=(30 + a*b)*(1 + c*d)'))

        prob.model.connect('comp.a', 'obj.a')
        prob.model.connect('comp.b', 'obj.b')
        prob.model.connect('comp.c', 'obj.c')
        prob.model.connect('comp.d', 'obj.d')

        # Played with bounds so we don't get subtractive cancellation of tiny numbers.
        prob.model.add_design_var('x', lower=np.array([0.2, -1.0]), upper=np.array([1.0, -0.2]))
        prob.model.add_objective('obj.f')

        prob.driver = om.PygmoDriver()
        # prob.driver.options['max_gen'] = 75

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('obj.f', prob['obj.f'])
            print('x', prob['x'])

        assert_near_equal(prob['obj.f'], 12.37306086, 1e-4)
        assert_near_equal(prob['x'][0], 0.2, 1e-4)
        assert_near_equal(prob['x'][1], -0.88653391, 1e-4)

    def test_analysis_error(self):
        class ValueErrorComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('x', 1.0)
                self.add_output('f', 1.0)

            def compute(self, inputs, outputs):
                raise ValueError

        prob = om.Problem()

        prob.model.add_subsystem('comp', ValueErrorComp(), promotes_inputs=['x'])
        prob.model.add_design_var('x', lower=-5.0, upper=10.0)
        prob.model.add_objective('comp.f')

        prob.driver = om.PygmoDriver( pop_size=25)

        prob.setup()
        # prob.run_driver()
        self.assertRaises(ValueError, prob.run_driver)

    def test_vector_desvars_multiobj(self):
        prob = om.Problem()

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 3)
        indeps.add_output('y', [4.0, -4])

        prob.model.add_subsystem('paraboloid1',
                                 om.ExecComp('f = (x+5)**2- 3'))
        prob.model.add_subsystem('paraboloid2',
                                 om.ExecComp('f = (y[0]-3)**2 + (y[1]-1)**2 - 3',
                                             y=[0, 0]))
        prob.model.connect('indeps.x', 'paraboloid1.x')
        prob.model.connect('indeps.y', 'paraboloid2.y')

        prob.driver = om.PygmoDriver(uda=pg.de(gen=300,ftol=1e-08)) # uda = pg.moead(gen=200)

        prob.model.add_design_var('indeps.x', lower=-5, upper=5)
        prob.model.add_design_var('indeps.y', lower=[-10, 0], upper=[10, 3])
        prob.model.add_objective('paraboloid1.f')
        prob.model.add_objective('paraboloid2.f')
        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('indeps.x', prob['indeps.x'])
            print('indeps.y', prob['indeps.y'])

        np.testing.assert_array_almost_equal(prob['indeps.x'], -5,decimal=4)
        np.testing.assert_array_almost_equal(prob['indeps.y'], [3, 1],decimal=4)

    def test_missing_objective(self):
        prob = om.Problem()
        model = prob.model

        model.add_subsystem('x', om.IndepVarComp('x', 2.0), promotes=['*'])
        model.add_subsystem('f_x', Paraboloid(), promotes=['*'])

        model.add_design_var('x', lower=-50, upper=50)

        prob.driver = om.PygmoDriver()

        prob.setup()

        with self.assertRaises(Exception) as raises_msg:
            prob.run_driver()

        exception = raises_msg.exception

        msg = "Driver requires objective to be declared"

        self.assertEqual(exception.args[0], msg)


    def test_vectorized_constraints(self):
        prob = om.Problem()

        dim = 2
        prob.model.add_subsystem('x', om.IndepVarComp('x', np.ones(dim)), promotes=['*'])
        prob.model.add_subsystem('f_x', om.ExecComp('f_x = sum(x * x)', x=np.ones(dim), f_x=1.0), promotes=['*'])
        prob.model.add_subsystem('g_x', om.ExecComp('g_x = 1 - x', x=np.ones(dim), g_x=np.zeros(dim)), promotes=['*'])

        prob.driver = om.PygmoDriver()

        prob.model.add_design_var('x', lower=-10, upper=10)
        prob.model.add_objective('f_x')
        prob.model.add_constraint('g_x', upper=np.zeros(dim))

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('x', prob['x'])

        # Check that the constraint is approximately satisfied (x >= 1)
        for i in range(dim):
            self.assertLessEqual(1.0 - 1e-6, prob["x"][i])



 
class TestMultiObjectiveDifferentialEvolution(unittest.TestCase):

    # def setUp(self):
    #     os.environ['PygmoDriver_seed'] = '11'

    # def tearDown(self):
    #     del os.environ['PygmoDriver_seed']  # clean up environment

    def test_multi_obj(self):
        class Box(om.ExplicitComponent):
            def setup(self):
                self.add_input('length', val=1.)
                self.add_input('width', val=1.)
                self.add_input('height', val=1.)

                self.add_output('front_area', val=1.0)
                self.add_output('top_area', val=1.0)
                self.add_output('area', val=1.0)
                self.add_output('volume', val=1.)

            def compute(self, inputs, outputs):
                length = inputs['length']
                width = inputs['width']
                height = inputs['height']

                outputs['top_area'] = length * width
                outputs['front_area'] = length * height
                outputs['area'] = 2*length*height + 2*length*width + 2*height*width
                outputs['volume'] = length*height*width

        prob = om.Problem()
        prob.model.add_subsystem('box', Box(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('length', 1.5)
        indeps.add_output('width', 1.5)
        indeps.add_output('height', 1.5)

        # setup the optimization
        prob.driver = om.PygmoDriver()
        # prob.driver.options['max_gen'] = 100
        prob.driver.options['multi_obj_exponent'] = 1.
        prob.driver.options['penalty_parameter'] = 10.
        prob.driver.options['multi_obj_weights'] = {'box.front_area': 0.1,
                                                    'box.top_area': 0.9}
        prob.driver.options['multi_obj_exponent'] = 1

        prob.model.add_design_var('length', lower=0.1, upper=2.)
        prob.model.add_design_var('width', lower=0.1, upper=2.)
        prob.model.add_design_var('height', lower=0.1, upper=2.)
        prob.model.add_objective('front_area', scaler=-1)  # maximize
        prob.model.add_objective('top_area', scaler=-1)  # maximize
        prob.model.add_constraint('volume', upper=1.)

        # run #1
        prob.setup()
        prob.run_driver()
        front = prob['front_area']
        top = prob['top_area']
        l1 = prob['length']
        w1 = prob['width']
        h1 = prob['height']

        if extra_prints:
            print('Box dims: ', l1, w1, h1)
            print('Front and top area: ', front, top)
            print('Volume: ', prob['volume'])  # should be around 1

        # run #2
        # weights changed
        prob2 = om.Problem()
        prob2.model.add_subsystem('box', Box(), promotes=['*'])

        indeps2 = prob2.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps2.add_output('length', 1.5)
        indeps2.add_output('width', 1.5)
        indeps2.add_output('height', 1.5)

        # setup the optimization
        prob2.driver = om.PygmoDriver()
        # prob2.driver.options['max_gen'] = 100
        prob2.driver.options['multi_obj_exponent'] = 1.
        prob2.driver.options['penalty_parameter'] = 10.
        prob2.driver.options['multi_obj_weights'] = {'box.front_area': 0.9,
                                                     'box.top_area': 0.1}
        prob2.driver.options['multi_obj_exponent'] = 1

        prob2.model.add_design_var('length', lower=0.1, upper=2.)
        prob2.model.add_design_var('width', lower=0.1, upper=2.)
        prob2.model.add_design_var('height', lower=0.1, upper=2.)
        prob2.model.add_objective('front_area', scaler=-1)  # maximize
        prob2.model.add_objective('top_area', scaler=-1)  # maximize
        prob2.model.add_constraint('volume', upper=1.)

        # run #1
        prob2.setup()
        prob2.run_driver()
        front2 = prob2['front_area']
        top2 = prob2['top_area']
        l2 = prob2['length']
        w2 = prob2['width']
        h2 = prob2['height']

        if extra_prints:
            print('Box dims: ', l2, w2, h2)
            print('Front and top area: ', front2, top2)
            print('Volume: ', prob['volume'])  # should be around 1

        self.assertGreater(w1, w2)  # front area does not depend on width
        self.assertGreater(h2, h1)  # top area does not depend on height

 
class TestConstrainedDifferentialEvolution(unittest.TestCase):


    def test_constrained_with_penalty(self):
        class Cylinder(om.ExplicitComponent):
            def setup(self):
                self.add_input('radius', val=1.0)
                self.add_input('height', val=1.0)

                self.add_output('Area', val=1.0)
                self.add_output('Volume', val=1.0)

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                height = inputs['height']

                area = height * radius * 2 * 3.14 + 3.14 * radius ** 2 * 2
                volume = 3.14 * radius ** 2 * height
                outputs['Area'] = area
                outputs['Volume'] = volume

        prob = om.Problem()
        prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = om.PygmoDriver()
        prob.driver.options['penalty_parameter'] = 3.
        prob.driver.options['penalty_exponent'] = 1.
        # prob.driver.options['max_gen'] = 50

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')
        prob.model.add_constraint('Volume', lower=10.)

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
            print('height', prob['height'])  # exact solution is 2*radius
            print('Area', prob['Area'])
            print('Volume', prob['Volume'])  # should be around 10

        # self.assertTrue(driver.supports["equality_constraints"], True)
        # self.assertTrue(driver.supports["inequality_constraints"], True)
        # check that it is not going to the unconstrained optimum
        self.assertGreater(prob['radius'], 1.)
        self.assertGreater(prob['height'], 1.)
 

    def test_constrained_without_penalty(self):
        class Cylinder(om.ExplicitComponent):
            def setup(self):
                self.add_input('radius', val=1.0)
                self.add_input('height', val=1.0)

                self.add_output('Area', val=1.0)
                self.add_output('Volume', val=1.0)

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                height = inputs['height']

                area = height * radius * 2 * 3.14 + 3.14 * radius ** 2 * 2
                volume = 3.14 * radius ** 2 * height
                outputs['Area'] = area
                outputs['Volume'] = volume

        prob = om.Problem()
        prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = om.PygmoDriver()
        prob.driver.options['penalty_parameter'] = 0.  # no penalty, same as unconstrained
        prob.driver.options['penalty_exponent'] = 1.
        # prob.driver.options['max_gen'] = 50

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')
        prob.model.add_constraint('Volume', lower=10.)

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
            print('height', prob['height'])  # exact solution is 2*radius
            print('Area', prob['Area'])
            print('Volume', prob['Volume'])  # should be around 10
 
        # it is going to the unconstrained optimum

        assert_near_equal(prob['radius'], 0.5, 1e-4)
        assert_near_equal(prob['height'], 0.5, 1e-4)
        # self.assertAlmostEqual(prob['radius'], 0.5)
        # self.assertAlmostEqual(prob['height'], 0.5)
        # self.assertAlmostEqual(prob['radius'], 0.5, 1)
        # self.assertAlmostEqual(prob['height'], 0.5, 1)

    def test_no_constraint(self):
        class Cylinder(om.ExplicitComponent):
            def setup(self):
                self.add_input('radius', val=1.0)
                self.add_input('height', val=1.0)

                self.add_output('Area', val=1.0)
                self.add_output('Volume', val=1.0)

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                height = inputs['height']

                area = height * radius * 2 * 3.14 + 3.14 * radius ** 2 * 2
                volume = 3.14 * radius ** 2 * height
                outputs['Area'] = area
                outputs['Volume'] = volume

        prob = om.Problem()
        prob.model.add_subsystem('cylinder', Cylinder(), promotes=['*'])

        indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
        indeps.add_output('radius', 2.)  # height
        indeps.add_output('height', 3.)  # radius

        # setup the optimization
        driver = prob.driver = om.PygmoDriver()
        prob.driver.options['penalty_parameter'] = 10.  # will have no effect
        prob.driver.options['penalty_exponent'] = 1.
        # prob.driver.options['max_gen'] = 50

        prob.model.add_design_var('radius', lower=0.5, upper=5.)
        prob.model.add_design_var('height', lower=0.5, upper=5.)
        prob.model.add_objective('Area')

        prob.setup()
        prob.run_driver()

        if extra_prints:
            print('radius', prob['radius'])  # exact solution is (5/pi)^(1/3) ~= 1.167
            print('height', prob['height'])  # exact solution is 2*radius
            print('Area', prob['Area'])
            print('Volume', prob['Volume'])  # should be around 10

        # self.assertTrue(driver.supports["equality_constraints"], True)
        # self.assertTrue(driver.supports["inequality_constraints"], True) 
        assert_near_equal(prob['radius'], 0.5, 1e-4)
        assert_near_equal(prob['height'], 0.5, 1e-4)

    # def test_multiple_constraints(self):

    #     p = om.Problem()

    #     exec = om.ExecComp(['y = x**2',
    #                         'z = a + x**2'],
    #                         a={'shape': (1,)},
    #                         y={'shape': (101,)},
    #                         x={'shape': (101,)},
    #                         z={'shape': (101,)})

    #     p.model.add_subsystem('exec', exec)

    #     p.model.add_design_var('exec.a', lower=-1000, upper=1000)
    #     p.model.add_objective('exec.y', index=50)
    #     p.model.add_constraint('exec.z', indices=[-1], lower=0)
    #     p.model.add_constraint('exec.z', indices=[0], upper=300, alias="ALIAS_TEST")

    #     # p.driver = om.PygmoDriver(uda=pg.de(gen=300,xtol=1e-20),pop_size=100)
    #     p.driver = om.PygmoDriver(uda=pg.pso(gen=300),pop_size=500)
    #     p.driver = om.PygmoDriver(uda=pg.gaco(gen=1000),pop_size=80)

    #     p.setup()

    #     p.set_val('exec.x', np.linspace(-10, 10, 101))

    #     p.run_driver()

    #     assert_near_equal(p.get_val('exec.z')[0], 100.0, tolerance=1e-6)
    #     assert_near_equal(p.get_val('exec.z')[-1], 100.0, tolerance=1e-6)

#     def test_same_cons_and_obj(self):

#         p = om.Problem()

#         exec = om.ExecComp(['y = x**2',
#                             'z = a + x**2'],
#                             a={'shape': (1,)},
#                             y={'shape': (101,)},
#                             x={'shape': (101,)},
#                             z={'shape': (101,)})

#         p.model.add_subsystem('exec', exec)

#         p.model.add_design_var('exec.a', lower=-1000, upper=1000)
#         p.model.add_objective('exec.z', index=50)
#         p.model.add_constraint('exec.z', indices=[0], upper=300, alias="ALIAS_TEST")

#         p.driver = om.PygmoDriver()

#         p.setup()

#         p.set_val('exec.x', np.linspace(-10, 10, 101))

#         p.run_driver()

#         assert_near_equal(p.get_val('exec.z')[0], -900)
#         assert_near_equal(p.get_val('exec.z')[50], -1000)

#     @parameterized.expand([
#         (None, -INF_BOUND, INF_BOUND),
#         (INF_BOUND, None, None),
#         (-INF_BOUND, None, None),
#     ],
#     name_func=_test_func_name)
#     def test_inf_constraints(self, equals, lower, upper):
#         # define paraboloid problem with constraint
#         prob = om.Problem()
#         model = prob.model

#         model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])
#         model.add_subsystem('const', om.ExecComp('g = x + y'), promotes_inputs=['x', 'y'])
#         model.set_input_defaults('x', 3.0)
#         model.set_input_defaults('y', -4.0)

#         # setup the optimization
#         prob.driver = om.PygmoDriver()
#         model.add_objective('parab.f_xy')
#         model.add_design_var('x', lower=-50, upper=50)
#         model.add_design_var('y', lower=-50, upper=50)
#         model.add_constraint('const.g', equals=equals, lower=lower, upper=upper)

#         prob.setup()

#         with self.assertRaises(ValueError) as err:
#             prob.final_setup()

#         # A value of None for lower and upper is changed to +/- INF_BOUND in add_constraint()
#         if lower == None:
#             lower = -INF_BOUND
#         if upper == None:
#             upper = INF_BOUND

#         msg = ("Invalid bounds for constraint 'const.g'. "
#                "When using PygmoDriver, the value for "
#                "'equals', 'lower' or 'upper' must be specified between "
#                f"+/-INF_BOUND ({INF_BOUND}), but they are: "
#                f"equals={equals}, lower={lower}, upper={upper}.")

#         self.maxDiff = None
#         self.assertEqual(err.exception.args[0], msg)




if __name__ == "__main__":
    unittest.main()
