"""
OpenMDAO Wrapper for the pygmo2 familiy of optimizers
"""
from numba import jit
import sys
from packaging.version import Version
import numpy as np
import pygmo as pg  
from openmdao.core.constants import INF_BOUND
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.class_util import WeakMethodWrapper
from openmdao.utils.mpi import MPI
from openmdao.utils.mpi import FakeComm
from openmdao.core.analysis_error import AnalysisError

# what version of pygmo are we working with
if pg and hasattr(pg, '__version__'):
    pygmo_version = Version(pg.__version__)
else:
    pygmo_version = None


# Optimizers in pygmo2
_optimizers = {'compass_search', 'nlopt' ,'scipy_optimize','ipopt','snopt7','wohrp'} 
_global_optimizers = {'gaco', 'maco', 'de','sade','de1220','gwo','ihs','pso' ,
               'pso_gen','sea','sga','simulated_annealing','bee_colony','cmaes',
               'xnes','nsga2','moead','nspso','mbh','cstrs_self_adaptive'} 

# For 'basinhopping' and 'shgo' gradients are used only in the local minimization
_gradient_optimizers = {'nlopt' ,'scipy_optimize','ipopt','snopt7','wohrp'} 
_hessian_optimizers = set()
_bounds_optimizers = _global_optimizers
_constraint_optimizers = {'gaco', 'ihs','compass_search', 'nlopt','scipy_optimize','ipopt','mbh','cstrs_self_adaptive' }
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers
_eq_constraint_optimizers = _constraint_optimizers 
_multiple_objectives = {'maco', 'ihs','nsga2','moead','nspso'}
# if Version(pygmo_version) >= Version("1.2"):  # Only available in newer versions
#     _global_optimizers |= {'shgo', 'dual_annealing'}

# Global optimizers and optimizers in minimize
_all_optimizers = _optimizers | _global_optimizers

CITATIONS = """
@article{Biscani2020,
  doi = {10.21105/joss.02338},
  url = {https://doi.org/10.21105/joss.02338},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {53},
  pages = {2338},
  author = {Francesco Biscani and Dario Izzo},
  title = {A parallel global multiobjective framework for optimization: pagmo},
  journal = {Journal of Open Source Software}
}
"""

class UserDefinedProblem: 
    def __init__(self,_fitfunc,lower,upper,nobj,nic,nec):
        self._fitfunc = _fitfunc
        self.lower = lower
        self.upper = upper
        self.nobj = nobj
        self.nic = nic
        self.nec = nec
        
    # @jit
    def fitness(self, x): 
        return self._fitfunc(x)
 
    def get_bounds(self):
        return (self.lower,self.upper)
    def get_nobj(self):
        return self.nobj
    def get_nic(self):
        return self.nic
    def get_nec(self):
        return self.nec
 
class PygmoDriver(Driver):
    """
    Driver wrapper for the scipy.optimize.minimize family of local optimizers.

    Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by SLSQP. None of the other
    optimizers support constraints.

    PygmoOptimizeDriver supports the following:
        equality_constraints
        inequality_constraints

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Driver options.

    Attributes
    ----------
    fail : bool
        Flag that indicates failure of most recent optimization.
    eval_count : int
        Counter for function evaluations.
    population :  
        population returned from pagmo2 call. 
    _check_jac : bool
        Used internally to control when to perform singular checks on computed total derivs.
    _con_cache : dict
        Cached population of constraint evaluations because scipy asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : {}
        Cached population of nonlinear constraint derivatives because scipy asks for them in a separate
        function.
    _exc_info : 3 item tuple
        Storage for exception and traceback information.
    _obj_and_nlcons : list
        List of objective + nonlinear constraints. Used to compute total derivatives
        for all except linear constraints.
    _dvlist : list
        Copy of _designvars.
    _lincongrad_cache : np.ndarray
        Pre-calculated gradients of linear constraints.
    """

    def __init__(self, **kwargs):
        """
        Initialize the ScipyOptimizeDriver.
        """
        super().__init__(**kwargs)

        # What we support
        self.supports['optimization'] = True
        self.supports['inequality_constraints'] = True
        self.supports['equality_constraints'] = True
        self.supports['two_sided_constraints'] = True
        self.supports['linear_constraints'] = True
        self.supports['simultaneous_derivatives'] = True
        self.supports['multiple_objectives'] = True
        self.supports['integer_design_vars'] = True

        # What we don't support
        self.supports['active_set'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here. 
        self.population = None
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.fail = False
        self.eval_count = 0
        self._check_jac = False
        self._exc_info = None
        self._total_jac_format = 'array'
        
        self._fill_NANs = False

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('uda', types=(object),default=pg.de(gen=300),
                             desc='Instance of algorithm to use') 
        # self.options.declare('seed', default=3453412, desc='Seed number') 
        self.options.declare('pop_size', default=0,
                             desc='Number of points in the GA. Set to 0 and it will be computed '
                             'as 20 times the total number of inputs.') 
        self.options.declare('penalty_parameter', default=10., lower=0.,
                             desc='Penalty function parameter.')
        self.options.declare('penalty_exponent', default=1.,
                             desc='Penalty function exponent.') 
        self.options.declare('multi_obj_weighted', default=True,
                             desc='Flag that defines type of multi-objecive optimization. If true, weighted optimization is performed.'
                             'if false, it depends on algorithm') 
        self.options.declare('multi_obj_weights', default={}, types=(dict),
                             desc='Weights of objectives for multi-objective optimization.'
                             'Weights are specified as a dictionary with the absolute names'
                             'of the objectives. The same weights for all objectives are assumed, '
                             'if not given.')
        self.options.declare('multi_obj_exponent', default=1., lower=0.,
                             desc='Multi-objective weighting exponent.')
        self.options.declare('verb', 1, types=int,
                             desc='Set to verbosity of Pygmo2 convergence messages') 

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "Pygmo2Optimize_" + self.options['uda'].__class__.__name__

    def _setup_driver(self, problem):
        """
        Prepare the driver for execution.

        This is the final thing to run during setup.

        Parameters
        ----------
        problem : <Problem>
            Pointer
        """
        super()._setup_driver(problem)
        uda = self.options['uda']
        name = uda.__class__.__name__

        self.supports._read_only = False
        self.supports['gradients'] = name in _gradient_optimizers
        self.supports['inequality_constraints'] = name in _constraint_optimizers
        self.supports['two_sided_constraints'] = name in _constraint_optimizers
        self.supports['equality_constraints'] = name in _eq_constraint_optimizers
        self.supports['multiple_objectives'] = name in _multiple_objectives
        self.supports._read_only = True
        # self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']
        self._desvar_idx = {}

        # Raises error if multiple objectives are not supported, but more objectives were defined.
        if not self.supports['multiple_objectives'] and len(self._objs) > 1 and not self.options['multi_obj_weighted']:
            msg = '{}  does not support multiple objectives.'
            raise RuntimeError(msg.format(self.msginfo))
 

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        return self.options['pop_size']

    def get_driver_derivative_calls(self):
        """
        Return number of derivative evaluations made during a driver run.

        Returns
        -------
        int
            Number of derivative evaluations made during a driver run.
        """ 
        return None

    def run(self):
        """
        Optimize the problem using selected Pygmo2 algorithm.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        uda = self.options['uda'] 
        alg = pg.algorithm(uda)
        alg.set_verbosity(self.options['verb'] )
        model = problem.model
        relevant = model._relevant
        self.pyopt_solution = None
        self._total_jac = None
        self.eval_count = 0 
        self._quantities = []

        self._check_for_missing_objective() 
  

        # comm = None if isinstance(problem.comm, FakeComm) else problem.comm
        func_dict = self.get_objective_values()
        cec_dict = self.get_constraint_values(ctype='eq') 
        cic_dict = self.get_constraint_values(ctype='ineq') # are bounds linear constraints? if so, exclude.  also order with equality constraints first?(Are there in openmdao? or all as inequality)
        n_obj = len(func_dict)
        if (n_obj > 1) and (self.options['multi_obj_weighted'] or self.supports['multi_objective']):
            is_single_objective = False
            if self.options['multi_obj_weighted']:
                n_obj = 1 
        else:
            is_single_objective = True
            n_obj = 1

        if self.supports['inequality_constraints']:
            n_ic  = len(cic_dict)
            n_ec  = len(cec_dict)
        else:
            n_ic = 0
            n_ec = 0
        # Size Problem
        ndesvar = 0
        for desvar in self._designvars.values():
            size = desvar['global_size'] if desvar['distributed'] else desvar['size']
            ndesvar += size 
        lower = np.empty(ndesvar) 
        upper = np.empty(ndesvar)
        # Bounds
        i = 0 
        for name,meta in self._designvars.items():
            size = meta['global_size'] if meta['distributed'] else meta['size'] 
            self._desvar_idx[name] = (i, i + size) 
            lower[i:i + size] = np.atleast_1d(meta['lower'])
            upper[i:i + size] = np.atleast_1d(meta['upper'])  
            i += size 
        # Automatic population size.
        pop_size=self.options['pop_size']
        if pop_size == 0:
            pop_size = 20 * i
        # Initialize UDP
        udp = UserDefinedProblem(WeakMethodWrapper(self, '_fitfunc'),lower=lower,upper=upper ,nobj=n_obj,nic=n_ic,nec=n_ec)
        
        # optimize
        try: 
            if self._problem().comm.rank != 0:
                self.opt_settings['verb'] = 0
                
            population = pg.population(udp,pop_size)
            population = alg.evolve(population)
                 

        # If an exception was swallowed in one of our callbacks, we want to raise it
        # rather than the cryptic message from scipy.
        except Exception as msg:
            self.fail = True
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        self.population = population
        # for desvar in self._designvars:
        #     i, j = self._desvar_idx[desvar]
        #     val = self.population.champion_x[i:j]
        #     self.set_design_var(desvar, val)
        #     # self.set_design_var(desvar, np.zeros(size))
        # model.run_solve_nonlinear()
        # Pull optimal parameters back into framework and re-run, so that
        # framework is left in the right final state, if it is not an unweighted multi-objective with more than objective function
        if not (not self.options['multi_obj_weighted'] and self.supports['multi_objective'] and len(self.get_objective_values())>1):
            for desvar in self._designvars:
                i, j = self._desvar_idx[desvar]
                val = self.population.champion_x[i:j]
                self.set_design_var(desvar, val)

            with RecordingDebugging(self._get_name(), self.eval_count, self) as rec:
                try:
                    model.run_solve_nonlinear()
                except AnalysisError:
                    model._clear_iprint()
                rec.abs = 0.0
                rec.rel = 0.0
            self.eval_count += 1 
        elif self.options['verb']>0:
            print('Problem is multi-objective with more than one objective, and therefore a champion is not well defined. \
                The user must explore the self.population property and re-evaluate the model accordingly.')
            print('-' * 35)
        if self.options['verb']>0:
            if self._problem().comm.rank == 0:
                print('Optimization Complete')
                print(alg.get_extra_info())
                print('-' * 35)

        return self.fail

    def _fitfunc(self, x_new):
        """
        Evaluate and return the fitness function.

        Model is executed here.
        
        This method will invoke the fitness() method of the UDP to compute the fitness of the input decision vector dv. 
        The return value of the fitness() method of the UDP is expected to have a dimension of  and to contain the
        concatenated values of  and  (in this order). Equality constraints are all assumed in the form  while
        inequalities are assumed in the form  so that negative values are associated to satisfied inequalities.

        In addition to invoking the fitness() method of the UDP, this method will perform sanity checks on dv and on the
        returned fitness vector. A successful call of this method will increase the internal fitness evaluation counter 
        (see get_fevals()).

        The fitness() method of the UDP must be able to take as input the decision vector as a 1D NumPy array, and it must
        return the fitness vector as an iterable Python object (e.g., 1D NumPy array, list, tuple, etc.).

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.

        Returns
        -------
        1D NumPy float array
            the fitness of dv at design point
        """
        model = self._problem().model
        
        
        func_dict = self.get_objective_values()
        cec_dict = self.get_constraint_values(ctype='eq') 
        cic_dict = self.get_constraint_values(ctype='ineq') # are bounds linear constraints? if so, exclude.  also order with equality constraints first?(Are there in openmdao? or all as inequality)
        n_obj = len(func_dict)
        # Single objective, if there is only one objective, which has only one element
        if (n_obj > 1) and (self.options['multi_obj_weighted'] or self.supports['multi_objective']):
            is_single_objective = False 
        else:
            is_single_objective = True 
 
        n_ic  = len(cic_dict) 
        n_ec  = len(cec_dict) 
        
        obj_exponent = self.options['multi_obj_exponent']
        if self.options['multi_obj_weights']:  # not empty
            obj_weights = self.options['multi_obj_weights']
        else:
            # Same weight for all objectives, if not specified
            obj_weights = {name: 1. for name in func_dict.keys()}
        sum_weights = sum(obj_weights.values())
         
        # a very large number, but smaller than the result of nan_to_num in Numpy
        almost_inf = INF_BOUND
        
        fail  = 0
        
        ## Execute model
        try:
            # Pass in new inputs
            i = 0
            if MPI:
                model.comm.Bcast(x_new, root=0)
            for name, meta in self._designvars.items():
                size = meta['size']
                self.set_design_var(name, x_new[i:i + size])
                # self.set_design_var(name, np.zeros(size))
                i += size

            with RecordingDebugging(self._get_name(), self.eval_count, self) as rec:
                self.eval_count += 1
                try:
                    # self._in_user_function = True
                    model.run_solve_nonlinear() 
                # Let the optimizer try to handle the error
                except AnalysisError:
                    model._clear_iprint()
            
            func_dict = self.get_objective_values()
            cec_dict = self.get_constraint_values(ctype='eq') 
            cic_dict = self.get_constraint_values(ctype='ineq')       
            # print(is_single_objective,func_dict)
            if is_single_objective:  # Single objective optimization
                for i in func_dict.values():
                    obj = np.atleast_1d(i)  # First and only key in the dict 
            elif self.options['multi_obj_weighted']:  # Multi-objective optimization with weighted sums
                weighted_objectives = np.array([])
                
                for name, val in func_dict.items():
                    # element-wise multiplication with scalar
                    # takes the average, if an objective is a vector
                    try:
                        weighted_obj = val * obj_weights[name] / val.size
                    except KeyError:
                        msg = ('Name "{}" in "multi_obj_weights" option '
                                'is not an absolute name of an objective.')
                        raise KeyError(msg.format(name))
                    weighted_objectives = np.hstack((weighted_objectives, weighted_obj))
                
                obj = np.atleast_1d(sum(weighted_objectives / sum_weights)**obj_exponent)
                
            else: 
                # fitness = np.empty(n_obj+n_ec+n_ic)
                obj = np.empty(n_obj)
                obj[0:n_obj] = np.fromiter(func_dict.values(), dtype=float) 
            # print(fitness)
            # print(fitness )
            # Parameters of the penalty method
            penalty  = self.options['penalty_parameter']
            exponent = self.options['penalty_exponent']
            if self.supports['inequality_constraints']:
                fitness = np.concatenate((obj,np.empty(n_ec+n_ic))) 
                
                fitness[n_obj:n_obj+n_ec] = np.fromiter(cec_dict.values(), dtype=float) 
                fitness[n_obj+n_ec:n_obj+n_ec+n_ic] = np.fromiter(cic_dict.values(), dtype=float) 

            elif penalty != 0 :

                constraint_violations = np.array([])
                for name, val in self.get_constraint_values().items():
                    con = self._cons[name]
                    # The not used fields will either None or a very large number
                    if (con['lower'] is not None) and np.any(con['lower'] > -almost_inf):
                        diff = val - con['lower']
                        violation = np.array([0. if d >= 0 else abs(d) for d in diff])
                    elif (con['upper'] is not None) and np.any(con['upper'] < almost_inf):
                        diff = val - con['upper']
                        violation = np.array([0. if d <= 0 else abs(d) for d in diff])
                    elif (con['equals'] is not None) and np.any(np.abs(con['equals']) < almost_inf):
                        diff = val - con['equals']
                        violation = np.absolute(diff)
                    constraint_violations = np.hstack((constraint_violations, violation))
                
                # fitness =fitness+ np.full(n_obj,penalty * sum(np.power(constraint_violations, exponent)))
                fitness =obj+ penalty * sum(np.power(constraint_violations, exponent))
                    

                # Record after getting obj and constraint to assure they have
                # been gathered in MPI.
                rec.abs = 0.0
                rec.rel = 0.0
            else:
                fitness = obj

            

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = sys.exc_info() 
            return 0
        
        return fitness

    def _reraise(self):
        """
        Reraise any exception encountered when pygmo calls back into our method.
        """
        raise self._exc_info[1].with_traceback(self._exc_info[2])



