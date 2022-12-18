from openmdao.drivers.optimizers import golden_section

"""
OpenMDAO Wrapper for the 1 dimensional optimizers/line search algorithms. Only functional for problems with one optimization variable
"""

import sys 

import numpy as np 

from openmdao.core.constants import INF_BOUND
from openmdao.core.driver import Driver, RecordingDebugging
from openmdao.utils.class_util import WeakMethodWrapper
from openmdao.utils.mpi import MPI

# Optimizers  
_optimizers = {'golden_section'} 

# For 'basinhopping' and 'shgo' gradients are used only in the local minimization
_gradient_optimizers = set()
_hessian_optimizers = set()
_bounds_optimizers = {'golden_section'}
_constraint_optimizers = set()
_constraint_grad_optimizers = _gradient_optimizers & _constraint_optimizers
_eq_constraint_optimizers = set()
_global_optimizers = set()

# Global optimizers and optimizers in minimize
_all_optimizers = _optimizers | _global_optimizers
 
 

CITATIONS = """
@article{Hwang_maud_2018
 author = {Hwang, John T. and Martins, Joaquim R.R.A.},
 title = "{A Computational Architecture for Coupling Heterogeneous
          Numerical Models and Computing Coupled Derivatives}",
 journal = "{ACM Trans. Math. Softw.}",
 volume = {44},
 number = {4},
 month = jun,
 year = {2018},
 pages = {37:1--37:39},
 articleno = {37},
 numpages = {39},
 doi = {10.1145/3182393},
 publisher = {ACM},
"""


class LineSearchDriver(Driver):
    """
    Driver wrapper for the scipy.optimize.minimize family of local optimizers.

    Inequality constraints are supported by COBYLA and SLSQP,
    but equality constraints are only supported by SLSQP. None of the other
    optimizers support constraints.

    ScipyOptimizeDriver supports the following:
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
    iter_count : int
        Counter for function evaluations.
    result : OptimizeResult
        Result returned from scipy.optimize call.
    opt_settings : dict
        Dictionary of solver-specific options. See the scipy.optimize.minimize documentation.
    _check_jac : bool
        Used internally to control when to perform singular checks on computed total derivs.
    _con_cache : dict
        Cached result of constraint evaluations because scipy asks for them in a separate function.
    _con_idx : dict
        Used for constraint bookkeeping in the presence of 2-sided constraints.
    _grad_cache : {}
        Cached result of nonlinear constraint derivatives because scipy asks for them in a separate
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

        # What we don't support
        self.supports['multiple_objectives'] = False
        self.supports['active_set'] = False
        self.supports['integer_design_vars'] = False
        self.supports['distributed_design_vars'] = False
        self.supports._read_only = True

        # The user places optimizer-specific settings in here.
        self.opt_settings = {}

        self.result = None
        self._grad_cache = None
        self._con_cache = None
        self._con_idx = {}
        self._obj_and_nlcons = None
        self._dvlist = None
        self._lincongrad_cache = None
        self.fail = False
        self.iter_count = 0
        self._check_jac = False
        self._exc_info = None
        self._total_jac_format = 'array'

        self.cite = CITATIONS

    def _declare_options(self):
        """
        Declare options before kwargs are processed in the init method.
        """
        self.options.declare('optimizer', 'golden_section', values=_all_optimizers,
                             desc='Name of optimizer to use')
        self.options.declare('tol', 1.0e-6, lower=0.0,
                             desc='Tolerance for termination. For detailed '
                             'control, use solver-specific options.')
        # self.options.declare('maxiter', 200, lower=0,
        #                      desc='Maximum number of iterations.') 

    def _get_name(self):
        """
        Get name of current optimizer.

        Returns
        -------
        str
            The name of the current optimizer.
        """
        return "LineSearchDriver_" + self.options['optimizer']

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
        opt = self.options['optimizer']

        self.supports._read_only = False
        self.supports['gradients'] = opt in _gradient_optimizers
        self.supports['inequality_constraints'] = opt in _constraint_optimizers
        self.supports['two_sided_constraints'] = opt in _constraint_optimizers
        self.supports['equality_constraints'] = opt in _eq_constraint_optimizers
        self.supports._read_only = True
        # self._check_jac = self.options['singular_jac_behavior'] in ['error', 'warn']

        # Raises error if multiple objectives are not supported, but more objectives were defined.
        if not self.supports['multiple_objectives'] and len(self._objs) > 1:
            msg = '{} currently does not support multiple objectives.'
            raise RuntimeError(msg.format(self.msginfo))
 

    def get_driver_objective_calls(self):
        """
        Return number of objective evaluations made during a driver run.

        Returns
        -------
        int
            Number of objective evaluations made during a driver run.
        """
        return self.debug['n_eval']

 

    def run(self):
        """
        Optimize the problem using selected Scipy optimizer.

        Returns
        -------
        bool
            Failure flag; True if failed to converge, False is successful.
        """
        problem = self._problem()
        opt = self.options['optimizer']
        model = problem.model
        self.iter_count = 0
        self._total_jac = None

        self._check_for_missing_objective()

        # Initial Run
        with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
            model.run_solve_nonlinear()
            self.iter_count += 1

        self._con_cache = self.get_constraint_values()
        desvar_vals = self.get_design_var_values()
        self._dvlist = list(self._designvars)


 
        # Initial Design Vars 
        use_bounds = (opt in _bounds_optimizers)
        if use_bounds:
            bounds = []
        else:
            bounds = None

        for name, meta in self._designvars.items(): 
            x_init  = desvar_vals[name] 

            # Bounds if our optimizer supports them
            if use_bounds:
                meta_low = meta['lower']
                meta_high = meta['upper']
                for j in range(1):

                    if isinstance(meta_low, np.ndarray):
                        p_low = meta_low[j]
                    else:
                        p_low = meta_low

                    if isinstance(meta_high, np.ndarray):
                        p_high = meta_high[j]
                    else:
                        p_high = meta_high

                    bounds.append(p_low)
                    bounds.append(p_high)

 
        # Constraints
        constraints = []
        i = 1  # start at 1 since row 0 is the objective.  Constraints start at row 1.
        lin_i = 0  # counter for linear constraint jacobian
        lincons = []  # list of linear constraints
        self._obj_and_nlcons = list(self._objs)

        if opt in _constraint_optimizers:
            for name, meta in self._cons.items():
                if meta['indices'] is not None:
                    meta['size'] = size = meta['indices'].indexed_src_size
                else:
                    size = meta['global_size'] if meta['distributed'] else meta['size']
                upper = meta['upper']
                lower = meta['lower']
                equals = meta['equals']
                if opt in _gradient_optimizers and 'linear' in meta and meta['linear']:
                    lincons.append(name)
                    self._con_idx[name] = lin_i
                    lin_i += size
                else:
                    self._obj_and_nlcons.append(name)
                    self._con_idx[name] = i
                    i += size

                # In scipy constraint optimizers take constraints in two separate formats

                # Type of constraints is list of NonlinearConstraint 
                # Loop over every index separately,
                # because scipy calls each constraint by index.
                for j in range(size):
                    con_dict = {}
                    if meta['equals'] is not None:
                        con_dict['type'] = 'eq'
                    else:
                        con_dict['type'] = 'ineq'
                    con_dict['fun'] = WeakMethodWrapper(self, '_confunc')
                    if opt in _constraint_grad_optimizers:
                        con_dict['jac'] = WeakMethodWrapper(self, '_congradfunc')
                    con_dict['args'] = [name, False, j]
                    constraints.append(con_dict)

                    if isinstance(upper, np.ndarray):
                        upper = upper[j]

                    if isinstance(lower, np.ndarray):
                        lower = lower[j]

                    dblcon = (upper < INF_BOUND) and (lower > -INF_BOUND)

                    # Add extra constraint if double-sided
                    if dblcon:
                        dcon_dict = {}
                        dcon_dict['type'] = 'ineq'
                        dcon_dict['fun'] = WeakMethodWrapper(self, '_confunc')
                        if opt in _constraint_grad_optimizers:
                            dcon_dict['jac'] = WeakMethodWrapper(self, '_congradfunc')
                        dcon_dict['args'] = [name, True, j]
                        constraints.append(dcon_dict)

            # precalculate gradients of linear constraints
            if lincons:
                self._lincongrad_cache = self._compute_totals(of=lincons, wrt=self._dvlist,
                                                              return_format=self._total_jac_format)
            else:
                self._lincongrad_cache = None

        # Provide gradients for optimizers that support it
        if opt in _gradient_optimizers:
            jac = self._gradfunc
        else:
            jac = None

        # Hessian calculation method for optimizers, which require it
        if opt in _hessian_optimizers:
            if 'hess' in self.opt_settings:
                hess = self.opt_settings.pop('hess')
            else:
                # Defaults to BFGS, if not in opt_settings
                from scipy.optimize import BFGS
                hess = BFGS()
        else:
            hess = None

        # compute dynamic simul deriv coloring if option is set
        coloring = self._get_coloring(run_model=False)

        # optimize
        try:
            if opt== 'golden_section':
                if self._problem().comm.rank != 0:
                    self.opt_settings['disp'] = False

                x,f,debug= golden_section(self._objfunc, bounds[0], bounds[1] , 
                                  xtol=self.options['tol'])
                self.fail = False
            else:
                msg = 'Optimizer "{}" is not implemented yet. Choose from: {}'
                raise NotImplementedError(msg.format(opt, _all_optimizers))

        # If an exception was swallowed in one of our callbacks, we want to raise it
        # rather than the cryptic message from scipy.
        except Exception as msg:
            if self._exc_info is None:
                raise

        if self._exc_info is not None:
            self._reraise()

        self.debug = debug
 
        return self.fail

    def _objfunc(self, x_new):
        """
        Evaluate and return the objective function.

        Model is executed here.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.

        Returns
        -------
        float
            Value of the objective function evaluated at the new design point.
        """
        model = self._problem().model

        try:

            # Pass in new inputs 
            if MPI:
                model.comm.Bcast(x_new, root=0)
            for name, meta in self._designvars.items(): 
                self.set_design_var(name, x_new ) 

            with RecordingDebugging(self._get_name(), self.iter_count, self) as rec:
                self.iter_count += 1
                model.run_solve_nonlinear()

            # Get the objective function evaluations
            for obj in self.get_objective_values().values():
                f_new = obj
                break

            self._con_cache = self.get_constraint_values()

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return 0

        # print("Functions calculated")
        # print('   xnew', x_new)
        # print('   fnew', f_new)

        return f_new

    def _con_val_func(self, x_new, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        The lower or upper bound is **not** subtracted from the value. Used for optimizers,
        which take the bounds of the constraints (e.g. trust-constr)

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.
        name : str
            Name of the constraint to be evaluated.
        dbl : bool
            True if double sided constraint.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        return self._con_cache[name][idx]

    def _confunc(self, x_new, name, dbl, idx):
        """
        Return the value of the constraint function requested in args.

        Note that this function is called for each constraint, so the model is only run when the
        objective is evaluated.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.
        name : str
            Name of the constraint to be evaluated.
        dbl : bool
            True if double sided constraint.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Value of the constraint function.
        """
        if self._exc_info is not None:
            self._reraise()

        cons = self._con_cache
        meta = self._cons[name]

        # Equality constraints
        equals = meta['equals']
        if equals is not None:
            if isinstance(equals, np.ndarray):
                equals = equals[idx]
            return cons[name][idx] - equals

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        upper = meta['upper']
        if isinstance(upper, np.ndarray):
            upper = upper[idx]

        lower = meta['lower']
        if isinstance(lower, np.ndarray):
            lower = lower[idx]

        if dbl or (lower <= -INF_BOUND):
            return upper - cons[name][idx]
        else:
            return cons[name][idx] - lower

    def _gradfunc(self, x_new):
        """
        Evaluate and return the gradient for the objective.

        Gradients for the constraints are also calculated and cached here.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.

        Returns
        -------
        ndarray
            Gradient of objective with respect to input array.
        """
        try:
            grad = self._compute_totals(of=self._obj_and_nlcons, wrt=self._dvlist,
                                        return_format=self._total_jac_format)
            self._grad_cache = grad

            # First time through, check for zero row/col.
            if self._check_jac:
                raise_error = self.options['singular_jac_behavior'] == 'error'
                self._total_jac.check_total_jac(raise_error=raise_error,
                                                tol=self.options['singular_jac_tol'])
                self._check_jac = False

        except Exception as msg:
            self._exc_info = sys.exc_info()
            return np.array([[]])

        # print("Gradients calculated for objective")
        # print('   xnew', x_new)
        # print('   grad', grad[0, :])

        return grad[0, :]

    def _congradfunc(self, x_new, name, dbl, idx):
        """
        Return the cached gradient of the constraint function.

        Note, scipy calls the constraints one at a time, so the gradient is cached when the
        objective gradient is called.

        Parameters
        ----------
        x_new : ndarray
            Array containing input values at new design point.
        name : str
            Name of the constraint to be evaluated.
        dbl : bool
            Denotes if a constraint is double-sided or not.
        idx : float
            Contains index into the constraint array.

        Returns
        -------
        float
            Gradient of the constraint function wrt all inputs.
        """
        if self._exc_info is not None:
            self._reraise()

        meta = self._cons[name]

        if meta['linear']:
            grad = self._lincongrad_cache
        else:
            grad = self._grad_cache
        grad_idx = self._con_idx[name] + idx

        # print("Constraint Gradient returned")
        # print('   xnew', x_new)
        # print('   grad', name, 'idx', idx, grad[grad_idx, :])

        # Equality constraints
        if meta['equals'] is not None:
            return grad[grad_idx, :]

        # Note, scipy defines constraints to be satisfied when positive,
        # which is the opposite of OpenMDAO.
        lower = meta['lower']
        if isinstance(lower, np.ndarray):
            lower = lower[idx]

        if dbl or (lower <= -INF_BOUND):
            return -grad[grad_idx, :]
        else:
            return grad[grad_idx, :]

    def _reraise(self):
        """
        Reraise any exception encountered when scipy calls back into our method.
        """
        raise self._exc_info[1].with_traceback(self._exc_info[2])



