'''
Created on 16.01.2018

@author: Samuel
'''
import warnings
import traceback
import sys
from functools import partial

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as op
from scipy.stats import chi2
from scipy.optimize._trustregion_exact import IterativeSubproblem



def round_str(n, full=6, after=3):
    if np.isfinite(n) and n != 0:
        digipre = int(np.log10(np.abs(n)))+1
        after = min(max(full-digipre-1, 0), after)
    return ("{:"+str(full)+"."+str(after)+"f}").format(n)

def is_negative_definite(M, tol=1e-5):
    try:
        r = np.linalg.cholesky(-M)
        if np.isclose(np.linalg.det(r), 0, atol=tol):
            return False
        return True
    except np.linalg.LinAlgError:
        return False

class FlexibleSubproblem():
    """
    Class representing constained quadratic subproblems
    """
    def __init__(self, x, fun, jac, hess, hessp=None,
                 k_easy=0.1, k_hard=0.2):
        """
        ``k_easy`` and ``k_hard`` are parameters used
        to determine the stop criteria to the iterative
        subproblem solver. Take a look at the IterativeSubproblem class
        for more info.
        """
        self.x = x
        self.fun = fun
        self.jac = jac
        self.hess = hess
        self.hessp = hessp
        self.k_easy = k_easy
        self.k_hard = k_hard
        
    def solve(self, radius, positiveDefinite=None, tol=1e-5, jac0tol=0, *args, **kwargs):
        if radius == 0:
            return np.zeros_like(self.jac(self.x)), True
        
        subproblem = IterativeSubproblem(self.x, self.fun, self.jac, self.hess, 
                                         self.hessp, self.k_easy, self.k_hard)
        
        # IterativeSubproblem runs into issues, if jac is too small
        if subproblem.jac_mag <= max(subproblem.CLOSE_TO_ZERO, jac0tol):
            j = self.jac(self.x)
            h = self.hess(self.x)
            if positiveDefinite is None:
                positiveDefinite = is_negative_definite(-h, tol)
            if positiveDefinite:
                x = np.zeros(j.size)
            else:
                vals, vecs = np.linalg.eigh(h)
                x = vecs[:,np.argmin(np.real(vals))]
                x *= radius
            jNorm = np.linalg.norm(j)
            if jNorm:
                x_alt = j / jNorm * radius
                if (np.linalg.multi_dot((x, h, x)) + np.dot(j, x) 
                    > np.linalg.multi_dot((x_alt, h, x_alt)) + np.dot(j, x_alt)):
                    return x_alt, True
            return x, True
        return subproblem.solve(radius, *args, **kwargs)

class Flipper:
    def __init__(
            self, 
            index: "index in which the result should be flipped",
            ):
        self.index = index
    def __call__(self, x):
        if not hasattr(x, "__iter__"):
            return x
        index = self.index
        x = x.copy()
        x[index] *= -1
        if x.ndim > 1:
            x[:, index] *= -1
        return x

def get_independent_row_indices(M: "considered matrix", 
                                jac: "ordering vector" = None, 
                                tol: "numerical tolerance" = None
                                ) -> "boolean array of linearly independent rows":
    """
    Returns a boolean array arr with arr[i]==True if and only if
    the i-th row of M is linearly independent of all other rows j with 
    arr[j]==True.
    The vector jac provides an ordering of the returned indices. 
    If M[i] and M[j] are linearly dependent, then arr[i] will be True if
    jac[i] >= jac[j]. Otherwise, arr[i] will be False and arr[j] will be True.
    """
    n = M.shape[0]
    result = np.ones(n, dtype=bool)
    
    if jac is not None:
        indices = np.argsort(np.abs(jac))[::-1]
    else:
        indices = np.arange(n, dtype=int)
    
    rank = 0
    for i in range(n):
        if np.linalg.matrix_rank(M[indices[:i+1]], tol=tol) > rank:
            rank += 1
        else:
            result[indices[i]] = False
    
    if not result.any():
        result[indices[0]] = True
    return result

class CounterFun:
    """
    Counts how often a function has been called
    """
    def __init__(self, fun):
        self.fun = fun
        self.evals = 0
    def __call__(self, *args, **kwargs):
        self.evals += 1
        return self.fun(*args, **kwargs)


def create_profile_plots(profile_result, index, labels=None, file_name=None, 
                         show=True):
    considered = np.ones(len(profile_result["params"][0]), dtype=bool)
    considered[index] = False
    
    parms_t = profile_result["params"].T
    interestParm = parms_t[index]
    relativeChange = parms_t[considered]
    
    relativeChange /= np.maximum(np.max(np.abs(relativeChange),1), 
                                 1e-20)[:,None] #relativeChange[:,len(interestParm)//2][:,None]
    #relativeChange -= 1
    
    plt.figure()
    if labels is None:
        xLabel = "Parameter " + str(index)
    else: 
        xLabel = labels[index]
    
    labels = list(labels)
    del labels[index]
    
    plt.plot(interestParm, profile_result["logL"])
    plt.xlabel(xLabel)
    plt.ylabel("Log-Likelihood")
    
    if file_name is not None:
        file_name += "_" + xLabel.replace(" ", "")
        plt.savefig(file_name + "_logL.pdf")
        plt.savefig(file_name + "_logL.png", dpi=1000)
    
    plt.figure()
    
    for relChange, label in zip(relativeChange, labels):
        plt.plot(interestParm, relChange, label=label)
    plt.xlabel(xLabel)
    plt.ylabel("Relative parameter change")
    plt.legend()

    if file_name is not None:
        plt.savefig(file_name + "_params.pdf")
        plt.savefig(file_name + "_params.png", dpi=1000)
    
    plt.figure()
    plt.xlabel(xLabel)
    plt.ylabel("Log-Likelihood")
    plt.plot(np.concatenate(profile_result["x_track"]), 
             np.concatenate(profile_result["f_track"]))
    
    if file_name is not None:
        plt.savefig(file_name + "_search.pdf")
        plt.savefig(file_name + "_search.png", dpi=1000)
    
    if show:
        plt.show()
        
def find_profile_CI_bound(index, direction, x0, fun, jac, hess, alpha=0.95, 
                          fun0=None, *args, vm=False, **kwargs):
    diff = chi2.ppf(alpha, df=1)/2
    
    if fun0 is None:
        fun0 = fun(x0)
    
    target = fun0-diff
    
    if vm:
        return venzon_moolgavkar(index, target, x0, fun, jac, hess, 
                                 direction, fun0, *args, **kwargs)
    else:
        return find_CI_bound(index, target, x0, fun, jac, hess, direction==1, 
                             fun0, None, *args, track_x=True, track_f=True, 
                             **kwargs)


def find_profile_CI_bound_steps(index, x0, fun, jac, hess, direction=1, alpha=0.95, 
                          stepn=1, fun0=None, hess0=None, nmax=200, 
                          epsilon=1e-4, disp=True, vm=False):
    dim = len(x0)
    diff = chi2.ppf(alpha, df=1)/2
    
    if fun0 is None:
        fun0 = fun(x0)
    
    steps = np.linspace(fun0, fun0-diff, stepn+1)
    print("diff", diff)
    print("steps", steps) 
    
    dtype = [("params", str(dim)+'double'),
             ('logL', 'double'),
             ("x_track", object),
             ("f_track", object),
             ]
    result = np.zeros(stepn + 1, dtype=dtype)
    
    result['logL'][0] = steps[0]
    result["params"][0] = x0
    result["x_track"][0] = [x0]
    result["f_track"][0] = [fun0]
    
    for i, target in enumerate(steps[1:]):
        if vm:
            op_result = venzon_moolgavkar(index, target, x0, fun, jac, hess, 
                                          direction, fun0, hess0, nmax, 
                                          epsilon, disp)
        else:
            op_result = find_CI_bound(index, target, x0, fun, jac, hess, direction==1, 
                                      fun0, None, hess0, nmax, disp=disp, 
                                      track_x=True, track_f=True)
            x0 = op_result.x
        if not op_result.success:
            if op_result.status == 3:
                warnings.warn("Iteration limit exceeded for variable "
                              + str(index) + " and logL="
                              + str(target) + " in direction " + str(direction))
            elif op_result.status >= 4:
                warnings.warn("Optimization failed for variable "
                              + str(index) + " and logL="
                              + str(target) + " in direction " + str(direction))
                #if not np.isnan(x0).any():
                #    x0 = op_result.x
                op_result.x += np.nan
                op_result.fun += np.nan
        result["x_track"][i+1] = np.array(op_result.x_track[::direction])
        result["f_track"][i+1] = op_result.f_track[::direction]
        result["logL"][i+1] = op_result.fun
        result["params"][i+1] = op_result.x
        fun0 = op_result.fun
        hess0 = None
        
        if disp:
            try:
                print("*** logL={}; index={}, direction={}, fdiff={}; jdiff={}, nit={}".format(
                        fun0, index, direction, op_result.fun-target, np.max(np.abs(op_result.jac)), op_result.nit))
            except Exception:
                print("*** finished, but can't print due to exception.", op_result.jac)
    
    if direction < 0:
        result = result[::-1]
            
    return result
    


def venzon_moolgavkar(index, target, x0, fun, jac, hess, step_scale=1, 
                      fun0=None, hess0=None, nmax=200, epsilon=1e-6, disp=True,
                      track_x=False):
    
    x_track = []
    
    # TODO: handle singular matrices (check determinant, return NaN), extend algorithm to be more robust
    i = 0
    try:    
        # preparation
        
        if fun0 is None:
            fun0 = fun(x0)
        target_diff = fun0-target
        if hess0 is None:
            dl2_dx2_0 = hess(x0)
        else:
            dl2_dx2_0 = hess0
        
        considered = np.ones_like(x0, dtype=bool)
        considered[index] = False
        hessConsidered = np.ix_(considered, considered)
        
        # test
        if False:
            f = lambda x: (np.square(fun(x)-target) 
                           + np.sum(np.square(jac(x)[considered])))
            print(op.minimize(f, x0))
                           
        # x is the parameter vector
        # w is the vector of nuisance parameters (x without row "index")
        # b is the parameter of interest, which is x[index]
        # l is fun
        # dl_d* is the derivative of l w.r.t. *
        # dl2_d*2 is the second derivative of l w.r.t. *
        
        # choosing the first step x1
        
        dl2_dw2 = dl2_dx2_0[hessConsidered]
        
        dl2_dbdw = dl2_dx2_0[index][considered]
        dl2_dw2_inv = np.linalg.inv(dl2_dw2)
        dw_db = -np.dot(dl2_dw2_inv, dl2_dbdw)
        
        factor = (np.sqrt(target_diff / (-2*(dl2_dx2_0[index, index]
                            - np.dot(np.dot(dl2_dbdw, dl2_dw2_inv), dl2_dbdw))))
                                               * step_scale)
        
        if np.isnan(factor):
            factor = 0.5
            #raise np.linalg.LinAlgError()
        
        init_direction = np.zeros_like(x0)
        init_direction[considered] = dw_db
        init_direction[index] = 1
        x = x0 + factor*init_direction
        
        
        # iteration
        for i in range(nmax):
            if track_x:
                x_track.append(x.copy())
            l = fun(x)
            dl_dx_ = jac(x)
            violation = max(np.max(np.abs(dl_dx_[considered])), np.abs(l-target))
            
            if disp:
                #print(x)
                print("iteration {}: fun_diff={}; jac_violation={}; direction={}".format(
                        i, l-target, np.max(np.abs(dl_dx_[considered])), step_scale))
            
            if violation < epsilon:
                break
            elif np.isnan(violation):
                raise np.linalg.LinAlgError()
            
            D = dl2_dx2 = hess(x)
            dl2_dx2_ = dl2_dx2.copy()
            dl2_dx2_[index] = dl_dx_
            dl_dx_[index] = l-target 
            
            dl2_dx2__inv = np.linalg.inv(dl2_dx2_)
            g = dl2_dx2__inv[:,index]
            v = np.dot(dl2_dx2__inv, dl_dx_)
            
            Dg = np.dot(D, g)
            vDg = np.dot(v, Dg)
            gDg = np.dot(g, Dg)
            vDv = np.dot(np.dot(v, D), v)
            
            p = 2 * (vDg-1) / gDg
            q = vDv / gDg
            
            _s = p*p/4 - q
            
            if _s >= 0:
                _s = np.sqrt(_s)
                s_ = - p/2
                s_1 = s_ + _s
                s_2 = s_ - _s
                
                v_sg_1 = v + s_1*g
                v_sg_2 = v + s_2*g
                measure1 = np.abs(np.dot(np.dot(v_sg_1, dl2_dx2_0), v_sg_1))
                measure2 = np.abs(np.dot(np.dot(v_sg_2, dl2_dx2_0), v_sg_2))
                
                if measure1 < measure2:
                    x -= v_sg_1
                else: 
                    x -= v_sg_2
            else:
                # TODO: do line search here! 
                x -= 0.1 * v
        else:
            l = fun(x)
            dl_dx_ = jac(x)
            violation = max(np.max(np.abs(dl_dx_[considered])), np.abs(l-target))
            return op.OptimizeResult(x=x, 
                                     fun=l,
                                     jac=jac,
                                     violation=violation,
                                     success=False, status=1,
                                     nfev=nmax+1, njev=nmax+1, nhev=nmax, 
                                     nit=nmax,
                                     x_track=np.array(x_track),
                                     message="Iteration limit exceeded"
                                     )
        
        return op.OptimizeResult(x=x, 
                                 fun=l,
                                 jac=jac(x),
                                 violation=violation,
                                 success=True, status=0,
                                 nfev=i, njev=i, nhev=i, 
                                 nit=i,
                                 x_track=np.array(x_track),
                                 message="Success"
                                 )
    except np.linalg.LinAlgError:
        return op.OptimizeResult(x=x0+np.nan, 
                                 fun=np.nan,
                                 jac=x0+np.nan,
                                 violation=np.nan,
                                 success=False, status=2,
                                 nfev=i, njev=i, nhev=i, 
                                 nit=i,
                                 x_track=np.array(x_track),
                                 message="Ill-conditioned matrix"
                                 )

STATUS_MESSAGES = {0:"Success",
                   1:"Result on discontinuity",
                   2:"Result is unbounded",
                   3:"Iteration limit exceeded",
                   4:"Ill-conditioned matrix",
                   5:"Unspecified error"
                   }

def find_CI_bound(
        index: "index of the parameter to consider", 
        target: "target log-likelihood l*", 
        x0: "maximum likelihood estimator (MLE)", 
        fun: "log-likelihood function", 
        jac: "gradient of fun", 
        hess: "hessian of fun", 
        forward: "True, if right end point of CI is sought, else False" = True, 
        fun0: "log-likelihood at MLE" = None, 
        jac0: "gradient of log-liekelihood at MLE" = None, 
        hess0: "Hessian of log-likelihood at MLE" = None, 
        nmax: "maximal number of iterations" = 200, 
        nchecks: "maximal number of trust-region changes per iteration" = 65,
        apprxtol: "relative tolerance between f and its approximation" = 0.5, #0.2 a 0.8 b
        resulttol: "tolerance of the result (f and norm(jac))" = 1e-3, #g
        singtol: "tolerance for singularity checks" = 1e-4, #1e-5, i
        minstep: "controls the minimal radius of the trust region" = 1e-5, 
        radiusFactor: "In [1, 2]. Controls how quickly the trust region decreases" = 1.5,
        infstep: "Stepsize after which a parameter is deemed unestimbale" = 1e10, 
        maxRadius: "radius of the trust region in the last iteration" = 1e4,
        disp: "whether to print a status message in each iteration" = True, 
        track_x: "whether to return the parameter trace" = False, 
        track_f: "whether to return the log-likelihood trace" = False):
    
    nuisance_minstep = 1e-3
    
    # ----- PREPARATION --------------------------------------------------------
    if fun0 is None:
        fun0 = fun(x0)
    if jac0 is None:
        jac0 = jac(x0)
    if hess0 is None:
        hess0 = hess(x0)
        
    # wrap the functions so that the correct end point is found
    if not forward:
        fun2 = fun
        jac2 = jac
        hess2 = hess
        flip = Flipper(index)
        fun = lambda x: fun2(flip(x)) 
        jac = lambda x: flip(jac2(flip(x)))
        hess = lambda x: flip(hess2(flip(x)))
        x0 = x0.copy()
        x0[index] *= -1
        jac0[index] *= -1
        hess0 = flip(hess0)
    else: 
        # this is just to ensure the results are traced and returned correctly
        flip = lambda x: x
    
    i: "current iteration" = 0
    x_track: "trace of parameter values" = []
    f_track: "trace of log-likelihood values" = []
    maximizing: "flag controlling whether we are maximizing f w.r.t. x[index]" = False
    nuisanceRadius: "search radius for unconstrained subproblems" = 1
    
    # counters to keep track how often functions have been called
    fun = CounterFun(fun)
    jac = CounterFun(jac)
    hess = CounterFun(hess)
    
    multi_dot = np.linalg.multi_dot
    
    is_negative_definite_ = partial(is_negative_definite, tol=singtol)
    
    def fPredicted_fun():
        """
        Taylor approximation of f
        """
        dx = xTmp-x
        return f + np.dot(J, dx) + multi_dot((dx, H, dx))/2
    
    def JPredicted_fun():
        """
        Taylor approximation of jac
        """
        return J + np.dot(H, xTmp-x)
    
    norm: "applied norm" = lambda arr: np.sqrt(np.mean(arr*arr))
    
    def jac_approx_precise(JPredicted_, JActual_, J_, bound=apprxtol):
        """
        Returns True, iff the predicted gradient is close to the
        actual one or, even better, closer to the target than predicted
        """
        # note how far we are from the target likelihood
        if (not searchmode=="normal" or maximizing) and not closeToDisc:
            return True
            
        return norm(JActual_) / norm(J_) <= bound or norm(JPredicted_-JActual_) / norm(J_) <= bound
    
    # TODO Accept improvements only then always, if they are better by a certain constant (compare values - line search)
    def f_approx_precise(fPredicted, fActual, bound=apprxtol):
        """
        Returns True, iff the predicted log-likelihood value is close to the
        actual one or, even better, closer to the target than predicted
        """
            
        # steps to negative predicted values cannot occur in theory and are 
        # either due to numerical issues or because we tried a very long step. 
        # Therefore, we reject these steps right away, if fActual is below the 
        # target.
        if fPredicted < target - resulttol and f > target and fActual < target:
            return False
        
        # return True if result is better than expected
        if fActual > fPredicted and xiTmp >= 0:
            return True
        
        # if we are maximizing w.r.t. the nuisance parameters, 
        if searchmode=="max_nuisance" or searchmode=="max_nuisance_const" or maximizing:
            # we do not accept steps that are worse than the current stage
            # (we guarantee elsewhere that it IS possible to increase f)
            if fActual < f: 
                if not relax_maximum_constraint or np.abs(fPredicted-fActual) > resulttol:
                    return False
            bound *= 2
            if relax_maximum_constraint:
                bound *= 2
        elif searchmode=="normal":
            # if we are outside the admissible region, we enforce that
            # we get closer to it again
            if fActual < target and np.abs(fActual-target) > np.abs(target-f):
                return False
            elif np.abs(fActual-target) < resulttol and np.abs(fPredicted-target) < resulttol:
                return True
            elif fActual > f > target:
                # we relax the precision requirement, if the step is not of
                # disadvantage and brings us forward (we value progression in 
                # the paramater of interest higher than a maximal likelihood)
                bound *= 2
                
        
        # we require that the error relative to the distance from the target
        # is not larger than the given bound.
        return np.abs(fPredicted-fActual) / np.abs(target-f) <= bound
    
    def test_precision() -> "True, if appxomiations are sufficiently precise":
        """
        Tests the precision of the Taylor approximation at the suggested 
        parameter vector.
        """
        fActual = fun(xTmp)
        fPredicted = fPredicted_fun()
        
        if not f_approx_precise(fPredicted, fActual):
            return False, fActual, None
        
        if np.abs(f-target) > resulttol and not closeToDisc: 
            return True, fActual, None
        
        JActual = jac(xTmp)
        JPredicted = JPredicted_fun() 
        
        if free_params_exist: 
            JActual_ = JActual[~free_considered]
            JPredicted_ = JPredicted[~free_considered]
            
            # the jacobian shall be mostly precise w.r.t. the changed parameters.
            # However, it shoudl not be totally imprecise with respect to the
            # remaining parameters, either.
            jac_precise = (jac_approx_precise(JPredicted_, JActual_, J_) and
                           jac_approx_precise(JPredicted[considered], 
                                              JActual[considered],
                                              J[considered], 
                                              20*apprxtol))
        else:
            JActual_ = JActual[considered]
            JPredicted_ = JPredicted[considered]
            jac_precise = jac_approx_precise(JPredicted_, JActual_, J_)
        
        return jac_precise, fActual, JActual
        
        
    
    def is_concave_down():
        """
        Checks whether the profile likelihood function is concave down.
        """
        return a < 0
    
    # variable definitions -------------------------------------------------
    x:      "parameter vector at current iteration"      = x0
    xTmp:   "potential new parameter vector"
    xi:     "parameter of interest at current iteration" = x0[index]
    f:      "log-likelihood at current iteration"        = fun0
    J:      "gradient at current iteration"              = jac0
    H:      "Hessian at current iteration"               = hess0
    maxFun: "maximal log-likelihood"                     = fun0
    xi_max: "maximal value for the parameter of interest with f(xi, x_) >= f*"    = xi
    x_max:  "parameter vector where xi_max leads to a log-likelihood value >= f*" = x
    considered:     "Array indexing nuisance parameters" = np.ones_like(x0, dtype=bool)
    considered[index]  = False
    consideredOriginal = considered.copy()
    hessConsidered: "Array indexing nuisance Hessian"    = np.ix_(considered, considered)
    H_:     "nuisance Hessian"                           = H[hessConsidered]
    H0_:    "nuisance entries of Hessian row corresponding to the parameter of interest" = H[index][considered]
    discontinuity_counter: "steps until discontinuous parameters are released" = 0
    maximizing: "flag to memorize whether the target has been changed temporarily" = False
    resultmode: "denotes whether or what kind of result has been found" = "searching"
    radius: "radius of the trust region in the last iteration" = np.inf
    closeToDisc: "True if we ar close to a discontinuity" = False
    xi_hist = []
    f_hist = []
    
    
    subproblem = FlexibleSubproblem(None, 
             lambda x: -f, 
             lambda x: -J_ - (xiTmp-xi)*H0_,
             lambda x: -H_
             )
    
    
    a = 0                        # just to prevent a possible name error...
    xiRadius = nuisanceRadius
        
    # Wrapper to catch potential errors without braking down completely
    try:    
        # ----- ITERATIONS -----------------------------------------------------
        for i in range(1, nmax+1):
            
            redoIteration = False
            ballsearch = False
            jacDiscont = False
            free_params_exist = False
            free_considered = ~considered
            infstepTmp = max(infstep, 10*np.abs(x0[index]))
            #closeToDisc = False
            
            if track_x:
                x_track.append(flip(x))
            if track_f:
                f_track.append(f)
            
            # if discontinuities are found, some parameters are held constant
            # for a while. The discontinuity_counter keeps track of the time
            if discontinuity_counter:
                discontinuity_counter -= 1
            else:
                considered = consideredOriginal.copy()
                hessConsidered = np.ix_(considered, considered)
                relax_maximum_constraint = False
            
            k = -1
            
            # determine approximation
            J_ = J[considered]
            xi = x[index]
            H0 = H[index]
            H0_ = H0[considered]
            H00 = H[index, index]
            H_ = H[hessConsidered]
            
            # Determine search mode:
            # max_nuisance:  set xi to specific value, maximize other parameters
            # max_nuisance_const:  hold xi constant, maximize other parameters
            # normal:        search x with f(x) = f*, J(x) = 0
            # binary search: get back to the adissible region with a binary 
            #                search
            counter = 0
            while True:
                counter += 1
                if counter > 10:
                    raise RuntimeError("Infinite loop", f, J, H, considered, 
                                       free_considered)
                
                if is_negative_definite_(H_):
                    H_inv = np.linalg.inv(H_)
                    searchmode = "normal"
                    break
                
                free_considered_ = ~get_independent_row_indices(H_, J_, 
                                                                tol=singtol)
                free_considered = ~considered
                free_considered[considered] = free_considered_
                free_params_exist = free_considered_.any()
                
                if not free_params_exist:
                    searchmode = "max_nuisance"
                    break
                
                H_dd = H_[np.ix_(~free_considered_, ~free_considered_)]
                
                # special case for H_ = 0
                if H_dd.size == 1 and np.allclose(H_dd, 0, atol=singtol):
                    # though H_ is singular, we set its inverse to a very large
                    # value to prevent NaNs that would mess up the result
                    H_ddInv = np.array([[max(-infstep/singtol, 1/H_dd[0,0])]])
                elif not is_negative_definite_(H_dd):
                    free_considered = ~considered
                    free_considered_[:] = False
                    searchmode = "max_nuisance"
                    break
                else:
                    H_ddInv = np.linalg.inv(H_dd)
                
                
                
                H0_d = H0_[~free_considered_]
                J_d = J_[~free_considered_]
                
                H_inv = H_ddInv
                H0_ = H0_d
                J_ = J_d
                H_ = H_dd
                
                searchmode = "normal"
                break
                
            # ----- NORMAL SEARCH ----------------------------------------------   
            while searchmode == "normal":
                
                a = (H00 - multi_dot((H0_, H_inv, H0_)))/2
                p = J[index] - multi_dot((H0_, H_inv, J_))
                fPL0: "current value on profile log-likelihood" 
                fPL0 = f - multi_dot((J_, H_inv, J_))/2
                
                if maximizing:
                    if (is_concave_down() 
                            and not (np.allclose(a, 0, atol=singtol) and 
                                     np.allclose(((fPL0-target)/p)*(a/p), 0, 
                                                 atol=singtol))
                            ) or f < target:
                        maximizing = False
                    else:
                        targetTmp = max(fPL0 + 1, (maxFun+f)/2)
                
                if not maximizing:
                    targetTmp = target
                    
                q = fPL0 - targetTmp
                
                if a == 0:
                    sqroot = np.nan
                else:
                    sqroot = (p/a)**2 - 4*q/a 
                
                
                # if desired point is not on approximation
                if sqroot < 0:
                    if (not np.allclose(a, 0, atol=singtol) 
                            and get_independent_row_indices(H, J, tol=singtol
                                                            )[index]):
                        if not maximizing and a<0 and f < targetTmp:
                            warnings.warn("The current function value is not on the approximation. "
                                          + "This may be due to a too large singtol. "
                                          + "The algorithm may not converge. Index=" 
                                          + str(index) + "; forward=" 
                                          + str(forward) + "; a=" 
                                          + str(a) + "; f=" + str(f)
                                          + "; root=" + str(sqroot))
                        
                        assert not maximizing
                        
                        if not is_concave_down() and f >= target:
                            # if beyond or very close to a local min
                            if p >= 0 or -p/a < singtol:
                                maximizing = True
                                continue
                            else:
                                # jump over minimum by factor 2
                                p *= 2
                            
                        sqroot = 0
                    # if f is approximately constant, we do not trust the 
                    # quadratic approximation too much and jsut try a very long
                    # step, even if it goes beyond the predicted admissible 
                    # region
                    else:
                        a = 0
                        sqroot = np.inf
                else:    
                    sqroot = np.sqrt(sqroot)
                
                # if Hessian is singular in the considered component
                # this control is important to prevent numerical errors if
                # |a| << 1 and the summands of the root are equal order of 
                # magnitude
                if (np.allclose(a, 0, atol=singtol) and 
                        np.allclose((q/p)*(a/p), 0, atol=singtol)):
                    if (maximizing and q*p>0) or (not maximizing and 
                                                    f >= target and q*p > 0):
                        maximizing = not maximizing
                        continue
                    else:
                        xiTmp = xi - q/p
                # control for NaN case
                elif p == 0 and a == 0: 
                    xiTmp = np.inf
                else:
                    xi1 = xi + (-p/a + sqroot) / 2
                    xi2 = xi + (-p/a - sqroot) / 2
                    
                    # check which root is closer to current value
                    if is_concave_down(): #not maximizing
                        assert not maximizing
                        xiTmp = max(xi1, xi2)
                    elif p*a > 0 and not maximizing and f > target:
                        maximizing = True
                        continue
                    elif maximizing and p/a < -singtol:
                        maximizing = False
                        continue
                    elif maximizing:
                        xiTmp = max(xi1, xi2)
                    else:
                        if np.abs(xi-xi1) < np.abs(xi-xi2): 
                            xiTmp = xi1
                        else:
                            xiTmp = xi2
                
                # if likelihood is independent of parameter
                if np.abs(xiTmp-xi) >= infstep * (1-singtol):
                    
                    
                    # try a very large xi value
                    xiTmp = xi + infstepTmp
                    xTmp = x.copy()
                    xTmp[index] = xiTmp
                    xTmp[~free_considered] += -np.dot(H_inv, H0_*(xiTmp-xi)+J_)
                    fActual = fun(xTmp)
                    
                    # if still larger than target
                    if fActual >= target:
                        JActual = jac(xTmp)
                        resultmode = "unbounded"
                        break 
                    
                    if not f > target:
                        if norm(J_) > singtol:
                            searchmode = "max_nuisance"
                            break 
                        
                        searchmode = "binary_search"
                        break 
                    xiTmp = xiTmp/4 + 3*xi/4
                
                xTmp = x.copy()
                xTmp[index] = xiTmp
                xTmp[~free_considered] += -np.dot(H_inv, H0_*(xiTmp-xi)+J_)
                
                xx = x[~free_considered]
                upperRadius = np.linalg.norm(xx-xTmp[~free_considered])
                
                # Test precision
                stop, fActual, JActual = test_precision()
                if stop:
                    radius = max(radius, upperRadius)
                    break
                
                if radius >= upperRadius:
                    lowerRadius = upperRadius/2
                else:
                    lowerRadius = radius
                
                if upperRadius > 0:
                    radius = upperRadius
                
                tmpRadius = lowerRadius
                xiStep = xiTmp-xi
                
                if tmpRadius == 0:
                    xiTmp = xi + min(xiStep/2, infstep)
                else:
                    xiTmp = xi + xiStep*np.power(tmpRadius/radius, np.log(2)/np.log(radiusFactor))
                
                ballsearch = True
                break
                    
            if not resultmode=="searching":
                x = xTmp
                J = JActual
                JActual_ = JActual[considered]
                f = fActual
                break
            
            if searchmode == "normal" and free_params_exist:
                J_new = norm(JPredicted_fun()[considered])
                J_current = norm(J[considered])
                
                # consider adjusted scenario with xiTmp = 0
                J_f = J[considered][free_considered_]
                H_df = H[hessConsidered][np.ix_(~free_considered_, free_considered_)]
                J_new2 = J_f - multi_dot((H_df.T, H_ddInv, J_d))
                J_new2 = np.linalg.norm(J_new2)/np.sqrt(J_.size)
                
                # if ignoring the free parameters hinders us from making an 
                # improvement w.r.t. the gradient
                abstol = max(np.log2(np.abs(xiTmp-xi))*np.abs(f-target)
                             /np.abs(maxFun-target), resulttol)
                if (J_new > abstol and J_new/J_current > apprxtol) or (
                        J_new2 > abstol and J_new2/J_current > apprxtol):
                    free_considered = ~considered
                    free_considered_[:] = False
                    free_params_exist = False
                    J_ = J[considered]
                    H_ = H[hessConsidered]
                    H0_ = H0[considered]
                    searchmode = "max_nuisance"
            
            
            counter = 0
            repeat = True
            while repeat:
                repeat = False
                
                if counter > 10:
                    raise RuntimeError("Infinite loop", f, J, H, considered, 
                                       free_considered)
                
                # ----- CONSTRAINED MAXIMIZATION -----------------------------------
                if searchmode == "max_nuisance" or searchmode == "max_nuisance_const":
                    xiTmp = xi
                    tmpRadius = nuisanceRadius * np.sqrt(J_.size)
                    ballsearch = True
                    
                    if not searchmode == "max_nuisance_const":
                        if f > target:
                            xiTmp += xiRadius
                        else:
                            searchmode = "max_nuisance_const"
                    
                # ----- BALL SEARCH ------------------------------------------------   
                if ballsearch:
                    increaseTrustRegion = True
                    xiPrev = xi
                    nAdjustement = 5
                    for k in range(nchecks):
                        
                        # it can happen that the radius is so conservative that
                        # the nuisance parameters cannot be maximized 
                        # sufficiently far to keep f in the admissible region
                        nRatioAdjustement = 10
                        for l in range(nRatioAdjustement):
                            #print("tmpRadius", tmpRadius)
                            xTmp_diff, _ = subproblem.solve(tmpRadius, 
                                                            searchmode=="normal",
                                                            jac0tol=singtol)
                            if xi > xiTmp and f>target:
                                print("a, p", a, p)
                                print("q, fPL0", q, fPL0)
                                print("targetTmp", targetTmp)
                                print("xiTmp, xi", xiTmp, xi)
                            xTmp = x.copy()
                            xTmp[index] = xiTmp
                            xTmp[~free_considered] += xTmp_diff
                            fPredicted = fPredicted_fun()
                            
                            # if we maximize w.r.t. the nuisance parameters,
                            # we want to assure that f gets increased to get
                            # back to the likelihood ridge
                            if (not searchmode == "max_nuisance" and 
                                    not searchmode == "max_nuisance_const"):
                                break
                            precise, fActual, JActual = test_precision()
                            if fPredicted >= f:
                                break
                            # if the step is super close and f still does not 
                            # increase, there must be a numerical issue. We
                            # leave the treatment of this issue to the procedures
                            # below.
                            elif (np.isclose(xiTmp, xi, atol=minstep/100) and tmpRadius<minstep/100):
                                break
                            
                            if l == nRatioAdjustement-2:
                                xiPrev = xiTmp
                                xiTmp = xi
                            else:
                                # we either increase the search radius for the
                                # nuisance parameters or decrease the search
                                # radius for the parameter of interest dependent
                                # on whether the approximation is precise and
                                # we can increase the trust region or not
                                if increaseTrustRegion and precise and tmpRadius < maxRadius*np.sqrt(xTmp_diff.size):
                                    tmpRadius *= 2
                                else:
                                    # if the error occurs even though xi is 
                                    # constant, there must be an error in the 
                                    # approximate solution and we decrease the
                                    # search radius
                                    if np.isclose(xiTmp, xi, atol=minstep/100):
                                        tmpRadius /= 5
                                    xiTmp = (xi+xiTmp) / 2
                                    
                        # if maximizing w.r.t. the nuisance parameters,
                        # adjust the trust region
                        if (searchmode == "max_nuisance" or 
                                searchmode == "max_nuisance_const"):
                            if precise:
                                # update trust region
                                nuisanceRadius = tmpRadius / np.sqrt(J_.size)
                                # update xiRadius only if we have changed xi
                                if searchmode == "max_nuisance":
                                    if xi != xiTmp:
                                        xiRadius = max(xiTmp-xi, minstep)
                                    else:
                                        # if we had to set xiRadius to 0, we set 
                                        # it to the last non-zero value
                                        xiRadius = max(xiPrev-xi, minstep)
                                    # since the xiRadius cannot recover on its
                                    # own, we always gradually increase it
                                    if 2*xiRadius < nuisanceRadius:
                                        xiRadius *= 2
                                
                                # if we have not decreased the trust region 
                                # before, try to increase it
                                if increaseTrustRegion and k < nAdjustement:
                                    xTmp2 = xTmp
                                    fActual2 = fActual
                                    xiTmp = xi + 2*xiRadius
                                    tmpRadius *= 2
                                    tmpRadius = min(maxRadius*np.sqrt(xTmp_diff.size),
                                                    tmpRadius)
                                    continue
                                else:
                                    stop = True
                            else:
                                stop = False
                                if increaseTrustRegion: 
                                    if k == 0:  
                                        # if in the first iteration, note that 
                                        # we are decreasing the trust region
                                        increaseTrustRegion = False
                                    else:
                                        # set xTmp to the last precise value
                                        fActual = fActual2
                                        xTmp = xTmp2
                                        xiTmp = xTmp[index]
                                        stop = True
                        else:    
                            stop, fActual, JActual = test_precision()
                        
                        if searchmode == "normal":
                            
                            if stop:
                                # if the approximation is precise, we may increase
                                # the radius, but save the largest current result
                                # to fall back to
                                xTmp2 = xTmp
                                fActual2 = fActual
                                lowerRadius = tmpRadius
                            else:
                                upperRadius = tmpRadius
                                
                            # if we have to decrease the trust region
                            if upperRadius <= lowerRadius and not stop:
                                # continue with the continuity checks and
                                # cut the trust trust region by half
                                pass
                            # if precision of geometric binary search is as 
                            # desired (first equality necessary to prevent NaNs)
                            elif upperRadius==0 or upperRadius/lowerRadius <= 2:
                                # stop iteration and set xTmp to the optimal 
                                # value
                                if lowerRadius > 0:
                                    radius = lowerRadius
                                fActual = fActual2
                                xTmp = xTmp2
                                xiTmp = xTmp[index]
                                break
                            else:
                                # update xiTmp and radius
                                tmpRadius = np.sqrt(lowerRadius*upperRadius)
                                xiTmp = xi + xiStep*np.power(tmpRadius/radius, np.log(2)/np.log(radiusFactor))
                                continue
                        elif stop:
                            break
                        
                        minstepTmp = max(minstep / max(norm(J_), 1), 1e-12)
                        #if (not searchmode=="normal" or maximizing) and not closeToDisc:
                        #    minstepTmp *= 1000
                        
                        # if the step is too small
                        diffs = xTmp - x
                        if not stop and np.allclose(xTmp, x, atol=minstepTmp, rtol=1e-14):
                            
                            # if we are not really on a discontinuity but only 
                            # on a local maximum with a non-negative definite
                            # Hessian, we may just release the constraint
                            # that we need to get to a larger function value
                            if (not searchmode=="normal" or maximizing) and not closeToDisc: 
                                relax_maximum_constraint = True
                                discontinuity_counter = 20
                                if test_precision()[0]:
                                    break
                                
                            # check parameter of interest first
                            xTmp = x.copy()
                            xTmp[index] += diffs[index]
                            
                            # if small change in parameter of interest makes 
                            # approximation imprecise
                            if diffs[index] and not test_precision()[0]:
                                
                                # make sure discontinuity happens on 
                                # profile function
                                if norm(J[considered]) > resulttol:
                                    relax_maximum_constraint = False
                                    searchmode = "max_nuisance_const"
                                    closeToDisc = True
                                    xiDisc = xi
                                    repeat = True
                                    radius = tmpRadius
                                    break
                                
                                fActual = fun(xTmp)
                                fCloseTarget = (np.allclose(target, f, 
                                                            atol=resulttol) and
                                                not np.allclose(target, fActual, 
                                                                atol=resulttol))
                                
                                # if step is favourable or benign, we accept 
                                # depending on the distance measure this case
                                # will never occur, as favourable steps may 
                                # always be classified as precise
                                if fActual >= target or f < fActual:
                                    pass
                                # if the target is on the jump
                                elif ((f > target or fCloseTarget)
                                      and fActual < target):
                                    # stop here
                                    fPredicted = fPredicted_fun()
                                    resultmode = "discontinuous"
                                    xTmp = x
                                    JActual = J
                                else:
                                    # get back to the feasible region by 
                                    # conducting a binary search
                                    searchmode = "binary_search"
                                    
                                print("iter {:3d}{}: ***discontinuity of {} in x_{} at {}***".format(
                                            i, ">" if forward else "<", f-fActual, 
                                            index, x))
                                break
                            
                            # check for discontinuities in nuisance parameters
                            discontinuities = np.zeros_like(free_considered)
                            discontinuities_rejected = discontinuities.copy()
                            if not f_approx_precise(fPredicted, fActual):
                                if diffs[index]:
                                    fPredicted = fPredicted_fun()
                                    fActual = fun(xTmp)
                                else:
                                    fPredicted = fActual = f
                                for l in np.nonzero(~free_considered)[0]:
                                    xTmp[l] += diffs[l]
                                    fActual_previous = fActual
                                    fActual = fun(xTmp)
                                    fPredicted = fPredicted_fun()
                                    if not f_approx_precise(fPredicted, fActual):
                                        discontinuities[l] = True
                                        if fActual < fActual_previous:
                                            discontinuities_rejected[l] = True
                                            xTmp[l] = x[l]
                                            fActual = fActual_previous
                                jacDiscont = False
                            # if the gradient is responsible for the rejection
                            # we require a smaller steap
                            elif np.abs(f-target) < resulttol and np.allclose(xTmp, x, rtol=minstepTmp/10):
                                for l in np.nonzero(~free_considered)[0]:
                                    xTmp[l] += diffs[l]
                                    JActual_previous = JActual
                                    JActual = jac(xTmp) 
                                    JPredicted = JPredicted_fun()
                                    jac_precise = jac_approx_precise(JPredicted[~free_considered], JActual[~free_considered], J[~free_considered]) 
                                    if not jac_precise:
                                        discontinuities[l] = True
                                        if not (JActual_previous[l] * J[l] >= 0 
                                                or diffs[l]*J[l] <= 0):
                                            discontinuities_rejected[l] = True
                                            xTmp[l] = x[l]
                                            JActual = JActual_previous
                                jacDiscont = True
                            

                            if discontinuities.any():
                                
                                varStr = "variables" if not jacDiscont else "gradient entries"
                                    
                                print("iter {:3d}{}: ***index={}: discontinuities in {} {} at {}***".format(
                                            i, ">" if forward else "<", index, varStr,
                                            tuple(np.nonzero(discontinuities)[0]), x))
                                
                                redoIteration = discontinuities.sum() == discontinuities_rejected.sum()
                                
                                if discontinuities_rejected.any():
                                    discontinuity_counter = 25
                                    considered &= ~discontinuities_rejected
                                    free_considered |= discontinuities_rejected
                                    free_considered_ = free_considered[considered]
                                    hessConsidered = np.ix_(considered, considered)
                                    relax_maximum_constraint = False
                                
                                break
                        if tmpRadius > (radiusFactor**nchecks-k)*minstep:
                            tmpRadius /= 5
                            xiTmp = (xi+xiTmp) / 5
                        else:
                            tmpRadius /= radiusFactor
                            xiTmp = (xi+xiTmp) / 2
                        
                    else:
                        xTmp = x
                        JActual = J
                        fActual = f
            
            if redoIteration:
                continue
                    
            # ----- BINARY SEARCH ----------------------------------------------
            if searchmode == "binary_search":
                tol = 0.01
                bin_min = x
                bin_max = x_max
                for k in range(nchecks):
                    xTmp = (bin_max+bin_min) / 2
                    fActual = fun(xTmp)
                    
                    if 0 <= fActual - target < tol:
                        break
                    
                    if fActual < target:
                        bin_min = xTmp
                    else:
                        bin_max = xTmp
                else: 
                    xTmp = bin_max
                JActual = jac(xTmp)
                        
                        
            # ----- UPDATES & CEHCKS -------------------------------------------
            if (xTmp == x).all() and not jacDiscont and resultmode=="searching":
                print("iter {:3d}{}: ***no improvement when optimizing x_{} at {}***".format(
                            i, ">" if forward else "<", index, flip(x)))
                resultmode = "discontinuous"
            
            
            
            xiChange = xTmp[index] - xi
            xChange = norm(x-xTmp)
            
            # when we are maximizing the nuisance parameters, we may step 
            # farther than the allowedbound
            if xiChange >= infstep and f >= target:
                resultmode = "unbounded"
            elif (searchmode == "max_nuisance" or searchmode == "max_nuisance_const"
                  or maximizing) and (xiChange < nuisance_minstep 
                                      and f > target+resulttol):
                if (len(xi_hist) > 5 and (np.abs(np.array(xi_hist[-5:]) - xi) < nuisance_minstep).all() and
                    (np.abs(np.array(f_hist[-5:]) - f) < nuisance_minstep).all()):
                    
                    xTmp[index] += nuisance_minstep
                    fActual = fun(xTmp)
                    m = 0
                    fActual_previous = fActual
                    xTmp_previous = xTmp.copy()
                    while fActual > target and m < 1000:
                        fActual_previous = fActual
                        xTmp_previous = xTmp.copy()
                        xTmp[index] += nuisance_minstep * 2**m
                        fActual = fun(xTmp)
                        m += 1
                    
                    xTmp = xTmp_previous
                    fActual = fActual_previous
                    
                    m = -1
                    while fActual < target-50 and m > -10:
                        fActual_previous = fActual
                        xTmp_previous = xTmp.copy()
                        xTmp[index] += nuisance_minstep * 2**m
                        fActual = fun(xTmp)
                        m -= 1
                    
                    
                    JActual = jac(xTmp)
                    
                    print("iter {:3d}{}: ***taking additional step in x_{} to avoid convergence issues. f={:6.3}***".format(
                        i, ">" if forward else "<", index, fActual))
                    for m in range(1, 5):
                        xi_hist[-m] += xTmp[index]-xi
                xi_hist.append(xTmp[index])
                f_hist.append(fActual)
            
            x = xTmp
            xi = xTmp[index]
            
            if JActual is None:
                JActual = jac(xTmp)
            JImprovement = np.log2(norm(J[considered])/norm(JActual[considered]))
            fImprovement = np.log2(np.abs((f-target)/(fActual-target)))
            
            J = JActual
            f = fActual
            JActual_ = JActual[considered]
            
            # reset maximal value
            if f > target and xi > xi_max:
                xi_max = xi
                x_max = x
            
            fPredicted = fPredicted_fun()
            
            if disp:
                radius_str = ""
                if searchmode == "normal":
                    if maximizing:
                        mode = "maximizing"
                    else:
                        mode = "searching"
                elif (searchmode == "max_nuisance" 
                      or searchmode == "max_nuisance_const"):
                    mode = "maximizing nuisance parameters"
                    radius_str = "; step={}; radius={}".format(
                                        round_str(xiRadius), round_str(nuisanceRadius))
                elif searchmode == "binary_search":
                    mode = "binary search"
                else:
                    mode = "other"
                
                ndiscont = np.sum(~considered)-1
                if ndiscont:
                    discontStr = "; ndisct={}".format(ndiscont)
                else: 
                    discontStr = ""
                    
                if free_params_exist and free_considered_.any():
                    mode += " with " + str(np.sum(free_considered_)
                                           ) + " free params"
                    jac_cnsStr = "; jac_cns={}".format(round_str(
                            norm(JActual[~free_considered])))
                else:
                    jac_cnsStr = ""
                    
                print(("iter {:3d}{}: x_{}_d={}; f_d={}; jac_d={}; " + 
                       "nsteps={:2d}; x_d={}; f_impr={}; jac_impr={}; " +
                       "f_e={}{}{}{} - {}").format(i, 
                                                 ">" if forward else "<", index, 
                                                 round_str(xi-x0[index]), 
                                                 round_str(f-target), 
                                                 round_str(norm(JActual_)), k+2, 
                                                 round_str(xChange), 
                                                 round_str(fImprovement), 
                                                 round_str(JImprovement), 
                                                 round_str(fActual-fPredicted), 
                                                 jac_cnsStr, discontStr, 
                                                 radius_str, mode))
            
            
            if resultmode == "searching":
                resViol = max(norm(JActual_), np.abs(f-target))
                if resViol <= resulttol:
                    resultmode = "success"
                    break
            else:
                break
            
            if f > maxFun:
                print("-> iter {:3d}{}: !!!found NEW MAXIMUM for x_{} of {:6.3f} (+{:6.3f}) at {}!!!".format(
                        i, ">" if forward else "<", index, f, f-fun0, flip(x)))
                maxFun = f
            
            if closeToDisc:
                if np.abs(xi - xiDisc) > minstep:
                    closeToDisc = False
            
            H = hess(x)
        
        else:
            resultmode = "exceeded"
    except np.linalg.LinAlgError:
        resultmode = "linalg_error"
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  file=sys.stdout)
    except Exception:
        resultmode = "error"
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback,
                                  file=sys.stdout)

    
    if resultmode == "success":
        status = 0
        success = True
    elif resultmode == "discontinuous":
        status = 1
        success = False
    elif resultmode == "unbounded":
        status = 2
        success = True
    elif resultmode == "exceeded":
        success = False
        status = 3
    elif resultmode == "error" or resultmode == "linalg_error":
        #x = x0+np.nan
        #f = np.nan
        #JActual_ = x0[considered] + np.nan
        try:        
            JActual_ = J_
        except UnboundLocalError:
            JActual_ = jac(x)[considered]
        success=False
        if resultmode == "error":
            status = 5
        else:
            status = 4
    
    if disp:
        print(op.OptimizeResult(x=flip(x), 
                             fun=f,
                             jac=JActual_,
                             success=success, status=status,
                             nfev=fun.evals, njev=jac.evals, nhev=hess.evals, 
                             nit=i,
                             message=STATUS_MESSAGES[status]
                             ))
    
    if track_f:
        f_track.append(f)
    if track_x:
        x_track.append(flip(x))
    
    return op.OptimizeResult(x=flip(x), 
                             fun=f,
                             jac=JActual_,
                             success=success, status=status,
                             nfev=fun.evals, njev=jac.evals, nhev=hess.evals, 
                             nit=i,
                             x_track=np.array(x_track),
                             f_track=np.array(f_track),
                             message=STATUS_MESSAGES[status]
                             )
    
def test2():
    H = [[-2.23327686e+03,3.99193784e+02,4.74986638e-01,-8.55852404e+01,-7.08365052e+01,-3.87178313e+01,-4.71627573e+00,5.01847818e+02,1.44515881e-01,2.09204344e+02,-2.02882521e+03,6.66055028e+03],
[3.99193784e+02,-2.86681223e+02,-8.65003524e-02,1.57660291e+01,1.37304541e+01,6.87668109e+00,7.85222384e-01,-8.58418932e+01,-2.47201778e-02,-3.48771207e+01,3.70636919e+02,-1.19345238e+03],
[4.74986638e-01,-8.65003524e-02,-1.64835411e-04,1.92763890e-02,1.72011648e-02,6.75886299e-03,8.91626484e-04,-9.91940354e-02,-2.85644463e-05,-4.52167010e-02,4.33103480e-01,-1.41922723e+00],
[-8.55852404e+01,1.57660291e+01,1.92763890e-02,-7.94912934e+00,-2.12003533e+00,-8.30980052e-01,-6.01223691e-02,8.52047771e+00,2.45365018e-03,8.47158090e+00,-7.95298372e+01,2.50601940e+02],
[-7.08365052e+01,1.37304541e+01,1.72011648e-02,-2.12003533e+00,-7.23593328e+00,-7.92316206e-01,-5.74494253e-02,8.56654775e+00,2.46704626e-03,6.51037754e+00,-6.54960801e+01,2.13904934e+02],
[-3.87178313e+01,6.87668109e+00,6.75886299e-03,-8.30980052e-01,-7.92316206e-01,-1.29001560e+00,-5.33347490e-02,7.97675579e+00,2.29710812e-03,3.50273390e+00,-3.55470579e+01,1.19683855e+02],
[-4.71627573e+00,7.85222384e-01,8.91626484e-04,-6.01223691e-02,-5.74494253e-02,-5.33347490e-02,-2.51590522e-02,2.16451499e+00,6.23292769e-04,4.30597816e-01,-4.06813815e+00,1.36888080e+01],
[5.01847818e+02,-8.58418932e+01,-9.91940354e-02,8.52047771e+00,8.56654775e+00,7.97675579e+00,2.16451499e+00,-4.04649962e+02,-1.16533068e-01,-4.51388591e+01,4.46790932e+02,-1.47552673e+03],
[1.44515881e-01,-2.47201778e-02,-2.85644463e-05,2.45365018e-03,2.46704626e-03,2.29710812e-03,6.23292769e-04,-1.16533068e-01,-3.37252324e-05,-1.29984733e-02,1.28662083e-01,-4.24906261e-01],
[2.09204344e+02,-3.48771207e+01,-4.52167010e-02,8.47158090e+00,6.51037754e+00,3.50273390e+00,4.30597816e-01,-4.51388591e+01,-1.29984733e-02,-3.17095991e+01,1.88854937e+02,-6.14833131e+02],
[-2.02882521e+03,3.70636919e+02,4.33103480e-01,-7.95298372e+01,-6.54960801e+01,-3.55470579e+01,-4.06813815e+00,4.46790932e+02,1.28662083e-01,1.88854937e+02,-2.02882525e+03,6.00165782e+03],
[6.66055028e+03,-1.19345238e+03,-1.41922723e+00,2.50601940e+02,2.13904934e+02,1.19683855e+02,1.36888080e+01,-1.47552673e+03,-4.24906261e-01,-6.14833131e+02,6.00165782e+03,-2.02998942e+04]]
    J = [-2.39836336e+00,4.09572931e-01,-5.69090295e-04,-4.22659930e-02,-4.20388261e-02,-4.28440455e-02,-3.82639520e-02,1.97355806e+00,5.71485452e-04,2.16780197e-01,-2.13385259e+00,7.04714512e+00]

    H = np.array(H)
    J = np.array(J)
    
    def f(x):
        return np.linalg.multi_dot((x,H,x))/ 2 + np.dot(x, J) -12156.150521445503+12157.95054853 
    
    def j(x):
        return np.dot(H, x) + J 
    def h(x):
        return H
    
    x0 = np.zeros(J.size)
    find_CI_bound(8, 0, x0, f, j, h, True, nmax=300, track_x=True)
    
def test():
    
    Hm = np.array([[4., 1., -1., 4.],
                   [1,  5 , -2,  4.],
                   [-1,  -2,  4,  1],
                   [4,  4,  1,  9]])
    Hm = np.array([[4., 1., -1., 0.],
                   [1,  5 , -2,  3.],
                   [-1,  -2,  4,  2],
                   [0,  3,  2,  5]])
    Hm = np.array([[4., 1.,   -1.,   2.],
                   [1,  .25, -.25,  .5],
                   [-1,-.25,  4,   -.5],
                   [2,  .5, -.5,  1]])
    Dm = np.array((2, 3, 1, 4.00001))
    multi_dot = np.linalg.multi_dot
    
    def f4(params):
        x, y, z, a = params
        return -(1e-80)**(1e-20+algopy.exp(x))+y*y
        if x>0:
            return ((x*x-2+y)**2-4)**2 + x*x/100 + ((y*y-2-x)**2-4)**2 - y*y + (5+y+0.2*z+a)**2
        return -x+y*y
        return ((x*x-2+y)**2-4)**2 + x*x/100 + ((y*y-2-x)**2-4)**2 - y*y + (5+y+0.2*z-a)**2
        return multi_dot((params, Hm, params))/2 + np.dot(Dm, params)
        return x*x + (x+1)**2*(x+y+3*z + a*a)**2 + (a-1)**2
        return (x-y+3*z)**2 + x*x + 3*y*y - x*2 + y +5*z
        return 100*(y-x*x)**2+(x-1)**2 + 100*(z-y*y)**2+(y-1)**2
        return -(algopy.exp(-x*x*100)-x*x+x**4/5-x**6/100) + y*y + z*z
        return -algopy.exp(-x*x) + y*y + z*z
        return ((x*x-2)**2-4)**2 + x*x/100 + ((y*y-2)**2-4)**2 - y*y + ((z*z-2)**2-4)**2 - z*z   
        return -algopy.exp(-x*x*100) + x*x + y*y + z*z
        return (x-y+3*z)**10 +  (x*x + 3*y*y + z*z)/10000 
        return x*x + 3*y*y +z*z  + y
    
    
    def f3(params):
        x, y, z, a = params
        z = z + a
        return (x-y+3*z)**2 + x*x + 3*y*y - x*2 + y +5*z
    
    def f3a(params):
        x, y, z, a = params
        z = z + a
        x = x * y + a
        return f1([x, z])
        
    
    def F1(params):
        x, y = params
        return (x-y)**2 + x*x + 3*y*y
    def F2(params):
        x, y = params
        return x*x #y*y
    
    arr = np.array((1,1,1,1,1,0,0,0,0,0))[:,None,None]
    
    def f1(x):
        x, y = x
        x = x*10
        y = y**3
        return ((x*x-2+y)**2-4)**2 + x*x/100 + ((y*y-2-x)**2-4)**2 - y*y    
        return 100*(y-x*x)**2+(x-1)**2
        return np.abs(x+y)+(y*y)
        return (x+y)**2
        return -(algopy.exp(-x*x*100)-x*x/20+x**4/5-x**6/100) + (x-y)**2-30 + (x-2*y)**4
        return algopy.sum(100.0 * (x[1:] - x[:-1]*x[:-1])*(x[1:] - x[:-1]*x[:-1]) + (1 - x[:-1])*(1 - x[:-1]))
        return algopy.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)
    
    
    def ffAppr(x):
        x, y = x
        x = 1-x
        y = 1-y
        return -802.00011956*x*x+2*400.00002891*x*y-200*y*y - 4.04854143e-06*x + 1.95198457e-06*y - 1.4750875345706636e-14
    
    ln = algopy.log
    #def f6(x):
    #    k, q, ec, eo = x
    def f6(k, q, ec, eo):
        k = algopy.exp(k)
        q = algopy.arctan(q)/np.pi+0.5
        ec = algopy.arctan(ec)/np.pi+0.5
        eo = algopy.arctan(eo)/np.pi+0.5
        a1 = ec*eo + (1-ec)/100
        a2 = ec*eo + (1-ec)
        return -(ln(k) + 2*k*ln(1-q)-(k+1)*ln(a1*q+1-q)-k*ln(a2*q+1-q)+ln(a1)+ln(q))
    
    
    n = np.array([1., 0, 0, 0, 1])
    n = np.array([1., 0, 2, 0, 4.])
    N = 1000
    n[0] = 1
    print(n)
    
    p = np.array([0.1, 0, 1, 0.5, 0.7])
    p = np.random.beta(0.5, 0.5, N)/1
    
    t = np.array([.1, 1, 0.5, 0.7, 0.5])
    t = np.random.beta(0.5, 0.5, N)/1
    
    n = np.random.negative_binomial(100, 0.0001, N)
    k, q, ec, eo = 10, 0.5, 0.1, 0.1
    a = t*(ec*eo+(1-ec)*p)
    n = np.random.negative_binomial(100, (1-q)/(a*q+1-q))
    
    from algopy_ext import nbinom_logpmf
    
    
    def f7(k, q, ec, eo):
        k = k*k
        q = algopy.arctan(q)/np.pi+0.5
        ec = algopy.arctan(ec)/np.pi+0.5
        eo = algopy.arctan(eo)/np.pi+0.5
        a = t *( ec*eo + (1-ec)*p)
        return -algopy.sum(nbinom_logpmf(n, k, (1-q)/(a*q + 1-q)), 0)
    
    #def f(x):
    #    a, b = x
    #    return f7(a, 0, 0, b)
    def f5(x):
        a, b, c, d = x
        return f7(a, b, c, d)
    
    #f = F1
    #"""
    f = f3a
    x0=np.zeros(4)
    index = 3
    """
    f = f1
    x0=np.zeros(2)+0.00000000001
    index = 0
    #"""
    
    ff = lambda params: -f(params)
    j = nd_algopy.Gradient(f)
    jj = nd_algopy.Gradient(ff)
    h = nd_algopy.Hessian(f)
    hh = nd_algopy.Hessian(ff)
    
    
    r = op.minimize(f, x0, jac=j, hess=h)
    print(r)
    
    def tt():
        k, q, ec, eo = r.x
        k = k*k
        q = algopy.arctan(q)/np.pi+0.5
        ec = algopy.arctan(ec)/np.pi+0.5
        eo = algopy.arctan(eo)/np.pi+0.5
        a = t *( ec*eo + (1-ec)*p)
        #return np.sum(n-
    
    x0 = r.x #+ (np.random.rand(x0.size)-0.5)*0.0001
    #x0 = np.array([0.,0.])
    
    #jjj = lambda x: np.ones(3) + np.nan
    
    #profile_CI_investigation(x0, ff, jj, hh, ["x", "y", "z", "a"], disp=True, vm=False)
    #print(find_profile_CI(0, x0, ff, jj, hh, -1, vm=False))
    
    target = 0.5
    target = -2
    target = -20
    #target = -r.fun-2
    
    #"""
    r2 = find_CI_bound(index, target, x0, ff, jj, hh, True, nmax=300, track_x=True)
    print(r2)
    r1 = find_CI_bound(index, target, x0, ff, jj, hh, False, nmax=300, track_x=True)
    print(r1)
    """
    from optimize_ext import find_bound
    r2 = find_bound(index, target, x0, ff, jj, hh, True, nmax=300, track_x=True)
    print(r2)
    r1 = find_bound(index, target, x0, ff, jj, hh, False, nmax=300, track_x=True)
    print(r1)
    #"""
    rr1 = venzon_moolgavkar(index, target, x0, ff, jj, hh, -1, nmax=300, disp=True, track_x=True)
    print(rr1)
    rr2 = venzon_moolgavkar(index, target, x0, ff, jj, hh, 1, nmax=300, disp=True, track_x=True)
    print(rr2)
    
    print("[{}, {}]: {}, {}".format(r1.x[index], r2.x[index], r1.nit, r2.nit))
    print("[{}, {}]: {}, {}".format(rr1.x[index], rr2.x[index], rr1.nit, rr2.nit))
    
    #print(np.arctan(r1.x[index])/np.pi+0.5, np.arctan(r2.x[index])/np.pi+0.5)
    
    
    x = []
    y = []
    z = []
    a = []
    
    for row in r1.x_track:
        x.append(row[0])
        y.append(row[1])
    
        
    x = x[::-1]
    y = y[::-1]
    
    for row in r2.x_track:
        x.append(row[0])
        y.append(row[1])
    
    xx = []
    yy = []
    
    for row in rr1.x_track:
        xx.append(row[0])
        yy.append(row[1])
        
    xx = xx[::-1]
    yy = yy[::-1]
    
    for row in rr2.x_track:
        xx.append(row[0])
        yy.append(row[1])

    
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot(x, y, z)
    
    x = np.array(x)
    y = np.array(y)
    
    #"""
    for row in r1.x_track:
        z.append(row[2])
        a.append(row[3])
    z = z[::-1]
    a = a[::-1]
    for row in r2.x_track:
        z.append(row[2])
        a.append(row[3])
    z = np.array(z)
    a = np.array(a)
    x = x * y + a
    y = z + a
    #"""
    
    xmax = np.max(x)
    xmin = np.min(x)
    ymax = np.max(y)
    ymin = np.min(y)
    xdiff = xmax-xmin
    ydiff = ymax-ymin
    
    xmax += 0.1*xdiff
    xmin -= 0.1*xdiff
    ymax += 0.1*ydiff
    ymin -= 0.1*ydiff
    
    """
    xmax = 1.04
    xmin = 1
    ymax = -1.02
    ymin = -1.04
    """
    
    def qqq(args): 
        x, y = args
        return (x*(217.12168835*y-105.03247006*x)+(217.12168835*x-432.2433767*y)*y)/2+209.928*x+(-429.3677)*y-208.2143389700666
    X = np.linspace(xmin, xmax, 200)
    Y = np.linspace(ymin, ymax, 200)
    #X = np.linspace(-1.8, -2.1, 300)
    #Y = np.linspace(-0.2, 1.5, 300)
    #X = np.linspace(-2, 2, 300)
    #Y = np.linspace(-2, 2, 300)
    X, Y = np.meshgrid(X, Y)
    #Z = ff([X, Y])
    Z = -f1([X, Y])
    #Z = np.maximum(ffAppr([X, Y]), target)
    #Z = qqq([X, Y])
    #cmapname = "prism" #"jet"        # ok, aber sehr bunt
    #cmap = plt.get_cmap(cmapname)
    
    plt.pcolormesh(X, Y, np.maximum(Z, target*1.2), shading='gouraud') #cmap=cmap) #, 
    plt.plot(xx, yy, marker = 'x', color='C5')
    plt.plot(x, y, marker = 'o', color='C1')
    plt.contour(X, Y, Z, levels=[target]) #cmap=cmap) #, 
    plt.plot(x0[0], x0[1], marker = 'o', color='C3')
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    plt.ylabel("x1")
    plt.xlabel("x0")
    print(xx)
    print(yy)
    
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                   linewidth=0, antialiased=True)
    
    plt.show()
    

def plot3d():
    from mpl_toolkits.mplot3d import Axes3D
    
    from mayavi import mlab
    import algopy
    from numdifftools import nd_algopy
    
    
    def f(xx):
        x, y = xx
        return -8*algopy.exp(-(x*x+2*y*y))-4*algopy.exp(-((x+2)**2+y**2))+x**2/20+y**2/30
        return 100*(y-x*x)**2+(x-1)**2
    def f2(xx):
        x, y = xx
        return -0.1*x**2 + y**2
    
    ff = lambda x, y: -f2([x, y])
    xmin, xmax, ymin, ymax = -3, 2, -2, 2
    xmin, xmax, ymin, ymax = -4, 4, -2, 2
    
    X = np.linspace(xmin, xmax, 40)
    Y = np.linspace(ymin, ymax, 40)
    X, Y = np.meshgrid(X, Y)
    Z = ff(X, Y)
    #Z[Z<-20] = np.nan
    #plt.pcolormesh(X, Y, np.maximum(Z, -20), shading='gouraud') #cmap=cmap) #, 
    #plt.show()    
    
    j = nd_algopy.Gradient(f)
    h = nd_algopy.Hessian(f)
    
    
    x0=np.ones(2)
    r = op.minimize(f, x0, jac=j, hess=h)
    
    print(r)
    
    x0, y0 = r.x
    x0, y0 = -1, 0
    x0, y0 = -0.2, 0.65
    
    H = -h((x0, y0))
    J = -j((x0, y0))
    f0 = -f((x0, y0))
    h00, h01, h10, h11 = H.ravel()
    j0, j1 = J
    multi_dot = np.linalg.multi_dot
    def fappr(x, y):
        x = x - x0
        y = y - y0
        return x**2*h00 + x*y*(h10+h01) + y**2*h11 + j0*x + j1*y + f0 
        return multi_dot((xx, H, xx)) + np.dot(xx, J) + f0 
    
    x, y = np.mgrid[xmin:xmax:0.1, ymin:ymax:0.05]
    diff1 = 0.7
    diff1 = 3
    diff2 = 0.5
    x1, y1 = np.mgrid[x0-diff1:x0+diff1:0.05, y0-diff2:y0+diff2:0.05]
    
    x1, y1 = x, y
    
    f1 = lambda x, y: x*0+3
    
    xCyl, zCyl = np.mgrid[-1:1.02:0.02, 0:3:0.05]
    xCyl, zCyl = np.mgrid[-1:1.02:0.02, -2:1:0.05]
    yCyl = np.sqrt(1-xCyl**2)
    
    
    z = ff(x,y)
    zApprox = fappr(x1, y1)
    
    print(z)
    print(zApprox)
    print(f0)
    
    zApprox[np.logical_or(zApprox<-3, zApprox>8)] = np.nan
    #zApprox[np.logical_or(zApprox<-3, zApprox>10)] = np.nan
    
    fig = mlab.figure(1, size=(1000, 800), bgcolor=(1,1, 1), fgcolor=(0.,0.,0.))
    mlab.clf()
    
    wScale = 0.4
    s = mlab.surf(x, y, z, colormap='viridis', #extent=[0, 2, 0, 1, 0, 1],
             line_width=1, warp_scale=wScale,
             representation='wireframe', transparent=True, opacity=0.9)
             #representation='surface', transparent=True, opacity=0.7)
    mlab.axes(nb_labels=0, xlabel="x1", ylabel="x2", zlabel="ln(L)",
              )
    sc1 = mlab.mesh(x0+xCyl, y0+yCyl, zCyl, colormap='bone', #extent=[0, 2, 0, 1, 0, 1],
             line_width=1, #warp_scale=wScale,
             representation='surface', transparent=True, opacity=0.3)
    sc2 = mlab.mesh(x0+xCyl, y0-yCyl, zCyl, colormap='bone', #extent=[0, 2, 0, 1, 0, 1],
             line_width=1, #warp_scale=wScale,
             representation='surface', transparent=True, opacity=0.3)
    '''
    sA = mlab.surf(x1, y1, zApprox, colormap='bone', #extent=[0, 2, 0, 1, 0, 1],
              line_width=1, warp_scale=wScale,
              representation='wireframe')
    #mlab.axes(nb_labels=0, xlabel="x1", ylabel="x2", zlabel="ln(L)",
    #          )
    mlab.points3d(x0, y0, f0*wScale, resolution=20, color=(.9, .5,.3), scale_factor=.1)
    s1 = mlab.surf(x, y, z*0+3, #8.2, 
                   colormap='gist_earth', 
              representation='wireframe', line_width=1, warp_scale=wScale #'auto'
              #extent=[0, 1, 0, 1, 0, 1]
              )
    #'''
    mlab.view(110, 75, 14, None)
    #mlab.axes()
    #mlab.outline()
    mlab.show()
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    #ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel())
    ax.plot_trisurf(x.ravel(), y.ravel(), z.ravel())
    ax.plot_trisurf(x.ravel(), y.ravel(), zApprox.ravel())
    #ax.plot_trisurf(X.ravel(), Y.ravel(), (Z*0+3).ravel())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    #ax.set_zlim(-20, 20)
    
    #ax.plot_trisurf(X.ravel(), Y.ravel(), Z.ravel(), vmin=-20,
    #                linewidth=0.2, antialiased=True,
    #                )
    #"""
    plt.show()    

def test3():
    h = lambda x: -np.diag((0, 0, 0, 0))
    j = lambda x: -np.array((1, 0, 0 ,0))
    f = lambda x: -x[0]
    
    x0 = np.zeros(4)
    #s = FlexibleSubproblem(x0, f, j, h)
    #print(s.solve(2))
    
    
    r = find_CI_bound(0, -1, x0, f, j, h)
    print(r)

def test4():
    h = [[-0.00000000e+00,-0.00000000e+00,-0.00000000e+00,-0.00000000e+00,-0.00000000e+00]
,[-0.00000000e+00,-0.00000000e+00,-0.00000000e+00,-0.00000000e+00,-0.00000000e+00]
,[-0.00000000e+00,-0.00000000e+00,6.59765873e-07,-7.95084287e-08,7.97178695e-08]
,[-0.00000000e+00,-0.00000000e+00,-7.95084287e-08,8.32746616e-05,-8.34294256e-05]
,[-0.00000000e+00,-0.00000000e+00,7.97178695e-08,-8.34294256e-05,8.33989592e-05]]
    j = [-0.00000000e+0,-0.00000000e+00, -5.56847260e-04,  8.37653714e-05,-8.34289217e-05]
    h = np.array(h)
    j = np.array(j)
    hess = lambda x: h
    jac = lambda x: j
    fun = lambda x: -10
    
    s = FlexibleSubproblem(j, fun, jac, hess)
    
    print(s.solve(300))
    
    
    
    

if __name__ == '__main__':
    plot3d()
    import sys
    sys.exit()
    import algopy
    #plot3d()
    test()