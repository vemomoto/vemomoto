'''
'''
import warnings
import traceback
import sys
from functools import partial
from multiprocessing import Pool
from itertools import count as itercount
import numdifftools as nd

import numpy as np
import scipy.optimize as op
from scipy.stats import chi2
from scipy.optimize._trustregion_exact import IterativeSubproblem

from vemomoto_core.tools.doc_utils import inherit_doc



def __round_str(n, full=6, after=3):
    if np.isfinite(n) and n != 0:
        digipre = int(np.log10(np.abs(n)))+1
        after = min(max(full-digipre-1, 0), after)
    return ("{:"+str(full)+"."+str(after)+"f}").format(n)

def is_negative_definite(M, tol=1e-5):
    """
    Returns whether the given Matrix is negative definite.
    
    Uses a Cholesky decomposition.
    """
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
    """
    Flips an array in a specified component
    """
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


STATUS_MESSAGES = {0:"Success",
                   1:"Result on discontinuity",
                   2:"Result is unbounded",
                   3:"Iteration limit exceeded",
                   4:"Ill-conditioned matrix",
                   5:"Unspecified error"
                   }

def find_CI_bound(x0, fun, index, direction, 
        jac = None, 
        hess = None, 
        alpha = 0.95,
        fun0 = None, 
        jac0 = None, 
        hess0 = None, 
        customTarget = None,
        nmax = 200, 
        nchecks = 65,
        apprxtol = 0.5,
        resulttol = 1e-3,
        singtol = 1e-4, 
        minstep = 1e-5, 
        radiusFactor = 1.5,
        infstep = 1e10, 
        maxRadius = 1e4,
        disp = False, 
        track_x = False, 
        track_f = False):
    """Finds an end point of a profile likelihood confidence interval.
    
    Parameters
    ----------
    x0 : float[]
        Maximum likelihood estimate (MLE) of the paramters.
    fun : callable
        Log-likelihood function.
    index : int
        Index of the parameter of interest.
    direction : int or bool
        If ``<=0``, the lower end point of the confidence interval is sought, 
        else the upper end point is sought.
    jac : callable
        Gradient of :py:obj:`fun`. If `None`, it will be computed based on
        finite differences using numdifftools.
    hess : callable
        Hessian of :py:obj:`fun`. If `None`, it will be computed based on
        finite differences using numdifftools.
    alpha : float
        Desired confidence level. Must be in ``(0,1)``
    fun0 : float
        log-likelihood at the MLE.
    jac0 : float[]
        Gradient of the log-liekelihood at the MLE.
    hess0 : float[][]
        Hessian of the log-likelihood at the MLE.
    customTarget : float
        Custom target log-likelihood l*. If this is given, :py:obj:`alpha` will
        be ignored.
    nmax : int
        Maximal number of iterations.
    nchecks : int
        Maximal number of trust-region changes per iteration.
    apprxtol : float
        Relative tolerance between :py:obj:`fun` and its approximation.
    resulttol : float
        Tolerance of the result (``fun`` and ``norm(jac)``).
    singtol : float
        Tolerance for singularity checks.
    minstep : int
        Controls the minimal radius of the trust region. 
    radiusFactor : float
        Controls how quickly the trust region decreases. Must be in ``[1, 2]``.
    infstep : float
        Stepsize after which a parameter is deemed unestimbale.
    maxRadius : float
        Rradius of the trust region in the last iteration.
    disp : bool
        Whether to print a status message in each iteration.
    track_x : bool
        Whether to return the parameter trace.
    track_f : bool
        Whether to return the log-likelihood trace.
    
    """
    nuisance_minstep = 1e-3
    
    # ----- PREPARATION --------------------------------------------------------
    if jac is None:
        jac = nd.Gradient(fun)
    if hess is None:
        hess = nd.Hessian(fun)
    
    
    # Make sure arguments and returned values are of correct type
    index = int(index)  # in case the index is not given as integer
    x0 = np.asarray(x0) # in case x0 is not given as numpy array
    
    forward = direction > 0
    
    # wrap the functions so that the correct end point is found
    jac2 = jac
    hess2 = hess
    if not forward:
        fun2 = fun
        flip = Flipper(index)
        fun = lambda x: fun2(flip(x)) 
        jac = lambda x: flip(np.asarray(jac2(flip(x))))
        hess = lambda x: flip(np.asarray(hess2(flip(x))))
        x0 = x0.copy()
        x0[index] *= -1
    else: 
        # this is just to ensure the results are traced and returned correctly
        flip = lambda x: x
        jac = lambda x: np.asarray(jac2(x))
        hess = lambda x: np.asarray(hess2(x))
    
    # initial values
    if fun0 is None:
        fun0 = fun(x0)
    if jac0 is None:
        jac0 = jac(x0)
    else:
        jac0 = np.asarray(jac0)
    if hess0 is None:
        hess0 = hess(x0)
    else:
        hess0 = np.asarray(hess0)
    
    # flip initial values if necessary
    if not forward:
        jac0[index] *= -1
        hess0 = flip(hess0)
        
    # compute the target
    if customTarget is None:
        target = fun0 - chi2.ppf(alpha, df=1)/2
    else:
        target = customTarget
    
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
    
    if disp:
        def dispprint(*args, **kwargs):
            print(*args, **kwargs)
    else:        
        def dispprint(*args, **kwargs):
            pass
    
    
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
                                # Somethong went wrong. Print debug information.
                                warnings.warn("Something went wrong. Please report this error "
                                              "and the information below to the bugtracker. "
                                              "Presumably the result is correct anyway.")
                                try:
                                    print("searchmode, maximizing", searchmode, maximizing)
                                    print("a, p =", a, ",", p)
                                    print("q, fPL0 =", q, ",", fPL0)
                                    print("f, target, targetTmp =", f, ",", target,",",  targetTmp)
                                    print("xiTmp, xi =", xiTmp, ",", xi)
                                    print("xi1, xi2 =", xi1, ",", xi2)
                                    print("xiStep, infstep =", xiStep, ",", infstep)
                                    print("tmpRadius, radius =", tmpRadius, ",", radius)
                                    print("index, direction, k, l =", index, forward, k, l )
                                    print("xTmp[index]", xTmp[index])
                                    print("---------------------")
                                except Exception:
                                    pass
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
                                    
                                dispprint("iter {:3d}{}: ***discontinuity of {} in x_{} at {}***".format(
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
                                    
                                dispprint("iter {:3d}{}: ***index={}: discontinuities in {} {} at {}***".format(
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
                            xiTmp = (4*xi+xiTmp) / 5
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
                dispprint("iter {:3d}{}: ***no improvement when optimizing x_{} at {}***".format(
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
                    
                    dispprint("iter {:3d}{}: ***taking additional step in x_{} to avoid convergence issues. f={:6.3}***".format(
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
                                        __round_str(xiRadius), __round_str(nuisanceRadius))
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
                    jac_cnsStr = "; jac_cns={}".format(__round_str(
                            norm(JActual[~free_considered])))
                else:
                    jac_cnsStr = ""
                    
                dispprint(("iter {:3d}{}: x_{}_d={}; f_d={}; jac_d={}; " + 
                       "nsteps={:2d}; x_d={}; f_impr={}; jac_impr={}; " +
                       "f_e={}{}{}{} - {}").format(i, 
                                                 ">" if forward else "<", index, 
                                                 __round_str(xi-x0[index]), 
                                                 __round_str(f-target), 
                                                 __round_str(norm(JActual_)), k+2, 
                                                 __round_str(xChange), 
                                                 __round_str(fImprovement), 
                                                 __round_str(JImprovement), 
                                                 __round_str(fActual-fPredicted), 
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
                dispprint("-> iter {:3d}{}: !!!found NEW MAXIMUM for x_{} of {:6.3f} (+{:6.3f}) at {}!!!".format(
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
        try:        
            JActual_ = J_
        except UnboundLocalError:
            JActual_ = jac(x)[considered]
        success=False
        if resultmode == "error":
            status = 5
        else:
            status = 4
    
    dispprint(op.OptimizeResult(x=flip(x), 
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

def __parallel_helper_fun(args):
    return args[0], args[3], find_CI_bound(*args[1:-1], **args[-1])

@inherit_doc(find_CI_bound)
def find_CI(x0, fun, jac=None, hess=None, indices=None, directions=None, alpha=0.95, 
            parallel=False, return_full_results=False, return_success=False,
            **kwargs):
    """Returns the profile likelihood confidence interval(s) for one or 
    multiple parameters.
    
    Parameters
    ----------
    indices : int[]
        Indices of the parameters of interest. If not given, all paramters
        will be considered.
    directions : float[][]
        Search directions. If not given, both end points of the confidence
        intervals will be determined. If given as a scalar, only lower
        end points will be returned if ``directions<=0`` and upper end points 
        otherwise. If given as an array, the confidence interval end points 
        specified in row ``i`` will be returned for parameter ``i``. Entries ``<=0``
        indicate that lower end points are desired, whereas positive entries 
        will result in upper end points.
    parallel : bool
        If ``True``, results will be computed in parallel using ``multiprocessing.Pool``.
        Note that this requires that all arguments are pickable.
    return_full_results : bool
        If ``True``, an ``OptimizeResult`` object will be returned for each 
        confidence interval bound. Otherwise, only the confidence interval 
        bounds for the parameters in question will be returned.
    return_success : bool
        If ``True``, an array of the same shape as the result will be returned
        in addition, indicating for each confidence interval bound whether it 
        was determined successfully. 
    **kwargs : keyword arguments
        Other keyword arguments passed to :py:meth:`find_CI_bound`. Look at 
        the documentation there.
        
    """
    x0 = np.asarray(x0)
    
    if indices is None:
        indices = np.arange(x0.size)
    elif np.isscalar(indices):
        indices = (indices,)
    
    if directions is None:
        directions = np.zeros((len(indices), 2))
        directions[:,0] = -1
        directions[:,1] = 1
    elif np.isscalar(directions):
        directions = np.full((len(indices), 1), directions)
    elif type(directions) == np.ndarray:
        if directions.ndim == 1:
            directions = directions[:,None]
    else:
        directions = [[i] if not hasattr(i, "__iter__") else i 
                      for i in directions]
    
    if "fun0" not in kwargs:
        kwargs["fun0"] = fun(x0)
    if "jac0" not in kwargs:
        if jac is None:
            kwargs["jac0"] = nd.Gradient(fun)(x0)
        else:
            kwargs["jac0"] = jac(x0)
    if "hess0" not in kwargs:
        if hess is None:
            kwargs["hess0"] = nd.Hessian(fun)(x0)
        else:
            kwargs["hess0"] = hess(x0)
    
    task_chain = []
    
    for i, index, directions_ in zip(itercount(), indices, directions):
        for direction in directions_:
            task_chain.append((i, x0, fun, index, direction, jac, hess, 
                               alpha, kwargs))
    
    if parallel:
        with Pool() as pool:
            result_list = list(pool.map(__parallel_helper_fun, task_chain))
    else:
        result_list = list(map(__parallel_helper_fun, task_chain))
    
    result = [[] for _ in range(min(len(indices), len(directions)))]
    if return_success:
        success = [[] for _ in range(min(len(indices), len(directions)))]
    
    for i, index, res in result_list:
        if return_success:
            success[i].append(res.success)
        if return_full_results:
            result[i].append(res)
        else:
            result[i].append(res.x[index])
    
    try:
        result = np.array(result)
    except ValueError:
        result = np.array(result, dtype=object)
    
    if return_success:
        try:
            success = np.array(success)
        except ValueError:
            success = np.array(success, dtype=object)
        return result, success
    else:
        return result

@inherit_doc(find_CI)
def find_function_CI(x0, function, logL, 
                     functionJac = None,
                     functionHess = None,
                     logLJac = None, 
                     logLHess = None,
                     relativeError = 1e-4, 
                     **kwargs):
    """Returns the profile likelihood confidence interval(s) for a function 
    of parameters.

    Parameters
    ----------
    function : callable
       Function of the parameters for which the confidence interval shall be 
       computed
    logL : callable
        Log-likelihood function.
    functionJac : callable
        Gradient of :py:obj:`function`. If `None`, it will be computed based on
        finite differences using numdifftools.
    functionHess : callable
        Hessian of :py:obj:`function`. If `None`, it will be computed based on
        finite differences using numdifftools.
    logLJac : callable
        Gradient of :py:obj:`logL`. If `None`, it will be computed based on
        finite differences using numdifftools.
    logLHess : callable
        Hessian of :py:obj:`logL`. If `None`, it will be computed based on
        finite differences using numdifftools.
    relativeError : float
        Permitted relative error in the confidence interval bound.
    **kwargs : keyword arguments
        Other keyword arguments passed to :py:meth:`find_CI` and
        :py:meth:`find_CI_bound`. Look at the documentation there.
    """
    
    function0 = function(x0)
    error = function0 * relativeError
    x0 = np.insert(x0, 0, function0)
    
    fun_2 = lambda x: 0.5 * ((function(x[1:])-x[0])/error)**2
    fun = lambda x: logL(x[1:]) - fun_2(x)
    
    if functionJac is None and logLJac is None:
        jac = nd.Gradient(fun)
    else:
        if functionJac is None:
            jac_2 = nd.Gradient(fun_2)
        else:
            def jac_2(x):
                result = np.zeros(x0.size)
                diffTerm = (function(x[1:]) - x[0]) / error**2
                result[0] = -diffTerm
                result[1:] = functionJac(x[1:]) * diffTerm
                return result
        
        if logLJac is None:
            logLJac = nd.Gradient(logL)
        
        def jac(x):
            result = -jac_2(x)
            result[1:] += logLJac(x[1:]) 
            return result
        
    if functionHess is None:
        hess_2 = nd.Hessian(fun_2)
    else:
        if functionJac is None:
            functionJac = nd.Gradient(function)
        
        def hess_2(x):
            result = np.zeros((x0.size, x0.size))
            functionJacX = functionJac(x[1:]) 
            result[0, 0] = 1 / error**2
            result[1:, 0] = result[0, 1:] = -functionJacX / error**2
            result[1:, 1:] = (functionHess(x[1:]) * (function(x[1:]) - x[0])
                              + functionJacX * functionJacX[:,None]) / error**2
            return result
        
    if logLHess is None:
        logLHess = nd.Hessian(logL)
        
    def hess(x):
        result = -hess_2(x)
        result[1:, 1:] += logLHess(x[1:])
        return result
    
    # These arguments could lead to wrong results if passed on.
    # Hence, we ignore them.
    kwargs.pop("fun0", None)
    kwargs.pop("jac0", None)
    kwargs.pop("hess0", None)
    
    return find_CI(x0, fun, jac, hess, indices=0, **kwargs)
