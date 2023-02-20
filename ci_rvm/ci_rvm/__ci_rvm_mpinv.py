'''
'''
import numpy as np
import scipy.optimize as op
from scipy.stats import chi2
from scipy import linalg
import warnings
from matplotlib import pyplot as plt
import traceback
import sys
from functools import partial

try:
    from .ci_rvm import *
except ImportError:
    from ci_rvm import *


def is_negative_semidefinite(M, tol=1e-6, return_singular=False): #
    try:
        np.linalg.cholesky(-M)
        if return_singular:
            return True, False
        return True
    except np.linalg.LinAlgError:
        result = (np.linalg.eigh(M)[0] <= tol).all()
        if return_singular:
            return result, True
        return result

        
def find_profile_CI_bound(index, direction, x0, fun, jac, hess, alpha=0.95, 
                          fun0=None, *args, **kwargs):
    diff = chi2.ppf(alpha, df=1)/2
    
    if fun0 is None:
        fun0 = fun(x0)
    
    target = fun0-diff
    
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
    
    is_negative_semidefinite_ = partial(is_negative_semidefinite, tol=singtol)
    
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
            if fActual < target-resulttol and np.abs(fActual-target) > np.abs(target-f):
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
    free_params_exist: "True if the nuisance Hessian does not have full rank" = False
    nuisance_dim: "dimension of the nuisance vector" = x.size-1
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
            searchmode = "max_nuisance"
            negDefinite, singular = is_negative_semidefinite_(H_, return_singular=True)
            if negDefinite:
                if singular:
                    H_inv, rank = linalg.pinvh(H_, singtol/10, return_rank=True)
                    
                    # This should be true in genera; however, we use a different
                    # method to check for singularity here. Hence, we rather 
                    # double check that...
                    free_params_exist = rank == nuisance_dim
                else:
                    rank = nuisance_dim
                    free_params_exist = False
                    H_inv = np.linalg.inv(H_)
                    
                # in theory, the inverse of a negative definite matrix is also
                # negative definite. However, due to numerical issues this
                # may not hold here, if the matrix has unfavourable properties.
                # Therefore, we rather double check that...
                if is_negative_semidefinite_(H_inv):
                    searchmode = "normal"
                
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
                            and np.linalg.matrix_rank(H, singtol) == rank): 
                            #                                )[index]):
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
                    xTmp[considered] += -np.dot(H_inv, H0_*(xiTmp-xi)+J_)
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
                xTmp[considered] += -np.dot(H_inv, H0_*(xiTmp-xi)+J_)
                
                xx = x[considered]
                upperRadius = np.linalg.norm(xx-xTmp[considered])
                
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
                
                # if ignoring the free parameters hinders us from making an 
                # improvement w.r.t. the gradient
                abstol = max(np.log2(np.abs(xiTmp-xi))*np.abs(f-target)
                             /np.abs(maxFun-target), resulttol)
                #if (J_new > abstol and J_new/J_current > apprxtol):
                if (J_new > abstol and J_new/J_current > apprxtol):
                    searchmode = "max_nuisance"            
            
            counter = 0
            repeat = True
            while repeat:
                repeat = False
                
                if counter > 10:
                    raise RuntimeError("Infinite loop", f, J, H, considered)
                
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
                            xTmp[considered] += xTmp_diff
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
                                    
                                dispprint("iter {:3d}{}: ***discontinuity of {} in x_{} at {}***".format(
                                            i, ">" if forward else "<", f-fActual, 
                                            index, x))
                                break
                            
                            # check for discontinuities in nuisance parameters
                            discontinuities = np.zeros_like(considered)
                            discontinuities_rejected = discontinuities.copy()
                            if not f_approx_precise(fPredicted, fActual):
                                if diffs[index]:
                                    fPredicted = fPredicted_fun()
                                    fActual = fun(xTmp)
                                else:
                                    fPredicted = fActual = f
                                for l in np.nonzero(considered)[0]:
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
                                for l in np.nonzero(considered)[0]:
                                    xTmp[l] += diffs[l]
                                    JActual_previous = JActual
                                    JActual = jac(xTmp) 
                                    JPredicted = JPredicted_fun()
                                    jac_precise = jac_approx_precise(JPredicted[considered], JActual[considered], J[considered]) 
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
                    
                dispprint(("iter {:3d}{}: x_{}_d={}; f_d={}; jac_d={}; " + 
                       "nsteps={:2d}; x_d={}; f_impr={}; jac_impr={}; " +
                       "f_e={}{}{} - {}").format(i, 
                                                 ">" if forward else "<", index, 
                                                 __round_str(xi-x0[index]), 
                                                 __round_str(f-target), 
                                                 __round_str(norm(JActual_)), k+2, 
                                                 __round_str(xChange), 
                                                 __round_str(fImprovement), 
                                                 __round_str(JImprovement), 
                                                 __round_str(fActual-fPredicted), 
                                                 discontStr, radius_str, mode))
            
            
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
    
