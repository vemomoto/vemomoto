'''
Created on 18.09.2019

@author: Samuel
'''
import sys
from itertools import repeat, starmap
from multiprocessing import Pool
from collections import defaultdict
from functools import partial

import numpy as np
from scipy import optimize as op, integrate, sparse
from scipy.stats import chi2, nbinom

try:
    from matplotlib import pyplot as plt
except ModuleNotFoundError:
    raise ModuleNotFoundError("The package matplotlib must be installed to run the test.")

try:
    from numdifftools import Gradient, Hessian
    import autograd.numpy as ag
except ModuleNotFoundError:
    raise ModuleNotFoundError("This module requires numdifftools and autograd "
                              "for numerical and automatic differenciation. "
                              "The packages are available on the Python package"
                              "index.")

try:
    from .ci_rvm import find_CI_bound, CounterFun, find_CI as find_CI_new, find_function_CI
    from .__ci_rvm_mpinv import find_profile_CI_bound as find_profile_CI_bound_mpinv
except ImportError:
    from ci_rvm import find_CI_bound, CounterFun, find_CI as find_CI_new, find_function_CI
    from __ci_rvm_mpinv import find_profile_CI_bound as find_profile_CI_bound_mpinv

if __name__ == '__main__':
    try:
        # if vemomoto_core_tools is installed, you can call the module with an 
        # argument specifying a log file to which the output is written in addition
        # to the console.
        from vemomoto_core.tools.tee import Tee
        if len(sys.argv) > 1:
            teeObject = Tee(sys.argv[1])
    except ModuleNotFoundError:
        pass

np.random.seed()

CONVERT = True
ALPHA = 0.95
def convertPos(x, indices=None, convert=CONVERT):
    if not convert:
        return x
    x = x.copy()
    
    """
    res = x[indices]
    cons = res > 50
    if not cons.any():
        res[:] = np.log(np.exp(res)+1)
    else: 
        res2 = np.log(np.exp(res)+1)
        res2[cons] = res[cons]
        res = res2
    x[indices] = res
    return x
    """
    
    x[indices] = np.exp(x[indices])
    return x

def convertPos_(x, convert=CONVERT):
    if not convert:
        return x
    return np.exp(x) #!!!!!!!
    if x <= 50:
        return np.log(np.exp(x)+1)
    else: 
        return x

def convertPosInv(x, indices=None, convert=CONVERT):
    if not convert:
        return x
    x = x.copy()
    res = x[indices]
    """
    cons = res > 50
    res2 = np.log(np.exp(res)-1)
    if cons.any():
        res2[cons] = res[cons]
    """
    res2 = np.log(res) #!!!!!!!!
    res2[~np.isfinite(res2)] = -10 #000
    x[indices] = res2
    return x

def convertPosInv_(x, convert=CONVERT):
    if not convert:
        return x
    res = np.log(x)
    if not np.isfinite(res):
        res = -10
    return res
    
    
    if not convert or x > 50:
        return x
    res = np.log(np.exp(x)-1)
    if not np.isfinite(res):
        res = -10000
    return res

class TrackFun():
    def __init__(self, fun):
        self.track = []
        self.fun = fun
    def __call__(self, args):
        self.track.append(args.copy())
        return self.fun(args)


def venzon_moolgavkar(index, direction, x0, fun, jac, hess, target=None, alpha=0.95, 
                      fun0=None, hess0=None, nmax=200, epsilon=1e-6, disp=True,
                      track_x=False):
    
    if fun0 is None:
        fun0 = fun(x0)
    
    if target is None:
        diff = chi2.ppf(alpha, df=1)/2
        target = fun0-diff
    
    step_scale = direction
    
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
                                     jac=dl_dx_,
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

def wald(x, hess, direction=None, alpha=ALPHA):
    for _ in [None]:
        diff = np.nan
        try:
            hinv = np.linalg.inv(hess)
        except Exception:
            message = "H is singular"
            break
        try:
            diff = np.sqrt(-np.diag(hinv) * chi2.ppf(alpha, df=1))
        except Exception:
            message = "H is not negative definite"
            break
    else: 
        message = "success"
        
    success = message == "success"
    if not direction:
        if not success:
            return None, None
        return x-diff, x+diff
    elif direction==1: 
        res = x+diff
    else: 
        res = x-diff
    return op.OptimizeResult(x=res, 
                             fun=np.nan,
                             success=success, 
                             nfev=0, njev=0, nhev=1, nit=0,
                             x_track=[],
                             f_track=[],
                             message=message
                             )


def root(index, fun, x0, alpha=ALPHA, method="df-sane"): #method='df-sane'):
    x2 = np.delete(x0, index)
    fun0 = fun(x0)
    target = fun0-chi2.ppf(alpha, df=1)/2
    def PL(xi):
        x = x2
        fun_nuiscance = lambda xx: -fun(np.insert(xx, index, xi))
        res = op.minimize(fun_nuiscance, x)
        x[:] = res.x
        print(xi, -res.fun-target)
        return -res.fun-target
    fullres = op.root(PL, x0[index], method=method)
    fullres.x = np.insert(x2, fullres.x, index)

def binsearch(index, direction, x0, fun, jac=None, alpha=ALPHA, initstep=1, 
              resulttol=1e-3, infstep=1e10, stepn=200, checkNo=5):
    fun = CounterFun(fun)
    
    if jac is not None:
        jac = CounterFun(jac)
        jac_nuiscance = lambda x: -np.delete(jac(np.insert(x, index, xi)), index)
    else: 
        jac_nuiscance = None
    
    xi = x0[index] + direction*initstep
    xi0 = x0[index]
    fun_nuiscance = lambda x: -fun(np.insert(x, index, xi))
    fun0 = fun(x0)
    target = fun0-chi2.ppf(alpha, df=1)/2
    xi_above = xi0
    xi_below = None
    x = np.delete(x0, index)
    f_trace = [fun0]
    x_trace = [x0]
    checkit = checkNo
    
    for i in range(stepn):
        opres = op.minimize(fun_nuiscance, x, jac=jac_nuiscance)
        f = -opres.fun
        print("{}: x_{}_d={:6.3f}, f_d={:6.3f}".format(i, index, xi-xi0, f-target))
        
        if np.isnan(f):
            f = -np.inf
        
        f_trace.append(f)
        x_trace.append(np.insert(opres.x, index, xi))
        
        if checkit < 0 and f > target:
            checkit = checkNo
            xi_below = None
            
        if f < target:
            xi_below = xi
            checkit = checkNo
        else:
            xi_above = xi
            x = opres.x
            checkit -= 1
            
        if xi_below is not None:
            if np.allclose(xi_above, xi_below, atol=resulttol):
                message = "success"
                success = True
                break
            if checkit == 0:
                xi = xi_below
                checkit -= 1
            else:
                xi = (xi_above+xi_below) / 2
        else:
            if np.abs(xi-xi0) > infstep and f > target:
                message = "unbounded"
                success = True
                break
            initstep *= 10
            xi += direction*initstep
    else:
        message = "iteration limit exceeded"
        success = False
    
    result = op.OptimizeResult(x=np.insert(x, index, xi), 
                             fun=f,
                             success=success, 
                             nfev=fun.evals, 
                             njev=(0 if jac is None else jac.evals), 
                             nhev=0,
                             nit=i,
                             x_track=np.array(x_trace),
                             f_track=np.array(f_trace),
                             message=message
                             )
    #print(result)
    return result

def gridsearch(index, direction, x0, fun, jac, hess, step=0.2, alpha=ALPHA,
               resulttol=1e-3, stepn=200, infstep=1000):
    
    f = fun0 = fun(x0)
    target = fun0-chi2.ppf(alpha, df=1)/2
    
    fun = CounterFun(fun)
    xi = xi0 = x0[index]
    
    dim = x0.size
    A = sparse.dok_matrix((dim, dim))
    A[index, index] = 1
    target_fun = lambda p: -fun(p)
    def target_fun(p): 
        res = -fun(p)
        #print(res+target, p[index]-xi) #, np.round(p,2)
        return res
    target_jac = CounterFun(lambda p: -jac(p))
    target_hess = CounterFun(lambda p: -hess(p))
    x = x0
    bound0 = np.full_like(x0, -np.inf)
    bound1 = np.full_like(x0, np.inf)
    for i in range(stepn+1):
        if i == stepn:
            x_prev = x
            f_prev = f
            xi += direction*infstep
        else:
            xi += direction*step
        #bounds = (-np.inf, xi) if direction < 0 else (xi, np.inf)
        bound0[index] = bound1[index] = xi
        bounds = (bound0, bound1)
        constraints = op.LinearConstraint(A, *bounds) 
        res = op.minimize(target_fun, x, jac=target_jac, hess=target_hess, constraints=constraints,
                             options=dict(maxiter=500), tol=resulttol, method="trust-constr") #, gtol=resulttol
        f = -res.fun
        print("{}: x_{}_d={:6.3f}, f_d={:6.3f}".format(i, index, xi-xi0, f-target))
        if f <= target:
            if step > resulttol:
                xi -= direction*step
                step /= 2
            else:
                message = "Success"
                #xi -= direction*step/2
                success = True
                break
        else:
            x = res.x
    else:
        if f < target:
            message = "iteration limit exceeded"
            success = False
            x = x_prev
            f = f_prev
        else:
            message = "Result is unbounded"
            success = True
    result = op.OptimizeResult(x=x, 
                             fun=f,
                             success=success, 
                             nfev=fun.evals, 
                             njev=target_jac.evals, 
                             nhev=target_hess.evals,
                             nit=i,
                             message=message
                             )
    return result
    

def bisection(index, direction, x0, fun_, jac=None, alpha=ALPHA, initstep=1, 
              resulttol=1e-3, constrainedMin=True, infstep=1e10, stepn=200):
    fun = CounterFun(fun_)
    xi = x0[index] + direction*initstep
    xi0 = x0[index]
    
    jac_nuiscance = None
    if constrainedMin:
        fun_nuiscance = lambda x: -fun(x)
        if jac is not None:
            jac = CounterFun(jac)
            unit = np.eye(1, x0.size, index).ravel()
            #def jac_nuiscance(x):
            #    print(x)
            #    jj = -jac(x)
            #    return jj
            #jac_nuiscance = CounterFun(jac_nuiscance)
            jac_nuiscance = CounterFun(lambda x: -jac(x))
        constraints = dict(type="eq", fun=lambda x: x[index]-xi, 
                           jac=lambda x: unit)
        dim = x0.size
        A = sparse.dok_matrix((dim, dim))
        A[index, index] = 1
    else:
        constraints = None
        fun_nuiscance = lambda x: -fun(np.insert(x, index, xi))
        if jac is not None:
            jac = CounterFun(jac)
            jac_nuiscance = lambda x: -np.delete(jac(np.insert(x, index, xi)), index)
    
    fun0 = fun(x0)
    target = fun0-chi2.ppf(alpha, df=1)/2
    f_above = fun0
    xi_above = xi0
    f_below = None 
    xi_below = None
    f_trace = [fun0]
    xi_trace = [xi0]
    x_trace_full = [x0]
    if not constrainedMin:
        x_trace = [np.delete(x0, index)]
    else:
        x_trace = x_trace_full
    
    for i in range(stepn):
        #if constrainedMin:
        #    bounds = (-np.inf, xi) if direction < 0 else (xi, np.inf)
        #    constraints = op.LinearConstraint(A, *bounds) 
        x = x_trace[np.argmin(np.abs(xi-np.array(xi_trace)))]
        x[np.isnan(x)] = x_trace[0][np.isnan(x)]
        for j in range(20):
            try:
                opres = op.minimize(fun_nuiscance, x, jac=jac_nuiscance, 
                                    constraints=constraints)
                if not np.isnan(opres.fun):
                    break
            except Exception:
                pass
            xi = (xi + xi_above) / 2
        else:
            message = "Error in function or jacobian"
            success = False
            break
            
        x_trace.append(opres.x)
        if not constrainedMin:
            x_trace_full.append(np.insert(opres.x, index, xi))
        f = -opres.fun
        f_trace.append(f)
        xi_trace.append(xi)
        print("{}: x_{}_d={:6.3f}, f_d={:6.3f}".format(i, index, xi-xi0, f-target))
        if np.isnan(f):
            print("Nan")
        if (xi_below is not None and (np.allclose(f, target, atol=resulttol) or
            np.allclose(xi_above, xi_below, atol=resulttol))):
            message = "success"
            success = True
            break
        elif np.abs(xi-xi0) > infstep and f > target:
            message = "unbounded"
            success = True
            break
        two_point = np.sum(np.array(f_trace)>target) + (f_below is not None) < 3
        if f >= target and f_below is not None:
            if f_below > target:
                if f < f_below:
                    f_above = f_below
                    xi_above = xi_below
                else:
                    two_point = True
                f_below = f
                xi_below = xi
            else:
                f_above = f
                xi_above = xi
        else:
            if f_below is not None and f_below > target:
                f_above = f_below
                xi_above = xi_below
            f_below = f
            xi_below = xi
            
        if two_point:
            p = (fun0-f)/(xi0-xi-0.5/xi0*(xi0**2-xi**2))
            a = -p/(2*xi0)
            q = fun0-p*(xi0)-a*xi0**2
        else:
            m = np.array([[xi0**2, xi0, 1],
                          [xi_above**2, xi_above, 1],
                          [xi_below**2, xi_below, 1]])
            a, p, q = np.dot(np.linalg.inv(m), [fun0, f_above, f_below])
        
            
        root = p**2/(4*a**2)-(q-target)/a
        if root < 0:
            initstep *= 10
            xi += direction*initstep
        else:
            dirFact = 1 if a<0 else -1
            xi = -p/(2*a) + dirFact * direction * np.sqrt(root)
            if np.abs(xi-xi0) > infstep:
                xi = direction*infstep*1.1
    else:
        message = "iteration limit exceeded"
        success = False
    result = op.OptimizeResult(x=x_trace_full[-1], 
                             fun=f,
                             success=success, 
                             nfev=fun.evals, 
                             njev=(0 if jac is None else jac.evals), 
                             nhev=0,
                             nit=i,
                             x_track=np.array(x_trace_full),
                             f_track=np.array(f_trace),
                             message=message
                             )
    print(result.x, result.fun)
    return result

def constrained_max(index, direction, x0, fun, jac=None, alpha=ALPHA, nit=500):
    fun0 = fun(x0)
    print("start", index, direction)
    
    target = fun0-chi2.ppf(alpha, df=1)/2
    preconditioning = 0.001
    target_fun = lambda x: -direction*x[index]
    unit = -np.eye(1, x0.size, index).ravel()*direction
    target_jac = lambda x: unit
    constraints = dict(type="eq", fun=lambda x: (fun(x)-target)*preconditioning, 
                       jac=lambda x: jac(x)*preconditioning)
    
    try:
        result = op.minimize(target_fun, x0, jac=target_jac, constraints=constraints,
                             options=dict(maxiter=500))
    except Exception:
        return op.OptimizeResult(fun=np.nan,
                                 success=False,
                                 x=x0,
                                 nit=0)
    print("done", index, direction)
    return result
    
    
def mixed_min(index, direction, x0, fun, jac=None, alpha=ALPHA):
    
    preconditioning = 0.01
    fun0 = fun(x0)
    target = fun0-chi2.ppf(alpha, df=1)/2
    f = lambda x: (2*np.square(fun(x)-target) - direction*x[index])*preconditioning
    
    
    if jac is not None:
        unit = np.eye(1, x0.size, index).ravel()
        j = lambda x: (4*jac(x)*(fun(x)-target) - direction*unit)*preconditioning
    else:
        j = None
    #j = Gradient(f, step=1e-7)
    result = op.minimize(f, x0, jac=j) #, method='trust-exact')
    print("index {:2d}: f_diff={:7.3f} (direction {})".format(index, fun(result.x)-target, direction))
    result.fun = fun(result.x)
    
    print(target-result.fun)
    #print(result)
    return result

def fixedFun(f, origArg, flex):
    def fun(x):
        xx = origArg.copy().astype(x.dtype)
        xx[flex] = x
        return f(xx)
        
    if isinstance(f, CounterFun):
        fun.evals = lambda: f.evals
    return fun

class BaseTester():
    def __init__(self):
        self.ML = None
        self._MLE = None
        self.x = None
        self.dataN = None
        self.data = None
        self.convert = False
        self.convertIndices = None
        self.fixed = None
        self.dim = None
        self.test_prediction = False
    
    @property
    def MLE(self):
        if self.fixed is None:
            return self._MLE
        return np.delete(self._MLE, self.fixed)
        
    @MLE.setter
    def MLE(self, val):
        self._MLE = val
        
    def convertPos(self, x):
        return convertPos(x, self.convertIndices, self.convert) 
    def convertPosInv(self, x):
        return convertPosInv(x, self.convertIndices, self.convert) 
    
    @property
    def funs(self):
        return self.get_funs()
        
    def get_funs(self):
        raise NotImplementedError("get_funs has not been implemented yet")


class DynamicalSystemTester(BaseTester):
    def __init__(self, *sim_args, **sim_kwargs):
        
        BaseTester.__init__(self)
        
        self.convert = CONVERT
        self.dim = 8
        self.simulate(*sim_args, **sim_kwargs)
    
    @staticmethod
    def rss_dyn(f, t, y, dim):
        tmax = np.max(t)
        def ff(p):
            fp = lambda t, x: f(t, x, p[dim:])
            try:
                res = np.sum(np.square(
                    integrate.solve_ivp(fp, (0, tmax), p[:dim], rtol=1e-9, atol=1e-11, dense_output=True, vectorized=True).sol(t) - y))
            except Exception:
                return 1000000
            #print(res)
            return res
        return ff
    
    @staticmethod
    def f_constr(f, t, y, dim, fact, target, tol):
        tmax = np.max(t)
        tt = np.linspace(0, tmax, 1000)
        def ff(p):
            fp = lambda t, x: f(t, x, p[dim:-1])
            try:
                sol = integrate.solve_ivp(fp, (0, tmax), p[:dim], rtol=1e-9, atol=1e-11, dense_output=True, vectorized=True)
                res = fact*np.sum(np.square(sol.sol(t) - y)) + (p[-1]-np.mean(sol.sol(tt)[0]))**2/tol**2*target
            except Exception:
                res = 1000000
            #print(res)
            return res
        return ff
    
    @staticmethod
    def f_prime(t, x, params):
        r, K, a, h, c, d = params
        x, y = x
        conversion = a*y*x/(1+h*x)
        return np.array((r*x*(1-x/K) - conversion, 
                         (c*conversion-y*d)))
    
    def get_rss(self):
        rss_ = DynamicalSystemTester.rss_dyn(DynamicalSystemTester.f_prime, *self.data, 2)
        return lambda p: rss_(convertPos(p))
    
    def get_funs(self):
        rss = self.get_rss()  
        fact = self.fact
        f_ = lambda x: fact*rss(x)
        if self.fixed is not None:
            MLE = self._MLE
            flex = np.ones_like(MLE, dtype=bool)
            flex[self.fixed] = False
            f_ = fixedFun(f_, MLE, flex)
        f = CounterFun(f_)
        g = Gradient(f, step=1e-7, method='complex')
        h = Hessian(f, step=1e-7, method='complex')
        return f, g, h
    
    def get_funs_constr(self):
        varMLE = self.ML / (self.dataN*2)
        fact = -0.5/varMLE
        dim = 2
        f = DynamicalSystemTester.f_constr(DynamicalSystemTester.f_prime, 
                                           *self.data, dim, fact, 
                                           chi2.ppf(.95, df=1)/2, 1e-3)
        g = Gradient(f, step=1e-8, method='complex')
        h = Hessian(f, step=1e-8, method='complex')
        t = self.data[0]
        tmax = np.max(t)
        p = self.MLE
        sol = integrate.solve_ivp(DynamicalSystemTester.f_prime, (0, tmax), p[:dim], rtol=1e-9, atol=1e-11, dense_output=True, vectorized=True)
        tt = np.linspace(0, tmax, 1000)
        self.cMLE = np.append(self.MLE, np.mean(sol.sol(tt)[0]))
        
        return f, g, h
        
        
    def simulate(self, params=None, dataN=50, tEnd=20, std=3, plot=True):
        
        if params is None:
            params = np.array((50, 25, 2, 200, 0.1, 0.01, 0.3, 1))
            
        print("simulating data")
        
        t = np.linspace(0, tEnd, dataN)
        paramsFull = params
        x0, params = params[:2], params[2:]
        res = integrate.solve_ivp(partial(self.f_prime, params=params), 
                                  (0, tEnd), x0, dense_output=True, 
                                  vectorized=True)
        x = np.maximum(res.sol(t) + np.random.randn(*t.shape)*std, 0)
        self.data = [t, x]
        #print(x[:,-1])
        rss = self.get_rss()   
        print("creating likelihood function")
        def rsss(p):
            #print(convertPos(p))
            res = rss(p)
            print(res)
            return res
        self.params = convertPosInv(paramsFull)
        
        print(rss(self.params))
        g = Gradient(rss, method='complex') #Gradient(rss, 1e-6, num_steps=4)
        h = Hessian(rss, method='complex')
        fit = op.minimize(rsss, self.params, jac=g, hess=h) #, options=dict(maxiter=1))
        print(fit)
        xMLE, paramsMLE = fit.x[:2], fit.x[2:]
        
        varMLE = fit.fun / dataN**2
        self.fact = -0.5/varMLE
        
        self.ML = fit.fun * self.fact
        self.MLE = fit.x
        self.x = paramsFull
        self.dataN = dataN
        if plot:
            print("plotting")
            tt = np.linspace(0, tEnd, tEnd*100)
            resHat = integrate.solve_ivp(partial(self.f_prime, params=convertPos(paramsMLE)), 
                                         (0, tEnd), convertPos(xMLE), dense_output=True, 
                                         vectorized=True)
            for y in res.sol(tt):
                plt.plot(tt, y)
            for y in resHat.sol(tt):
                plt.plot(tt, y)
            for y in x:
                plt.plot(t, y, '.')
            plt.show()
        
        
class LogRegressTester(BaseTester):
    def __init__(self, *sim_args, seed=None, **sim_kwargs):
        BaseTester.__init__(self)
        np.random.seed(seed)
        self.simulate(*sim_args, **sim_kwargs)
    
    def get_funs(self):
        covariates, switch = self.data
        if self.powers:
            covariateN = self.covariateN
            #f = lambda p: (-ag.sum(ag.log(1+ag.exp(-ag.sum(p[:covariateN]*covariates[switch:]**p[covariateN:-1], 1)-p[-1])))
            #               -ag.sum(ag.log(1+ag.exp(ag.sum(p[:covariateN]*covariates[:switch]**p[covariateN:-1], 1)+p[-1]))))
            f_ = lambda p: (-ag.sum(ag.log(1+ag.exp(-ag.sum(p[:covariateN]*ag.power(covariates[switch:],p[covariateN:-1]), 1)-p[-1])))
                           -ag.sum(ag.log(1+ag.exp(ag.sum(p[:covariateN]*ag.power(covariates[:switch],p[covariateN:-1]), 1)+p[-1]))))
            def f_(p): 
                p = self.convertPos(p)
                return (-ag.sum(ag.log(1+ag.exp(-ag.sum(p[:covariateN]*ag.power(covariates[switch:],p[covariateN:-1]), 1)-p[-1])))
                        -ag.sum(ag.log(1+ag.exp(ag.sum(p[:covariateN]*ag.power(covariates[:switch],p[covariateN:-1]), 1)+p[-1]))))
        else:
            f_ = lambda p: (-ag.sum(ag.log(1+ag.exp(-ag.sum(covariates[switch:]*p[:-1], 1)-p[-1])))
                           -ag.sum(ag.log(1+ag.exp(ag.sum(covariates[:switch]*p[:-1], 1)+p[-1]))))
        
        
        if self.test_prediction:
            testData = self.testData
            tol = 0.001
            target = 2
            f = lambda p: (f_(p[:-1]) - 
               (p[-1]-np.mean(1/(1+np.exp(
                   -np.sum(p[:covariateN]*testData**p[covariateN:-2], 1)
                   -p[-2]))))**2/tol**2*target)
            def f(p): 
                diff = (p[-1]-np.mean(1/(1+np.exp(
                   -np.sum(p[:covariateN]*testData**p[covariateN:-2], 1)
                   -p[-2]))))
                res = f_(p[:-1]) - (diff/tol)**2*target
                if isinstance(diff, float):
                    #print(diff)
                    pass
                return res
        else:
            f = f_
            
        f = CounterFun(f)
        
        if self.fixed is not None:
            MLE = self._MLE
            flex = np.ones_like(MLE, dtype=bool)
            flex[self.fixed] = False
            f = fixedFun(f, MLE, flex)
            g = Gradient(f, method='complex', step=1e-7)
            h = Hessian(f, method='complex', step=1e-7)
        else:
            #g = AgGradient(f)
            #h = AgHessian(f)
            g = Gradient(f, method='complex', step=1e-7)
            h = Hessian(f, method='complex', step=1e-7, num_steps=1)
        return f, g, h
    
    def simulate(self, params=None, dataN=1000, testDataN=0, powers=True, mode="3"):
        
        print("simulating data")
        
        if params is None:
            if mode=="3":
                params = np.array([5, 0.5, -10])
            elif mode=="11":
                params = np.array([5, 2, -1, -3, -2, 0.2, 1, 0.1, 0.2, 0.5, -1])
            elif mode=="11cx":
                params = np.array([0.8, 0.2, -0.6, -1, -1, 0.2, 0.5, 0.1, -0.2, 0.2, 2])
                powers = False
        if powers:
            covariateN = (len(params)-1)//2
        else:
            covariateN = len(params)-1
        
        self.dataN = dataN
        self.dim = len(params)
        if powers:
            self.convert = True
            self.convertIndices = slice((self.dim-1)//2, -1)
        allDataN = testDataN + dataN
        
        covariates = np.zeros((allDataN, covariateN), dtype=int)
        m = 5
        p = 0.5
        p2 = 0.2
        for i in range(covariateN):
            if not i % 2:
                covariates[:,i] = np.random.negative_binomial(m, p, allDataN)
            else:
                covariates[:,i] = np.random.binomial(covariates[:,i-1], p2, allDataN) #+ np.random.poisson(0.5, allDataN)
        
        covariates = covariates+1e-16
        
        self.testData = covariates[dataN:]
        covariates = covariates[:dataN]
        
        if powers:
            data = np.random.binomial(1, 1/(1+np.exp(-np.sum(params[:covariateN]*covariates**params[covariateN:-1], 1)-params[-1])))
        else:
            data = np.random.binomial(1, 1/(1+np.exp(-np.sum(covariates*params[:-1], 1)-params[-1])))
        print(data.mean())
        if powers:
            print(np.round(1/(1+np.exp(-np.sum(params[:covariateN]*covariates**params[covariateN:-1], 1)-params[-1])),2))
        else:
            print(np.round(1/(1+np.exp(-np.sum(params[:covariateN]*covariates, 1)-params[-1])),2))
        self.covariateN = covariateN
        self.powers = powers
        
        order = np.argsort(data)
        covariates = covariates[order]
        data = data[order]
        
        switch = np.argmax(data)
        
        self.data = [covariates, switch]
        
        f, j, h = self.funs
        
        print(f(self.convertPosInv(params)))
        ff = lambda p: -f(p)
        jj = lambda p: -j(p)
        hh = lambda p: -h(p)
        fit = op.minimize(ff, self.convertPosInv(params), jac=jj)
        fit = op.minimize(ff, fit.x, jac=jj, hess=hh, method='trust-exact')
        print(j(params))
        print(fit)
        print(params)
        
        self.test_prediction = testDataN
        if self.test_prediction:
            testP = 1/(1+np.exp(-np.sum(params[:covariateN]*self.testData**params[covariateN:-1], 1)-params[-1]))
            testPMLE = 1/(1+np.exp(-np.sum(fit.x[:covariateN]*self.testData**fit.x[covariateN:-1], 1)-fit.x[-1]))
            self.x = np.append(params, np.mean(testP))
            self.MLE = np.append(fit.x, np.mean(testPMLE))
        else:
            self.x = params
            self.MLE = fit.x
        self.ML = -fit.fun
    
    
    def simulate_(self, params=None, dataN=1000, powers=True):
        
        if params is None:
            params = 5
        if isinstance(params, int):
            params = np.random.randn(params)
            
        self.dim = len(params)
        if powers:
            covariateN = (len(params)-1)//2
        else:
            covariateN = len(params)-1
        if powers:
            params[covariateN:-1] = np.random.rand(covariateN)
        
        print("simulating data")
        cov = np.random.randn(covariateN, covariateN)+10
        cov = np.dot(cov, cov.T)
        
        
        m = np.random.randn(covariateN)
        
        covariates = np.random.multivariate_normal(m, cov, dataN)/50
        #covariates[:,1] = covariates[:,0] + np.random.rand(dataN)*0.01
        
        if powers:
            shift = 0.01 - np.min(covariates)
            params[-1] -= shift
            covariates += shift
            data = np.random.binomial(1, 1/(1+np.exp(-np.sum(params[:covariateN]*covariates**params[covariateN:-1], 1)-params[-1])))
        else:
            data = np.random.binomial(1, 1/(1+np.exp(-np.sum(covariates*params[:-1], 1)-params[-1])))
        
        self.covariateN = covariateN
        self.powers = powers
        
        order = np.argsort(data)
        covariates = covariates[order]
        data = data[order]
        
        switch = np.argmax(data)
        
        self.data = [covariates, switch]
        
        f, j, h = self.funs
        
        print(f(params))
        ff = lambda p: -f(p)
        
        self.x = params
        print(j(params))
        fit = op.minimize(ff, params, jac=j, hess=h, method='trust-exact')
        print(fit)
        print(params)
        
        self.ML = -fit.fun
        self.MLE = fit.x

class H14Tester(BaseTester):
    
    def __init__(self, fixed=[0], vals=[25]): #fixed=[0, 1, 2], vals=[25, 0, 0]):
        BaseTester.__init__(self)
        self.convert = CONVERT
        self.dim = 7-len(fixed)
        
        r_source = robjects.r['source']
        print("Executing R script")
        robjects.r('setwd("test_CI/Histone_H1")')
        r_source("get_f.R")
        #x0 = np.array([0.02, 0., 0, 0.090083433, 0.015029212, 0.001207238, 0.002303329])
        x0 = np.array([0, 0.03429, 0, 0.37168, 0.05365, 0, 0.00794])
        x0 = np.array([0, 0.04024, 0.00794, 0.36572, 0.05365, 0, 0 ])
        x0 = np.array([0, 0, 0.00794, 0.40596, 0.04912, 0.00453, 0 ])
        x0 = np.array([0, 0, 0, 4.05836e-01, 4.91267e-02, 3.79864e-03, 8.67439e-03])
        #x0 = np.array([0] + [1e-3]*6)
        if fixed is not None:
            fixed = np.array(fixed, dtype=int)
            fixed.sort()
            insertIndices = fixed - np.arange(fixed.size)
            x0[fixed] = vals
        x0 = convertPosInv(x0)
        ff = robjects.r['f']
        vals = x0[fixed]
        x0 = np.delete(x0, fixed)
        def f(x):
            if fixed is not None and len(fixed):
                x = np.insert(x, insertIndices, vals)
            try:
                return -ff(robjects.FloatVector(convertPos(x)))[0]
            except Exception: #rinterface.RRuntimeError:
                return -np.inf
        
        fun2 = lambda x: -f(x)
        
        j = Gradient(fun2, 1e-5, num_steps=2)
        h = Hessian(fun2, 1e-5, num_steps=2)
        
        opres = op.minimize(fun2, x0, jac=j, hess=h, method='trust-exact')
        print(opres)
        n = robjects.r['dataN'][0]
        var = opres.fun / n
        fact = 0.5/var
        self.constVals = (insertIndices, vals)
        
        self.MLE = opres.x
        self.ML = -fact * opres.fun 
        self.fact = fact
        self.x = convertPos(opres.x)
        
        print("done", self.x, self.ML, self.fact)
        
    def get_funs(self):
        
        insertIndices, vals = self.constVals
        from rpy2 import robjects
        r_source = robjects.r['source']
        print("Executing R script")
        #robjects.r('setwd("test_CI/Histone_H1")')
        r_source("get_f.R")
        ff = robjects.r['f']
        
        def f(x):
            if insertIndices is not None and len(insertIndices):
                x = np.insert(x, insertIndices, vals)
            try:
                return -ff(robjects.FloatVector(convertPos(x)))[0]
            except Exception: #rinterface.RRuntimeError:
                return -np.inf
        
        fact = self.fact
        
        fun = CounterFun(lambda x: fact*f(x))
        
        if self.fixed is not None:
            MLE = self._MLE
            flex2 = np.ones_like(MLE, dtype=bool)
            flex2[self.fixed] = False
            fun = fixedFun(fun, MLE, flex2)
        
        j = Gradient(fun, 1e-5, num_steps=2)
        h = Hessian(fun, 1e-5, num_steps=2)
    
        return fun, j, h        
        
def find_CI(tester, index, direction, method="RVM", f_track=False):
    nmax = 600
    infstep = 1e3
    resulttol=1e-5
    if method == "Wald":
        tmpPos = tester.convertPos
        tester.convertPos = False
    f_, g, h = tester.funs
    if method == "Wald":
        tester.convertPos = tmpPos 
    
    if f_track:
        f = TrackFun(f_)
    else:
        f = f_
    try:
        if method == "RVM":
            res = find_CI_bound(tester.MLE, f, index, direction, 
                                g, h, nmax=nmax, apprxtol=0.5,
                                alpha=ALPHA, disp=False,
                                infstep=infstep)
        elif method == "RVM_psI":
            res = find_profile_CI_bound_mpinv(index, direction, tester.MLE,
                                         f, g, h, nmax=nmax, apprxtol=0.5,
                                         alpha=ALPHA, disp=False, infstep=infstep)
        elif method == "VM":
            res = venzon_moolgavkar(index, direction, tester.MLE,
                                         f, g, h, nmax=nmax,
                                         alpha=ALPHA, disp=False)
        elif method == "bisection":
            res = bisection(index, direction, tester.MLE, f, g, stepn=nmax, 
                            infstep=infstep, resulttol=resulttol)
        elif method == "binsearch":
            res = binsearch(index, direction, tester.MLE, f, g, stepn=nmax, 
                            infstep=infstep, resulttol=resulttol)
        elif method == "gridsearch":
            res = gridsearch(index, direction, tester.MLE, f, g, h, stepn=nmax,
                             resulttol=resulttol)
        elif method == "Wald":
            res = wald(tester.MLE, h(tester.convertPos(tester.MLE)), direction)
            return res, 0
        elif method == "mixed_min":
            res = mixed_min(index, direction, tester.MLE, f, g)
        elif method == "constrained_max":
            res = constrained_max(index, direction, tester.MLE, f, g)
        else:
            raise ValueError("Method {} unknown.".format(method))
        if f_track:
            return res, f.track
        return res, f_.evals
    except Exception:
        return op.OptimizeResult(x=tester.MLE*np.nan, 
                             fun=np.nan,
                             success=False, 
                             nfev=f_.evals, 
                             njev=0, 
                             nhev=0,
                             nit=0,
                             x_track=np.array(tester.MLE),
                             f_track=np.array(tester.ML),
                             message="Error when computing the CI."
                             ), f_.evals

def find_CIs(tester, method="RVM", indices=None, printCI=True, 
             multiprocessing=True):
    print("Using method", method)
    if indices is None:
        n = tester.dim
        indices = range(n)
    else:
        n = len(indices)
    
    #results = list(starmap(find_CI, zip(repeat(tester), list(range(4, dim))*2, [1]*dim + [-1]*dim, repeat(method))))
    args = zip(repeat(tester), list(indices)*2, [-1]*n + [1]*n, repeat(method))
    #args = zip([tester], [10], [1], repeat(method))
    
    
    if multiprocessing:
        with Pool() as pool:
            results = pool.starmap(find_CI, args)
    else:
        results = starmap(find_CI, args)
    
    
    results = list(results)
    
    results2 = []
    
    target = tester.ML-chi2.ppf(ALPHA, df=1)/2
    tol = 1e-3
    if printCI:
        print("============== {:<15}================".format(method))
    for i, j in enumerate(indices):
        result1, feval1 = results[i]
        result2, feval2 = results[i+n]
        left = tester.convertPos(result1.x)[j]
        right = tester.convertPos(result2.x)[j]
        leftOriginal = result1.x[j]
        rightOriginal = result2.x[j]
        if printCI:
            print("[{:14.5f}{}, {:9.5f}, {:14.5f}{}], ({:4d}, {:4d}), ({:6d}, {:6d})".format(
                left, "*" if not result1.success else " ",
                tester.x[j], right, 
                "*" if not result2.success else " ", result1.nit, result2.nit,
                feval1, feval2))
            
            if method=="RVM" and (result1.fun < target-tol or result2.fun < target-tol):
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                print(result1.fun -(target-tol), result2.fun - (target-tol))
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            
        results2.append([left, right, leftOriginal, rightOriginal, result1.fun >= target-tol, 
                         result2.fun >= target-tol, result1.success, 
                         result2.success, feval1, feval2])
    if printCI:
        print("==============================================")
        print()
            
    return results2

def test_dynamical_system():
    #                      r, K,   a,    h,     c,  d
    p = np.array((50, 25, 2, 200, 0.1, 0.1, 0.3, 1))
    tester = DynamicalSystemTester(None) #p, std=3)
    find_CIs(tester, "mixed_min")
    find_CIs(tester, "RVM")
    find_CIs(tester, "constrained_max")
    find_CIs(tester, "Wald")
    find_CIs(tester, "binsearch")
    find_CIs(tester, "bisection")
    return

def test_LogRegress(methods):
    tester = LogRegressTester(dataN=500)
    #find_CI(tester, 3, 1, "constrained_max")
    for m in methods:
        find_CIs(tester, m)

def test_LogRegress_pred():
    tester = LogRegressTester(testDataN=10)
    indices = [tester.dim]
    find_CIs(tester, "constrained_max", indices)
    find_CIs(tester, "mixed_min", indices)
    find_CIs(tester, "Wald", indices)
    find_CIs(tester, "RVM", indices)
    find_CIs(tester, "binsearch", indices)
    find_CIs(tester, "bisection", indices)
    return

def test_H14():
    tester = H14Tester()
#     find_CIs(tester, "RVM")
#     find_CIs(tester, "RVM_psI")
#     find_CIs(tester, "constrained_max")
#     find_CIs(tester, "mixed_min")
#     find_CIs(tester, "binsearch")
#     find_CIs(tester, "bisection")
#     find_CIs(tester, "gridsearch")
#     find_CIs(tester, "Wald")
    find_CIs(tester, "VM")
    return

def create_plot(tester, considered, additional=[], methods=["RVM"], fileName=None):
    
    print("Creating figure for", tester, considered, additional)
    
    fixed = list(range(tester.dim))
    for i in considered:
        fixed.remove(i)
    for i in additional:
        fixed.remove(i)
    
    if methods == ["RVM"]:
        rvmplot = True
    
    consIndices = np.sort(np.concatenate((considered, additional)))
    index = np.searchsorted(consIndices, considered[0])
    ind2 = np.searchsorted(consIndices, considered[1])
    plotinds = np.array([index, ind2])
    flex = []
    for i in additional:
        flex.append(np.searchsorted(consIndices, i))
    flex = np.array(flex)
    
    tester.fixed = fixed
    fun = tester.funs[0]
    
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf
    x, y = [], []
    x2, y2 = [], []
    strings = []
    for method in methods:
        r1, track1 = find_CI(tester, index, -1, method, f_track=True)
        r2, track2 = find_CI(tester, index, 1, method, f_track=True)
        xmax = max(xmax, tester.convertPos(r2.x)[considered[0]], tester.convertPos(r1.x)[considered[0]])
        xmin = min(xmin, tester.convertPos(r2.x)[considered[0]], tester.convertPos(r1.x)[considered[0]])
        ymax = max(ymax, tester.convertPos(r2.x)[considered[1]], tester.convertPos(r1.x)[considered[1]])
        ymin = min(ymin, tester.convertPos(r2.x)[considered[1]], tester.convertPos(r1.x)[considered[1]])
        strings.append(method + ": [{:6.3f}, {:6.3f}]".format(r1.x[considered[0]], r2.x[considered[0]]))
        xx, yy = tester.convertPos(np.concatenate((track1[::-1], track2)).T)[considered]
        x.append(xx)
        y.append(yy)
        if rvmplot:
            xx2, yy2 = tester.convertPos(np.concatenate((r1.x_track[::-1], r2.x_track)).T)[considered]
            x2.append(xx2)
            y2.append(yy2)
    for s in strings:
        print(s)
    
    if rvmplot:
        xmax = min(np.max(x[0]), 99)
        xmin = np.min(x[0])
        ymax = min(np.max(y[0]), 99)
        ymin = np.min(y[0])
        
        xOrder = 10**np.floor(np.log10(xmax))/2
        yOrder = 10**np.floor(np.log10(ymax))
        xmax = (xmax // xOrder + 1) *xOrder
        xmin = max((xmin // xOrder) * xOrder, -10)
        ymax = (ymax // yOrder + 1) *yOrder
        ymin = (ymin // yOrder) * yOrder
        
        if tester.dataN >= 10000:
            xmin, xmax = 0, 11
        
    else:
        xdiff = xmax-xmin
        ydiff = ymax-ymin
        
        xmax += 0.1*xdiff
        xmin -= 0.1*xdiff
        ymax += 0.1*ydiff
        ymin -= 0.1*ydiff
    #print(np.arctan(r1.x[index])/np.pi+0.5, np.arctan(r2.x[index])/np.pi+0.5)
    
    if not (r1.success and r2.success):
        #raise Exception("no success")
        pass
    
    #x, y = tester.convertPos(np.concatenate((r1.x_track[::-1], r2.x_track))).T[considered]
    """
    xmax = np.max(x[0])
    xmin = np.min(x[0])
    ymax = np.max(y[0])
    ymin = np.min(y[0])
    """
    #xmin, xmax = -100, 2
    #ymin, ymax = -2, 2
    #ymax = np.mean(y)
    #xmax = np.mean(x)
    
    
    #xmin = max(0, xmin)
    #ymin = max(0, ymin)
    plotn = 200
    XX = np.linspace(xmin, xmax, plotn)
    YY = np.linspace(ymin, ymax, plotn)
    X, Y = np.meshgrid(XX, YY)
    
    Z = np.zeros_like(X)
    
    x0 = tester.MLE
    target = tester.ML-chi2.ppf(ALPHA, df=1)/2
    print(target, tester.ML)
    figsize = (3.5, 3)
    rect = [0.15, 0.18, 0.75, 0.78]
    plt.figure(figsize=figsize).add_axes(rect)
    if rvmplot:
        for i, xx2 in enumerate(XX):
            print(i)
            for j, yy2 in enumerate(YY):
                if tester.convert and (xx2<0 or yy2<0):
                    Z[j, i] = np.nan 
                else:
                    if additional:
                        x02 = x0.copy()
                        x02 = tester.convertPos(x02)
                        x02[index] = xx2
                        x02[ind2] = yy2
                        x02 = tester.convertPosInv(x02)
                        #x02[flex] = -x02[ind2]
                        ff = fixedFun(lambda x: -fun(x), x02, flex)
                        #x03 = op.basinhopping(ff, x02[flex], 10).x
                        if len(additional) == 1:
                            Z[j, i] = -op.minimize_scalar(ff, bounds=(-300, 100), method="bounded").fun
                        else:
                            Z[j, i] = -op.minimize(ff, x02).fun
                        #Z[j, i] = -op.minimize(ff, x02[flex]).fun
                    else:
                        Z[j, i] = fun(tester.convertPosInv(np.array([xx2, yy2])))
        Z[np.isnan(Z)] = -np.inf
        plt.pcolormesh(X, Y, np.maximum(Z, tester.ML-50), shading='gouraud') #cmap=cmap) #, 
    #"""
    #plt.plot(xx, yy, marker = 'x', color='C5')
    #plt.plot(x, y, marker = 'o', color='C1')
    markers = ["o", "v", "s", "h", "D"]
    colors = ["C1", "C0", "C2", "C3", "C4", "C5"]
    plt.contour(X, Y, Z, levels=[target]) #cmap=cmap) #, 
    if rvmplot:
        for xx, yy, marker, color, method in zip(x2, y2, markers, colors, methods):
            plt.plot(xx, yy, marker=marker, color=color)
        
        
    for xx, yy, marker, color, method in zip(x, y, markers, colors, methods):
        opacity = 1 
        if not rvmplot:
            if method=="RVM": 
                color = 'k'
            else: 
                opacity = 0.5
            markersize = 5
        else:
            markersize = 3
        plt.plot(xx, yy, marker=marker, color=color, alpha=opacity, linewidth=0.3, 
                 label=method, markersize=markersize)
            
    plt.plot(*tester.convertPos(x0)[plotinds], marker = 'o', color="C3", label="MLE", linewidth=0)
    
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    if rvmplot:
        plt.xlabel(r"$\beta_1$")
        plt.ylabel(r"$\alpha_1$")
        plt.locator_params(nticks=3, nbins=3)
    else:
        plt.xlabel(r"$x_{}$".format(considered[0]))
        plt.ylabel(r"$x_{}$".format(considered[1]))
        plt.legend()
    
    #print(xx)
    #print(yy)
    
    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                   linewidth=0, antialiased=True)
    if rvmplot:
        fileName = "rvm" + str(tester.dataN)
    if fileName:
        plt.savefig(fileName+".png", dpi=1000)
        plt.savefig(fileName+".pdf", dpi=1000)
    if not fileName or rvmplot:
        plt.show()

def benchmark_(methods, simkwargs):
    tester = LogRegressTester(**simkwargs)
    return [find_CIs(tester, method, printCI=True,
                     multiprocessing=False) for method in methods]

def benchmark(methods, nsim=100, **simkwargs):
    print("Benchmarking {} realizations.".format(nsim))
    print("simargs:", simkwargs)
    
    dtype = [("left", float), ("right", float), ("leftOriginal", float), 
             ("rightOriginal", float), ("leftAdmissible", bool), 
             ("rightAdmissible", bool), ("leftSuccess", bool), 
             ("rightSuccess", bool), ("leftEvals", int), ("rightEvals", int)]
    
    stats = defaultdict(lambda: dict(success=[], error=[], evals=[], 
                                     errorTot=[], evalsTot=[],
                                     errorReduced=[], largeErrors=[],
                                     largeErrorsTot=[]))
    
    bound = 1000
    tol = 1e-3
    rtol = 0.05
    with Pool() as pool:
        for results_ in pool.starmap(benchmark_, zip(repeat(methods), repeat(simkwargs, nsim)),
                                     chunksize=1):
            results = np.core.records.fromarrays(np.array(results_).T, dtype=dtype)
            
            lefts = np.ma.array(results["left"], mask=~results["leftAdmissible"])
            rights = np.ma.array(results["right"], mask=~results["rightAdmissible"])
            
            results["left"] = np.maximum(results["left"], -bound)
            results["right"] = np.minimum(results["right"], bound)
            
            
            trueLeft = np.min(lefts, 1)
            trueLeft[trueLeft.mask] = np.nan
            trueLeft = trueLeft.data
            trueRight = np.max(rights, 1)
            trueRight[trueRight.mask] = np.nan
            trueRight = trueRight.data
            
            for method, result in zip(methods, results.T):
                methodStats = stats[method]
                methodStats["success"].extend(
                    (result["leftSuccess"] & np.isclose(result["left"], trueLeft, rtol=rtol, atol=tol))
                     | ((result["leftOriginal"] <= -bound) & result["leftAdmissible"]))
                methodStats["success"].extend(
                    (result["rightSuccess"] & np.isclose(result["right"], trueRight, rtol=rtol, atol=tol))
                     | ((result["rightOriginal"] >= bound) & result["rightAdmissible"]))
                leftError = np.abs(result["left"]-trueLeft)
                rightError = np.abs(np.abs(result["right"]-trueRight))
                methodStats["error"].extend(leftError[result["leftSuccess"]])
                methodStats["error"].extend(rightError[result["rightSuccess"]])
                methodStats["errorTot"].extend(leftError[trueLeft > -bound])
                methodStats["errorTot"].extend(rightError[trueRight < bound])
                largeLeftError = ((leftError > 10) & result["leftSuccess"]) | ((trueLeft <= -bound) & (result["left"] > 10-bound))
                largeRightError = ((rightError > 10) & result["rightSuccess"]) | ((trueRight >= bound) & (result["right"] < bound-10))
                largeLeftErrorTot = (leftError > 10) & ~((trueLeft <= -bound) & (result["left"] < 10-bound))
                largeRightErrorTot = (rightError > 10) & ~((trueRight >= bound) & (result["right"] > bound-10))
                if method=='RVM' and (largeLeftError.any() or largeRightError.any()):
                    print("####################################")
                    print(results)
                    print("####################################")
                methodStats["errorReduced"].extend(leftError[result["leftSuccess"] & (~largeLeftError)])
                methodStats["errorReduced"].extend(rightError[result["rightSuccess"] & (~largeRightError)])
                methodStats["largeErrors"].extend(largeLeftError)
                methodStats["largeErrors"].extend(largeRightError)
                methodStats["largeErrorsTot"].extend(largeLeftErrorTot)
                methodStats["largeErrorsTot"].extend(largeRightErrorTot)
                
                #methodStats["errorTot"].extend(np.abs(result["left"]-trueLeft)[result["leftOriginal"] > -bound])
                #methodStats["errorTot"].extend(np.abs(result["right"]-trueRight)[result["rightOriginal"] < bound])
                methodStats["evals"].extend(result["leftEvals"][result["leftSuccess"]])
                methodStats["evals"].extend(result["rightEvals"][result["rightSuccess"]])
                methodStats["evalsTot"].extend(result["leftEvals"])
                methodStats["evalsTot"].extend(result["rightEvals"])
        
    for method in methods:
        methodStats = stats[method]
        print("{:<15}: success={:7.5f}, error={:8.4f}, errorReduced={:8.4f}, errorTot={:8.4f}, largeErrors={:7.5f}, largeErrorsTot={:7.5f}, evals={:8.1f}, evalsTot={:8.1f}".format(
            method, 
            np.nanmean(methodStats["success"]), np.nanmean(methodStats["error"]),
            np.nanmean(methodStats["errorReduced"]), np.nanmean(methodStats["errorTot"]),
            np.nanmean(methodStats["largeErrors"]), np.nanmean(methodStats["largeErrorsTot"]),
            np.nanmean(methodStats["evals"]), np.nanmean(methodStats["evalsTot"])))
        

def _transform_parameters(params):
    k, p = params
    return np.exp(k), 1/(1+np.exp(-p))

def _LL_nbinom(params, data):
    k, p = _transform_parameters(params)
    return nbinom.logpmf(data, k, p).sum()
def _grad_LL_nbinom(params, data):
    return Gradient(_LL_nbinom)(params, data)
def _hess_LL_nbinom(params, data):
    return Hessian(_LL_nbinom)(params, data)

def test_find_CI():
    n = 100
    k, p = 5, 0.1
    data = np.random.negative_binomial(k, p, size=n)
    
    def LL(x):
        k, p = np.maximum(x, 1e-10)
        return nbinom.logpmf(data, k, p).sum()
    
    LLParallel = partial(_LL_nbinom, data=data)
    
    def nLLParallel(params):
        return -LLParallel(params)
    
    x0 = [1, 0.5]
    
    def nLL(params):
        return -LL(params)
    
    def function(params):
        return np.prod(params)
    
    functionJac = Gradient(function)
    functionHess = Hessian(function)
    
    jac = Gradient(LL)
    hess = Hessian(LL)
    
    result = op.differential_evolution(nLL, [(0, 10), (0, 1)])
    resultParallel = op.minimize(nLLParallel, x0)
    print("Estimate: k={:5.3f}, p={:5.3f}".format(*result.x))
    
    CI1 = find_CI_new(result.x, LL)
    print("All CIs:", CI1)
    CI2 = find_CI_new(resultParallel.x, LLParallel, parallel=True)
    print("All CIs parallel:", CI2)
    
    CIF0 = find_function_CI(result.x, lambda x: x[0], LL)
    print("CI for the first parameter (via function interface):", CIF0)
    
    CIF = find_function_CI(result.x, function, LL)
    print("CI for the product of all parameters:", CIF)
    CIF = find_function_CI(result.x, function, LL, logLJac=jac, logLHess=hess)
    print("CI for the product of all parameters:", CIF)
    CIF = find_function_CI(result.x, function, LL, functionJac=functionJac, functionHess=functionHess)
    print("CI for the product of all parameters:", CIF)
    CIF = find_function_CI(result.x, function, LL, logLJac=jac, logLHess=hess, functionJac=functionJac, functionHess=functionHess)
    print("CI for the product of all parameters:", CIF)
    
    CI3 = find_CI_new(result.x, LL, return_full_results=True)
    print("All CIs, full results:", CI3)
    CI4, success = find_CI_new(result.x, LL, return_success=True)
    print("All CIs and success:", CI4, success)
    CI5 = find_CI_new(result.x, LL, None, None, 1)
    print("Second parameter:", CI5)
    CI6 = find_CI_new(result.x, LL, None, None, None, 1)
    print("Upper bounds:", CI6)
    CI7 = find_CI_new(result.x, LL, None, None, None , [1, [False, True]])
    print("Upper bound for the first, both bounds for the second:", CI7)
    CI8 = find_CI_new(result.x, LL, None, None, [0, 1], [1, -1])
    print("Upper bound for the first, lower for the second:", CI8)
    CI9 = find_CI_new(result.x, LL, None, None, 0, 0)
    print("Lower bound for the first:", CI9)
    
    
if __name__ == '__main__':
    methods = ["Wald", "RVM", "RVM_psI", "bisection", "mixed_min", "constrained_max", "binsearch", "VM",  "gridsearch"]
    
    """
    # Uncomment this to test the H14 model, which requires additional R files.
    
    try:
        from rpy2 import robjects
        test_H14()
    except ImportError:
        raise ImportError("The package rpy2 must be installed in order to test "
                          "the H14 model.")
    """
    test_find_CI()
    benchmark(methods, 200, dataN=50, mode="11cx") 
    benchmark(methods, 200, dataN=10000, mode="11") 
    benchmark(methods, 200, dataN=10000, mode="3") 
    sys.exit()
