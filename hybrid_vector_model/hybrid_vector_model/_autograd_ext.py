'''
Created on 10.02.2018

@author: Samuel
'''
from autograd.extend import primitive, defvjp
from autograd.scipy.special import gammaln
import autograd.numpy as np

from vemomoto_core.npcollections.npextc import pointer_sum


@primitive
def sparsepowersum(arr, b, fact=None):
    tmp = np.empty_like(arr.data)
    np.exp(np.multiply(np.log(arr.data, tmp), b, tmp), tmp)
    if not type(arr.indptr) == np.int64:
        indptr = arr.indptr.astype(np.int64)
    else:
        indptr = arr.indptr
    if fact is not None:
        tmp *= fact
    result = pointer_sum(tmp, indptr)
    return result

def sparsepowersum_vjp_b(ans, arr, b, fact=None):
    if fact is None:
        return lambda g: np.sum(g * sparsepowersum(arr, b, np.log(arr.data)))
    return lambda g: np.sum(g * sparsepowersum(arr, b, fact*np.log(arr.data)))

defvjp(sparsepowersum, None, sparsepowersum_vjp_b)


def nbinom_logpmf(k, n, p): 
    return (gammaln(n+k) - gammaln(n) - gammaln(k+1) 
            + n*np.log(p) + k*np.log(1-p)) 
