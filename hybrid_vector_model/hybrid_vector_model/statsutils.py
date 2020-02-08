'''
Created on 13.06.2017

@author: Samuel
'''

#import rpy2.robjects as robjects
#from rpy2.robjects.packages import importr
import warnings
from itertools import repeat
from collections import defaultdict

import numpy as np
import scipy as sc
from scipy.special import binom
from scipy.stats import nbinom, poisson
from scipy.special import lambertw
from statsmodels.distributions.empirical_distribution import ECDF
np.random.seed()

#import line_profiler
from vemomoto_core.concurrent.nicepar import Counter
from vemomoto_core.concurrent.concurrent_futures_ext import ProcessPoolExecutor

"""
# R package names
packnames = ('dgof',)

utils = importr('utils')
utils.chooseCRANmirror(ind=1)

# R vector of strings
from rpy2.robjects.vectors import StrVector
import rpy2.robjects.packages as rpackages

# Selectively install what needs to be install.
# We are fancy, just because we can.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
#"""


def anderson_darling_test_discrete(data, modelX, modelY, 
                                   simulate_p_value=False):
    # import R's "dgof" package
    dgof = importr('dgof')
    
    #raise NotImplementedError("This functionality requires a proper R setup.",
    #                          "Other")
    
    #print("data.shape", data.shape)
    #print("np.unique(data)", np.unique(data))
    
    #print("modelX.shape", modelX.shape)
    #print("modelY.shape", modelY.shape)
    
    rData = robjects.IntVector(data)
    
    first1 = np.nonzero(modelY==1)[0]
    if len(first1):
        first1 = first1[0] + 1
    else:
        first1 = len(modelX)
    
    modelX = modelX[:first1]
    modelY = modelY[:first1]
    
    rModelX = robjects.IntVector(modelX)
    rModelY = robjects.FloatVector([0] + list(modelY))
    
    #print("rModelX.r_repr()", rModelX.r_repr())
    #print("rModelY.r_repr()", rModelY.r_repr())
    
    rStepFun = robjects.r['stepfun']
    #print("rStepFun.r_repr()", rStepFun)
    rCVMTest = robjects.r['cvm.test']
    
    rCMF = rStepFun(rModelX, rModelY)
    
    testResult = rCVMTest(rData, rCMF, "A2", simulate_p_value)
    
    return testResult.rx("p.value")[0][0]

#@profile
def zero_truncated_NB(size, n, p, poissonLimit=False, quantile = 0.999, 
                      MHSteps=100):
    """
    returns a sample of size "size" from the negative binomial distribution with 
    parameters n, p under the condition that at least one element in the
    sample is nonzero.
    MHSteps denotes the number of Metropolis-Hastings iterations
    """
    if p==1:
        poissonLimit = True
    
    
    # if obtaining a random sample with total count 0 is sufficiently unlikely,
    # sample until a suitable sample is found.
    if poissonLimit:
        zeroP = np.exp(-size*n)
    else:
        zeroP = p**(size*n)
    if zeroP < 0.7:
        while not poissonLimit:
            result = np.random.negative_binomial(n, p, size)
            if result.any():
                return result
        while poissonLimit:
            result = np.random.poisson(n, size)
            if result.any():
                return result
    
    # pmf of truncated negative binomial for total count
    q = min(quantile*(1-zeroP)+zeroP, 0.999999)
    
    if poissonLimit:
        maxbin = poisson.ppf(q, size*n)
    else:
        dist = nbinom(n, p)
        maxbin = nbinom.ppf(q, size*n, p)
    
    maxbin = max(maxbin, 5)
    x = np.arange(1, maxbin+1)
    if poissonLimit:
        trunc_pmf = poisson.pmf(x, size*n)
    else:
        trunc_pmf = nbinom.pmf(x, size*n, p)
        
    trunc_pmf /= np.sum(trunc_pmf)
    
    # sampling the total count value
    totalCount = np.random.choice(x, p=trunc_pmf)
    
    if poissonLimit:
        return np.random.multinomial(totalCount, np.full(size, 1/size))
    elif totalCount == 1:
        # if only one observation has been made, it does not matter when
        result = np.zeros(size)
        result[0] = 1
        return result
    elif totalCount == 2:
        # if two observations have been made, we have to decide whether they
        # occurred in the same sample or in distinct samples
        
        # when computing the joint probabilities of the possible events, I 
        # neglect factors that appear in all probabilities
        
        # p11 = (size choose 2) * pmf(1)**2
        p11 = (size-1)/2 * dist.pmf(1)**2
        # p20 = size * pmf(2) * pmf(0)
        p20 = dist.pmf(2) * dist.pmf(0)
        
        norm = p11 + p20
        p11 /= norm
        p20 /= norm
        
        result = np.zeros(size)
        if np.random.choice([True, False], p=[p11, p20]):
            result[:2] = 1
        else:
            result[0] = 2
        return result
    elif totalCount == 3:
        # p111 = (size choose 3) * pmf(1, n, p)**3
        p111 = (size-1)*(size-2)/6 * dist.pmf(1)**3
        p210 = (size-1) * dist.pmf(2)*dist.pmf(1)*dist.pmf(0)
        p300 = dist.pmf(3)*dist.pmf(0)**2
        ps = np.array([p111, p210, p300])
        ps /= np.sum(ps)
        
        result = np.zeros(size)
        choice = np.random.choice(np.arange(3), p=ps)
        if choice == 0:
            result[:3] = 1
        elif choice == 1:
            result[0] = 2
            result[1] = 1
        elif choice == 2:
            result[0] = 3
        return result
    else:
        return _dist_bins_MH(size, totalCount, dist, MHSteps)
        
    
def _dist_bins_MH(size, totalCount, dist, steps=100):
    
    # initialize
    if dist.mean() / dist.var() > 0.5:
        # if overdispersion is low, generate initial sample under indpendece
        # assumption
        counts = np.random.multinomial(totalCount, np.full(size, 1/size))
        nonzero = list(np.nonzero(counts)[0])
    else:
        # if overdispersion is high, generate initial sample with one big heap
        counts = np.zeros(size)
        counts[0] = totalCount
        nonzero = [0]
    
    for i in range(steps):
        # pick a pair of indices to swap a count value
        swapFrom = np.random.choice(nonzero)
        swapTo = np.random.randint(0, size)
        if swapFrom == swapTo:
            continue
        swappedCounts = counts.copy()
        swappedCounts[swapFrom] -= 1
        swappedCounts[swapTo] += 1
        
        changeP = dist.logpmf(swappedCounts).sum() - dist.logpmf(counts).sum()
        if changeP >= 0 or changeP >= np.log(np.random.rand()):
            if counts[swapTo] == 0:
                nonzero.append(swapTo)
            if counts[swapFrom] == 1:
                nonzero.remove(swapFrom)
            counts = swappedCounts
        
    return counts
    
def __anderson_darling_statistic_P(data, fitData):
    cmfData = ECDF(data)
    x, pmf = np.unique(fitData, return_counts=True)
    pmf = (pmf / pmf.sum())[:-1]
    cmf = pmf.cumsum()
    return data.size * np.sum(np.square(cmfData(x)[:-1]-cmf) * pmf / (cmf * (1-cmf)))
    
# test statistic for Anderson-Darling test
#@profile
def __anderson_darling_statistic_NB(data, parameters=None,
                                    getParamEstimates=False,
                                    poissonLimit=False):
    
    isPDFData = hasattr(data[0], "__iter__")
    if isPDFData:
        x, counts = data
        size = np.sum(counts)
        maxx = x[-1]
    else:
        maxx = np.max(data)
        size = data.size
    
    if parameters is None:
        # observed mean and variance and other base measures
        if isPDFData:
            mean = np.sum(x*counts)/size
            var = np.sum(np.square(x-mean)*counts)/(size-1)
        else:
            mean = np.mean(data)
            var = np.var(data, ddof=1)
        mean = max(mean, 1e-10)
        var = max(var, 1e-10)
        
        # method of moments parameter estimates
        if not poissonLimit:
            #p = mean / var <- this does not work if we neglect zeros
            optf = lambda pp: mean/pp-var-mean**2*pp**(size*(pp*(mean**2+var)-mean)/((1-pp)*mean))
            left = mean/(var+mean**2)
            
            if left >= 1:
                poissonLimit=True
            else:
                right = 1
                err = 1e-5
                while right-left > err:
                    p = (right+left)/2
                    if optf(p) > 0:
                        left = p
                    else:
                        right = p
                p = (right+left)/2
                poissonLimit = right == 1
        if not poissonLimit:
            n = (p*(mean**2+var)-mean)/((1-p)*mean)
        else:
            if mean*size <= 1:
                if getParamEstimates:
                    return 0, 0, 1
                else:
                    return 0
            p = 1
            n = lambertw(-np.exp(-mean*size)*mean*size).real/size + mean
    else:
        n, p = parameters
    
    if not isPDFData:
        x = np.arange(maxx+1)
        # empirical cmf
        ecmf = ECDF(data)(x)
    else:
        ecmf = np.cumsum(counts)/size
        
        
    if not poissonLimit:
        pmf = binom(x+(n-1), (n-1))*p**n*np.power(1-p, x) #nbinom.pmf(x, n, p) 
        pmf[0] -= p**(size*n)
        pmf /= (1-p**(size*n))
    else:
        pmf = poisson.pmf(x, n)
        pmf[0] -= np.exp(-size*n)
        pmf /= (1-np.exp(-size*n))
    
    cmf = pmf.cumsum()
    if cmf[-1] > 1: # in case of significant numerical errors that lead to
                    # a probabiity > 1: normalize
        pmf /= cmf[-1]
        cmf /= cmf[-1]
    #if np.mean(data) >= np.var(data, ddof=1):
    #    print("meanvar issue", np.mean(data), np.var(data, ddof=1))
    
    # anderson-darling statistic
    T = size * np.sum(np.square(ecmf-cmf) * pmf / (cmf * (1-cmf)))
    
    #print(T, T2)
    if getParamEstimates:
        return (T, n, p)
    else:
        return T


__ADRESULTS = {}
__PRESULTS = {}
__PSAMPLERESULTS = {}
MAXSIZE = 500000
RESULTSAVER = [__ADRESULTS, __PRESULTS, __PSAMPLERESULTS, MAXSIZE]

#@profile
def anderson_darling_NB(data, parameters=None, poissonLimit=False, 
                        pSampleSize=0, bootstrapN=400, bootstrapN_P=0,
                        MHSteps=100, usePreviousPVals=True,
                        usePreviousPSamples=False,
                        resultSaver=None):
    
    """
    global __PRESULTS
    global __ADRESULTS
    global __PSAMPLERESULTS
    global MAXSIZE 
    """
    if resultSaver is None:
        resultSaver = RESULTSAVER #{}, {}, {}, 50000
    __ADRESULTS, __PRESULTS, __PSAMPLERESULTS, MAXSIZE = resultSaver
    #"""
    
    pSampleSize2 = max(pSampleSize, bootstrapN_P)
    if parameters is None and np.sum(data) <= 1:
        if pSampleSize2:
            if bootstrapN_P:
                return 1, np.ones(pSampleSize2), np.ones((bootstrapN_P, pSampleSize))
            else:
                return 1, np.ones(pSampleSize2)
        else:
            return 1
    if parameters is not None:
        parameters = tuple(parameters)
    
    
    obs, counts = np.unique(data, return_counts=True)
    dataHash = hash((tuple(obs), tuple(counts), poissonLimit, parameters))
    if not bootstrapN_P:
        if usePreviousPVals:
            result = __PRESULTS.get(dataHash, None)
            if result is not None:
                if not pSampleSize:
                    return result
        """
                elif usePreviousPSamples:
                    result2 = __PSAMPLERESULTS.get(dataHash, None)
                    #result2 = {}.get(dataHash, None)
                    if result2 is not None and len(result2) >= pSampleSize:
                        #return result, result2[:pSampleSize]
                        pass
        if usePreviousPVals and (dataHash in __PRESULTS):
            if not pSampleSize:
                return __PRESULTS[dataHash]
            elif usePreviousPSamples:
                result = __PSAMPLERESULTS.get(dataHash, None)
                if result is not None and len(result) >= pSampleSize:
                    print(len(__PRESULTS), len(__PSAMPLERESULTS))
                    return __PRESULTS[dataHash], result[:pSampleSize]
        """
    else:
        np.random.seed()
        
    # base statistic a
    T, n, p = __anderson_darling_statistic_NB((obs, counts), parameters, True, poissonLimit)
    
    parameters2 = (n, p)
    
    poissonLimit2 = poissonLimit or p == 1
    
    # bootstrapping
    
    pVal = 0
    if pSampleSize2:
        if bootstrapN_P:
            pSamples = np.zeros((bootstrapN_P, pSampleSize))
        pValues = np.zeros(pSampleSize2)
    
    bootstrapN = max(bootstrapN, pSampleSize, bootstrapN_P)
    for i in range(bootstrapN):
        row = zero_truncated_NB(data.size, n, p, poissonLimit2, MHSteps)
        robs, rcounts = np.unique(row, return_counts=True)
        rowHash = hash((tuple(robs), tuple(rcounts), poissonLimit2, parameters))
        Tb = __ADRESULTS.get(rowHash, None)
        if Tb is None:
            Tb = __anderson_darling_statistic_NB((robs, rcounts), parameters2, 
                                                 poissonLimit=poissonLimit2)
            __ADRESULTS[rowHash] = Tb
        if Tb >= T:
            pVal += 1 
        if (i < bootstrapN_P) or (i < pSampleSize):
            if usePreviousPVals and (poissonLimit != poissonLimit2):
                rowHash = hash((tuple(robs), tuple(rcounts), poissonLimit, parameters))
            pVal2 = pSample = None
            if i < bootstrapN_P:
                if usePreviousPSamples:
                    #print("RU", len(__PRESULTS), len(__PSAMPLERESULTS))
                    pSample = __PSAMPLERESULTS.get(rowHash, None)
                if pSample is None:
                    pVal2, pSample = anderson_darling_NB(row, parameters, poissonLimit, 
                                                pSampleSize, bootstrapN, 0, MHSteps, 
                                                usePreviousPVals, usePreviousPSamples,
                                                resultSaver=resultSaver)
                pSamples[i] = pSample
            if i < pSampleSize2:
                if pVal2 is None and usePreviousPSamples:
                    pVal2 = __PRESULTS.get(rowHash, None)
                if pVal2 is None:
                    pVal2 = anderson_darling_NB(row, parameters, poissonLimit, 
                                                0, bootstrapN, 0, MHSteps, 
                                                usePreviousPVals, usePreviousPSamples,
                                                resultSaver=resultSaver)
                pValues[i] = pVal2
            
    pVal /= bootstrapN
    
    if len(__ADRESULTS) > MAXSIZE:
        __ADRESULTS.clear()
    if len(__PRESULTS) > MAXSIZE:
        __PRESULTS.clear()
    if len(__PSAMPLERESULTS) > MAXSIZE:
        __PSAMPLERESULTS.clear()
    
    __PRESULTS[dataHash] = pVal
    
    if pSampleSize:
        __PSAMPLERESULTS[dataHash] = pValues
        if bootstrapN_P:
            #print(len(__ADRESULTS), len(__PRESULTS), len(__PSAMPLERESULTS))
            #print(len(__PRESULTS), len(__PSAMPLERESULTS), usePreviousPSamples)
            return pVal, pValues, pSamples
        return pVal, pValues
    else:
        return pVal
    

def anderson_darling_P(data, poissonLimit, pSampleSize, bootstrapN, bootstrapN_P, counter=None):
    
    global __ADRESULTS
    global __PRESULTS
    __ADRESULTS.clear()
    __PRESULTS.clear()
    
    
    #anderson_darling_NB(data[0], None, poissonLimit, pSampleSize, bootstrapN, bootstrapN_P)
    p1 = []
    p2 = []
    p3 = []
    with ProcessPoolExecutor() as pool:
        testRes = pool.map(anderson_darling_NB, data,
                           repeat(None), repeat(poissonLimit), 
                           repeat(pSampleSize), repeat(bootstrapN),
                           repeat(bootstrapN_P), repeat(100), repeat(True),
                           repeat(False), chunksize=1) 
        
        for pVal, pValues, pSamples in testRes:
            p1.append(pVal)
            p2.append(pValues)
            p3.append(pSamples)
            counter()
    
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    T = __anderson_darling_statistic_P(p1, p2[:,:pSampleSize].ravel())
    ts = []
    pVal = 0
    for i in range(bootstrapN_P):
        Tb = __anderson_darling_statistic_P(p2[:,i], p3[:,i].ravel())
        if Tb >= T:
            pVal += 1
        ts.append(Tb)
    pVal /= bootstrapN_P
    return pVal
        


def _test_ZTMNB():
    n, p = 8385.273718025031, 0.9999971718211882 #1, 0.9
    samplesize, testn = 30, 1000
    
    a = np.array([zero_truncated_NB(samplesize, n, p)
                  for _ in range(testn)])
    """
    resN, resP = [], []
    for r in range(30):
        t, nn, pp = __anderson_darling_statistic_NB(zero_truncated_NB(samplesize, n, p), 
                                                    None, True, False)
        resN.append(nn)
        resP.append(pp)
    
    
    print(np.mean(resN), np.mean(resP))
    print(resN)
    print(resP)
    """
    
    meanSumExp = samplesize*n*(1-p)/(p*(1-p**(n*samplesize)))
    varSumExp = meanSumExp/p-p**(n*samplesize)*meanSumExp**2
    meanExp = n*(1-p)/(p*(1-p**(n*samplesize)))
    varExp = meanExp/p-p**(n*samplesize)*meanExp**2
    sums = a.sum(1)
    meanSum = np.mean(sums)
    varSum = np.var(sums)
    var = np.mean(np.var(a, 1, ddof=1))
    mean = np.mean(a)
    
    print(meanSumExp, meanSum)
    print(varSumExp, varSum)
    print(meanExp, mean)
    print(varExp, var)
    
    
    
    m = mean
    s = var
    N = samplesize
    #"""
    #optf = lambda pp: m-s*pp-m**2*pp**(pp/(1-pp)*(m**2+s-m)/m)
    #optf = lambda pp: m-s*pp-m**2*pp**(pp/(1-pp)*(N*(m**2+s)-m)/m)
    optf = lambda pp: m/pp-s-m**2*pp**(N*(pp*(m**2+s)-m)/((1-pp)*m))
    #optfM = lambda nn: nn*(1-pp)/(pp*(1-pp**(nn*samplesize)))-mean
    left = mean/(var+mean**2)
    right = 1
    err = 1e-7
    while right-left > err:
        new = (right+left)/2
        if optf(new) > 0:
            left = new
        else:
            right = new
    pp = (right+left)/2
    #"""
    #pp = mean/var
    #nn = (pp*(meanSum**2+varSum)-meanSum)/(samplesize*(1-pp)*meanSum)
    nn = (pp*(m**2+s)-m)/((1-pp)*m)
    #nn = ((np.log(mean-var*pp)-2*np.log(mean))/np.log(pp)-1)/samplesize
    pold = mean/var
    nold = mean*pold/(1-pold)
    print("Estimate n", nn, n, nold)
    print("Estimate p", pp, p, pold)
    
    m = meanExp
    s = varExp
    #pp = m/(s+m**2)
    m*(1-p**(n*N))*p/(N*(1-p))
    m - p**(n*N+1)*m**2-s*p
    (p*(m**2+s)-m)/(N*(1-p)*m)
    m - p**(n*N+1)*m**2-s*p
    m-s*p-m**2*p**(p/(1-p)*(m**2+s-m)/m)
    
    ppp = np.linspace(0,1,1000)
    nnn = np.linspace(0,3,1000)
    y = m-s*ppp-m**2*ppp**(ppp/(1-ppp)*(m**2+s-m)/m)
    y = nnn*(1-pp)/(pp*(1-pp**(nnn*samplesize)))-mean
    (pp/(1-pp)*(m**2+s-m)/m)
    
    print("done")
    

def _test_anderson_darling_NB():
    
    testn = 50
    datan = 50
    
    np.random.seed(2)
     
    print("Starting tests with testn={} and datan={}".format(testn, datan))
    
    p = np.full(testn, 0.8)
    n = np.full(testn, 0.5)
    p = np.random.rand(testn)/3+0.05 #+0.3
    n = np.random.exponential(0.5, size=testn)
    n = np.random.negative_binomial(1, 0.2, size=testn) / 20 + 0.01
    p2 = np.random.rand(testn)
    n2 = np.random.poisson(1, size=testn)+1
    
    m = lambda x: (np.mean(x), np.var(x), np.mean(x)/np.var(x))
    """
    print("== Tests with Poisson distribution ==")
    pvals = [anderson_darling_NB(zero_truncated_NB(datan, nn, 1, True),
                                 False)
             for nn, pp in zip(n2, p)]
    pvals = np.array(pvals)[~np.isnan(pvals)]
    print("mean:", np.mean(pvals))
    freq, bins = np.histogram(pvals, 20, [0, 1])
    freq = freq/np.sum(freq)
    print(np.round(freq,2))
    print(np.round(np.cumsum(freq),2))
    #"""
    from matplotlib import pyplot as plt
    print("== Tests with NB ==")
    """
    for i in range(10, 101, 10):
        a = np.zeros(i)
        a[0] = 1
        print(i, np.mean([anderson_darling_NB(a, False) for _ in range(20)], 0))
    print("Proceed")
    pv, pvc = anderson_darling_NB(np.array([1,1]+[0]*20), None, True, True) 
    print(pv)
    a = np.histogram(pvc, 20, [0,1])[0]
    print(a/a.sum())
    #"""
    
    global __PRESULTS
    global __ADRESULTS
    global __PSAMPLERESULTS
    
    pvals = []
    pvalcandidates = []
    i = 1
    
    from itertools import chain as iterchain
    
    samples = []
    for nn, pp in zip(n, p):
        samples.append(zero_truncated_NB(datan, nn, pp, False))
    
    pvs = []
    for i in range(100):
        c = Counter(testn, 0.05)
        def counter():
            perc = c.next()
            if perc:
                print("{}%".format(round(perc*100)))
        #"""    
        samples = []
        for nn, pp in zip(n, p):
            samples.append(zero_truncated_NB(datan, nn, pp, False))
        #"""
        pv = anderson_darling_P(samples, False, 50, 100, 50, counter)
        #pv = anderson_darling_P(samples, False, 100, 100, 100, counter)
        print(i, ":", pv)
        pvs.append(pv)
    print(pvs)
    print(np.histogram(pvs, 20, (0,1)))
    return
    
    print("== Tests with binomial distribution ==")
    pvals = [anderson_darling_NB(np.random.binomial(nn, pp, datan)) 
             for nn, pp in zip(n2, p)]
    pvals = np.array(pvals)[~np.isnan(pvals)]
    print("mean:", np.mean(pvals))
    freq, bins = np.histogram(pvals, 20, [0, 1])
    freq = freq/np.sum(freq)
    print(np.round(freq,2))
    print(np.round(np.cumsum(freq),2))
    
    print("== Tests with multi density NB ==")
    pvals = [anderson_darling_NB(np.random.binomial(nn, pp, datan)+
                                 np.random.binomial(nn2, pp2, datan)) 
             for nn, pp, nn2, pp2 in zip(n, p, n2, p2)]
    pvals = np.array(pvals)[~np.isnan(pvals)]
    print("mean:", np.mean(pvals))
    freq, bins = np.histogram(pvals, 20, [0, 1])
    freq = freq/np.sum(freq)
    print(np.round(freq,2))
    print(np.round(np.cumsum(freq),2))
    
    print("== Tests with multi density Poisson ==")
    pvals = [anderson_darling_NB(np.random.poisson(nn, datan) 
             + np.random.randint(0, 2, datan) * np.random.poisson(nn2, datan))
             for nn, pp, nn2, pp2 in zip(n, p, n2, p2)]
    pvals = np.array(pvals)[~np.isnan(pvals)]
    print("mean:", np.mean(pvals))
    freq, bins = np.histogram(pvals, 20, [0, 1])
    freq = freq/np.sum(freq)
    print(np.round(freq,2))
    print(np.round(np.cumsum(freq),2))
    

    

def vonmises_logpdf(x, kappa, loc, scale):
    return (kappa * np.cos((x-loc)/scale) - np.log(2*np.pi) 
            - np.log(sc.special.i0e(kappa)) - kappa) - np.log(scale)

def R2(predicted, observed):
    mean = np.mean(observed)
    return 1 - np.sum((observed-predicted)**2)/np.sum((observed-mean)**2)

if __name__ == '__main__':
    import sys
    
    x = np.linspace(0 , 10, 10)
    print(R2(x, x+2))
    
    sys.exit()
    _test_ZTMNB()
    
    from simprofile import profile
    
    #line_profiler.LineProfiler(zero_truncated_NB)
    
    _test_anderson_darling_NB()
    #profile("_test_anderson_darling_NB()", globals(), locals())
    
    
    from scipy.stats import vonmises
    
    kappa = 300
    scale = 1/(2*np.pi)
    loc = 0.7
    a=np.linspace(0,1,11)
    from timeit import timeit
    
    lp = vonmises.logpdf
    
    print(timeit("lp(a, kappa, loc, scale)", number=1000, globals={**globals(), **locals()}))
    print(timeit("vonmises_logpdf(a, kappa, loc, scale)", number=1000, globals={**globals(), **locals()}))
    print(timeit("lp(a, kappa, loc, scale)", number=1000, globals={**globals(), **locals()}))
    print(timeit("vonmises_logpdf(a, kappa, loc, scale)", number=1000, globals={**globals(), **locals()}))
    
    b1 = vonmises.logpdf(a, kappa, loc, scale)
    b2 = vonmises_logpdf(a, kappa, loc, scale)
    
    print(a)
    print(b1)
    print(b2)
    print(np.abs(b1-b2))
    sys.exit()
    
    np.random.seed()
    
    x = np.arange(1, 10+1)
    d = np.random.choice(x, 25, replace=True)
    y = np.cumsum(np.ones(x.size) / x.size)
    
    x = np.arange(16)
    #x = np.arange(2)
    d = np.zeros(90)
    y = np.array([0.998635494757076, 0.999995651829634, 0.999999983505647, 
                  0.999999999932426, 0.999999999999711, 0.999999999999999, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    #y = np.array([0.7, 0.8])
    print(anderson_darling_test_discrete(d, x, y))
