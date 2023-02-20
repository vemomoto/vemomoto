'''
'''

from warnings import warn
from functools import partial
from itertools import product

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import vonmises
from numdifftools import Gradient, Hessian

from ci_rvm import find_CI_bound

class BaseTrafficDensityDayTime():
    def pdf(self, t):
        raise NotImplementedError()
    def interval_probability(self, tStart=None, tEnd=None):
        raise NotImplementedError()
    def plot(self, show=True, scalingFactor=1):
        raise NotImplementedError()
    def neg_log_likelihood(self, coefficients, obsTime, shiftStart, shiftEnd):
        return (- np.sum(np.log(self.pdf(obsTime, coefficients)))
                + np.sum(np.log(self.interval_probability(shiftStart, shiftEnd, 
                                                          coefficients))))
    def maximize_likelihood(self, obsTime, shiftStart, shiftEnd, x0, 
                            ineqConstr=None):
        
        eqConstr = (lambda coeff: 
                    1-self.interval_probability(coefficients=coeff))
        
        optResult = optimize.minimize(self.neg_log_likelihood, x0, 
                                      (obsTime, shiftStart, shiftEnd), 
                                      method='SLSQP',
                                      constraints=({"type":"eq",
                                                    "fun":eqConstr},
                                                   {"type":"ineq",
                                                    "fun":ineqConstr}), 
                                      options={'disp': False, 'ftol': 1e-08}
                                      )
        
        print(optResult)
        
        if not optResult.success:
            warn("Optimization was not successful")
            
        self.coefficients = optResult.x
        self.AIC = 2*(optResult.fun + len(self.coefficients))
        print("AIC:", self.AIC)
            
        return optResult
    
    
class TrafficDensityDayTime_PLinear(BaseTrafficDensityDayTime):
    def __init__(self, tMin, tMax, intervalNumber, coefficients=None):
        if tMin >= tMax:
            raise ValueError("tMax-tMin must be positive.")
        self.tMin = tMin
        self.tMax = tMax
        self.intervalNumber = intervalNumber
        self.intervalWidth = (tMax - tMin) / intervalNumber
        if coefficients is not None:
            if not len(coefficients) == intervalNumber+1:
                raise ValueError("The coefficient array must match the number "
                                 + "of intervals.")
        self.coefficients = np.array(coefficients)
        
    def pdf(self, t, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients
            if coefficients is None:
                raise ValueError("No coefficients are given.")
        elif type(coefficients) is not np.ndarray and hasattr(t, "__iter__"):
            coefficients = np.array(coefficients)
        
        # Add: Check whether t is in [tMin, tMax]
        
        intervalWidth = self.intervalWidth
        
        t = t-self.tMin
        
        if np.any(t < 0):
            raise RuntimeError("Some observation occurred outside the shifts.")
        
        intervalIndex = (t // intervalWidth).astype(int)
        modTime = t % intervalWidth
        
        return (coefficients[list(intervalIndex)] * (1 - modTime)
                + coefficients[list(intervalIndex+1)] * modTime)
        
    def interval_probability(self, tStart=None, tEnd=None, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients
            if coefficients is None:
                raise ValueError("No coefficients are given.")
        elif type(coefficients) is not np.ndarray:
            coefficients = np.array(coefficients)
        
        tMin, tMax = self.tMin, self.tMax
        
        if tStart is None:
            tStart = tMin
        if tEnd is None:
            tEnd = tMax
            
        width = self.intervalWidth
        
        lowerParts = np.minimum(coefficients[:-1], coefficients[1:])
        differences = coefficients[1:] - coefficients[:-1]
        upperParts = np.abs(differences) / 2
        partIntegrals = (lowerParts + upperParts) * width
        
        if tStart is tMin and tEnd is tMax:
            return np.sum(partIntegrals)
        
        startIndex = ((tStart-tMin) // width).astype(int)
        endIndex = ((tEnd-tMin) // width).astype(int)
        
        if not hasattr(tStart, "__iter__"):
            startIndex = np.array([startIndex])
        if not hasattr(tEnd, "__iter__"):
            endIndex = np.array([endIndex])
        
        r = np.arange(len(partIntegrals))
        mask = (startIndex[:,None] <= r) & (endIndex[:,None] > r)
        
        integral = np.einsum('j,ij->i', partIntegrals, mask)
        
        startModTime = (tStart-tMin) % width
        endModTime = (tEnd-tMin) % width
        differences = np.append(differences, 0)
        
        integral += (
                     ((0.5/width) * differences[startIndex] 
                      * startModTime + coefficients[startIndex]) 
                     * (-startModTime)
                     + ((0.5/width) * differences[endIndex] 
                        * endModTime + coefficients[endIndex]) 
                     * endModTime
                     )
        
        return integral
    
    def maximize_likelihood(self, obsTime, shiftStart, shiftEnd):
        
        x0 = np.ones(self.intervalNumber+1) / (self.tMax - self.tMin)
        
        ineqConstr = lambda coeff: coeff
        
        return super().maximize_likelihood(obsTime, shiftStart, shiftEnd, x0, 
                                           ineqConstr)
    
    def plot(self, show=True, scalingFactor=1):
        times = np.linspace(self.tMin, self.tMax, self.intervalNumber+1)
        if self.coefficients is None:
            raise AttributeError("self.coefficients must be defined to allow "
                                 + "plotting.")
        plt.plot(times, self.coefficients*scalingFactor, label="Distribution with " 
                 + str(self.intervalNumber) + " intervals")
        plt.xlabel("Day-Time [h]")
        plt.ylabel("Traffic Density")
        plt.ylim(ymin=0)
        plt.legend(loc=2, frameon=False)
        if show: plt.show()

class TrafficDensityDayTime_StepFun(BaseTrafficDensityDayTime):
    def __init__(self, tMin, tMax, intervalNumber, coefficients=None):
        if tMin >= tMax:
            raise ValueError("tMax-tMin must be positive.")
        self.tMin = tMin
        self.tMax = tMax
        self.intervalNumber = intervalNumber
        self.intervalWidth = (tMax - tMin) / intervalNumber
        if coefficients is not None:
            if not len(coefficients) == intervalNumber:
                raise ValueError("The coefficient array must match the number "
                                 + "of intervals.")
        self.coefficients = np.array(coefficients)
        
    def pdf(self, t, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients
            if coefficients is None:
                raise ValueError("No coefficients are given.")
        elif type(coefficients) is not np.ndarray and hasattr(t, "__iter__"):
            coefficients = np.array(coefficients)
        
        # Add: Check whether t is in [tMin, tMax]
        
        intervalWidth = self.intervalWidth
        
        t = t-self.tMin
        
        if np.any(t < 0):
            raise RuntimeError("Some observation occurred outside the shifts.")
        
        intervalIndex = (t // intervalWidth).astype(int)
        
        return coefficients[list(intervalIndex)]
        
    def interval_probability(self, tStart=None, tEnd=None, coefficients=None):
        if coefficients is None:
            coefficients = self.coefficients
            if coefficients is None:
                raise ValueError("No coefficients are given.")
        elif type(coefficients) is not np.ndarray:
            coefficients = np.array(coefficients)
        
        tMin, tMax = self.tMin, self.tMax
        
        if tStart is None:
            tStart = tMin
        if tEnd is None:
            tEnd = tMax
        tStart = np.maximum(tStart, tMin)
        tEnd = np.minimum(tEnd, tMax)
            
        width = self.intervalWidth
        
        coefficients = np.append(coefficients, 0)
        
        cumulativeDensity = np.cumsum(coefficients) * width
        
        tStart -= tMin
        tEnd -= tMin
        restWidthStart = tStart % width
        restWidthEnd = tEnd % width
        startIndex = (tStart // width).astype(int)
        endIndex = (tEnd // width).astype(int)
        if not hasattr(tStart, "__iter__"):
            startIndex = np.array([startIndex])
        if not hasattr(tEnd, "__iter__"):
            endIndex = np.array([endIndex])
        
        integral = cumulativeDensity[endIndex-1] - cumulativeDensity[startIndex]
        
        integral += (width-restWidthStart)*coefficients[startIndex]
        integral += (restWidthEnd)*coefficients[endIndex]
        
        return integral
    
    def maximize_likelihood(self, obsTime, shiftStart, shiftEnd):
        
        x0 = np.ones(self.intervalNumber) / (self.tMax - self.tMin)
        ineqConstr = lambda coeff: coeff
        return super().maximize_likelihood(obsTime, shiftStart, shiftEnd, x0, 
                                           ineqConstr)
    
    def plot(self, show=True, scalingFactor=1):
        times = np.insert(np.linspace(self.tMin, self.tMax, 
                                      self.intervalNumber+1),
                                      0, self.tMin)
        if self.coefficients is None:
            raise AttributeError("self.coefficients must be defined to allow "
                                 + "plotting.")
        plt.step(times, np.hstack([0, self.coefficients, 0])*scalingFactor, 
                 label="Distribution with " 
                 + str(self.intervalNumber) + " intervals",
                  where='post')
        plt.xlabel("Day-Time [h]")
        plt.ylabel("Traffic Density")
        plt.ylim(ymin=0)
        plt.legend(loc=2, frameon=False)
        if show: plt.show()

class TrafficDensityVonMises(BaseTrafficDensityDayTime):
    def __init__(self, location=None, kappa=None):
        self.location = location
        self.kappa = kappa
    
    def interval_probability(self, tStart, tEnd, location=None, 
                             kappa=None):
        if location is None:
            location = self.location
            if location is None:
                raise ValueError("No mean is given.")
        if kappa is None:
            kappa = self.kappa
            if kappa is None:
                raise ValueError("No variance is given.")
        return (vonmises.cdf(tEnd, kappa, location, 12/np.pi) 
                - vonmises.cdf(tStart, kappa, location, 12/np.pi))
    
    def neg_log_likelihood(self, coeff, obsTime, shiftStart, shiftEnd):
        location, kappa = np.abs(coeff)
        return (- np.sum(vonmises.logpdf(obsTime, kappa, location, 
                                             12/np.pi))
                + np.sum(np.log(self.interval_probability(shiftStart, shiftEnd, 
                                                          location, kappa))))
    
     
    def maximize_likelihood(self, obsTime, shiftStart, shiftEnd, getCI=False):
        
        x0 = np.array((12., 1.))
        if not hasattr(obsTime, "__iter__") or len(np.unique(obsTime)) == 1:
            optResult = optimize.OptimizeResult()
            optResult.x = np.array([obsTime, 1000])
            optResult.fun = -np.inf
            optResult.success = True
        elif not hasattr(obsTime, "__iter__") and len(np.unique(obsTime)) == 0:
            optResult = optimize.OptimizeResult()
            optResult.x = x0
            optResult.fun = np.nan
            optResult.success = True
            
        else:
            
            ineqConstr1 = lambda coeff: coeff
            ineqConstr2 = lambda coeff: 24-coeff[0]
            fun = partial(self.neg_log_likelihood, 
                          obsTime=obsTime, shiftStart=shiftStart,  
                          shiftEnd=shiftEnd)
            
            
            optResult = optimize.differential_evolution(
                fun, 
                [(0,24), (0, 10)], 
                maxiter=100,
                disp=True
                )
            print(optResult)
            
            optResult = optimize.minimize(fun, x0, 
                                          method='SLSQP',
                                          constraints=({"type":"ineq",
                                                        "fun":ineqConstr1},
                                                       {"type":"ineq",
                                                        "fun":ineqConstr2}), 
                                          options={'disp': False, 'ftol': 1e-08}
                                          )
            if getCI:
                x0 = optResult.x
                ffun = lambda x: -fun(x)
                jac, hess = Gradient(ffun), Hessian(ffun)
                
                dim = len(x0)
                CIs = np.zeros((dim, 2))
                
                fun0 = -optResult.fun
                hess0 = hess(x0)
                
                for i, j in product(range(dim), range(2)):
                    direction = j*2-1
                    op_result = find_CI_bound(x0, ffun, jac, hess, i, direction, 
                                           fun0=fun0, hess0=hess0)
                    CIs[i, j] = op_result.x[i]
            
                
                print("Confidence intervals:")
                print(CIs)
        
        print("Optimization result:")
        print(optResult)
        
        if optResult.fun < 0:
            self.neg_log_likelihood(optResult.x, obsTime, shiftStart, shiftEnd)
        
        if not optResult.success:
            raise RuntimeError("Optimization was not successful")
        
        self.location, self.kappa = optResult.x
        self.AIC = 2*(optResult.fun + 2)
        self.negLL = optResult.fun
        
        
        print("AIC:", self.AIC)
        
            
        return optResult

    def plot(self, normInterval=None, show=True, fileName=None):
        if normInterval is None:
            times = np.linspace(0, 24, 5000)
            normFact = 1
        else:
            times = np.linspace(*normInterval, 5000)
            normFact = 1/self.interval_probability(*normInterval)
            
        if self.location is None or self.kappa is None:
            raise AttributeError("location and kappa must be defined to allow "
                                 + "plotting.")
        plt.plot(times, self.pdf(times)*normFact, 
                 label="Von Mises Distribution")
        plt.xlabel("Day-Time [h]")
        plt.ylabel("Traffic Density")
        plt.ylim(ymin=0)
        plt.legend(loc=2, frameon=False)
        
        if fileName is not None:
            plt.savefig(fileName + ".png", dpi=1000)
            plt.savefig(fileName + ".pdf")
        if show: plt.show()
    
    def pdf(self, t):
        return vonmises.pdf(t, self.kappa, self.location, 12/np.pi)
    
    def sample(self, n=None):
        return np.random.vonmises((self.location-12)*np.pi/12, self.kappa, 
                                  n)*(12/np.pi) + 12
        



def cropData(data, left, right):
    if right <= left:
        raise ValueError("It must be right > left!")
    croppedData = data + 0
    considered = np.ones(data.shape[0], bool)
    for i, row in enumerate(data):
        if row[2] < left or row[2] > right:
            considered[i] = False 
        else:
            croppedData[i, 0] = np.maximum(left, croppedData[i, 0])
            croppedData[i, 1] = np.minimum(right, croppedData[i, 1])
    return croppedData[considered]

def readTimeData(fileName, delimiter=",", missing_values="", crop=None):
    data = np.genfromtxt(fileName, float, delimiter=delimiter, skip_header=True) #, missing_values=missing_values
    if not crop == None:
        data = cropData(data, *crop)
    
    data = np.append(data, np.zeros((1,data.shape[1])), axis=0)
    considered = np.ones(data.shape[0]-1, dtype=bool)
    
    observationTimes = np.unique(data[:,:2][~np.isnan(data[:,:2])])
    observationNumbers = np.zeros_like(observationTimes)
    
    compare = data[0,:2]
    index = 0
    for i, row in enumerate(data):
        if not (row[:2][np.logical_not(np.isnan(row[:2]))] == compare[np.logical_not(np.isnan(compare))]).all():
            try:
                sStart, sEnd = row[:2]
                noEnd = 0
                if np.isnan(compare[0]):
                    minIndex = np.nanargmin(data[index:i, 2]) + index
                    data[index:i, 0] = data[minIndex, 2]
                    considered[minIndex] = False
                    sStart = data[minIndex, 2]
                if np.isnan(compare[1]):
                    maxIndex = np.nanargmax(data[index:i, 2]) + index
                    data[index:i, 1] = data[maxIndex, 2]
                    considered[maxIndex] = False
                    sEnd = data[maxIndex, 2]
                    noEnd = 1
                
                # Note how many shifts were running during each time of the day
                observationNumbers[np.searchsorted(observationTimes, sStart):
                                   np.searchsorted(observationTimes, sEnd)
                                   -noEnd] += 1   
            except ValueError:
                considered[index:i] = False
            
            index = i
            compare = row[:2]
    
    data = data[:-1]
    with np.errstate(invalid='ignore'):
        considered &= (~np.isnan(data[:,2]) & (data[:,2] <= data[:,1]) 
                       & (data[:,2] >= data[:,0]) & (data[:,0] < data[:,1]))
    
    return data[considered], observationTimes, observationNumbers
        

def fitEstimator(fileName, intervalNumber):
    data, observationTimes, observationNumbers = readTimeData(fileName)
    density = TrafficDensityDayTime_PLinear(np.min(data[:, 0]), np.max(data[:, 1]), 
                             intervalNumber)
    density.maximize_likelihood(data[:, 2], data[:, 0], data[:, 1])
    
    return density


if __name__ == '__main__':
    
    
    # This is the name of the file containing the data
    fileName = "ExportBoaterTimes.csv"
    
    # Read the data. They must be in the following format:
    # [Shift Start],[Shift End],[Observation Time]
    data, observationTimes, observationNumbers = readTimeData(fileName)
    
    
    # Fit a von-Mises distribution
    print("Von Mises distribution")
    density2 = TrafficDensityVonMises()
    density2.maximize_likelihood(data[:, 2], data[:, 0], data[:, 1])
    
    # Fit a probability distribution without pre-defined shape. 
    # Use 
    #  - TrafficDensityDayTime_StepFun for a distribution resembling a histrogram
    #  - TrafficDensityDayTime_PLinear for a piece-wise linear distribution
    
    # These values determine how many intervals your histogram has. 
    # Feel free to add, remove, or cahnge the values. It must be positive 
    # integers, though. At least one number followed by a comma is required
    
    start, end = np.min(data[:, 0]), np.max(data[:, 1])
    for n in 5, 16:
        print("Semi-parametric distribution with n =", n)
        
        density = TrafficDensityDayTime_StepFun(start, end, n)
        
        # Uncomment this, if you want a piecewise linear function instead
        #density = TrafficDensityDayTime_PLinear(np.min(data[:, 0]), np.max(data[:, 1]), n)
        
        density.maximize_likelihood(data[:, 2], data[:, 0], data[:, 1])
        
        # Add this distribution to the final output plot, but allow to add
        # further distributions later
        density.plot(False, density2.interval_probability(start, end))
    
    # Plot the von Mises distribution and finish and show the plot
    density2.plot()
    
    # Create a plot showing how many shifts cover which time of the day
    ax2 = plt.twinx()
    ax2.step(observationTimes, observationNumbers, "k", label='Number of shifts')
    ax2.set_ylabel("Number of Shifts")
    plt.legend(loc=1, frameon=False)
    plt.show()
    