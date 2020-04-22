'''
Created on 21.06.2018

@author: Samuel
'''
from collections import defaultdict
from copy import copy
from itertools import product as iterproduct, repeat

import numpy as np
from scipy import optimize as op
import numdifftools as nd

from vemomoto_core.npcollections.npext import list_to_csr_matrix, \
    csr_list_to_csr_matrix, convert_R_0_1, convert_R_0_1_reverse, \
    sparseprod, sparsepower, sparsesum, sparsesum_chosen_rows, \
    sparsesum_chosen_rows_fact, sparsesum_row_prod
from vemomoto_core.npcollections.npextc import FlexibleArray
from ci_rvm import find_profile_CI_bound
from vemomoto_core.concurrent.concurrent_futures_ext import ProcessPoolExecutor
from vemomoto_core.tools.hrprint import HierarchichalPrinter

class RouteChoiceModel(HierarchichalPrinter):
    '''
    classdocs
    '''
    
    NOISEBOUND = 0.05
    
    
    def __init__(self, **printerArgs):
        '''
        Constructor
        '''
        self._fit_prepared = False
        self._fitted = False
        HierarchichalPrinter.__init__(self, **printerArgs)
    
    VARIABLE_LABELS = ["P(drive randomly)",
                       "route choice exponent",
                       "P(observable if driving randomly)"]
    
    @property
    def fitted(self):
        return self._fitted
    
    @property
    def prepared(self):
        return self._fit_prepared
    
    def set_fitting_data(self, dayData, shiftData, inspectedRoutes,
                          routeLengths, trafficModel, complianceRate,
                          properDataRate):
        
        #self.trafficModel = trafficModel
        def intervalProbability(*args, **kwargs):
            return trafficModel.interval_probability(*args, **kwargs) * complianceRate * properDataRate
        
        pairToPairIndex = dict()
        pairDayToPairDayIndex = dict()
        pairStationToPairStationIndex = dict()
        
        # table containing all relevant routes, i.e. for each observed
        # ij pair all routes
        pairRoutes = FlexibleArray(10000, dtype=object)
        
        dayTimeFactors = FlexibleArray(10000, dtype=object)
        
        # contains for each day and pair observed on that day
        # the time factor and length for each covered route
        # and a link to the respective normalization factor
        pairDayDType = [
            ("timeFactors1", object),
            ("timeFactors2", object),
            ("timeFactors3", object),
            ("routeIndices", object),
            ("pairIndex", np.int64),
            ("dayIndex", np.int64),
            ]
        pairDayData = FlexibleArray(10000, dtype=pairDayDType)
        
        for shifts, countData in dayData:
            
            probs = intervalProbability(
                        shiftData["shiftStart"][shifts],
                        shiftData["shiftEnd"][shifts],
                        )
            
            dayIndex = dayTimeFactors.add(probs)
            
            for pair in countData.keys():
                dayRouteIndices = defaultdict(lambda: list())
                for shift in shifts:
                    stationIndex = shiftData[shift]["stationIndex"]
                    if (stationIndex in inspectedRoutes and
                            pair in inspectedRoutes[stationIndex]):
                        for ri in inspectedRoutes[stationIndex][pair]:
                            dayRouteIndices[ri].append(shift)
                if pair in pairToPairIndex:
                    pairIndex = pairToPairIndex[pair]
                else: 
                    pairIndex = pairRoutes.add(routeLengths[pair])
                    pairToPairIndex[pair] = pairIndex
                
                if dayRouteIndices:
                    maxRouteIndex = max(dayRouteIndices.keys())
                    dayRouteTimeFactors1 = [0] * (maxRouteIndex+1)
                else: 
                    dayRouteTimeFactors1 = []
                dayRouteTimeFactors2 = []
                dayRouteTimeFactors3 = []
                for routeIndex, routeShifts in dayRouteIndices.items():
                    probs = intervalProbability(
                        shiftData["shiftStart"][routeShifts],
                        shiftData["shiftEnd"][routeShifts],
                        )
                    
                    tmp = np.prod(1-probs)
                    dayRouteTimeFactors1[routeIndex] = tmp
                    dayRouteTimeFactors2.append(tmp*np.sum(probs/(1-probs))) 
                    dayRouteTimeFactors3.append(1-tmp) 
                
                pairDayIndex = pairDayData.add_tuple((dayRouteTimeFactors1,
                                                      dayRouteTimeFactors2,
                                                      dayRouteTimeFactors3,
                                                      list(dayRouteIndices.keys()),
                                                      pairIndex,
                                                      dayIndex))
                pairDayToPairDayIndex[(pair, dayIndex)] = pairDayIndex
            
        
        
        observationDType = [
            ("count", int),
            ("pairDayIndex", np.int64),
            ("pairStationIndex", np.int64),
            ("dayStationIndex", np.int64),
            ]
        observationData = FlexibleArray(10000, dtype=observationDType)
        
        pairStationDType = [
            ("routeIndices", object),
            ("pairIndex", np.int64),
            ]
        pairStationData = FlexibleArray(10000, dtype=pairStationDType)
        
        dayStationDType = [
            ("timeFactor", "double"),
            ("dayIndex", np.int64),
            ]
        dayStationData = np.zeros(shiftData.size, dtype=dayStationDType)
        dayStationData["timeFactor"] = intervalProbability(
            shiftData["shiftStart"], shiftData["shiftEnd"]
            )
        dayStationData["dayIndex"] = shiftData["dayIndex"]
        
        for dayStationIndex, row in enumerate(shiftData):
            
            dayIndex, stationIndex, _, _, countData, _, _, _ = row
            
            for pair, count in countData.items():
                
                pairIndex = pairToPairIndex[pair]
                if not (pair, stationIndex) in pairStationToPairStationIndex:
                    if (stationIndex in inspectedRoutes and
                            pair in inspectedRoutes[stationIndex]):
                        r = inspectedRoutes[stationIndex][pair]
                    else:
                        r = []
                    pairStationIndex = pairStationData.add_tuple((
                        r, pairIndex
                        ))
                    pairStationToPairStationIndex[(pair, stationIndex)] = \
                        pairStationIndex
                else:
                    pairStationIndex = pairStationToPairStationIndex[
                                                    (pair, stationIndex)] 
                
                pairDayIndex = pairDayToPairDayIndex[(pair, dayIndex)]
                observationData.add_tuple((count, pairDayIndex, 
                                           pairStationIndex, dayStationIndex))
        
        
        # contains for each observed source-sink pair the lengths of all 
        # routes. Must be a sparse csr_matrix
        self.pairRoutes = csr_list_to_csr_matrix(pairRoutes.get_array())
        
        
        # contains for each observation day the time factors of the different
        # shifts. Must be a sparse csr_matrix
        self.dayTimeFactors = list_to_csr_matrix(dayTimeFactors.get_array())
        
        # for each observed combination of observation day and 
        # source-sink pair 
        #  - "routeIndices" contains for each entry a list of the indices 
        #    corresponding to the routes in pairRoutes that have been
        #    covered on the respective day 
        #  - "timeFactors" the probability to observe a boater on a route
        #    due to timing issues. This is a list with a time factor for each
        #    covered route on that day from source to sink
        #  - "pairIndex" links this table to pairRoutes by noting which 
        #    row of pairRoutes contains information on the source-sink pair
        #    considered in the respective row of this table
        #  - "dayIndex" links this table to dayTimeFactors by noting which
        #    row of dayTimeFactors contains information on the day
        #    considered in the respective row of this table
        pairDayData = pairDayData.get_array()
        self.pairDayData = {
            "timeFactors1":list_to_csr_matrix(pairDayData["timeFactors1"]),
            "timeFactors2":list_to_csr_matrix(pairDayData["timeFactors2"]),
            "timeFactors3":list_to_csr_matrix(pairDayData["timeFactors3"]),
            "routeIndices":list_to_csr_matrix(pairDayData["routeIndices"],
                                              dtype=np.int64),
            "pairIndex":pairDayData["pairIndex"].copy(),
            "dayIndex":pairDayData["dayIndex"].copy(),
            }
        
        # contains for each observation shift several information:
        observationData = observationData.get_array()
        self.observationData = {n:observationData[n].copy() for n 
                                in observationData.dtype.names}
        
        
        pairStationData = pairStationData.get_array()
        self.pairStationData = {
            "routeIndices":list_to_csr_matrix(pairStationData["routeIndices"],
                                              dtype=np.int64),
            "pairIndex":pairStationData["pairIndex"].copy()
            }
        
        self.dayStationData = dayStationData
        
        self._fit_prepared = True
        
    @staticmethod
    def _convert_parameters(parameters):
        """
        return (convert_R_0_1(parameters[0]),
        """ 
        return (convert_R_0_1(parameters[0])*RouteChoiceModel.NOISEBOUND, #,
                parameters[1],
                convert_R_0_1(parameters[2]))
    
    @staticmethod
    def _convert_parameters_reverse(parameters):
        """
        return (convert_R_0_1_reverse(parameters[0]),
        """
        return (convert_R_0_1_reverse(parameters[0]/RouteChoiceModel.NOISEBOUND), #
                parameters[1],
                convert_R_0_1_reverse(parameters[2]))
    
    @staticmethod
    def _route_normalization_probabilities(pRandom, pObservation,
                                           routePowers, normFactors,
                                           pairDayData, 
                                           dayTimeFactorsProd,
                                           dayTimeFactors,
                                           givenOnlyOneObservation=True,
                                           givenOnlyOneNoiseObservation=True):
        
        if givenOnlyOneObservation or givenOnlyOneNoiseObservation:
            if givenOnlyOneNoiseObservation:
                admissibleRouteProb = sparsesum_chosen_rows_fact(
                    routePowers, pairDayData["routeIndices"], 
                    pairDayData["pairIndex"], pairDayData["timeFactors3"]
                    )
            else:    
                admissibleRouteProb = sparsesum_chosen_rows_fact(
                    routePowers, pairDayData["routeIndices"], 
                    pairDayData["pairIndex"], pairDayData["timeFactors2"]
                    )
            
            admissibleRouteProb *= (1-pRandom) / normFactors[pairDayData["pairIndex"]]
            
            dayTimeFactors2 = copy(dayTimeFactors)
            dayTimeFactors2.data = (dayTimeFactors2.data/
                                   (1-pObservation*dayTimeFactors2.data))
            timeFactors = (sparsesum(dayTimeFactors2) * dayTimeFactorsProd 
                           * (pObservation*pRandom))
            
        else:
            admissibleRouteProb = sparsesum_chosen_rows_fact(
                routePowers, pairDayData["routeIndices"], pairDayData["pairIndex"],
                pairDayData["timeFactors3"]
                )
            
            admissibleRouteProb *= (1-pRandom) / normFactors[pairDayData["pairIndex"]]
            
            timeFactors = 1-dayTimeFactorsProd
            timeFactors *= pRandom
            
        radnomRouteProb = timeFactors[pairDayData["dayIndex"]]
        
        return admissibleRouteProb + radnomRouteProb
        
    def get_nLL_funtions(self):
        
        observations = self.observationData
        pairRoutes = self.pairRoutes
        pairDayData = self. pairDayData
        pairStationData = self.pairStationData
        dayTimeFactors = self.dayTimeFactors
        dayStationData = self.dayStationData
        return self._get_nLL_funtions_static(observations, pairRoutes, 
                                     pairDayData, pairStationData, 
                                     dayStationData, dayTimeFactors)
    
    @staticmethod
    def _get_nLL_funtions_static(observations, pairRoutes, pairDayData,
                                 pairStationData, dayStationData, 
                                 dayTimeFactors):
        nLL = lambda params: RouteChoiceModel._negLogLikelihood_static(
            params, observations, pairRoutes, pairDayData, pairStationData, 
            dayStationData, dayTimeFactors
            )
        
        jac = nd.Gradient(nLL)
        hess = nd.Hessian(nLL)
        return nLL, jac, hess
    
    @staticmethod
    def _negLogLikelihood_static(params, observations, pairRoutes, pairDayData,
                                 pairStationData, dayStationData, dayTimeFactors,
                                 givenOnlyOneNoiseObservation=True):
        
        
        
        pRandom, power, pObservation = RouteChoiceModel._convert_parameters(params)
        
        # This was introduced to fit a null model with noise traffic only
        #pRandom = 1
        #power = 0
        
        routePowers = sparsepower(pairRoutes, power)
        
        normConstants = sparsesum(routePowers)
        pairIndices = pairStationData["pairIndex"][observations["pairStationIndex"]]
        
        if givenOnlyOneNoiseObservation:
            obsProbs = sparsesum_chosen_rows(routePowers, 
                                             pairStationData["routeIndices"],
                                             pairStationData["pairIndex"],
                                             )
            obsProbs *= (1-pRandom) / normConstants[pairStationData["pairIndex"]]
            obsProbs = obsProbs[observations["pairStationIndex"]]
        else:
            #routePowers[pairIndices,
            #            pairStationData["routeIndices"][observations["pairStationIndex"]]
            #            ] 
            #    * pairDayData["timeFactors1"][observations["pairDayIndex"]]
            obsProbs = sparsesum_row_prod(routePowers, 
                                          pairStationData["routeIndices"],
                                          pairIndices,
                                          observations["pairStationIndex"], #necessary to make it contiguous
                                          pairDayData["timeFactors1"],
                                          observations["pairDayIndex"])
        
            obsProbs *= (1-pRandom) / normConstants[pairIndices]
        
            obsProbs /= (1-dayStationData["timeFactor"])[observations["dayStationIndex"]] 
        
        dayTimeFactors2 = copy(dayTimeFactors)
        dayTimeFactors2.data = 1-dayTimeFactors2.data*pObservation
        dayTimeFactorsProd = sparseprod(dayTimeFactors2)
        dayTimeFactorsProd2 = dayTimeFactorsProd * (pObservation*pRandom)
        obsProbsRand = (dayTimeFactorsProd2[dayStationData["dayIndex"]] 
                        / (1-pObservation*dayStationData["timeFactor"]))
        
        obsProbs += obsProbsRand[observations["dayStationIndex"]]
        
        observationLogLikelihood = np.sum(observations["count"] 
                                          * np.log(obsProbs))
        
        normProbabilities = RouteChoiceModel._route_normalization_probabilities(
            pRandom, pObservation, routePowers, normConstants,
            pairDayData, dayTimeFactorsProd, dayTimeFactors,
            givenOnlyOneNoiseObservation=givenOnlyOneNoiseObservation)
        normLogLikelihood = np.sum(observations["count"] 
                                   * np.log(normProbabilities
                                            )[observations["pairDayIndex"]])
        
        # uncomment this, if you are interested in the REAL log-likelihood
        observationLogLikelihood += np.sum(observations["count"] 
                                    * np.log(dayStationData["timeFactor"])[observations["dayStationIndex"]])
        
        return normLogLikelihood-observationLogLikelihood
    
    
    def fit(self, guess=None, improveGuess=False, disp=True):
        """Fits the route choice model.
        
        Parameters
        ----------
        guess : float[]
            Guess for the maximum likelihood estimate.
        improveGuess : bool
            If ``True``, :py:obj:`guess` will be used as initial guess for the 
            model fit. Otherwise, it will be used as the maximum likelihood
            estimate.
        disp : bool
            Whether partial results shall be printed.
        
        """
        
        
        if not self._fit_prepared:
            raise ValueError("The model has not been provided with data for", 
                             "the model fit.")
            
        nLL, jac, hess = self.get_nLL_funtions()
        
        if guess is None:
            bounds = [
                (-20, 20),
                (-20, 1),
                (-20, 20),
                ]
            
            result = op.differential_evolution(nLL, bounds, popsize=500,
                                               mutation=(1.,1.99999), 
                                               maxiter=300, #300, 
                                               disp=disp)
            if disp: 
                print(result)
            x0 = result.x
        elif not improveGuess:
            self.parameters = guess
            self._parameters_converted = self._convert_parameters_reverse(guess)
            self._fitted = True
            return
        else:
            x0 = self._convert_parameters_reverse(guess)
            
        result = op.minimize(nLL, x0, #jac=jac, hess=hess, 
                             bounds=None, options={"maxiter":800, "iprint":2},
                             method="SLSQP")
        result.xOriginal = self._convert_parameters(result.x)
        #result2.jacOriginal = jac(result2.xOriginal, False)
        if disp:
            print("SLSQP result", result)        
            
        self.parameters = result.xOriginal
        self._parameters_converted = result.x
        self._fitted = True
    
    @staticmethod
    def _find_profile_CI_static(observations, pairRoutes, pairDayData, 
                                pairStationData, dayStationData, dayTimeFactors, 
                                index, x0, direction,
                                profile_LL_args={}):
                 
        
        nLL, jac, hess = RouteChoiceModel._get_nLL_funtions_static(
            observations, pairRoutes, pairDayData, pairStationData, 
            dayStationData, dayTimeFactors)
        
        nLL_ = lambda x: -nLL(x)   
        jac_ = lambda x: -jac(x)   
        hess_ = lambda x: -hess(x)   
        
        return find_profile_CI_bound(index, direction, x0, nLL_, jac_, hess_, 
                                     **profile_LL_args)
    
    def get_confidence_intervals(self, fileName=None, show=True, **optim_args):
        
        if not self._fitted:
            raise ValueError("The model must be fitted before confidence "
                             + "intervals can be computed")
        
        x0 = self._parameters_converted
        nLL, jac, hess = self.get_nLL_funtions()
        
        if not "fun0" in optim_args:
            self.prst("Determining logLikelihood")
            optim_args["fun0"] = -nLL(x0)
        if not "hess0" in optim_args:
            self.prst("Determining Hessian of logLikelihood")
            optim_args["hess0"] = -hess(x0)
        
        dim = len(x0)
        result = np.zeros((dim, 2))
        
        labels = np.array(self.VARIABLE_LABELS)
        
        indices, directions = zip(*iterproduct(range(dim), (-1, 1)))
        
        const_args = [self.observationData, self.pairRoutes, self. pairDayData,
                      self.pairStationData, self.dayStationData,
                      self.dayTimeFactors]
        
        self.prst("Creating confidence intervals")
        #try: #, max_workers=13
        with ProcessPoolExecutor(const_args=const_args) as pool:
            mapObj = pool.map(RouteChoiceModel._find_profile_CI_static, 
                              indices, repeat(x0), directions, 
                              repeat(optim_args))
            
            
            for index, direction, r in zip(indices, directions, mapObj):
                result[index][(0 if direction==-1 else 1)
                              ] = self._convert_parameters(r.x)[index]
        
        self.prst("Printing confidence intervals:")
        self.increase_print_level()
        
        x0Orig = self._convert_parameters(x0)
        
        for index, intv in enumerate(result):
            start, end = intv
            strs = []
            for v in start, x0Orig[index], end:
                if (0.01 < np.abs(v) < 1e6) or v == 0:
                    strs.append("{:10.4f}".format(v))
                else:
                    strs.append("{:<10}".format(str(v)))
            self.prst("CI for {:<40}: [{} --- {} --- {}]".format(
                labels[index], *strs))
            
        self.decrease_print_level()
        

#print(RouteChoiceModel._convert_parameters([  4.54319707e+03,  -1.76750980e+01,  -2.64583166e+04]))