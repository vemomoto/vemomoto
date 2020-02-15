
import os
import sys

import numpy as np
import matplotlib
if os.name == 'posix':
    # if executed on a Windows server. Comment out this line, if you are working
    # on a desktop computer that is not Windows.
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import autograd.numpy as ag

try:
    from .hybrid_vector_model import *
except ImportError:
    from hybrid_vector_model import *



class TrafficFactorModel(BaseTrafficFactorModel):
    """
    Gravity model for a factor proportional to the mean traffic flow between
    an origin and a destination.
    
    Can be overwritten to build a model with different covariates.
    All implemented methods and constants must be adjusted!
    """
    
    # The data in the covariate files must have the columns specified below in 
    # the corresponding order!
    ORIGIN_COVARIATES = [("population", 'double'),
                         ("anglers", 'double'),
                         ("canadian", bool)] 
    DESTINATION_COVARIATES = [("area", 'double'),
                       ("perimeter", 'double'),
                       ("campgrounds", 'int'),
                       ("pointsOfInterest", 'int'),
                       ("marinas", 'int'),
                       ("population", 'double')]
    
    def __init__(self, originData, destinationData, postalCodeAreaData, 
                 distances, postalCodeDistances):
        """
        Constructor
        
        See `BaseTrafficFactorModel.__init__`
        """
        super(TrafficFactorModel, self).__init__(originData, destinationData, 
                 postalCodeAreaData, distances, postalCodeDistances)
        
        self.population = originData["population"]
        self.anglers = originData["anglers"]
        self.canadian = originData["canadian"]
        self.lakeArea = destinationData["area"]
        self.lakePerimeter = destinationData["perimeter"]
        self.campgrounds = destinationData["campgrounds"]
        self.pointsOfInterest = destinationData["pointsOfInterest"]
        self.marinas = destinationData["marinas"]
        self.lakePopulation5km = destinationData["population"]
        self.postalCodePopulation = postalCodeAreaData["population"]
        self.postalCodeDists = postalCodeDistances * 0.001 
        self.dists = distances * 0.0001
        
    @staticmethod_inherit_doc(BaseTrafficFactorModel.process_sink_covariates)
    def process_sink_covariates(covariates):
        """#
        Convert number of campgrounds, pointsOfInterest, and marinas to 
        presence/absence data to avoid identifiability issues
        """
        covariates["campgrounds"] = covariates["campgrounds"] > 0
        covariates["pointsOfInterest"] = covariates["pointsOfInterest"] > 0
        covariates["marinas"] = covariates["marinas"] > 0
        return covariates
    
    SIZE = 22
    
    LABELS = np.array([
        "base area",
        "area exponent",
        "base perimeter",
        "perimeter exponent",
        "campground factor",
        "campground exponent",
        "POI factor",
        "POI exponent",
        "marina factor",
        "marina exponent",
        "surrounding population factor",
        "base surrounding population",
        "surrounding population exponent",
        "base PCA distance",
        "PCA distance exponent",
        "PC population exponent",
        "base population",
        "population exponent",
        "base angler number",
        "angler number exponent",
        "canadian factor",
        "distance exponent",
        ])
    
    BOUNDS = np.array(((-2, 2000),
                       (0, 3),
                       (-2, 2000),
                       (0, 3),
                       (-5, 20),
                       (-7, 7),
                       (-5, 20),
                       (-7, 7),
                       (-5, 20),
                       (-7, 7),
                       (-5, 20),
                       (-7, 500),
                       (0, 3),
                       (-5, 7),
                       (-10, 0),
                       (0, 3),
                       (-7, 500),
                       (0, 5),
                       (-7, 500),
                       (0, 5),
                       (-3, 5),
                       (-5, 0)))
    
    #     A0     A^    Pr0     Pr^   
    PERMUTATIONS = [
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, True, False, True,  False, True, True],
         [True, False,  False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, True, False, True,  False, True, True],
         [True, False,  False, False,True,  False, True,  False,  True, False,  True, True,  False, False, False,  False, True, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False, False, True, False, False,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  False, False, False, False, True, False, False,  False, True, True],
         [False, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False, False, True, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  False, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, False,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, True, True, False,  False, False, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, True, False,  False, False, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  False, True, True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  True, True,  True, False, False,  False, False, False, False, False,True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  False, True, True, False, False,  False, False, False, False, False,True, True],
         [True, False, False, False, True,  False, False, False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, False, False, False, False,  True, False,  True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, False, False, False, False,  False,False, True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, False, False,  False,False, True, True,  True, False, False,  False, False, False, True,  False, True, True],
         [True, False, False, False, True,  False, True,  False,  True, False,  False,False, False, False, False,  False, False, False,True,  False, True, True],
    ]
    
    def convert_parameters(self,     
                           dynamicParameters:"(double[]) Free parameters", 
                           considered:"(bool[]) Which parameters are free"):
        """#
        Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model
        
        See `BaseTrafficFactorModel.convert_parameters`
        """
        
        result = [np.nan]*len(considered)
        j = 0
        if considered[0]:
            result[0] = convert_R_pos(dynamicParameters[j])
            j += 1
        
        if considered[1]:
            result[1] = dynamicParameters[j]
            j += 1
            
        for i in range(2, 13):    
            if considered[i]:
                result[i] = convert_R_pos(dynamicParameters[j])
                j += 1
        
        if considered[13]:
            result[13] = dynamicParameters[j]
            j += 1
        if considered[14]:
            result[14] = convert_R_pos(dynamicParameters[j])
            j += 1
        if considered[15]:
            result[15] = dynamicParameters[j]
            j += 1
        if considered[16]:
            result[16] = convert_R_pos(dynamicParameters[j])
            j += 1
        if considered[17]:
            result[17] = dynamicParameters[j]
            j += 1
        if considered[18]:
            result[18] = convert_R_pos(dynamicParameters[j])
            j += 1
        if considered[19]:
            result[19] = dynamicParameters[j]
            j += 1
        if considered[20]:
            result[20] = ag.exp(dynamicParameters[j])
            j += 1
        if considered[21]:
            result[21] = dynamicParameters[j]
            j += 1
        return result
    
    def get_mean_factor(self, parameters, considered, pair=None):
        """# 
        Gravity model for a factor proportional to the traffic flow
        between a jurisdiction and a lake.
        
        See `BaseTrafficFactorModel.get_mean_factor` for further details.
        """
        #return 1 #!!!!!!!!!!!!!!!!!!!!!!!!!!
        postalCodePopulation = self.postalCodePopulation
        if pair is None or tuple(pair)==(None, None):
            population = self.population
            anglers = self.anglers
            canadian = self.canadian
            lakeArea = self.lakeArea
            lakePerimeter = self.lakePerimeter
            campgrounds = self.campgrounds
            pointsOfInterest = self.pointsOfInterest
            marinas = self.marinas
            lakePopulation5km = self.lakePopulation5km
            dists = self.dists
        else:
            if type(pair) == np.ndarray:
                source = pair.T[0]
                sink = pair.T[1]
            else:
                transposed = list(zip(pair))
                source = np.array(transposed[0])
                sink = np.array(transposed[1])
            
            population = self.population[source].ravel()
            anglers = self.anglers[source].ravel()
            canadian = self.canadian[source].ravel()
            lakeArea = self.lakeArea[sink].ravel()
            lakePerimeter = self.lakePerimeter[sink].ravel()
            campgrounds = self.campgrounds[sink].ravel()
            pointsOfInterest = self.pointsOfInterest[sink].ravel()
            marinas = self.marinas[sink].ravel()
            lakePopulation5km = self.lakePopulation5km[sink].ravel()
            if sink is None:
                postalCodeDists = self.postalCodeDists
            else:
                postalCodeDists = self.postalCodeDists[:,sink]
            
            if sink is None:
                dists = self.dists[source]
            elif source is None:
                dists = self.dists[:, sink]
            else:
                dists = self.dists[source, sink]
        
        # lake size covariates
        cons_l0, cons_l1, cons_l2, cons_l3 = considered[:4]
        l0, l1, l2, l3 = parameters[:4]
        
        # lake infrastructure covariates
        li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10, li11 = parameters[4:16]
        cons_li0, cons_li1, cons_li2, cons_li3, cons_li4, cons_li5, cons_li6, \
            cons_li7, cons_li8, cons_li9, cons_li10, cons_li11 = considered[4:16]
        
        # jurisdiction covariates
        j0, j1, j2, j3, j4 = parameters[16:21]
        cons_j0, cons_j1, cons_j2, cons_j3, cons_j4 = jur_cons = considered[16:21]
        
        # distance covariate
        d = parameters[21]
        cons_d = considered[21]
        
        exp = np.exp
        
        if cons_l1 or cons_l0:
            if cons_l0:
                lakeAttractivity = lakeArea / (l0 + lakeArea) 
                if cons_l1:
                    lakeAttractivity = np.power(lakeAttractivity, l1, 
                                                lakeAttractivity)
            else: 
                lakeAttractivity = np.power(lakeArea, l1)
        else:
            lakeAttractivity = np.ones(lakeArea.size)
        if cons_l3 or cons_l2:
            if cons_l2:
                lakeAttractivity2 = lakePerimeter / (l2 + lakePerimeter) 
                if cons_l3:
                    lakeAttractivity *= np.power(lakeAttractivity2, l3, 
                                                 lakeAttractivity2)
                else:
                    lakeAttractivity *= lakeAttractivity2
            else: 
                lakeAttractivity *= np.power(lakePerimeter, l3)
        
            
        lakeAttractivitySum = 1 
        if cons_li0: 
            if cons_li1:
                campgrounds = np.power(campgrounds, li1)
            lakeAttractivitySum = lakeAttractivitySum + li0*campgrounds
        if cons_li2:
            if cons_li3:
                pointsOfInterest = np.power(pointsOfInterest, li3)
            lakeAttractivitySum = lakeAttractivitySum + li2*pointsOfInterest
        if cons_li4:
            if cons_li5:
                marinas = np.power(marinas, li5)
            lakeAttractivitySum = lakeAttractivitySum + li4*marinas
        if cons_li6:
            if cons_li7:
                lakePopulation5km = lakePopulation5km / (lakePopulation5km + li7)
            if cons_li8:
                lakePopulation5km = np.power(lakePopulation5km, li8)
            lakeAttractivitySum = lakeAttractivitySum + li6*lakePopulation5km    
            
        lakeAttractivity *= lakeAttractivitySum    
            
        
        if cons_li11: 
            if not cons_li9:
                li9 = -1
            if cons_li10:
                postalCodeDists = postalCodeDists + li10
            
            tmp = np.sum(np.power(postalCodeDists, li9 
                                  ).__imul__(postalCodePopulation[:,None]), 0)
            tmp /= np.mean(tmp)
            lakeAttractivity *= np.power(tmp, li11)
        
        if not jur_cons.any():
            jurisdictionCapacity = np.ones(population.size)[:,None]
        else:
            if cons_j1 or cons_j0:
                if cons_j0:
                    population = population / (j0 + population)
                else:
                    population = population.copy()
                if cons_j1:
                    jurisdictionCapacity = np.power(population, j1, population)[:,None]
                else:
                    if type(population) == np.ndarray:
                        jurisdictionCapacity = population[:,None]
            else:
                jurisdictionCapacity = 1
                
            
            if cons_j3 or cons_j2:
                if cons_j2:
                    anglers = anglers / (j2 + anglers)
                else:
                    anglers = anglers.copy()
                if cons_j3:
                    jurisdictionCapacity *= np.power(anglers, j3, anglers)[:,None]
                else:
                    jurisdictionCapacity *= anglers[:,None]
            
            if cons_j4:
                jurisdictionCapacity *= exp(canadian*np.log(j4))[:,None]
            
        if cons_d:
            shortestDistances = np.power(dists, d)
        else: 
            shortestDistances = np.ones(dists.shape)
        
        return (lakeAttractivity * jurisdictionCapacity) * shortestDistances
    
    def get_mean_factor_autograd(self, parameters, considered):
        """#
        Autograd version of `get_mean_factor`. The numpy functions are replaced
        by autograd functions to allow for automatic differentiation. This is
        needed for efficient likelihood maximization.
        """
        #return 1 #!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        
        population = self.population
        anglers = self.anglers
        canadian = self.canadian
        lakeArea = self.lakeArea
        lakePerimeter = self.lakePerimeter
        campgrounds = self.campgrounds
        pointsOfInterest = self.pointsOfInterest
        marinas = self.marinas
        lakePopulation5km = self.lakePopulation5km
        postalCodeDists = self.postalCodeDists
        postalCodePopulation = self.postalCodePopulation
        
        exp = ag.exp
        power = ag.power
        
        # lake size covariates
        cons_l0, cons_l1, cons_l2, cons_l3 = considered[:4]
        l0, l1, l2, l3 = parameters[:4]
        
        # lake infrastructure covariates
        li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10, li11 = parameters[4:16]
        cons_li0, cons_li1, cons_li2, cons_li3, cons_li4, cons_li5, cons_li6, \
            cons_li7, cons_li8, cons_li9, cons_li10, cons_li11 = considered[4:16]
        
        # jurisdiction covariates
        j0, j1, j2, j3, j4 = parameters[16:21]
        cons_j0, cons_j1, cons_j2, cons_j3, cons_j4 = jur_cons = considered[16:21]
        
        # distance covariate
        d = parameters[21]
        cons_d = considered[21]
        
        if cons_l1 or cons_l0:
            if cons_l0:
                lakeAttractivity = lakeArea / (l0 + lakeArea) 
                if cons_l1:
                    lakeAttractivity = power(lakeAttractivity, l1)
            else: 
                lakeAttractivity = power(lakeArea, l1)
        else:
            lakeAttractivity = ag.ones(lakeArea.size)
        if cons_l3 or cons_l2:
            if cons_l2:
                lakeAttractivity2 = lakePerimeter / (l2 + lakePerimeter) 
                if cons_l3:
                    lakeAttractivity *= power(lakeAttractivity2, l3)
                else:
                    lakeAttractivity *= lakeAttractivity2
            else: 
                lakeAttractivity *= power(lakePerimeter, l3)
                
        
        lakeAttractivitySum = 1 
        if cons_li0: 
            if cons_li1:
                campgrounds = power(campgrounds, li1)
            lakeAttractivitySum = lakeAttractivitySum + li0*campgrounds
        if cons_li2:
            if cons_li3:
                pointsOfInterest = power(pointsOfInterest, li3)
            lakeAttractivitySum = lakeAttractivitySum + li2*pointsOfInterest
        if cons_li4:
            if cons_li5:
                marinas = power(marinas, li5)
            lakeAttractivitySum = lakeAttractivitySum + li4*marinas
        if cons_li6:
            if cons_li7:
                lakePopulation5km = (lakePopulation5km+1e-200) / (lakePopulation5km + li7) #hack to prevent 0/0
            if cons_li8:
                lakePopulation5km = power(lakePopulation5km, li8)
            lakeAttractivitySum = lakeAttractivitySum + li6*lakePopulation5km    
            
        lakeAttractivity *= lakeAttractivitySum    
        
        if cons_li11: 
            if not cons_li9:
                li9 = -1
            if cons_li10:
                postalCodeDists = postalCodeDists + li10
            
            tmp = ag.sum(power(postalCodeDists, li9) 
                         * (postalCodePopulation[:,None]), 0)
            tmp = tmp / ag.sum(tmp) * tmp.size
            lakeAttractivity *= power(tmp, li11)
        
        if not jur_cons.any():
            jurisdictionCapacity = ag.ones(population.size)[:,None]
        else:
            if cons_j1 or cons_j0:
                if cons_j0:
                    population = population / (j0 + population)
                else:
                    population = population.copy()
                if cons_j1:
                    jurisdictionCapacity = ag.power(population, j1)[:,None]
                else:
                    jurisdictionCapacity = population[:,None]
            else:
                jurisdictionCapacity = 1
                
            
            if cons_j3 or cons_j2:
                if cons_j2:
                    anglers = anglers / (j2 + anglers)
                if cons_j3:
                    jurisdictionCapacity = (jurisdictionCapacity
                                            * power(anglers, j3)[:,None])
                else:
                    jurisdictionCapacity = (jurisdictionCapacity
                                            * anglers[:,None])
            
            if cons_j4:
                jurisdictionCapacity *= exp(canadian*ag.log(j4))[:,None]
        
        if cons_d:
            shortestDistances = power(self.dists, d)
        else: 
            shortestDistances = ag.ones(self.dists.shape)
        
        return (lakeAttractivity * jurisdictionCapacity) * shortestDistances
 



















def main():
    
    #nbinom_fit_test(1000, 1, 100, 0.5)
    #print(probability_equal_lengths_for_distinct_paths(1000, 1e5))
    #sys.exit()
    
    #draw_operating_hour_reward(1.05683124)
    #sys.exit()
    
    restart = True
    restart = False
    #print("test4")
     
      
    #""" 
    
    stationSets = [
        np.array([b'6', b'9', b'20', b'6b']),
        np.array([b'14', b'18', b'22', b'6b', b'18b']),
        ]
    
    fileNameEdges = "LakeNetworkExample_full.csv"
    fileNameVertices = "LakeNetworkExample_full_vertices.csv"
    fileNameOrigins = "LakeNetworkExample_full_populationData.csv"
    fileNameDestinations = "LakeNetworkExample_full_lakeData.csv"
    fileNamePostalCodeAreas = "LakeNetworkExample_full_postal_code_areas.csv"
    fileNameObservations = "LakeNetworkExample_full_observations.csv"
    fileNameObservations = "LakeNetworkExample_full_observations_new2.csv"
    fileNameObservations = "shortExample3_SimulatedObservations.csv"
    fileNameObservations = "shortExample1_SimulatedObservations.csv"
    fileNameObservations = "LakeNetworkExample_full_observations_new.csv"
    fileNameComparison = ""
    complianceRate = 0.5
    
    fileNameSave = "shortExample3"
    fileNameSave = "shortExample2"
    fileNameSaveNull = "shortExampleNull"
    fileNameSave = "shortExample1"
    
    '''
    fileNameEdges = "LakeNetworkExample_mini.csv"
    fileNameVertices = "LakeNetworkExample_mini_vertices.csv"
    fileNameOrigins = "LakeNetworkExample_mini_populationData.csv"
    fileNameDestinations = "LakeNetworkExample_mini_lakeData.csv"
    fileNamePostalCodeAreas = "LakeNetworkExample_mini_postal_code_areas.csv"
    fileNameObservations = "LakeNetworkExample_mini_observations.csv"
    
    #fileNameEdges = "LakeNetworkExample_small.csv"
    #fileNameVertices = "LakeNetworkExample_small_vertices.csv"
    
    fileNameSave = "debugExample"
    #'''
    """ 
    fileNameEdges = "ExportEdges_HighwayTraffic.csv" 
    fileNameEdges = "ExportEdges.csv" 
    fileNameVertices = "ExportVertices.csv"
    fileNameOrigins = "ExportPopulation.csv"
    fileNameDestinations = "ExportLakes.csv"
    fileNamePostalCodeAreas = "ExportPostal_Code_Data.csv"
    fileNameObservations = "ZebraMusselSimulation_1.3-0.35-0.8-1.2-11000_SimulatedObservations.csv"
    fileNameObservations = "ZebraMusselSimulation_1.4-0.2-0.8-1.2-1_SimulatedObservations.csv"
    fileNameObservations = "ZebraMusselSimulation_1.4-0.2-0.8-1.2-1_SimulatedObservationsFalseAdded.csv"
    fileNameObservations = "ExportBoaterSurveyFalseRemoved.csv"
    fileNameObservations = "ZebraMusselSimulation_1.4-0.2-0.8-1.2-11000_SimulatedObservations.csv"
    fileNameObservations = "ZebraMusselSimulation_1.6-0.2-0.8-1.1-1_SimulatedObservations.csv"
    fileNameObservations = "ExportBoaterSurvey.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_fit.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_val.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_with_days_fit.csv"
    fileNameObservations = "ExportBoaterSurvey_HR_with_days_validation.csv"
    complianceRate = 0.7959
    
    fileNameSave = "Sim_1.4-0.1-0.8-1.1-1_PathTest" # for road network only.
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_opt" # for optimization of inspection locations
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_HW_1" 
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_val" # for validation
    fileNameComparison = fileNameSave
    fileNameSave = "Sim_1.4-0.2-1-1-1_HR_HW" #for fit and road traffic estimates
    fileNameSaveNull = "Sim_1.4-0.2-1-1-1_HR_HW_null" #for fit and road traffic estimates
    fileNameSaveNull = "Sim_1.4-0.2-1-1-1_HR_val_null" #for fit and road traffic estimates
    #"""  
     
    #redraw_predicted_observed(fileNameSave, fileNameComparison)
    #sys.exit()
    
    print("Starting test. (2)")
     
    #print("Seed")
    
    #np.random.seed() 
    
    routeParameters = ( 
                          (1.4, .2, 1, 1)
                       )
    #"""
    #"""
    flowParameters = {}
    #best parameters one more parameter 
    flowParameters["considered"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,True,True,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23546394e+03,5.55260234e+00,3.50775439e+00,2.53985567e+01,1.01026970e+03,8.86681452e+02,0,-1.8065007296786513,2.69364013e+00,-3.44611446e+00]
        )
    nullParameters = {}
    #best parameters one more parameter 
    nullParameters["considered"] = flowParameters["considered"].copy()
    nullParameters["considered"][:] = False
    #nullParameters["considered"][:3] = True
    nullParameters["paramters"] = np.array([-50., 10.])
    nullParameters["paramters"] = np.array([7.42643338e-01, 5.15536529e+04])
    
    #print(TrafficFactorModel.convert_parameters(None, flowParameters["paramters"][2:], flowParameters["considered"][2:]))
    #sys.exit()
    """
    #best parameters one less parameter 
    flowParameters["considered"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,False,False,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23546394e+03,5.55260234e+00,3.50775439e+00,2.53985567e+01,1.171835466180462,-1.8065007296786513,2.69364013e+00,-3.44611446e+00]
        )
    
    #best parameters
    flowParameters["considered"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,True,False,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23546394e+03,5.55260234e+00,3.50775439e+00,2.53985567e+01,1.01026970e+03,8.86681452e+02,-1.8065007296786513,2.69364013e+00,-3.44611446e+00]
        )
        
    
    
    # best parameters old parameterization
    flowParameters["considered"] = np.array( 
        [True,True,True,False,False,False,True,False,True,False,True,False,True,True,False,False,False,False,True,False,False,False,True,True]
        )
    flowParameters["paramters"] = np.array(
        [-1.71042747e+01,1.15230704e+00,1.23646394e+03,6.55260234e+00,4.50775439e+00,2.63985567e+01,1.01126970e+03,8.87681452e+02,1.64227810e-01,2.69364013e+00,-3.44611446e+00]
        )
    """
    routeChoiceParameters = [0.048790208690779414, -7.661288616999463, 0.0573827962901976]
    nullRouteChoiceParameters = [1, 0, 0.0001]
    travelTimeParameters = np.array([14.00344885,  1.33680321])
    nullTravelTimeParameters = np.array([0, 1e-10])
    properDataRate = 0.9300919842312746
    
    """
    nullModel = HybridVectorModel.new(
                fileNameBackup=fileNameSaveNull, 
                trafficFactorModel_class=TrafficFactorModel,
                fileNameEdges=fileNameEdges,
                fileNameVertices=fileNameVertices,
                fileNameOrigins=fileNameOrigins,
                fileNameDestinations=fileNameDestinations,
                fileNamePostalCodeAreas=fileNamePostalCodeAreas,
                fileNameObservations=fileNameObservations,
                complianceRate=complianceRate,
                preprocessingArgs=None,
                #preprocessingArgs=(10,10,10),
                #considerInfested=True, 
                #findPotentialRoutes=True,
                edgeLengthRandomization=0.001,
                routeParameters=(1.0001, 0.9999, 0.5, 2), 
                readSurveyData=True,
                properDataRate=properDataRate,
                #createRouteChoiceModel=True,
                fitRouteChoiceModel=True,
                #readOriginData=True,
                #readOriginData=True, 
                travelTimeParameters=nullTravelTimeParameters, 
                fitTravelTimeModel=True,
                fitFlowModel=True,
                routeChoiceParameters=nullRouteChoiceParameters, #continueRouteChoiceOptimization=True,
                flowParameters=nullParameters, #continueTrafficFactorOptimization=True, #readDestinationData=True,  readPostalCodeAreaData=True, , #  #findPotentialRoutes=True, #  extrapolateCountData=True , # #readSurveyData=True   ###  #  #   findPotentialRoutes=False ,  readSurveyData=True 
                restart=restart, #readSurveyData=True, 
                )
    sys.exit()
    """
    model = HybridVectorModel.new(
                fileNameBackup=fileNameSave, 
                trafficFactorModel_class=TrafficFactorModel,
                fileNameEdges=fileNameEdges,
                fileNameVertices=fileNameVertices,
                fileNameOrigins=fileNameOrigins,
                fileNameDestinations=fileNameDestinations,
                fileNamePostalCodeAreas=fileNamePostalCodeAreas,
                fileNameObservations=fileNameObservations,
                complianceRate=complianceRate,
                preprocessingArgs=None,
                #preprocessingArgs=(10,10,10),
                #considerInfested=True, 
                #findPotentialRoutes=True,
                edgeLengthRandomization=0.001,
                routeParameters=routeParameters, 
                #readSurveyData=True,
                properDataRate=properDataRate,
                #createRouteChoiceModel=True,
                #fitRouteChoiceModel=True,
                #readOriginData=True,
                #readOriginData=True, 
                fitFlowModel=True,
                routeChoiceParameters=routeChoiceParameters, continueRoutChoiceOptimization=False,
                flowParameters=flowParameters, continueTrafficFactorOptimization=True, #readDestinationData=True,  readPostalCodeAreaData=True, , #  #findPotentialRoutes=True, #  extrapolateCountData=True , # #readSurveyData=True   ###  #  #   findPotentialRoutes=False ,  readSurveyData=True 
                travelTimeParameters=travelTimeParameters, 
                #fitTravelTimeModel=True,
                restart=restart, #readSurveyData=True, 
                )
    #model.compare_travel_time_distributions(model.fileName)
    
    #sys.exit()
    
    '''
    model = saveobject.load_object(fileNameSave + ".vmm")
    #'''
    #"""
    #model.save_model_predictions()
    '''
    model.save_simulated_observations(shiftNumber=2500, dayNumber=600, 
                                      stationSets=stationSets)
    '''
    #"""
    model.create_quality_plots(saveFileName=model.fileName, worstLabelNo=5,
                               comparisonFileName=fileNameComparison)
    model.test_1_1_regression(20, saveFileName=model.fileName,
                              comparisonFileName=fileNameComparison)
    model.save(model.fileName)
    sys.exit()
    model.save_simulated_observations()
    model.check_count_distributions_NB(fileName=model.fileName+"-pvals")
    sys.exit()
    #"""
    #'''
    #model.optimize_inspection_station_placement(20, saveFileName=model.fileName)
    allowedShifts = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    allowedShifts = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    allowedShifts = [6, 9, 10, 14, 18]
    allowedShifts = [6., 7., 8., 9., 10., 11., 12., 13., 14., 16., 18., 20., 23.]
    allowedShifts = [4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22]
    allowedShifts = [2, 4 ]
    allowedShifts = [4, 6, 8, 9, 10, 11, 12, 13, 15, 17, 20, 22]
    allowedShifts = [5, 8, 10, 11, 12, 15, 19, 22]
    
    """
    RouteChoiceModel.NOISEBOUND = 0.2
    print("Setting noise bound to", RouteChoiceModel.NOISEBOUND)
    model.fit_route_model(routeParameters, True)
    RouteChoiceModel.NOISEBOUND = 0.02
    print("Setting noise bound to", RouteChoiceModel.NOISEBOUND)
    model.fit_route_model(routeParameters, True)
    """
    
    routeChoiceParamCandidates = [
                                  None,
                                  [0.00739813, -5.80269191, 0.7815986],
                                  [0.19972783, -5.69001292, 0.012684587],
                                  ]
    """
    RouteChoiceModel.NOISEBOUND = 0.2
    for r in routeChoiceParamCandidates:
        print("Setting Route Choice parameters to", r)
        model.fit_route_model(routeParameters, True, np.array(r))
        model.fit_flow_model(flowParameters=flowParameters, continueFlowOptimization=True,
                             refit=True)
    """
        
    def setRouteParameters(m, val, refit=True):
        if val is not None:
            m.NOISEBOUND = 1
            m.fit_route_model(routeParameters, True, np.array(val), False, get_CI=False)
            if refit:
                m.fit_flow_model(flowParameters=flowParameters, continueFlowOptimization=True,
                                 refit=True, get_CI=False)
    def setNoise(m, val):
        if val is not None:
            print("Set noise to", val)
            m.NOISEBOUND = 1
            routeParams = np.array(m.routeModel["routeChoiceModel"].parameters)
            routeParams[0] = val
            m.fit_route_model(True, routeParams, False, get_CI=False)
    defaultArgs = dict(costShift=3.5, costSite=1., shiftLength=8., costRoundCoeff=None, 
                       nightPremium=(5.5, 21, 5), baseTimeInv=24, timeout=3000,
                       init_greedy=True, costBound=80.)#, loadFile=False) 
    
    """
    model.set_infested(b'J54145')
    model.set_infested(b'J54170')
    model.set_infested(b'J54185')
    model.fileName += "Idaho"
    #"""
    model.save_model_predictions()
    
    #model.fileName += "_noinit_"
    #model.create_caracteristic_plot("costBound", [20., 80., 160.], **defaultArgs)
    #model.create_budget_plots(5, 75, 15, **defaultArgs)
    #model.create_budget_plots(55, 100, 10, **defaultArgs)
    #model.create_budget_plots(100, 150, 11, **defaultArgs)
    #model.create_budget_plots(135, 150, 4, **defaultArgs)
    
    sys.exit()
    #model.create_budget_plots(5, 150, 30, **defaultArgs)
    
    
    model.create_caracteristic_plot("costBound", [25., 50., 100.], characteristicName="Budget", **defaultArgs)
    
    #model.create_caracteristic_plot(setRouteParameters, routeChoiceParamCandidates, 
    #                                "NoiseRefit", [0.047, 0.007, 0.2],
    #                                **defaultArgs)
    noise = [0.001, 0.05, 0.2]
    #model.create_caracteristic_plot(setNoise, noise, "Noise level", **defaultArgs)
    
    sys.exit()
    
    for ignoreRandomFlow in False, True:
        #profile("model.optimize_inspection_station_operation(2, 1, 30, 6., allowedShifts=allowedShifts, costRoundCoeff=1, baseTimeInv=18, ignoreRandomFlow=ignoreRandomFlow, saveFileName=model.fileName)", 
        #        globals(), locals()) 
        #"""        
        model.optimize_inspection_station_operation(3.5, 1., 10., 7, #80
                                                    #allowedShifts=allowedShifts, #[6, 8, 10, 11, 12, 14], 
                                                    costRoundCoeff=0.5, 
                                                    nightPremium=(1.2, 22, 6),
                                                    baseTimeInv=24,
                                                    ignoreRandomFlow=ignoreRandomFlow,
                                                    integer=True,
                                                    extended_info=True
                                                    )
    #'''
    """
    print(find_shortest_path(model.roadNetwork.vertices.array,
                             model.roadNetwork.edges.array,
                             0, 9))
    """
    
    """
    stations = [b'386', b'307', b'28']
        
    fromIDs = [b'J54130', b'J54181', b'J54173']
    
    toIDs = [b'L329518145A', b'L328961702A', b'L328974235A']
    
    ps = []
    for stationID, fromID, toID in iterproduct(stations, fromIDs, toIDs):
        #print("Consider journeys from", fromID, "to", toID, "observed at", 
        #      stationID)
        fname = model.fileName + str(stationID + fromID + toID)
        p = model.compare_distributions(stationID, fromID, toID, 15, saveFileName=fname)
        if p is not None:
            ps.append(p)
    
    try:
        print("p distribution:", np.min(ps), np.max(ps), np.histogram(ps))
    except Exception:
        pass
       
    #"""
    #model.compare_distributions(b'3', b'1', b'L1', 15, saveFileName=model.fileName)
    #model.test_1_1_regression(20, routeParameters[0], model.fileName + str(routeParameters[0]))
    #model.save_simulated_observations(shiftNumber=1000, fileName=model.fileName+"1000")
    #model.save_simulated_observations()
    plt.show()
    
    """
    considered = np.array([True] * 7)
    #considered[2] = False
    #x0Stat = (np.log(5), np.sqrt(-np.log(0.7)), np.sqrt(-np.log(.1)), -3, np.sqrt(-np.log(0.1))) 
    #x0Flex = np.array((np.log(2), 1, np.log(1), -1, 1, 1, -2))
    x0Stat = (1, 0.5, 5, -3, 1.5) 
    x0Stat = (5.5, 0.5, 1.5, -3, 1.5) 
    x0Flex = np.array((2, 1, 1, -1, 1, 1, -2))
    x0 = np.array((*x0Stat, *x0Flex[considered]))
    #model.simulate_count_data_test(5, x0, covariates=considered)
    model.save_simulated_observations(x0, considered, "area", shiftNumber=2000, fileName=model.fileName+"1000")
    #"""
    
    """
    
    #profile("network = TransportNetwork(fileNameEdges, fileNameVertices)", globals(), locals())
    if exists(fileNameSave) and not restart:
        print("Loading file...")
        network = saveobject.load_object(fileNameSave)
        print("Edge number", len(network.edges))
        print("Edge size", network.edges.size)
    else:
        network = TransportNetwork(fileNameEdges, fileNameVertices)
        print("Edge number", len(network.edges))
        network.lock = None 
        print("Saving file...")
        saveobject.save_object(network, fileNameSave)
    
    #network.find_potential_routes(1.5, .2)
    #profile("network.find_potential_routes(1.5, .2)", globals(), locals())  
    #print("Timed execution:", Timer(network.find_potential_routes).timeit(1))
    #network.find_alternative_paths_test(1.5)
    for stretch, lo, prune in (
                               (1.25, .1, .7),
                               (1.5, .2, .7),
                               (1.25, .2, .7),
                               (1.25, .1, 1),
                               (1.25, .2, 1),
                               (1.5, .2, 1),
                               ):
        print("Timed execution:", Timer("network.find_potential_routes(stretch, lo, prune)", globals=locals()).timeit(1))
        print("="*80)
        print()
        print("="*80)
    """
    
if __name__ == '__main__':
    #f = "ZebraMusselSimulation_1.4-0.2-1-1-1_HR_opt[3.5, 1, 50.0, 8, (1.2, 22, 6), 0].dat"
    #o = saveobject.load_object(f)
    #print(o)
    
    
    main()
    # LD_PRELOAD=../anaconda3/lib/libmkl_core.so python ...