
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

# The first command line argument specifies the output file to which all output
# will be written.     
from vemomoto_core.tools.tee import Tee                
if len(sys.argv) > 1:
    teeObject = Tee(sys.argv[1])


class TrafficFactorModel(BaseTrafficFactorModel):
    """Gravity model for a factor proportional to the mean boater flow between
    jurisdictions and lakes. 
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
                           dynamicParameters, 
                           parametersConsidered):
        """#
        Converts an array of given parameters to an array of standard (maximal)
        length and in the parameter domain of the model
        
        See `BaseTrafficFactorModel.convert_parameters`
        """
        
        result = [np.nan]*len(parametersConsidered)
        j = 0
        if parametersConsidered[0]:
            result[0] = convert_R_pos(dynamicParameters[j])
            j += 1
        
        if parametersConsidered[1]:
            result[1] = dynamicParameters[j]
            j += 1
            
        for i in range(2, 13):    
            if parametersConsidered[i]:
                result[i] = convert_R_pos(dynamicParameters[j])
                j += 1
        
        if parametersConsidered[13]:
            result[13] = dynamicParameters[j]
            j += 1
        if parametersConsidered[14]:
            result[14] = convert_R_pos(dynamicParameters[j])
            j += 1
        if parametersConsidered[15]:
            result[15] = dynamicParameters[j]
            j += 1
        if parametersConsidered[16]:
            result[16] = convert_R_pos(dynamicParameters[j])
            j += 1
        if parametersConsidered[17]:
            result[17] = dynamicParameters[j]
            j += 1
        if parametersConsidered[18]:
            result[18] = convert_R_pos(dynamicParameters[j])
            j += 1
        if parametersConsidered[19]:
            result[19] = dynamicParameters[j]
            j += 1
        if parametersConsidered[20]:
            result[20] = ag.exp(dynamicParameters[j])
            j += 1
        if parametersConsidered[21]:
            result[21] = dynamicParameters[j]
            j += 1
        return result
    
    def get_mean_factor(self, parameters, parametersConsidered, pair=None):
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
        cons_l0, cons_l1, cons_l2, cons_l3 = parametersConsidered[:4]
        l0, l1, l2, l3 = parameters[:4]
        
        # lake infrastructure covariates
        li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10, li11 = parameters[4:16]
        cons_li0, cons_li1, cons_li2, cons_li3, cons_li4, cons_li5, cons_li6, \
            cons_li7, cons_li8, cons_li9, cons_li10, cons_li11 = parametersConsidered[4:16]
        
        # jurisdiction covariates
        j0, j1, j2, j3, j4 = parameters[16:21]
        cons_j0, cons_j1, cons_j2, cons_j3, cons_j4 = jur_cons = parametersConsidered[16:21]
        
        # distance covariate
        d = parameters[21]
        cons_d = parametersConsidered[21]
        
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
    
    def get_mean_factor_autograd(self, parameters, parametersConsidered):
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
        cons_l0, cons_l1, cons_l2, cons_l3 = parametersConsidered[:4]
        l0, l1, l2, l3 = parameters[:4]
        
        # lake infrastructure covariates
        li0, li1, li2, li3, li4, li5, li6, li7, li8, li9, li10, li11 = parameters[4:16]
        cons_li0, cons_li1, cons_li2, cons_li3, cons_li4, cons_li5, cons_li6, \
            cons_li7, cons_li8, cons_li9, cons_li10, cons_li11 = parametersConsidered[4:16]
        
        # jurisdiction covariates
        j0, j1, j2, j3, j4 = parameters[16:21]
        cons_j0, cons_j1, cons_j2, cons_j3, cons_j4 = jur_cons = parametersConsidered[16:21]
        
        # distance covariate
        d = parameters[21]
        cons_d = parametersConsidered[21]
        
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
 


def example():
    """Shows how an example model can be fitted.
    
    The files for the example are provided in the subfolder ``Example``.
    
    """
    
    # Reuse earlier results if possible
    restart = False
    
    # Declare the file names. Because we assume here that the 
    # files are in a subdirectory 'Example', we need to merge 
    # the file names accordingly.
    # See the documentation for HybridVectorModel.new for a 
    # detailed description of the files and their contents.
    folder = "Example"
    fileNameEdges = os.path.join(folder, "Edges.csv")
    fileNameVertices = os.path.join(folder, "Vertices.csv")
    fileNameOrigins = os.path.join(folder, "PopulationData.csv")
    fileNameDestinations = os.path.join(folder, "LakeData.csv")
    fileNamePostalCodeAreas = os.path.join(folder, "PostalCodeAreas.csv")
    fileNameObservations = os.path.join(folder, "SurveyData.csv")
    
    # Set the compliance rate of travellers. This is the fraction of
    # travellers who would stop at a survey location and comply with a survey.
    # Typically, this rate cannot be computed directly from 
    # survey data and must therefore be specified independently.
    complianceRate = 0.8
    
    # File name of the model
    fileNameSave = "Example"
    
    # These parameters define which routes are deemed likely.
    # The first parameter is the factor by how much an admissible
    # route may be longer than the shortest route. 
    # The second parameter specifies the length of subpaths of the 
    # route that are required to be optimal (length given as fraction 
    # of the total length). 0: no restrictions, 1: only optimal paths
    # are considered. 
    # The last two parameters control internal approximations. Choosing 
    # 1 in both cases yields exact results.
    routeParameters = (1.4, .2, 1, 1)
    
    # create and fit a hybrid traffic model
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
                routeParameters=routeParameters, 
                restart=restart
                )
    
    
if __name__ == '__main__':
    example()
