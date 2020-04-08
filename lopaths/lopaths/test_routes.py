'''
Created on 20.03.2020

@author: Samuel
'''
import os
import sys

import numpy as np
from matplotlib import pyplot as plt

from hybrid_vector_model.hybrid_vector_model import TransportNetwork, IDTYPE
from vemomoto_core.tools import saveobject
from vemomoto_core.tools.hrprint import HierarchichalPrinter
from vemomoto_core.tools.tee import Tee

try:
    from test_graph import adjust_ticks
except ImportError:
    from .test_graph import adjust_ticks

if len(sys.argv) > 1:
    teeObject = Tee(sys.argv[1])

FILE_EXT = ".rrn"
SAVEEXT = ".roc"
FIGARGS = {"figsize":(4.4,4)}
TICKS = [0, 0.25, 0.5, 0.75, 1]

class RouteTester(TransportNetwork):
    def __init__(self, fileName, fileNameEdges=None, fileNameVertices=None, 
                 preprocessingArgs=None,
                 edgeLengthRandomization=.001, 
                 **printerArgs):
        self.fileNameSave = fileNameSave
        
        self.result_dir = "REVC_results_" + fileName
        self.figure_dir = "REVC_figures_" + fileName
        
        self.fileName = fileName
        
        HierarchichalPrinter.__init__(self)
        
        if not fileNameEdges:
            return
        
        TransportNetwork.__init__(self, fileNameEdges, fileNameVertices, 
                                  destinationToDestination=False, 
                                  preprocessingArgs=preprocessingArgs,
                                  edgeLengthRandomization=edgeLengthRandomization, 
                                  **printerArgs)
        self.preprocessing()
        self.save()
    
    def save(self, fileName=None):
        if "vertices" not in self.__dict__:
            return
        if fileName is None:
            fileName = self.fileName
        self.lock = None
        if fileName is not None:
            fileName += FILE_EXT
            self.prst("Saving the model as file", fileName)
            saveobject.save_object(self, fileName)
    
    @staticmethod
    def new(fileName, fileNameEdges, fileNameVertices, restart=False):
        if restart:
            return RouteTester(fileName, fileNameEdges, fileNameVertices)
        try:
            print("Loading file", fileName+FILE_EXT)
            return saveobject.load_object(fileName+FILE_EXT)
        except FileNotFoundError:
            return RouteTester(fileName, None)
    
    def test_empirical_validity(self, fileNameData, additionalStations=[],
                                level=0.02, stretchConstant=1.5, restart=False,
                                show=True):
        baseFileName = "ROC-{}-{}".format(level, stretchConstant)
        baseFileNameResults = "ROC-"
        
        if self.result_dir:
            if not os.access(self.result_dir, os.F_OK): 
                os.makedirs(self.result_dir)
            fileName = os.path.join(self.result_dir, baseFileName)
            FileNameResults = os.path.join(self.result_dir, baseFileNameResults)
        else:
            fileName = baseFileName
            FileNameResults = baseFileNameResults
        
        
        restartTmp = restart    
        if not restart:
            try:
                ROC = saveobject.load_object(fileName+SAVEEXT)
            except FileNotFoundError:
                restartTmp = True
                
        if restartTmp:
            ROC = self._get_empirical_validity(fileNameData, FileNameResults,
                                               additionalStations, level, 
                                               stretchConstant, restart)
            saveobject.save_object(ROC, fileName+SAVEEXT)
            with open(fileName+".txt", "w") as file:
                file.write(str(ROC.dtype.names) + "\n")
                np.rec.array(ROC).tofile(file, "\n")
        
        AUC = np.trapz(ROC["truePosRate"], ROC["falsePosRate"])
        print("=+=+= beta: {:4.2f}; level: {:4.2f}; AUC: {:4.2f} =+=+=".format(
            stretchConstant, level, AUC))
        plt.plot(ROC["falsePosRate"], ROC["truePosRate"], 
                 label=r'$\beta$={:3.1f} (area={:4.2f})'.format(stretchConstant, AUC))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        if show:
            plt.plot([0, 1], [0, 1], "--", color='k')
        l = plt.legend(loc="lower right")
        l.get_frame().set_linewidth(0.0)
        
        if self.figure_dir:
            if not os.access(self.figure_dir, os.F_OK): 
                os.makedirs(self.figure_dir)
            fileName = os.path.join(self.figure_dir, baseFileName)
        else:
            fileName = baseFileName
        
        plt.savefig(fileName + ".pdf")
        
        if show:
            plt.show()
        
        
    def _get_empirical_validity(self, fileNameData, fileNameResults=None, 
                                additionalStations=[],
                                level=0.02, stretchConstant=1.5,
                                restart=False):
        
        self.prst("Testing the empirical validity of the algorithm.")
        self.increase_print_level()
        self.prst("level:", level)
        self.prst("stretchConstant:", stretchConstant)
        
        rawData = np.genfromtxt(fileNameData, delimiter=",",  
                             skip_header = True, 
                             dtype = {"names":["fromID", "toID", 
                                               "stationID", "count"], 
                                      "formats":[IDTYPE, IDTYPE, IDTYPE, float]},
                             autostrip = True)
        
        data = np.zeros_like(rawData, 
                             dtype={"names":["fromIndex", "toIndex", 
                                             "stationIndex", "count"], 
                                    "formats":[int, int, int, float]})
        
        vertexIDToVertexIndex = {iD:i for i, iD 
                                 in enumerate(self.vertices.array["ID"])}
        
        data["fromIndex"] = [vertexIDToVertexIndex[iD] for iD in rawData["fromID"]]
        data["toIndex"] = [self.sinkIndexToVertexIndex[self.sinkIDToSinkIndex[iD]] for iD in rawData["toID"]]
        data["stationIndex"] = [self.stationIDToStationIndex[iD] for iD in rawData["stationID"]]
        data["count"] = rawData["count"]
        
        pairs = {(fromIndex, toIndex) for fromIndex, toIndex in 
                 data[["fromIndex", "toIndex"]]}
        pairs = {pair:i for i, pair in enumerate(pairs)}
        
        stations = {station:i for i, station in 
                    enumerate(np.unique(data["stationIndex"]))}
        stationIDs = np.zeros(len(stations)+len(additionalStations), dtype=IDTYPE)
        for station, i in stations.items():
            stationIDs[i] = self.stationIndexToStationID[station]
        
        
        for station in additionalStations:
            stationIDs[len(stations)] = station
            stations[self.stationIDToStationIndex[station]] = len(stations)
        
        coverageData = np.zeros(len(pairs), dtype=[("pair", "2int"),
                                                   ("pairOD", "2int"),
                                                   ("coverage", 
                                                    str(len(stations))+"double")
                                                   ])
        origins = np.unique(data["fromIndex"])
        destinations = np.unique(data["toIndex"])
        vertexIndexToSourceIndex = {origin:i for i, origin in enumerate(origins)}
        vertexIndexToSinkIndex = {destination:i for i, destination in enumerate(destinations)}
        
        for pair, i in pairs.items():
            coverageData[i]["pair"] = pair
            coverageData[i]["pairOD"] = (vertexIndexToSourceIndex[pair[0]],
                                         vertexIndexToSinkIndex[pair[1]])
        
        for fromIndex, toIndex, stationIndex, count in data:
            coverageData[pairs[fromIndex, toIndex]]["coverage"][stations[stationIndex]] = count
        
        shortestDistances = None
        locOptConsts = np.array([0.02, 0.03, 0.05, 0.07, 0.1, 0.13, 0.16, 0.2, 
                                 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 1])[::-1]
        
        result = np.zeros(locOptConsts.size+3, dtype=[("alpha", float),
                                                      ("truePosRate", float),
                                                      ("falsePosRate", float)])
        result[0] = (1., 0., 0.)
        result[-1] = (0., 1., 1.)
        result[-2] = (0., 1., 1-np.mean(coverageData["coverage"] >= level))
        
        observedCoverage = coverageData["coverage"] >= level
        
        fileName = fileNameResults
        np.savetxt(fileName+"obscov-{}.txt".format(level), observedCoverage, "%1d", delimiter=',')
        np.savetxt(fileName+"stations.txt", stationIDs, "%11s", delimiter=',')
        
        pairsIDs = np.zeros((len(pairs), 2), dtype=IDTYPE)
        pairsIDs[:,0] = self.vertices.array["ID"][coverageData["pair"][:,0]]
        pairsIDs[:,1] = self.vertices.array["ID"][coverageData["pair"][:,1]]
        np.savetxt(fileName+"pairs.txt", pairsIDs, "%11s", delimiter=',')
        #with open(fileName+"coverageData.txt", "w") as file:
        #    file.write(str(coverageData.dtype.names) + "\n")
        #    np.rec.array(coverageData).tofile(file, "\n")
        
        positiveN = np.sum(observedCoverage)
        negativeN = np.sum(~observedCoverage)
        extCov = ".cov"
        for i, locopt in enumerate(locOptConsts):
            fileName = fileNameResults + "cov_{}_{}".format(stretchConstant,
                                                             locopt)
            restartTmp = False
            if not restart:
                try:
                    coverage = saveobject.load_object(fileName+extCov)
                except FileNotFoundError:
                    restartTmp = True
                    
            if restart or restartTmp:
                if shortestDistances is None:
                    shortestDistances = self.find_shortest_distance_array(origins,
                                                                          destinations)
                _, inspectedRoutes, _ = self.find_locally_optimal_paths(origins, 
                                                destinations,
                                                shortestDistances, 
                                                stretchConstant=stretchConstant,         
                                                localOptimalityConstant=locopt, 
                                                acceptionFactor=0.9,
                                                rejectionFactor=1.1)
                coverage = np.zeros_like(coverageData["coverage"], dtype=bool)
            
                for j, pairOD in enumerate(coverageData["pairOD"]):
                    for station, k in stations.items():
                        coverage[j, k] = (station in inspectedRoutes and 
                                          tuple(pairOD) in inspectedRoutes[station])
                
                saveobject.save_object(coverage, fileName+extCov)
                np.savetxt(fileName+".txt", coverage, "%1d", delimiter=',')
            
            i += 2
            TPR = np.sum(observedCoverage & coverage) / positiveN
            FPR = np.sum((~observedCoverage) & coverage) / negativeN
            result[i] = locopt, TPR, FPR
            print("=== alpha: {:4.2}; TPR: {:4.2}; FPR: {:4.2} ===".format(
                locopt, TPR, FPR))
        
        self.decrease_print_level()
        
        return result
    