'''
'''
import sys
import os
from builtins import FileNotFoundError
from functools import wraps
from time import time
from collections import defaultdict
from itertools import repeat
from copy import copy

import numpy as np
import numpy.lib.recfunctions as rf
import matplotlib.pyplot as plt

from vemomoto_core.npcollections.npext import add_fields
try:
    from graph import FlowPointGraph, FlexibleGraph
except ModuleNotFoundError:
    from .graph import FlowPointGraph, FlexibleGraph
    
IDTYPE = "|S11" 
from vemomoto_core.tools import saveobject
from vemomoto_core.tools.hrprint import HierarchichalPrinter

from vemomoto_core.tools.tee import Tee
if len(sys.argv) > 1:
    teeObject = Tee(sys.argv[1])

np.random.seed()
FILE_EXT = ".rn"

FIGARGS = {"figsize":(2.2,2)}
SHOWCAPTIONS = False
FIGADJUSTMENT = dict(left=0.2, right=0.99, top=0.99, bottom=0.15)

TICKS = {"mean dists":[600, 800, 1000],
         "time slowdown":([8, 13, 18]),
         "mean paths":([0, 15, 30]),
         "number unique candidates":([10, 20, 30], (0.1, 0.2)),
         "time path":([0, 7.5, 15], (0.1, 0.1)),
         "time total":[100, 400, 700]}

def adjust_ticks(xticks, yticks, fact=0.1, hideMid=True, ax=None):
    
    if hasattr(yticks[0], "__iter__"):
        lowerF, upperF = yticks[1]
        yticks = yticks[0]
    else:
        lowerF, upperF = fact, fact
    
    if ax is None:
        ax = plt.gca()
    
    if hideMid:
        ax.set_yticks(yticks[1:-1], minor=True)
        ax.set_xticks(xticks[1:-1], minor=True)
        ax.tick_params(which='minor', length=4)
        yticks = (yticks[0], yticks[-1])
        xticks = (xticks[0], xticks[-1])
        
    ax.set_yticks(yticks)
    lower, upper = min(yticks), max(yticks)
    diffV = upper-lower
    ax.set_ylim(lower-diffV*lowerF, upper+diffV*upperF)
    
    ax.set_xticks(xticks)
    left, right = min(xticks), max(xticks)
    diffH = right-left
    ax.set_xlim(left-diffH*fact, right+diffH*fact)
    
def arrstr(iterarg):
    r = ""
    for elem in iterarg:
        if hasattr(elem, "__iter__"):
            r += str(list(elem))
        else:
            r += str(elem)
        r += ", "
    return r[:-2]

def timing(f):
    @wraps(f)
    def wrap(self, *args, **kw):
        ts = time()
        result = f(self, *args, **kw)
        te = time()
        self.prst("Executing >{}< took {} seconds.".format(f.__name__, te-ts))
        return result
    return wrap

def time_call(f, *args, **kwargs):
    ts = time()
    result = f(self, *args, **kwargs)
    te = time()
    return result, te-ts

def split_figure_vertical(figsize_1, additional_width, rect_1, rect_2):
    """
    figsize_1 is the size of the figure without the color bar
    additional_width is the additional width used for the color bar
    rect_1, rect_2 define where the plotting area and color bar are located
    in their respective sections of the figure
    """
    oldWidth_1 = figsize_1[0]
    newWidth = oldWidth_1 + additional_width
    factor_1 = oldWidth_1 / newWidth
    factor_2 = additional_width / newWidth
    
    figsize = (newWidth, figsize_1[1])
    
    fig = plt.figure(figsize=figsize)
    
    rect_1[0] *= factor_1
    rect_1[2] *= factor_1
    
    rect_2[0] *= factor_2
    rect_2[2] *= factor_2
    rect_2[0] += factor_1
    
    ax1 = fig.add_axes(rect_1)
    ax2 = fig.add_axes(rect_2)
    
    return ax1, ax2
    

def create_min_paths_graph(data, x, xlabel, xticks=None, maxPathBound=30, xlim=None, colorBar=False):
    
    cmap = mpl.cm.get_cmap('viridis')
    
    Y = [np.ones_like(x)]
    YErr = [np.zeros_like(x)]
    colors = [cmap(0)]
    
    figsize = (3, 3)
    rect = [0.2, 0.2, 0.7, 0.7]
    
    if colorBar:
        ax1, ax2 = split_figure_vertical(figsize, 1, rect, 
                                         [0., 0.2, 0.2, 0.7])
    else:
        ax1 = plt.figure(figsize=figsize).add_axes(rect)
    
    for i in range(1, maxPathBound):
        name = "> {} paths".format(i)
        Y.append(data[name][:,0])
        YErr.append(data[name][:,1])
        colors.append(cmap(i/(maxPathBound-1)))
        
    for y, color in zip(Y, colors):
        ax1.fill_between(x, y, color=color)
    
    """
    for y, yErr, color in zip(Y, YErr, colors):
        plt.errorbar(x, y, yerr=yErr, 
                     elinewidth=0.5, capsize=1.5, capthick=0.5,
                     color=color)
    """
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Fraction of orig.-dest. pairs")
    #ax1.set_ylim(0, 1)
    adjust_ticks(xticks, [0, 0.5, 1], 0, ax=ax1)
    
    #if xlim is not None:
    #    ax1.set_xlim(xlim)
    
    #nticks = 4
    #ax1.locator_params(nticks=4)
    #ax1.locator_params(nbins=3)
    
    if colorBar:
        norm = mpl.colors.Normalize(vmin=1,vmax=maxPathBound)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=[0.5, maxPathBound//2-0.5, maxPathBound-0.5], 
                            boundaries=np.arange(maxPathBound+1), cax=ax2
                            )
        
        cbar.ax.set_yticklabels([r'$\geq 1$', r'$\geq {}$'.format(maxPathBound//2),
                                  r'$\geq {}$'.format(maxPathBound)])
        cbar.set_label("Number of paths")
    #plt.tight_layout()

def create_min_paths_graph2(data, x, xlabel, xticks=None, maxPathBound=30, xlim=None, colorBar=False):
    
    cmap = mpl.cm.get_cmap('viridis')
    
    YErr = [np.zeros_like(x)]
    colors = [cmap(0)]
    
    figsize = (3, 3)
    rect = [0.2, 0.2, 0.7, 0.7]
    
    y = np.arange(0.5, maxPathBound+1, 1)
    y[0] = 1
    y[-1] = maxPathBound
    xx = x[:-1] + np.diff(x)/2
    xx = np.concatenate([[x[0]], xx, [x[-1]]])
    X, Y = np.meshgrid(xx, y)
    Z = np.zeros_like(X[:-1, :-1])
    
    if colorBar:
        ax1, ax2 = split_figure_vertical(figsize, 1, rect, 
                                         [0., 0.2, 0.2, 0.7])
    else:
        ax1 = plt.figure(figsize=figsize).add_axes(rect)
    
    rest = 0
    power = 0.4
    f = lambda x: np.log(np.maximum(x, 1e-5))
    f = lambda x: np.power(x, power)
    
    for i in range(1, maxPathBound)[::-1]:
        name = "> {} paths".format(i)
        Z[i] = data[name][:,0] - rest
        rest = data[name][:,0]
    Z[0] = 1 - rest
    
    
    m = ax1.pcolormesh(X, Y, f(Z), cmap=cmap, vmin=f(0), vmax=f(0.4))
    print(np.max(Z))
    """
    for y, yErr, color in zip(Y, YErr, colors):
        plt.errorbar(x, y, yerr=yErr, 
                     elinewidth=0.5, capsize=1.5, capthick=0.5,
                     color=color)
    """
    
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Fraction of pairs")
    #ax1.set_ylim(0, 1)
    adjust_ticks(xticks, [1, 10, 20], 0, ax=ax1)
    
    #if xlim is not None:
    #    ax1.set_xlim(xlim)
    
    #nticks = 4
    #ax1.locator_params(nticks=4)
    #ax1.locator_params(nbins=3)
    
    if colorBar:
        ti = np.array([0, 0.01, 0.1, 0.4])
        cbar = plt.colorbar(m, cax=ax2, ticks=f(ti))
        
        cbar.ax.set_yticklabels(ti)
        cbar.set_label("Number of paths")
    #plt.tight_layout()
        
class GraphTester(FlowPointGraph):
    def __init__(self, fileName, fileNameEdges=None, preprocessingArgs=None):
        
        self.result_dir = "REVC_results_" + fileName
        self.figure_dir = "REVC_figures_" + fileName
        
        self.fileName = fileName
        
        HierarchichalPrinter.__init__(self)
        
        if not fileNameEdges:
            return
        
        self.prst("Reading file", fileNameEdges)
        
        edges = np.genfromtxt(fileNameEdges, delimiter=",",  
                              skip_header = True, 
                              dtype = {"names":["ID", "from_to_original", 
                                                "length", "inspection", 
                                                "lakeID"], 
                                       'formats':[IDTYPE, 
                                                  '2' + IDTYPE, 
                                                  "double", 
                                                  "3" + IDTYPE, 
                                                  IDTYPE]},
                              autostrip = True)
        
        from_to = np.vstack((edges["from_to_original"], 
                             edges["from_to_original"][:,::-1]))
        
        edgeData = rf.repack_fields(edges[["ID", "length", "lakeID"]])
        edgeData = np.concatenate((edgeData, edgeData))
        
        edgeData = add_fields(edgeData, ["inspection"], [object], [None])
        
        vertexID = np.zeros(0, dtype=IDTYPE)
        vertexData = np.zeros(0, dtype=[("significant", bool)])
        
        graph = FlexibleGraph(from_to, edgeData, vertexID, vertexData,
                              replacementMode="shortest", lengthLabel="length")
        
        graph.set_default_vertex_data(True)
        super().__init__(graph, "length", "significant")
        self.preprocessing(preprocessingArgs)
        
        if fileName:
            self.save(fileName)
    
    @staticmethod
    def new(fileName, fileNameEdges, restart=False):
        if restart:
            return GraphTester(fileName, fileNameEdges)
        try:
            print("Loading file", fileName+FILE_EXT)
            return saveobject.load_object(fileName+FILE_EXT)
        except FileNotFoundError:
            return GraphTester(fileName, None)
            
        
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
    
    @timing 
    def preprocessing(self, preprocessingArgs=None):
        if preprocessingArgs is None:
            FlowPointGraph.preprocessing(self, 1, 3, 3, 
                                         expansionBounds=(1.5, 1.5, 2.5),
                                         maxEdgeLength=20) #20)
        else:
            if type(preprocessingArgs) == dict:
                FlowPointGraph.preprocessing(self, **preprocessingArgs)
            else:
                FlowPointGraph.preprocessing(self, *preprocessingArgs)
    
    def test_REVC_range_avg(self, *args, **kwargs):
        
        result = self.test_REVC_range(*args, **kwargs)
        
        result["time path"] /= (result["sourceNo"]*result["sinkNo"])[:,None]
        result["time path"] *= 1000
        
        dtype = []
        for t in result.dtype.descr:
            if len(t) <= 2:
                dtype.append(t)
            else:
                dtype.append((*t[:2], (2,)))
        
        resultAvg = np.zeros(result.size, dtype=dtype)
        for t in result.dtype.descr:
            name = t[0]
            if len(t) <= 2:
                resultAvg[name] = result[name]
            else:
                resultAvg[name][:,0] = np.mean(result[name], 1)
                resultAvg[name][:,1] = np.std(result[name], 1, ddof=1)
            #if name=="number labelled edges":
            #print(name, resultAvg[name], result[name])
            
        return resultAvg
        
    
    def test_REVC_range(self, sourceNo=100, sinkNo=100, 
                        stretchConstant=1.5, 
                        localOptimalityConstant=0.2, 
                        acceptionFactor=0.9, 
                        rejectionFactor=1.1, 
                        repetitions=10,
                        testing=True,
                        restart=False):
        
        args = copy(locals())
        SAVEEXT = ".res"
        
        testing_optimizations = not type(testing) == bool
        if not testing_optimizations:
            del args["testing"]
            
        del args["self"]
        del args["restart"]
        
        fileName = arrstr(args.values())
        if self.result_dir:
            if not os.access(self.result_dir, os.F_OK): 
                os.makedirs(self.result_dir)
            fileName = os.path.join(self.result_dir, fileName)
        
        if not restart:
            try:
                result = saveobject.load_object(fileName+SAVEEXT)
                for n in result.dtype.names:
                    try:
                        if np.max(result[n]) >= 10000:
                            result[n] = result[n] / 1000
                    except AttributeError:
                        pass
                return result
            except FileNotFoundError:
                pass
        
        del args["repetitions"]
        dtype = []
        
        for var, val in args.items():
            if not hasattr(val, "__iter__"):
                varType = type(val)
                args[var] = repeat(val)
            else:
                if type(val)==tuple:
                    val = np.linspace(*val)
                    args[var] = val
                varType = type(val[0])
            dtype.append((var, varType))
            
        argArr = np.array(list(zip(*args.values())), dtype=dtype)
        result = argArr.copy()
        
        for i, arg in enumerate(argArr):
            partResult = self.test_REVC_repetition(*arg, repetitions=repetitions)
            for resultType, resultTypeStr in ("time", "time "), ("result", ""):
                for resName, resVal in partResult[resultType].items():
                    resStr = resultTypeStr+resName
                    try:
                        result[i][resStr] = resVal
                    except ValueError:
                        result = rf.append_fields(result, resStr, np.zeros(1), dtypes=str(repetitions)+"double")
                        result[i][resStr] = resVal
                
        saveobject.save_object(result, fileName+SAVEEXT)
        
        #df = pd.DataFrame(result)
        #df.to_csv(fileName+".csv", index=False)
        
        with open(fileName+".txt", "w") as file:
            file.write(str(result.dtype.names) + "\n")
            np.rec.array(result).tofile(file, "\n")
        
        return result
    
    
    def get_origin_destination_indices(self, sourceNo, sinkNo, trialIndex=None):
        if "trial_dict" not in self.__dict__:
            self.trial_dict = {}
            
        if trialIndex is None or trialIndex not in self.trial_dict:
            self.prst("Drawing origins and destinations")
            endpoints = np.random.choice(len(self.vertices), sourceNo + sinkNo, replace=False)
            fromIndices = endpoints[:sourceNo]
            toIndices = endpoints[sourceNo:]
            if trialIndex is not None:
                self.trial_dict[trialIndex] = (fromIndices, toIndices)
        else:
            fromIndices, toIndices = self.trial_dict[trialIndex]
            fromDiff = max(sourceNo - fromIndices.size, 0)
            toDiff = max(sinkNo - toIndices.size, 0)
            if fromDiff or toDiff:
                allIndices = np.setdiff1d(np.arange(len(self.vertices)),
                                          np.concatenate((fromIndices, 
                                                          toIndices)),
                                          True)
                
                if not allIndices.size: 
                    return fromIndices[:sourceNo], toIndices[:sinkNo]
                
                newIndices = np.random.choice(allIndices, fromDiff+toDiff, 
                                              False)
                
                fromIndices = np.concatenate((fromIndices, 
                                              newIndices[:fromDiff]))
                toIndices = np.concatenate((toIndices, newIndices[fromDiff:]))
            
                self.trial_dict[trialIndex] = (fromIndices, toIndices)
        
        
        return fromIndices[:sourceNo], toIndices[:sinkNo]
    
    def test_REVC_once(self, sourceNo, sinkNo, *args, trialIndex=None):
        
        if sourceNo + sinkNo > len(self.vertices):
            sourceNo = int(round(len(self.vertices) 
                                 * sourceNo / (sourceNo + sinkNo)))
            sinkNo = len(self.vertices) - sourceNo
        
        fromIndices, toIndices = self.get_origin_destination_indices(sourceNo, sinkNo, trialIndex)
        
        if len(args) <= 4:
            return self.find_locally_optimal_paths(fromIndices, toIndices, None, *args,
                                               testing=True)
        return self.find_locally_optimal_paths(fromIndices, toIndices, None, *args)
    
    def test_REVC_repetition(self, sourceNo, sinkNo, *args, repetitions=10, printResult=True):
        
        results = {"time":defaultdict(list), "result":defaultdict(list),
                   "setup":{"number sources":sourceNo,
                            "number sinks":sinkNo,
                            "number repetitions":repetitions,
                            "arguments":args,
                            }}
        
        for i in range(repetitions):
            times, resultsItems = self.test_REVC_once(sourceNo, sinkNo, *args, 
                                                      trialIndex=i).values()
            for key, val in times.items():
                results["time"][key].append(val)
            for key, val in resultsItems.items():
                results["result"][key].append(val)
        
        if printResult:
            print()
            print("=== Results for {} origins, {} destinations, {} repetitions, arguments={} ===".format(
                sourceNo, sinkNo, repetitions, args
                ))
            print("Timing results:")
        
        resultsTime = results["time"]  
        resultsResult = results["result"]  
        for key, resultlist in resultsTime.items():
            resultsTime[key] = resultlist
            if printResult:
                print("  {}: {:4.2f} (std: {:4.2f})".format(key, np.mean(resultlist), np.std(resultlist, ddof=1)))
        
        if printResult:
            print("Result characteristics:")
            
        for key, resultlist in resultsResult.items():
            resultsResult[key] = resultlist
            if printResult:
                print("  {}: {:4.2f} (std: {:4.2f})".format(key, np.mean(resultlist), np.std(resultlist, ddof=1)))
            
        if printResult:
            print()
            
        return results
    
    def test_REVC_approximations(self, acceptionFactors=None, 
                                 rejectionFactors=None,
                                 SoSiNo=100,
                                 repetitions=10,
                                 show=False):
        
        self.prst("Testing REVC approximation constants.")
        self.increase_print_level()

        if acceptionFactors is None:
            acceptionFactors = np.array([0.6, 0.8, 1.])
        if rejectionFactors is None:
            rejectionFactors = np.array([1., 1.1, 1.3, 1.5, 2.])
        
        results = {}
        for acceptionFactor in acceptionFactors:
            results[acceptionFactor] = self.test_REVC_range_avg(
                sourceNo=SoSiNo,
                sinkNo=SoSiNo,
                acceptionFactor=acceptionFactor, 
                rejectionFactor=rejectionFactors,
                repetitions=repetitions)
        #self.save()
        
        if not os.access(self.figure_dir, os.F_OK): 
            os.makedirs(self.figure_dir)
        
        
        characteristics = list(TICKS.keys())
        for characteristic in characteristics:
            plt.figure(**FIGARGS)
            for acceptionFactor, result in results.items():
                plt.errorbar(rejectionFactors, 
                             result[characteristic][:,0], 
                             yerr=result[characteristic][:,1], 
                             label=r"$\gamma = " + str(acceptionFactor)+ r"$",
                             elinewidth=0.5, capsize=1.5, capthick=0.5)
            if SHOWCAPTIONS:
                plt.legend()
                plt.xlabel(r"$\delta$")
                plt.ylabel(characteristic)
            if characteristic in TICKS:
                if characteristic == "time total":
                    l = plt.legend(loc='upper left', fancybox=True, framealpha=0.5) #frameon=False
                    l.get_frame().set_linewidth(0.0)
                maxt = np.max(rejectionFactors)
                adjust_ticks([1, (maxt+1)/2, maxt], TICKS[characteristic])
            plt.subplots_adjust(**FIGADJUSTMENT)
            
            fileName = os.path.join(self.figure_dir, "Apprx_"+characteristic)
            plt.savefig(fileName + ".png", dpi=1000)
            plt.savefig(fileName + ".pdf")
        
        self.decrease_print_level()
        if show:
            plt.show()
        plt.close('all')
    
    def test_REVC_source_sink(self, pairBased=False, show=True):
        self.prst("Testing REVC O-D numbers.")
        self.increase_print_level()
        
        sqrtPairs = np.array([50, 100, 200, 300, 400])
        if pairBased:
            
            sources1 = sinks1 = sqrtPairs
            sources2 = sqrtPairs // 2
            sinks2 = sqrtPairs * 2
        else:
            sources1 = sinks1 = sqrtPairs
            sources2 = sqrtPairs*2 // 5
            sinks2 = sources2*4
            
        
        results = {}
        for sources, sinks, sourceFrac in ((sources1, sinks1, "|O|:|D| = 1:1"), 
                                         (sources2, sinks2, "|O|:|D| = 1:4")):
            results[sourceFrac] = self.test_REVC_range_avg(
                sourceNo=sources, 
                sinkNo=sinks)
        
        self.save()
        if not os.access(self.figure_dir, os.F_OK): 
            os.makedirs(self.figure_dir)
        
        
        characteristics = list(TICKS.keys())
        mult = lambda a, b: a*b/1000
        add = lambda a, b: a+b
        
        
        for f, xlabel in ((mult, "O-D pairs"),
                          (add, "End points")):
            for characteristic in characteristics:
                plt.figure(**FIGARGS)
                for sourceFrac, result in results.items():
                    plt.errorbar(f(result["sourceNo"], result["sinkNo"]), 
                                 result[characteristic][:,0], 
                                 yerr=result[characteristic][:,1],
                                 label=sourceFrac,
                                 elinewidth=0.5, capsize=1.5, capthick=0.5)
                if SHOWCAPTIONS:
                    plt.legend()
                    plt.xlabel(xlabel)
                    plt.ylabel(characteristic)
                if characteristic in TICKS:
                    if characteristic == "time total":
                        l = plt.legend(loc='upper left', fancybox=True, framealpha=0.5) #frameon=False
                        l.get_frame().set_linewidth(0.0)
                    xmax = np.max(f(result["sourceNo"], result["sinkNo"]))
                    adjust_ticks([0, xmax//2, xmax], TICKS[characteristic])
                
                plt.subplots_adjust(**FIGADJUSTMENT)
                
                fileName = os.path.join(self.figure_dir, "OD_"+characteristic)
                plt.savefig(fileName + xlabel + ".png", dpi=1000)
                plt.savefig(fileName + xlabel + ".pdf")
            
            self.decrease_print_level()
        if show:
            plt.show()        
        plt.close('all')
        
    
    
    def test_REVC_stretch(self, show=True):
        
        self.prst("Testing REVC stretch constant.")
        self.increase_print_level()
        
        stretch = np.array([1., 1.1, 1.2, 1.3, 1.5, 1.7, 2])
        
        result = self.test_REVC_range_avg(stretchConstant=stretch)
        
        if not os.access(self.figure_dir, os.F_OK): 
            os.makedirs(self.figure_dir)
        
        create_min_paths_graph(result, stretch, r"$\beta$", [1.1, 1.4, 1.7, 2], 
                               20, stretch[[0, -1]], True)
        #plt.show()
        fileName = os.path.join(self.figure_dir, "Stretch_PathDist")
        plt.savefig(fileName + ".png", dpi=1000)
        plt.savefig(fileName + ".pdf")
        
        characteristics = list(TICKS.keys())
        
        for characteristic in characteristics:
            plt.figure(**FIGARGS)
            plt.errorbar(stretch, result[characteristic][:,0], 
                         yerr=result[characteristic][:,1],
                         elinewidth=0.5, capsize=1.5, capthick=0.5)
            if SHOWCAPTIONS:
                plt.xlabel(r"$\beta$")
                plt.ylabel(characteristic)
            if characteristic in TICKS:
                adjust_ticks([1, 1.5, 2], TICKS[characteristic])
            plt.subplots_adjust(**FIGADJUSTMENT)
             
            fileName = os.path.join(self.figure_dir, "Stretch_"+characteristic)
            plt.savefig(fileName + ".png", dpi=1000)
            plt.savefig(fileName + ".pdf")
            
        self.decrease_print_level()
        
        if show:
            plt.show()
        plt.close('all')
            
    def test_REVC_LOC(self, show=True):
        
        self.prst("Testing REVC local optimality constant.")
        self.increase_print_level()
        
        LOC = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
        
        result = self.test_REVC_range_avg(localOptimalityConstant=LOC)
        
        if not os.access(self.figure_dir, os.F_OK): 
            os.makedirs(self.figure_dir)
        
        create_min_paths_graph(result, LOC, r"$\alpha$", [0.05, 0.2, 0.35, 0.5], 20, 
                               LOC[[0, -1]], False)
        fileName = os.path.join(self.figure_dir, "LOC_PathDist")
        plt.savefig(fileName + ".png", dpi=1000)
        plt.savefig(fileName + ".pdf")
        
        characteristics = list(TICKS.keys())
        for characteristic in characteristics:
            plt.figure(**FIGARGS)
            plt.errorbar(LOC, result[characteristic][:,0], 
                         yerr=result[characteristic][:,1],
                         elinewidth=0.5, capsize=1.5, capthick=0.5)
            if SHOWCAPTIONS:
                plt.xlabel(r"$\alpha$")
                plt.ylabel(characteristic)
            if characteristic in TICKS:
                adjust_ticks([0, 0.25, 0.5], TICKS[characteristic])
            plt.subplots_adjust(**FIGADJUSTMENT)
            fileName = os.path.join(self.figure_dir, "LOC_"+characteristic)
            plt.savefig(fileName + ".png", dpi=1000)
            plt.savefig(fileName + ".pdf")
        
        self.decrease_print_level()
        if show:
            plt.show()
        plt.close('all')
            

    def test_optimizations(self, repetitions=10, optimizations=None, **kwargs):
        
        self.prst("Testing REVC optimizations.")
        self.increase_print_level()
        
        if optimizations is None:
            optimizations = [
                {"None"},
                {"tree_bound"},
                {"pruning_bound"},
                {"pruning_bound_extended"},
                {"find_plateaus"},
                {"reject_identical"},
                {"joint_reject"},
                {"reuse_queries"}
                ]
        if type(optimizations) == str:
            optimizations = [{optimizations}]
        if type(optimizations) == set:
            optimizations = [optimizations]
        elif hasattr(optimizations, "__iter__"):
            optimizations = [set(i) for i in optimizations]
        else:
            raise ValueError("Type of `optimizations` not understood.")
        
        result = self.test_REVC_range_avg(repetitions=repetitions,
                                          testing=optimizations,
                                          **kwargs)
        
        #default_time = result[0]["time total"][0]
        default_dists = result[0]["mean dists"]
        default_paths = result[0]["mean paths"]
        
        dtm, dtv = result[0]["time total"]
        dtv *= dtv
        def ratio(m, s):
            v = s*s
            return (m/dtm-1)*100, np.sqrt((m/dtm)**2 * (v/m**2 + dtv/dtm**2))*100
        
        for opt, row in zip(optimizations, result):
            print("{:24s} time total {:7.1f} ({:5.1f}); time slowdown "
                  "{:5.2f} ({:4.2f}); % change {:4.1f}  ({:4.1f}); "
                  "number plateau peaks {:5.2f}; number labelled edges {:5.2f}; "
                  "number unique candidates {:5.2f}; paths error {:4.2f}; "
                  "dist error {:4.2f}".format(opt.pop()+":", 
                      *row["time total"], *row["time slowdown"], 
                      *ratio(*row["time total"]),
                      row["number plateau peaks"][0], 
                      row["number labelled edges"][0], 
                      row["number unique candidates"][0],
                      np.mean(np.abs(row["mean paths"] - default_paths) / default_paths),
                      np.mean(np.abs(row["mean dists"] - default_dists) / default_dists),
                      )
                  )
            '''
            print(opt.pop(), ": Time total", row["time total"],
                  "| time slowdown", row["time slowdown"],
                  "| time comparison factor", row["time total"][0]/default_time)
        '''
        self.decrease_print_level()
        