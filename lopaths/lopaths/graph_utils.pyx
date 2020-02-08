# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False
''' 
Created on 05.07.2016

@author: Samuel
'''

from itertools import count as itercount

import numpy as np
cimport numpy as np 
np.import_array() 

from vemomoto_core.npcollections.intquickheapdict cimport intquickheapdict, KEYVALUE
from vemomoto_core.npcollections.npextc cimport FlexibleArray, FlexibleArrayDict 
from vemomoto_core.npcollections.FixedOrderedIntDict cimport FixedOrderedIntDict
FLOAT_DTYPE = np.double
INT_DTYPE = np.long
BOOL_DTYPE = np.uint8   



# self must actually be a FlowPointGraph... but for now it is ok
cpdef FLOAT_DTYPE_t find_shortest_distance(np.ndarray vertexArr, 
                                           np.ndarray edgeArr,
                                           INT_DTYPE_t fromIndex, 
                                           INT_DTYPE_t toIndex,
                                           double rTol = 1+1e-7):
    cdef:
        long initSize = 2000
        double[:] reachArr = vertexArr["reachBound"]
        np.ndarray[object, ndim=1] successorArr = vertexArr["successors"]
        np.ndarray[object, ndim=1] predecessorArr = vertexArr["predecessors"]
        np.ndarray[object, ndim=1] neighborArr
        double[:] lengthArr = edgeArr["length"]
        FlexibleArrayDict forwardData
        FlexibleArrayDict backwardData
        FlexibleArrayDict thisData
        FlexibleArrayDict oppositeData
        intquickheapdict forwardQueue
        intquickheapdict backwardQueue
        intquickheapdict thisQueue
        intquickheapdict oppositeQueue
        long  forwardVertex
        long  backwardVertex
        long  thisVertex
        long  neighbor
        long  edge
        long  i
        double forwardCost
        double backwardCost
        double thisCost
        double oppositeCost
        double bestLength
        double reverseCost
        double totalLength
        double reach
        double length
        double newCost
        double neighborCost
        bint update
        KEYVALUE nextVal
        FixedOrderedIntDict neighborDict
        double aTol = 1e-10
    #    short DEBUG = False
        
    
    forwardData = FlexibleArrayDict(initSize, 
                                    dtype={"names":["cost", "parent", 
                                                    "edge"],
                                           "formats":[FLOAT_DTYPE, 
                                                      INT_DTYPE, 
                                                      INT_DTYPE]})
    forwardData.setitem(fromIndex, (0, -1, -1))
    
    backwardData = FlexibleArrayDict(initSize, 
                                     dtype={"names":["cost", "parent", 
                                                     "edge"],
                                            "formats":[FLOAT_DTYPE, 
                                                       INT_DTYPE, 
                                                       INT_DTYPE]})
    backwardData.setitem(toIndex, (0, -1, -1))
    
    forwardQueue = intquickheapdict(((fromIndex, 0),), initSize)
    backwardQueue = intquickheapdict(((toIndex, 0),), initSize)
    
    forwardVertex, forwardCost = fromIndex, 0
    backwardVertex, backwardCost = toIndex, 0
    bestLength = np.inf
    
    while bestLength > (forwardCost + backwardCost) * rTol:
        if forwardCost <= backwardCost:
            thisQueue = forwardQueue
            oppositeQueue = backwardQueue
            thisVertex = forwardVertex
            thisCost = forwardCost
            oppositeCost = backwardCost
            neighborArr = successorArr
            thisData = forwardData
            oppositeData = backwardData
            #if debug: print(">", vertexArr["ID"][thisVertex], np.round(thisCost, 2), np.round(bestLength - (forwardCost + backwardCost) * rTol, 2))
        else:
            thisQueue = backwardQueue 
            oppositeQueue = forwardQueue
            thisVertex = backwardVertex
            thisCost = backwardCost
            oppositeCost = forwardCost
            neighborArr = predecessorArr
            thisData = backwardData
            oppositeData = forwardData
            #if debug: print("<", vertexArr["ID"][thisVertex], np.round(thisCost, 2), np.round(bestLength - (forwardCost + backwardCost) * rTol, 2))
            
        #DEBUG = debug and (thisVertex == 534719 or thisVertex == 535098)
         
        # delete item from queue
        #thisQueue.popitem()
        thisQueue.popitem_c() 
        
        # prune, if necessary (This step is necessary, since
        # early pruning is weakened in order to allow for the fancy
        # termination criterion
        #if reachArr[thisVertex] * rTol < thisCost:
        #    if DEBUG: print("pruned")
        if not reachArr[thisVertex] * rTol < thisCost:
            
            #if DEBUG: print("not pruned")
            
            # check whether the vertex has been labeled from the opposite 
            # side
            reverseCost = oppositeQueue.get(thisVertex, -1.)
            if reverseCost >= 0:                             # if yes
                totalLength = thisCost + reverseCost
                # update best path if necessary
                #if DEBUG: print("update total length", reverseCost, totalLength, bestLength)
                if totalLength + aTol < bestLength:
                    bestLength = totalLength
            
            # set the vertex cost           
            thisData.getitem(thisVertex)["cost"] = thisCost
            
            # process successors
            neighborDict = neighborArr[thisVertex]
            #for neighbor, edge in neighborArr[thisVertex].items():
            for i in range(neighborDict.len()):
                neighbor = neighborDict.key_array_c[i]
                edge = neighborDict.value_array_c[i]
                # early pruning
                reach = reachArr[neighbor] * rTol
                length = lengthArr[edge]
                newCost = thisCost + length
                #if DEBUG: print("neighbour", vertexArr["ID"][neighbor], thisCost, newCost, oppositeCost, reach)
                if reach < oppositeCost:
                    #if DEBUG: print("reach < oppositeCost", oppositeCost, reach, oppositeCost - reach)
                    if reach < thisCost:
                        #if DEBUG: print("reach < thisCost", thisCost, reach, thisCost - reach)
                        break
                    elif reach < newCost:
                        #if DEBUG: print("reach < newCost", newCost, reach, newCost - reach)
                        continue
                
                # if not pruned
                neighborCost = thisQueue.get(neighbor, -1.)
                #if DEBUG: print("neighborCost", neighborCost)
                
                if neighborCost >= 0:   # if neighbor is in the queue
                    if neighborCost > newCost + aTol:
                        neighborData = thisData.getitem(neighbor)
                        neighborData["parent"] = thisVertex
                        neighborData["edge"] = edge
                        update = True
                    else:
                        update = False
                else:
                    #check whether neighbor already scanned
                    if not thisData.indexDict.count(neighbor):
                        thisData.setitem(neighbor, 
                                         (np.nan, thisVertex, edge))
                        update = True
                    else:
                        update = False
                
                if update:
                    #if DEBUG: print("update neighbour")
                    thisQueue.setitem(neighbor, thisCost + length)
                    
                    # check whether neighbor has been scanned from the
                    # opposite direction and update the best path 
                    # if necessary
                    if oppositeData.indexDict.count(neighbor):
                        reverseCost = oppositeData.getitem(neighbor)["cost"]
                        if not np.isnan(reverseCost):
                            totalLength = (thisCost + length 
                                           + reverseCost)
                            if totalLength < bestLength:
                                bestLength = totalLength
        
        if bestLength > (forwardCost + backwardCost) * rTol:        
            if thisQueue.len():
                if forwardCost <= backwardCost:
                    nextVal = thisQueue.peekitem_c()
                    forwardVertex, forwardCost = nextVal.key, nextVal.value
                    #forwardVertex, forwardCost = thisQueue.peekitem()
                else:
                    #backwardVertex, backwardCost = thisQueue.peekitem()
                    nextVal = thisQueue.peekitem_c()
                    backwardVertex, backwardCost = nextVal.key, nextVal.value
            else:
                # warnings.warn
                print("Vertices {} and {} are disconnected.".format(
                    vertexArr[fromIndex]["ID"], 
                    vertexArr[toIndex]["ID"]))
                #print("Tree disconnected.", forwardCost, backwardCost)
                #if backwardCost < forwardCost:
                #    print(backwardQueue)
                break
    return bestLength
    
from collections import deque
def find_shortest_path(np.ndarray vertexArr, 
                                           np.ndarray edgeArr,
                                           INT_DTYPE_t fromIndex, 
                                           INT_DTYPE_t toIndex):
    cdef:
        long initSize = 2000
        double[:] reachArr = vertexArr["reachBound"]
        np.ndarray[object, ndim=1] successorArr = vertexArr["successors"]
        np.ndarray[object, ndim=1] predecessorArr = vertexArr["predecessors"]
        np.ndarray[object, ndim=1] neighborArr
        double[:] lengthArr = edgeArr["length"]
        FlexibleArrayDict forwardData
        FlexibleArrayDict backwardData
        FlexibleArrayDict thisData
        FlexibleArrayDict oppositeData
        intquickheapdict forwardQueue
        intquickheapdict backwardQueue
        intquickheapdict thisQueue
        intquickheapdict oppositeQueue
        long  forwardVertex
        long  backwardVertex
        long  thisVertex
        long  neighbor
        long  edge
        long  i
        long  bestPathMiddleEdge
        double forwardCost
        double backwardCost
        double thisCost
        double oppositeCost
        double bestLength
        double reverseCost
        double totalLength
        double reach
        double length
        double newCost
        double neighborCost
        bint update
        KEYVALUE nextVal
        FixedOrderedIntDict neighborDict
        double rTol = 1+1e-7
        double aTol = 1e-10
    
    forwardData = FlexibleArrayDict(initSize, 
                                    dtype={"names":["cost", "parent", 
                                                    "edge"],
                                           "formats":[FLOAT_DTYPE, 
                                                      INT_DTYPE, 
                                                      INT_DTYPE]})
    forwardData.setitem(fromIndex, (0, -1, -1))
    
    backwardData = FlexibleArrayDict(initSize, 
                                     dtype={"names":["cost", "parent", 
                                                     "edge"],
                                            "formats":[FLOAT_DTYPE, 
                                                       INT_DTYPE, 
                                                       INT_DTYPE]})
    backwardData.setitem(toIndex, (0, -1, -1))
    
    forwardQueue = intquickheapdict(((fromIndex, 0),), initSize)
    backwardQueue = intquickheapdict(((toIndex, 0),), initSize)
    
    forwardVertex, forwardCost = fromIndex, 0
    backwardVertex, backwardCost = toIndex, 0
    bestLength = np.inf
    bestPathMiddleEdge = -1
    
    while bestLength > forwardCost + backwardCost:
        if forwardCost <= backwardCost:
            thisQueue = forwardQueue
            oppositeQueue = backwardQueue
            thisVertex = forwardVertex
            thisCost = forwardCost
            oppositeCost = backwardCost
            neighborArr = successorArr
            thisData = forwardData
            oppositeData = backwardData
        else:
            thisQueue = backwardQueue
            oppositeQueue = forwardQueue
            thisVertex = backwardVertex
            thisCost = backwardCost
            oppositeCost = forwardCost
            neighborArr = predecessorArr
            thisData = backwardData
            oppositeData = forwardData
            
         
        # delete item from queue
        #thisQueue.popitem()
        thisQueue.popitem_c()
        
        # prune, if necessary (This step is necessary, since
        # early pruning is weakened in order to allow for the fancy
        # termination criterion
        if not reachArr[thisVertex] * rTol < thisCost:
                
            # check whether the vertex has been labeled from the opposite 
            # side
            reverseCost = oppositeQueue.get(thisVertex, -1.)
            if reverseCost >= 0:                             # if yes
                totalLength = thisCost + reverseCost
                # update best path if necessary
                if totalLength + aTol < bestLength:
                    bestLength = totalLength
                    bestPathMiddleEdge = oppositeData[thisVertex]["edge"]
            
            # set the vertex cost           
            thisData.getitem(thisVertex)["cost"] = thisCost
            
            # process successors
            neighborDict = neighborArr[thisVertex]
            #for neighbor, edge in neighborArr[thisVertex].items():
            for i in range(neighborDict.len()):
                neighbor = neighborDict.key_array_c[i]
                edge = neighborDict.value_array_c[i]
                # early pruning
                reach = reachArr[neighbor] * rTol
                length = lengthArr[edge]
                newCost = thisCost + length
                if reach < oppositeCost:
                    if reach < thisCost:
                        break
                    elif reach < newCost:
                        continue
                
                # if not pruned
                neighborCost = thisQueue.get(neighbor, -1.)
                
                if neighborCost >= 0:   # if neighbor is in the queue
                    if neighborCost > newCost + aTol:
                        neighborData = thisData.getitem(neighbor)
                        neighborData["parent"] = thisVertex
                        neighborData["edge"] = edge
                        update = True
                    else:
                        update = False
                else:
                    #check whether neighbor already scanned
                    if not thisData.indexDict.count(neighbor):
                        thisData.setitem(neighbor, 
                                         (np.nan, thisVertex, edge))
                        update = True
                    else:
                        update = False
                
                if update:
                    thisQueue.setitem(neighbor, thisCost + length)
                    
                    # check whether neighbor has been scanned from the
                    # opposite direction and update the best path 
                    # if necessary
                    if oppositeData.indexDict.count(neighbor):
                        reverseCost = oppositeData.getitem(neighbor)["cost"]
                        if not np.isnan(reverseCost):
                            totalLength = (thisCost + length 
                                           + reverseCost)
                            if totalLength < bestLength:
                                bestLength = totalLength
                                bestPathMiddleEdge = edge
        
        if bestLength > forwardCost + backwardCost:        
            if thisQueue.len():
                if forwardCost <= backwardCost:
                    nextVal = thisQueue.peekitem_c()
                    forwardVertex, forwardCost = nextVal.key, nextVal.value
                    #forwardVertex, forwardCost = thisQueue.peekitem()
                else:
                    #backwardVertex, backwardCost = thisQueue.peekitem()
                    nextVal = thisQueue.peekitem_c()
                    backwardVertex, backwardCost = nextVal.key, nextVal.value
            else:
                # warnings.warn
                print("Vertices {} and {} are disconnected.".format(
                    vertexArr[fromIndex]["ID"], 
                    vertexArr[toIndex]["ID"]))
                #print("Tree disconnected.", forwardCost, backwardCost)
                #if backwardCost < forwardCost:
                #    print(backwardQueue)
                break
        
            
            
    """
    fromIndexArr = edgeArr["fromIndex"]
    toIndexArr = edgeArr["toIndex"]
    #originalEdgeArr = edgeArr[["originalEdge1", "originalEdge2"]]
    originalEdgeArr = edgeArr[["originalEdge2", "originalEdge1"]]
    isShortcutArr = edgeArr["originalEdge1"] >= 0
    def expandEdge(edgeIndex):
        result = []
        stack = [edgeIndex]
        while stack:
            edge = stack.pop()
            if isShortcutArr[edge]:
                stack.extend(originalEdgeArr[edge])
            else:
                result.append(edge)
        return result
    """
    pathVertexIndices = deque([edgeArr["fromIndex"][bestPathMiddleEdge],
                               edgeArr["toIndex"][bestPathMiddleEdge]])
    while not pathVertexIndices[0] == fromIndex:
        pathVertexIndices.appendleft(forwardData[pathVertexIndices[0]
                                                 ]["parent"])
    while not pathVertexIndices[len(pathVertexIndices)-1] == toIndex:
        pathVertexIndices.append(backwardData[pathVertexIndices[len(pathVertexIndices)-1]
                                                 ]["parent"])
    
    return bestLength, vertexArr[list(pathVertexIndices)]

    
'''
def find_admissible_via_vertices(np.ndarray vertexArr,
                                 np.ndarray edgeArr,
                       np.ndarray[FLOAT_DTYPE_t, ndim=2] shortestDistances, 
                       np.ndarray[FLOAT_DTYPE_t, ndim=2] sourceDistances, 
                       np.ndarray[FLOAT_DTYPE_t, ndim=2] sinkDistances, 
                       list labelData,  # should be typed some time
                       double stretchConstant,
                       double localOptimalityConstant, 
                       #np.ndarray[FLOAT_DTYPE_t, ndim=1] vertexDistances, 
                       INT_DTYPE_t candidateIndex,
                       INT_DTYPE_t vertexIndex, bint noProfile=False):
    
    originalArgs = (shortestDistances, sourceDistances, sinkDistances, 
                    labelData, stretchConstant,
                    localOptimalityConstant, candidateIndex,
                    vertexIndex, True)
    
    cdef:
        np.ndarray[FLOAT_DTYPE_t, ndim=1] thisSourceDistances 
        np.ndarray[FLOAT_DTYPE_t, ndim=1] thisSinkDistances 
        INT_DTYPE_t sourceNumber
        INT_DTYPE_t sinkNumber
        INT_DTYPE_t i
        INT_DTYPE_t already_elsewhere
        np.ndarray[BOOL_DTYPE_t, ndim=1, cast=True] sinksConsidered 
        np.ndarray[BOOL_DTYPE_t, ndim=1, cast=True] sourcesConsidered 
        np.ndarray[BOOL_DTYPE_t, ndim=2, cast=True] sinkInSource 
        np.ndarray[BOOL_DTYPE_t, ndim=2, cast=True] sourceInSink 
        np.ndarray[BOOL_DTYPE_t, ndim=1, cast=True] toInf 
        np.ndarray[BOOL_DTYPE_t, ndim=2, cast=True] admissiblePairs 
        np.ndarray[FLOAT_DTYPE_t, ndim=2] sinkDiffs 
        np.ndarray[FLOAT_DTYPE_t, ndim=2] sourceDiffs 
        np.ndarray[FLOAT_DTYPE_t, ndim=2] distances 
        np.ndarray[INT_DTYPE_t, ndim=1] consideredSources 
        np.ndarray[INT_DTYPE_t, ndim=1] consideredSinks
        np.ndarray[INT_DTYPE_t, ndim=1] consideredSourceIndices
        np.ndarray[INT_DTYPE_t, ndim=1] consideredSinkIndices
        
    thisSourceDistances = sourceDistances[candidateIndex]
    thisSinkDistances = sinkDistances[candidateIndex]
    
    sourceNumber = shortestDistances.shape[0]
    sinkNumber = shortestDistances.shape[1]
    
    sinksConsidered = (~np.isnan(thisSinkDistances)) #.view(BOOL_DTYPE))
    sinksConsidered.dtype = np.bool # necessary, because cython does not support np.bool
    sourcesConsidered = (~np.isnan(thisSourceDistances)) #.view(BOOL_DTYPE))
    sourcesConsidered.dtype = np.bool
    
    sourceDistances = sourceDistances[candidateIndex+1:,
                                      sourcesConsidered]
    sinkDistances = sinkDistances[candidateIndex+1:,
                                  sinksConsidered]
    thisSourceDistances = thisSourceDistances[sourcesConsidered]
    thisSinkDistances = thisSinkDistances[sinksConsidered]
    
    distances = thisSourceDistances[:,None] + thisSinkDistances
    shortestDistances = shortestDistances[sourcesConsidered][
                                                        :,sinksConsidered]
    
    sinkDiffs = thisSinkDistances - sinkDistances
    sourceDiffs = sourceDistances - thisSourceDistances
    
    considered = (~(np.all(np.isnan(sinkDiffs), 1) | 
                   np.all(np.isnan(sourceDiffs), 1)))#.view(BOOL_DTYPE))
    considered.dtype = np.bool
    
    sinkDiffs = np.round(sinkDiffs[considered], 6)
    sourceDiffs = np.round(sourceDiffs[considered], 6) 
    
    """
    sinkInSource = np.array(list(starmap(np.in1d, 
                                 zip(sinkDiffs, sourceDiffs))), 
                            ndmin=2).view(BOOL_DTYPE)
    """                        
    sinkInSource = np.empty_like(sinkDiffs, dtype=BOOL_DTYPE)
    for i in range(sinkDiffs.shape[0]):
        sinkInSource[i] = np.in1d(sourceDiffs[i], sinkDiffs[i])
    sinkInSource.dtype = np.bool  
    
    """
    sourceInSink = np.array(list(starmap(np.in1d, 
                                 zip(sourceDiffs, sinkDiffs))), 
                            ndmin=2).view(BOOL_DTYPE)
    """
    sourceInSink = np.empty_like(sourceDiffs, dtype=BOOL_DTYPE)
    for i in range(sinkDiffs.shape[0]):
        sourceInSink[i] = np.in1d(sinkDiffs[i], sourceDiffs[i])
    sourceInSink.dtype = np.bool
    
    already_elsewhere = 0
    for i in np.nonzero(np.any(sourceInSink, 0))[0]:
        #distances[i][np.any(sinkInSource[sourceInSink[:,i]], 0)] = np.inf
        toInf = np.any(sinkInSource[sourceInSink[:,i]], 0).view(BOOL_DTYPE)
        toInf.dtype = np.bool
        distances[i][toInf] = np.inf
        already_elsewhere += np.sum(toInf)
        
    admissiblePairs = (np.round(distances,6) <= np.round(shortestDistances
                                         *stretchConstant, 6)).view(BOOL_DTYPE)
    
    consideredSources, consideredSinks = np.nonzero(admissiblePairs) 
    
    if not len(consideredSources):
        return 0, already_elsewhere, 0, 0 
    
    consideredSourceIndices = np.nonzero(sourcesConsidered)[0]
    consideredSinkIndices = (np.nonzero(sinksConsidered)[0] 
                             + sourceNumber)
    
    cdef: 
        np.ndarray[long, ndim=1] TTestStartPoints
        np.ndarray[long, ndim=1] TTestStopPoints 
        np.ndarray[long, ndim=1] TTestPoints
        np.ndarray[FLOAT_DTYPE_t, ndim=1] TTestCompareValues 
        np.ndarray[FLOAT_DTYPE_t, ndim=1] distFromStart 
        int sSIndex
        np.ndarray[INT_DTYPE_t, ndim=1] consideredEnds
        np.ndarray[INT_DTYPE_t, ndim=1] consideredOpposites
        np.ndarray[INT_DTYPE_t, ndim=1] consideredEndIndices
        np.ndarray[FLOAT_DTYPE_t, ndim=2] dists
        np.ndarray[INT_DTYPE_t, ndim=1] indices
        np.ndarray[INT_DTYPE_t, ndim=1] opposites
        FlexibleArrayDict thisLabelData
        np.ndarray[FLOAT_DTYPE_t, ndim=1] TValues 
        INT_DTYPE_t endIndex
        INT_DTYPE_t thisVertex
        INT_DTYPE_t parent
        FLOAT_DTYPE_t baseCost
        FLOAT_DTYPE_t T
        FLOAT_DTYPE_t difference
        FLOAT_DTYPE_t cost
        bint extended
    
    # Better: Use structured arrays here
    TTestStartPoints = np.zeros_like(consideredSources, dtype=INT_DTYPE)
    TTestStopPoints = np.zeros_like(consideredSinks, dtype=INT_DTYPE)
    TTestCompareValues = np.zeros_like(consideredSources, dtype=FLOAT_DTYPE)
    distFromStart = np.zeros_like(consideredSources, dtype=FLOAT_DTYPE)
    
    # key: vertexID, value: list of visited sources and sinks
    # could use c maps !!!
    vertexVisits = defaultdict(lambda: (set(), set()))
    
    for sSIndex, consideredEnds, consideredOpposites, consideredEndIndices, TTestPoints, dists in (
            (0, consideredSources, consideredSinks, consideredSourceIndices, TTestStartPoints, shortestDistances),
            (1, consideredSinks, consideredSources, consideredSinkIndices, TTestStopPoints, shortestDistances.T)):
        
        for endIndex in np.unique(consideredEnds):
            i = 0
            indices = np.nonzero(consideredEnds==endIndex)[0]
            opposites = consideredOpposites[indices] #opposites = consideredOpposites[sliceIndex:sliceIndex+count]
            #sliceIndex = sliceIndex+count
            thisLabelData = labelData[consideredEndIndices[endIndex]]
            
            TValues = dists[endIndex, opposites] * localOptimalityConstant
            
            thisVertex, _, baseCost = thisLabelData[vertexIndex]
            
            # if at end point of the path, there is no need for a T-test
            if thisVertex < 0:
                TTestPoints[indices] = vertexIndex
                TTestCompareValues[indices] = np.nan
                vertexVisits[thisVertex][sSIndex].update(
                                                    consideredEnds[indices])
                continue 
            
            for T in TValues:
                while True:
                    if thisVertex in vertexVisits:
                        vertexVisits[thisVertex][sSIndex].update(
                                                consideredEnds[indices[i:]])
                        extended = True
                    else:  
                        extended = False
                    #print("thisLabelData.array[thisLabelData.indexDict[thisVertex]]", 
                    #      thisLabelData.array[thisLabelData.indexDict[thisVertex]])
                    parent, _, cost = thisLabelData[thisVertex]
                    difference = baseCost-cost
                    if difference >= T or parent < 0:
                        TTestPoints[indices[i]] = thisVertex
                        TTestCompareValues[indices[i]] += difference
                        if not sSIndex: distFromStart[indices[i]] = difference
                        if not extended:
                            vertexVisits[thisVertex][sSIndex].update(
                                                consideredEnds[indices[i:]])
                        i += 1
                        break
                    thisVertex = parent
        
    cdef:        
        np.ndarray[BOOL_DTYPE_t, ndim=1] pairsConsidered 
        np.ndarray[INT_DTYPE_t, ndim=1] order 
        INT_DTYPE_t index
        INT_DTYPE_t start
        INT_DTYPE_t stop
        INT_DTYPE_t a
        INT_DTYPE_t b
        FLOAT_DTYPE_t dist
        long j
        
    
    pairsConsidered = np.ones_like(consideredSources, dtype=BOOL_DTYPE)
    pairsConsidered.dtype = np.bool
    
    nanNumber = np.sum(np.isnan(TTestCompareValues))
    if nanNumber == len(TTestCompareValues):  # if all entries are NaN
        return nanNumber, already_elsewhere, 0, 0 
    
    TTestCompareValues = np.round(TTestCompareValues, 6)
    
    order = np.argsort(TTestCompareValues)
    
    # sort the pairs
    TTestStartPoints = TTestStartPoints[order]
    TTestStopPoints = TTestStopPoints[order]
    consideredSources = consideredSources[order]
    consideredSinks = consideredSinks[order]
    
    previousPair = None
    for i, index, start, stop in zip(range(len(order)-nanNumber), order, TTestStartPoints, TTestStopPoints):
        pair = (start, stop)
        if pair == previousPair:
            continue
        if not pairsConsidered[index]:
            continue 
        """
        isShortest = ((np.abs(np.abs(vertexDistances[vertexIndex]
                                     -vertexDistances[start])
                              -distFromStart[index]) < 1e-6) 
                      &
                      (np.abs(np.abs(vertexDistances[stop]
                                     -vertexDistances[start])
                              -TTestCompareValues[index]) < 1e-6)
                      ).any()
        
        if not isShortest:
        """
        dist = np.round(find_shortest_distance(vertexArr, edgeArr, 
                                               start, stop), 6)
        
        if not dist >= TTestCompareValues[index]:
            #pairsConsidered[np.nonzero(np.in1d(consideredSources[i:], vertexVisits[start][0]) & np.in1d(consideredSinks[i:], vertexVisits[stop][1]))[0]+i] = False
            vstart = vertexVisits[start][0]
            vstop = vertexVisits[stop][1]
            for j, a, b in zip(itercount(), consideredSources[i:], 
                               consideredSinks[i:]):
                if a in vstart and b in vstop:
                    pairsConsidered[j+i] = False
        previousPair = pair
        
    res = np.sum(pairsConsidered)
    """   
    if not noProfile and res > 5000:
        profile("self._find_admissible_via_vertices(*originalArgs)", globals(), locals())
        raise Exception()
        sys.exit()
    """
    nans = np.sum(np.isnan(distances))
    tooFar = distances.size-res-nans-already_elsewhere 
    notLO = np.sum(~pairsConsidered)
    """
    print(vertexIndex, tuple(zip(np.nonzero(sourcesConsidered)[0][consideredSources[pairsConsidered]],
                                 np.nonzero(sinksConsidered)[0][consideredSinks[pairsConsidered]] 
                                 )))
    #"""
    #print(res)
    return res, already_elsewhere, tooFar, notLO 
''' 
         
cpdef in_sets(long[:] A, long[:] B, set s1, set s2, BOOL_DTYPE_t[:] result, 
                INT_DTYPE_t offset):
    cdef:  
        INT_DTYPE_t i
    for i in range(len(A)):
        if A[i] in s1 and B[i] in s2:
            result[i+offset] = 0