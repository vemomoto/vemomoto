# distutils: language=c++
# cython: boundscheck=False, wraparound=False, nonecheck=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
'''  
Created on 05.07.2016

@author: Samuel 
''' 

cimport numpy as np 
from vemomoto_core.npcollections.intquickheapdict cimport intquickheapdict, KEYVALUE
from vemomoto_core.npcollections.npextc cimport FlexibleArray, FlexibleArrayDict 
ctypedef np.long_t INT_DTYPE_t
ctypedef np.double_t FLOAT_DTYPE_t
ctypedef np.uint8_t BOOL_DTYPE_t
 
        
    
# self must actually be a FlowPointGraph... but for now it is ok
cpdef FLOAT_DTYPE_t find_shortest_distance(np.ndarray vertexArr, 
                                           np.ndarray edgeArr,
                                           INT_DTYPE_t fromIndex, 
                                           INT_DTYPE_t  toIndex)
cpdef in_sets(long[:] A, long[:] B, set s1, set s2, BOOL_DTYPE_t[:] result, 
                INT_DTYPE_t offset)