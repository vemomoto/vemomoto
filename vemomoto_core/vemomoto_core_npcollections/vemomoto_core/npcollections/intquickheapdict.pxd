# distutils: language=c++
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
Created on 26.11.2016

@author: Samuel
'''
cimport numpy as np
ctypedef np.long_t INT_DTYPE_t
ctypedef np.double_t FLOAT_DTYPE_t
cdef struct KEYVALUE:
    long key
    double value
from libcpp.deque cimport deque

cdef class intquickheapdict(object):
    cdef: 
        long[:] internalKeys
        long[:] positions
        long[:] keys
        double[:] heap
        np.ndarray internalKeys_nparr  # np array view to the data
        np.ndarray positions_nparr
        np.ndarray keys_nparr
        np.ndarray heap_nparr
        INT_DTYPE_t size
        INT_DTYPE_t space
        dict dict
        deque deleted[INT_DTYPE_t]
    
    cpdef void setitem(intquickheapdict self, INT_DTYPE_t key, 
                       FLOAT_DTYPE_t priority)
    cdef void _extend(intquickheapdict self)
    cdef KEYVALUE popitem_c(intquickheapdict self)
    cdef KEYVALUE peekitem_c(intquickheapdict self)
    cpdef FLOAT_DTYPE_t get(intquickheapdict self, INT_DTYPE_t key, 
                           FLOAT_DTYPE_t default)
    cdef void delitem(intquickheapdict self, INT_DTYPE_t key)
    cdef INT_DTYPE_t len(intquickheapdict self)
    cpdef FLOAT_DTYPE_t getitem(intquickheapdict self, INT_DTYPE_t key)
    