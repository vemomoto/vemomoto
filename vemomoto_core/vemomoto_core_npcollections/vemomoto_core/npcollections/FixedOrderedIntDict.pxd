# distutils: language=c++
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
Created on 26.11.2016

@author: Samuel
'''
cimport numpy as np
ctypedef np.long_t INT_DTYPE_t

cdef class FixedOrderedIntDict(object):
    cdef: 
        dict dict
        long[:] key_array_c
        long[:] value_array_c
        readonly np.ndarray key_array
        readonly np.ndarray value_array
    cpdef object items(self)
    cpdef long[:] keys(self)
    cpdef long[:] values(self)
    cdef long len(self)
    cdef bint contains(self, item)
    cdef long getitem(self, key)
    cdef void __set_attributes(self, dict itemdict, 
                               np.ndarray key_array, 
                               np.ndarray value_array)
    cdef tuple __get_attributes(self)
