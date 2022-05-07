# distutils: language=c++
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
'''
Created on 05.07.2016

@author Samuel
'''

cimport numpy as np
from libcpp.deque cimport deque
from libcpp.unordered_map cimport unordered_map
#from unordered_map cimport unordered_map
ctypedef np.int8_t INT8
ctypedef np.uint8_t UINT8_t
ctypedef np.double_t DOUBLE_t
ctypedef np.int_t INT_t
cdef extern from "<algorithm>" namespace "std":
    T find[T](T first, T last, long value)
cdef extern from "sectionsum.c":
    void sectionsum(double *arr, long long *indptr, long long arrzise, long long ptrsize, double *out) nogil
    void sectionsum_chosen(double *arr, long long *arrindptr, long long *consd, long long *consdindptr, long long rownumber, double *out) nogil
    void sectionsum_chosen_rows(double *arr, long long *arrindptr, long long *consd,
        long long *consdindptr, long long *rows, long long rownumber, double *out) nogil
    void sectionsum_chosen_rows_fact(double *arr, long long *arrindptr, long long *consd,
        long long *consdindptr, long long *rows, double *factor, long long rownumber, double *out) nogil
    void sectionprod(double *arr, long long *indptr, long long ptrsize, double *out) nogil
    void sectionsum_rowprod(double *arr1, long long *arr1indptr, long long *columns1,
        long long *columns1indptr, long long *columns1rows, long long *rows1,
        double *arr2, long long *arr2indptr, long long *rows2,
        long long outsize, double *out) nogil
    
cdef class FlexibleArray(object):
    cdef:
        public np.ndarray array
        INT8[:] considered_c
        public np.ndarray considered
        readonly long size 
        readonly long space
        readonly long changeIndex
        deque[long] deleted
        readonly bint isStructured
        readonly bint isRecArray
        double extentionFactor
        list aliases
        object zeroitem
        readonly tuple shape
    cdef void __set_zero_item(self)
    cdef long add_by_dict(self, dict keywordData)
    cpdef long add_tuple(self, object data)
    cdef void delitem(self, long index) 
    cdef void _expand(self, long newlines)
    cdef long len(self)
    cdef object getitem(self, long index)
    cdef long setitem(self, long index, object value)
    cdef long setitem_by_dict(self, long index, dict keywordData)
    cdef void make_considered(self, long index)
    cpdef bint is_contiguous(self)
    cdef long __wrap_index(self, long index) except -1
    cdef void __set_attributes_FA(self, np.ndarray array, 
                                  np.ndarray considered, long size, long space, 
                                  long changeIndex, list deleted, bint isStructured, 
                                  bint isRecArray, double extentionFactor,
                                  list aliases, object zeroitem)
    cdef tuple __get_attributes(self)
    
cdef class FlexibleArrayDict(FlexibleArray):
    cdef readonly unordered_map[long, long] indexDict
    #cdef readonly dict indexDict
    cdef void delitem(self, long index)
    cdef object getitem(self, long index)
    cdef long setitem(self, long index, object value)
    cdef long setitem_by_dict(self, long index, dict keywordData)
    cpdef object get(self, long index, object default)
    cdef void __set_attributes_FAD(self, unordered_map[long, long] indexDict)
    cdef tuple __get_attributes(self)
    
cpdef np.ndarray[double] pointer_sum(np.ndarray[double] arr, np.ndarray[long long] indptr)
cpdef np.ndarray[double] pointer_sum_chosen(np.ndarray[double] arr, np.ndarray[long long] indptr, 
                                            np.ndarray[long long] considered, np.ndarray[long long] consideredindptr)
cpdef np.ndarray[double] pointer_sum_chosen_rows(np.ndarray[double] arr, 
                                                 np.ndarray[long long] indptr, 
                                                 np.ndarray[long long] considered, 
                                                 np.ndarray[long long] consideredindptr,
                                                 np.ndarray[long long] rows)
cpdef np.ndarray[double] pointer_sum_chosen_rows_fact(
        np.ndarray[double] arr, np.ndarray[long long] indptr, 
        np.ndarray[long long] considered, np.ndarray[long long] consideredindptr,
        np.ndarray[long long] rows, np.ndarray[double] factor)
cpdef np.ndarray[double] pointer_prod(np.ndarray[double] arr, np.ndarray[long long] indptr)
cpdef np.ndarray[double] pointer_sum_row_prod(
        np.ndarray[double] arr1, np.ndarray[long long] arr1indptr, 
        np.ndarray[long long] columns1, np.ndarray[long long] columns1indptr,
        np.ndarray[long long] columns1rows,
        np.ndarray[long long] rows1, np.ndarray[double] arr2,
        np.ndarray[long long] arr2indptr, np.ndarray[long long] rows2)