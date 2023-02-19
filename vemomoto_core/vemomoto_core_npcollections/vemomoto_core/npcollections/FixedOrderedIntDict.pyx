# distutils: language=c++
#cython: boundscheck=False, wraparound=False, nonecheck=False
'''
Created on 26.11.2016

@author: Samuel
'''
import numpy as np 
cimport numpy as np
np.import_array()
INT_DTYPE = np.int_
FLOAT_DTYPE = np.double
from ._FixedOrderedIntDict_reduce import rebuild_FixedOrderedIntDict
 


cdef class FixedOrderedIntDict(object):
    
    def __init__(self, keys, values=None, inputdict=None, copy=True, check=True):
        
        if not copy and type(keys) == np.ndarray:
            self.key_array = keys
        else:
            self.key_array = np.array(keys, dtype=INT_DTYPE)
        if inputdict is None:
            if values is None:
                raise ValueError("Either values or inputdict must be given.")
            self.dict = {key:value for key, value in zip(keys, values)}
            dictCheck = False
        else:
            dictCheck = True
            self.dict = inputdict.copy if copy else inputdict
        
        if values is None or (check and inputdict is not None):
            self.value_array = np.array([inputdict[key] for key in keys],
                                          dtype=INT_DTYPE)
        else:
            if not copy and type(values) == np.ndarray:
                self.value_array = values
            else:
                self.value_array = np.array(values, dtype=INT_DTYPE)
        
        if check:
            if not (len(self.value_array) == len(self.key_array) 
                    and len(self.value_array) == len(self.dict)):
                raise ValueError("The sizes of the inputs must be equal.")
            if dictCheck:
                for key in keys:
                    if key not in dict:
                        raise ValueError("The key "+str(key)+" is not in the "
                                         + "given dict.")
        self.value_array_c = self.value_array
        self.key_array_c = self.key_array 
            
        
    cpdef object items(self):
        return zip(self.key_array, self.value_array)
    
    cpdef long[:] keys(self):
        return self.key_array
    
    cpdef long[:] values(self):
        return self.value_array
    
    def __len__(self):
        return self.len()
    
    cdef long len(self):
        return len(self.dict)
        
    def __repr__(self):
        return ("FixedOrderedIntDict with keys " + str(self.key_array) + 
                " and values " +  str(self.value_array))
        
    def __str__(self):
        return self.__repr__()
    
    def __contains__(self, item):
        return self.contains(item)
    
    cdef bint contains(self, item):
        return item in self.dict
    
    def __getitem__(self, key):
        return self.getitem(key)
    
    cdef long getitem(self, key):
        return self.dict[key]
    
    def __iter__(self):
        return self.items()
    
    cdef void __set_attributes(self, dict itemdict, 
                               np.ndarray key_array, 
                               np.ndarray value_array):
        self.dict = itemdict
        self.key_array = key_array
        self.key_array_c = key_array
        self.value_array = value_array
        self.value_array_c = value_array
        
    cdef tuple __get_attributes(self):
        return (self.dict, self.key_array, self.value_array)
    
    def __reduce__(self):
        return (rebuild_FixedOrderedIntDict, self.__get_attributes())
    
    @staticmethod
    def new(dict itemdict, np.ndarray key_array, 
            np.ndarray value_array):
        cdef: 
            FixedOrderedIntDict od = FixedOrderedIntDict.__new__(
                                                            FixedOrderedIntDict)
        od.__set_attributes(itemdict, key_array, value_array)
        return od
