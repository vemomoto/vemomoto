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
from functools import partial
from libcpp.deque cimport deque
import warnings
cdef extern from "heapoperations_c.c":
    void heapifyUp(double *heap, long *keys, long *positions, long start)
    void heapifyDown(double *heap, long *keys, long *positions, long start,
        long size)


cdef class intquickheapdict(object):
    
    def __init__(self, data=None, initSize=1000):

        if not data is None:
            data = tuple(zip(*data))
            if data:
                keys, priorities = data
                dataLen = len(keys)
                initSize = max(initSize, dataLen)
            else: 
                data = None
        
        self.internalKeys_nparr = np.empty(initSize, dtype=INT_DTYPE)
        self.internalKeys = self.internalKeys_nparr
        self.positions_nparr = np.empty(initSize, dtype=INT_DTYPE)
        self.positions = self.positions_nparr
        self.heap_nparr = np.empty(initSize, dtype=FLOAT_DTYPE)
        self.heap = self.heap_nparr
        self.keys_nparr = np.empty(initSize, dtype=INT_DTYPE)
        self.keys = self.keys_nparr
        
            
        if not data is None:
            self.heap_nparr[:dataLen] = priorities
            self.internalKeys_nparr[:dataLen] = np.arange(dataLen)
            self.positions_nparr[:dataLen] = np.arange(dataLen)
            self.keys_nparr[:dataLen] = keys
            self.dict = {key:internalKey for internalKey, key in 
                         enumerate(keys)}
            
        cdef:
            long i
            long*  internalKeys = <long*> self.internalKeys_nparr.data
            long* positions = <long*> self.positions_nparr.data
            double* heap = <double*> self.heap_nparr.data
            
        if not data is None:
            for i in range(1, dataLen):
                heapifyUp(heap, internalKeys, positions, i)
            
            self.size = dataLen
            
        else:
            self.dict = {}
            self.size = 0
        
        self.space = initSize
        
    def __setitem__(intquickheapdict self, INT_DTYPE_t key, 
                    FLOAT_DTYPE_t priority):
        self.setitem(key, priority)
    
    cpdef void setitem(intquickheapdict self, INT_DTYPE_t key, 
                       FLOAT_DTYPE_t priority):
        cdef: 
            INT_DTYPE_t internalKey
            INT_DTYPE_t index
            bint increasePriority
        
        if key in self.dict:
            internalKey = self.dict[key]
            index = self.positions[internalKey]
            increasePriority = self.heap[index] < priority
        else:
            increasePriority = False
            index = self.size
            self.size += 1
            if not self.deleted.empty():
                internalKey = self.deleted.front()
                self.deleted.pop_front()
            else:
                internalKey = index
                if internalKey == self.space:
                    self._extend()
            self.positions[internalKey] = index 
            self.keys[internalKey] = key
            self.dict[key] = internalKey
            self.internalKeys[index] = internalKey
        
        self.heap[index] = priority
        
        if increasePriority:
            heapifyDown(<double*> self.heap_nparr.data, <long*> self.internalKeys_nparr.data, 
                        <long*> self.positions_nparr.data, index, self.size)
        else:
            heapifyUp(<double*> self.heap_nparr.data, <long*> self.internalKeys_nparr.data, 
                      <long*> self.positions_nparr.data, index)
        
    cdef void _extend(intquickheapdict self):
        self.space *= 2
        cdef INT_DTYPE_t newSpace = self.space
        
        try:
            self.heap_nparr.resize(newSpace, refcheck=False)
        except ValueError:
            self.heap_nparr = np.concatenate((self.heap_nparr, 
                                              np.empty_like(self.heap_nparr)))
        
        try:
            self.positions_nparr.resize(newSpace, refcheck=False)
        except ValueError:
            self.positions_nparr = np.concatenate((self.positions_nparr, 
                                        np.empty_like(self.positions_nparr)))
        
        try:
            self.keys_nparr.resize(newSpace, refcheck=False)
        except ValueError:
            self.keys_nparr = np.concatenate((self.keys_nparr, 
                                              np.empty_like(self.keys_nparr)))
        try:
            self.internalKeys_nparr.resize(newSpace, refcheck=False)
        except ValueError:
            self.internalKeys_nparr = np.concatenate((self.internalKeys_nparr, 
                                                np.empty_like(
                                                    self.internalKeys_nparr)))
        self.heap = self.heap_nparr
        self.positions = self.positions_nparr
        self.keys = self.keys_nparr
        self.internalKeys = self.internalKeys_nparr
        
        
    cdef KEYVALUE popitem_c(intquickheapdict self):
        
        if not self.size:
            raise IndexError("The heap is empty.")
        
        cdef:
            INT_DTYPE_t removeIndex
            INT_DTYPE_t resultValue
            INT_DTYPE_t size
            INT_DTYPE_t internalIndex
            FLOAT_DTYPE_t resultPriority
        
        # remove the top item
        removeIndex = self.internalKeys[0]
        resultValue = self.keys[removeIndex]
        del self.dict[resultValue]
        resultPriority = self.heap[0]
        self.deleted.push_back(removeIndex)
        self.size = size = self.size - 1
        
        # rebuild heap
        self.heap[0] = self.heap[size]
        self.internalKeys[0] = internalIndex = self.internalKeys[size]
        self.positions[internalIndex] = 0
        heapifyDown(<double*> self.heap_nparr.data, <long*> self.internalKeys_nparr.data, 
                    <long*> self.positions_nparr.data, 0, size)
        
        return KEYVALUE(resultValue, resultPriority)
    
    def popitem(self):
        result = self.popitem_c()
        return result.key, result.value
    
    cdef KEYVALUE peekitem_c(intquickheapdict self):
        if not self.size:
            raise IndexError("The heap is empty.")
        return KEYVALUE(self.keys[self.internalKeys[0]], self.heap[0])
    
    def peekitem(self):
        result = self.peekitem_c()
        return result.key, result.value
        
    cpdef FLOAT_DTYPE_t get(intquickheapdict self, INT_DTYPE_t key, 
                           FLOAT_DTYPE_t default):
        if not key in self.dict:
            return default
        cdef INT_DTYPE_t index = self.dict[key]
        return self.heap[self.positions[index]]
    
    def __delitem__(self, key):
        self.delitem(key)
        
    cdef void delitem(intquickheapdict self, INT_DTYPE_t key):
        cdef:
            INT_DTYPE_t removeIndex
            INT_DTYPE_t internalDelIndex
            INT_DTYPE_t internalIndex
            INT_DTYPE_t size
            long[:] positions = self.positions
            long[:] internalKeys = self.internalKeys
            double[:] heap = self.heap
            bint increasePriority
            
        # remove the top item
        removeIndex = self.dict.pop(key)
        internalDelIndex = positions[removeIndex]
        self.deleted.push_back(removeIndex)
        self.size = size = self.size - 1
        
        # rebuild heap
        increasePriority = heap[internalDelIndex] < heap[size]
        heap[internalDelIndex] = heap[size]
        internalKeys[internalDelIndex] = internalIndex = internalKeys[size]
        positions[internalIndex] = internalDelIndex
        
        if increasePriority:
            heapifyDown(<double*> self.heap_nparr.data, <long*> self.internalKeys_nparr.data, 
                        <long*> self.positions_nparr.data, 
                        internalDelIndex, size)
        else:
            heapifyUp(<double*> self.heap_nparr.data, <long*> self.internalKeys_nparr.data, 
                      <long*> self.positions_nparr.data, 
                      internalDelIndex)
            
    def items(self):
        return self.__iter__()
    
    def __len__(self):
        return self.len()
    
    cdef INT_DTYPE_t len(intquickheapdict self):
        return self.size
        
    def __repr__(self):
        n = 20
        if self.size > n:
            m = n
            addStr = ", ... ]"
        else:
            m = self.size
            addStr = "]" 
        return ("quickheapdict with length " + str(self.size) + ": " 
                + str(list(zip(self.keys_nparr[
                    self.internalKeys[:m]], self.heap[:m])))[:-1] + addStr)
        
    def __str__(self):
        return self.__repr__()
    
    def __contains__(self, item):
        return item in self.dict
    
    cpdef FLOAT_DTYPE_t getitem(intquickheapdict self, INT_DTYPE_t key):
        cdef INT_DTYPE_t index = self.dict[key]
        return self.heap[self.positions[index]]
    
    def __getitem__(self, key):
        return self.getitem(key)
    
    def __iter__(self):
        return zip(self.keys_nparr[self.internalKeys[:self.size]], 
                   self.heap[:self.size])
