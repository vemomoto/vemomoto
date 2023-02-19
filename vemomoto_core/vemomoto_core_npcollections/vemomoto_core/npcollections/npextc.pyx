# distutils: language=c++
#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
'''
Created on 05.07.2016

@author: Samuel
'''

import copy
from libcpp.unordered_map cimport unordered_map
from libc.math cimport NAN

import numpy as np
cimport numpy as np
from cython.parallel import prange, parallel
import cython
np.import_array()

from ._npextc_utils import rebuild_FlexibleArrayDict, rebuild_FlexibleArray
from vemomoto_core.tools.iterext import Repeater

def fields_view(arr, fields):
    #d2 = [(name, arr.dtype.fields[name][0]) for name in fields]
    return np.ndarray(arr.shape, np.dtype({name:arr.dtype.fields[name] 
                                           for name in fields}), 
                      arr, 0, arr.strides)

def remove_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to remove.

    Returns a view of the array `a` (not a copy).  
    """
    dt = a.dtype
    keep_names = [name for name in dt.names if name not in names]
    return fields_view(a, keep_names)

def add_alias(arr, original, alias):
    """
    Adds an alias to the field with the name original to the array arr.
    Only one alias per field is allowed.
    """
    
    if arr.dtype.names is None:
        raise TypeError("arr must be a structured array. Use add_name instead.")
    descr = arr.dtype.descr
    
    try:
        index = arr.dtype.names.index(original)
    except ValueError:
        raise ValueError("arr does not have a field named '" + str(original) 
                         + "'")
    
    if type(descr[index][0]) is tuple:
        raise ValueError("The field " + str(original) + 
                         " already has an alias.")
    
    descr[index] = ((alias, descr[index][0]), descr[index][1])
    arr.dtype = np.dtype(descr)
    return arr

def add_names(arr, names, sizes=None):
    """
    Adds a name to the data of an unstructured array.
    """
    
    if arr.dtype.names is not None:
        raise TypeError("arr must not be a structured array. "
                        + "Use add_alias instead.")
    
    dt = str(arr.dtype)
    if type(names) is str:
        if arr.ndim > 1: 
            sizeStr = str(arr.shape[-1])
        else:
            sizeStr = ''
        descr = [(names, sizeStr + dt)]
    else:
        if len(names) == arr.shape[-1]:
            descr = [(name, dt) for name in names]
        elif sizes is None:
            raise ValueError("If the length of the names array does not match "
                             + "the length of the first dimension of arr, "
                             + "then the sizes of the fields must be given.")
        elif len(sizes) == len(names):
            if sum(sizes) == arr.shape[-1]:
                descr = [(name, str(sizes[i])+dt) for i, name 
                         in enumerate(names)]
            else:
                raise ValueError("The entries in names must sum to the size "
                                 + "of the first dimension of arr.")
        else:
            raise ValueError("The lengths of the size array and the names "
                             + "array must match.")
            
    arr.dtype = np.dtype(descr)
    return arr
    

cdef class FlexibleArray(object):
    '''
    classdocs
    '''
    
    def __init__(self, nparray, copy=True, 
                 dtype=None, recArray=False, extentionFactor=2):
        '''
        Constructor
        '''
        cdef:
            bint considerAll = False
            
        if type(nparray) == int or type(nparray) == tuple:
            nparray = np.zeros(nparray, dtype=dtype)
            copy = False
        else:
            considerAll = True
        
        if not nparray.size:
            raise ValueError("The array must base on a non-empty array.")
        #if nparray.ndim > 1 or type(nparray) == tuple or type(nparray) == list:
        #    raise ValueError("Flexible C Arrays must be 1 dimensional.")
         
        if not type(nparray) == np.ndarray:
            nparray = np.array(nparray)
        
        self.space = nparray.shape[0]
        self.shape = nparray.shape
        
        self.array = nparray.copy() if copy else nparray
        
        self.isStructured = not self.array.dtype.names is None
        if recArray:
            self.array = np.rec.array(self.array, copy=False)
        
        self.isRecArray = type(self.array) == np.recarray
        self.extentionFactor = extentionFactor
        
        self.__set_zero_item()
        
        self.aliases = []
        if considerAll:
            self.considered = np.ones(self.space, dtype=np.int8)
            self.size = self.space
        else:
            self.considered = np.zeros(self.space, dtype=np.int8)
            self.size = 0
        
        self.considered_c = self.considered
        self.considered.dtype = np.bool
        
        self.changeIndex = 0
    
    cdef void __set_zero_item(self):
        if self.isStructured:
            self.zeroitem = tuple(np.zeros(self.shape[1:], 
                                           dtype=self.array.dtype).tolist())
        else:
            self.zeroitem = np.zeros(1, dtype=self.array.dtype)[0]
            
    def add(self, data=None, **keywordData):
        if data is None:
            if self.isStructured:
                data = keywordData
                keywordData = None
            else: 
                data = self.zeroitem
                keywordData = None
        
        if self.deleted.empty():
            if self.size == self.space: self._expand(0)
            index = self.size
            self.__setitem_by_flexible_keywords(index, data, keywordData)
            self.size += 1
        else:
            index = self.deleted.front()
            
            
            self.__setitem_by_flexible_keywords(index, data, keywordData)
            self.deleted.pop_front()
        self.considered[index] = True
        return index
    
    def add_by_keywords(self, **keywordData):
        return self.add_by_dict(keywordData)
        
    cdef long add_by_dict(self, dict keywordData):
        cdef:
            bint deletedEmpty = self.deleted.empty()
            long index
        if deletedEmpty:
            if self.size == self.space: self._expand(0)
            index = self.size
            self.size += 1
        else:
            index = self.deleted.front()
            self.array[index] = self.zeroitem
            
        [self.array[key].__setitem__(index, val) 
                for key, val in keywordData.items()]
        
        if not deletedEmpty:
            self.deleted.pop_front()
        self.considered_c[index] = True
        return index
    
    cpdef long add_tuple(self, object data):
        cdef:
            bint deletedEmpty = self.deleted.empty()
            long index
        if deletedEmpty:
            if self.size == self.space: self._expand(0)
            index = self.size
            self.size += 1
        else:
            index = self.deleted.front()
            
        self.array[index] = data
        if not deletedEmpty:
            self.deleted.pop_front()
        
        self.considered_c[index] = True
        return index
    
    def __delitem__(self, index): 
        if index is None:
            return
        self.delitem(index)
    
    cdef void delitem(self, long index): 
        
        index = self.__wrap_index(index)
        if index == self.size-1:
            self.size -= 1
        else:
            self.deleted.push_back(index)
        self.considered_c[index] = False
    
    def expand(self, newlines=None):
        if not newlines or newlines < 0:
            newlines = int(np.ceil(self.space * (self.extentionFactor - 1)))
        self._expand(newlines)
        
    cdef void _expand(self, long newlines):
        
        if not newlines:
            newlines = <long> (self.space * (self.extentionFactor-1))
        
        self.space += newlines
        self.shape = (self.space,) + self.shape[1:]
        
        newSize = np.prod(self.shape)
        #print("EXPAND from", self.space, "to", self.space+newlines)
        
        try:
            self.array.resize(newSize, refcheck=False)
            if len(self.shape):
                self.array = np.reshape(self.array, self.shape)
            #print("FlexArr expand array SUCCESSFUL")
        except ValueError:
            addShape = (newlines,) + self.shape[1:]
            self.array = np.concatenate((self.array, np.zeros(addShape, 
                                                    dtype=self.array.dtype)))
            if self.isRecArray:
                self.array = np.rec.array(self.array, copy=False)
            #print("FlexArr expand array FAILED")
        try:    
            self.considered.resize(self.space, refcheck=False)
            #print("FlexArr expand considered SUCCESSFUL")
        except ValueError: 
            self.considered = np.concatenate((self.considered, 
                                              np.zeros(newlines, 
                                               dtype=self.considered.dtype)))
            #print("FlexArr expand considered FAILED")
        self.considered_c = self.considered.view(np.int8)
        self.changeIndex += 1
        
    
    def get_array(self):
        if self.deleted.empty():
            return self.array[:self.size]
        else:
            return self.array[:self.size][self.considered.view(bool)[
                                                                    :self.size]]
    
    def __len__(self):
        return self.len()
    
    cdef long len(self):
        return self.size - self.deleted.size()
    
    def __getitem__(self, index):
        if index is None:
            return
        return self.getitem(index)
        
    cdef object getitem(self, long index):
        
        index = self.__wrap_index(index)
        
        if not self.considered_c[index]:
            raise KeyError("The specified element does not exist.")
        
        return self.array[index]
    
    def __setitem__(self, index, value):
        self.setitem(index, value)
    
    def setitem_flexible(self, index, value=None, **keywordData):
        index = self.__wrap_index(index)
        
        if value is None:
            value = keywordData
            keywordData = None
        self.__setitem_by_flexible_keywords(index, value, keywordData)
        self.make_considered(index)
        return index
    
    cpdef long setitem(self, long index, object value):
        index = self.__wrap_index(index)
            
        self.array[index] = value
        self.make_considered(index)
        return index
    
    def setitem_by_keywords(self, index, **keywordData):
        return self.setitem_by_dict(index, keywordData)
    
    cdef long setitem_by_dict(self, long index, dict keywordData):
        index = self.__wrap_index(index)
            
        if not self.considered_c[index]:
            self.make_considered(index)
            self.array[index] = self.zeroitem
        [self.array[key].__setitem__(index, val) 
                for key, val in keywordData.items()]
        return index
    
    cdef void make_considered(self, long index):
        if not self.considered_c[index]:
            self.considered_c[index] = True
            self.deleted.erase(find(self.deleted.begin(),
                                        self.deleted.end(),
                                        index))
    
    def __setitem_by_flexible_keywords(self, long index, value, keywordData=None):
        if self.isStructured:
            if not self.considered_c[index]:
                self.array[index] = self.zeroitem
            if type(value) == dict:
                if keywordData is not None:
                    keywordData = {**value, **keywordData}
                else:
                    keywordData = value
            elif type(value) is np.ndarray and value.dtype.names is not None:
                [self.array[key].__setitem__(index, val) 
                            for key, val in zip(value, value.dtype.names)]
            else:
                self.array[index] = value
            if keywordData is not None and len(keywordData):
                [self.array[key].__setitem__(index, val) 
                        for key, val in keywordData.items()]
        else:
            self.array[index] = value
        return index
    
    def exists(self, *indices):
        try:
            return self.considered[list(indices)].all()
        except IndexError:
            return False
        
    def add_fields(self, names, dtypes, fillVal = None):
        
        array = self.array
        
        if type(names) == str or not hasattr(names, "__iter__"):
            names = (names,)
        
        if type(dtypes) == str or not hasattr(dtypes, "__iter__"):
            dtypes = Repeater(dtypes)
        
        for name in names:
            if name in array.dtype.names:
                raise ValueError("A field with name '" + str(name)
                                 + "' does already exist.")
        
        descr = list(zip(names, dtypes))
        
        newArr = np.empty(self.space, dtype=array.dtype.descr + descr)  
        
        #for name in array.dtype.names:
        #    newArr[name] = array[name]
        view = fields_view(newArr, array.dtype.names)
        view[:] = array 
        
        simpleFill = True
        if fillVal is None:
            fillVal = np.zeros(1, dtype=descr)
        elif type(fillVal) == np.ndarray:
            fillVal = add_names(np.expand_dims(fillVal, fillVal.ndim), names)
        elif hasattr(fillVal, "__iter__"):
            for name, value in zip(names, fillVal):
                newArr[name] = value
            simpleFill = False
        
        if simpleFill:
            view = fields_view(newArr, names)
            view[self.considered.view(bool)] = fillVal
        
        if self.isRecArray:
            self.array = np.rec.array(newArr, copy=False)
        else:
            self.array = newArr
        
        self.__set_zero_item()
        self.changeIndex += 1
    
    def remove_fields(self, names):
        self.array = remove_fields(self.array, names)
        if self.isRecArray:
            self.array = np.rec.array(self.array, copy=False)
        self.__set_zero_item()
        self.changeIndex += 1
        
    def __iter__(self):
        return FlexibleArrayIterator(self)
    
    def __str__(self):
        return self.get_array().__str__()
    
    def __repr__(self):
        return (str(type(self)) + " object with " + str(len(self)) 
                + " rows occupying the space of " + str(self.space) 
                + " rows. Data: \n" + self.get_array().__repr__())
    
    def get_array_indices(self, column, equalCondition):
        return np.nonzero(self.array[column][:self.size][
                            self.considered[:self.size]] == equalCondition)[0]
                        
    def extend(self, newRows):
        cdef: 
            long newRowNumber = self.size - self.space + len(newRows)
            long start = self.size 
            long stop = self.size+len(newRows)
        if newRowNumber > 0:
            self._expand(newRowNumber)
        self.array[start:stop] = newRows
        self.considered[start:stop] = True
        self.size = stop
        
    cpdef bint is_contiguous(self):
        return self.deleted.empty()
    
    def make_contiguous(self):
        
        if self.is_contiguous():
            return [], []
        
        newLen = len(self)
        
        emptySpaces = np.nonzero(~self.considered[:newLen])[0]
        fill = np.nonzero(self.considered[:self.size])[0][-len(emptySpaces):]
        self.array[emptySpaces] = self.array[fill]
        self.considered[fill] = True
        
        self.deleted.clear()
        self.size = newLen
        
        return fill, emptySpaces
        
    def cut(self):
        
        self.space = self.size
        self.shape = (self.space,) + self.shape[1:]
        newSize = np.prod(self.shape)
        try:
            self.array.resize(newSize, refcheck=False)
            #print("FlexArr cut array SUCCESSFUL")
        except ValueError:
            self.array = np.resize(self.array, newSize)
            #print("FlexArr cut array FAILED")
        
        if len(self.shape):
            self.array = np.reshape(self.array, self.shape)
        
        try:
            self.considered.resize(self.space, refcheck=False)
            #print("FlexArr cut considered SUCCESSFUL")
        except ValueError:
            self.considered = np.resize(self.considered, self.space)
            #print("FlexArr cut considered FAILED")
        self.changeIndex += 1
        
    cdef long __wrap_index(self, long index) except -1:
        if index < 0:
            index += self.size
        if index >= self.size or index < 0:
            raise IndexError("Index " + str(index) + " out of range " 
                             + str(self.size))
        return index
    
    cdef void __set_attributes_FA(self, np.ndarray array, 
                                  np.ndarray considered, long size, long space, 
                                  long changeIndex, list deleted, 
                                  bint isStructured, bint isRecArray, 
                                  double extentionFactor, list aliases, 
                                  object zeroitem):
        cdef long i
        self.array = array
        self.considered_c = considered.view(np.int8)
        self.considered = considered
        self.size = size
        self.space = space
        self.changeIndex = changeIndex
        for i in deleted:
            self.deleted.push_back(i)
        self.isStructured = isStructured
        self.isRecArray = isRecArray
        self.extentionFactor = extentionFactor
        self.aliases = aliases
        self.zeroitem = zeroitem
    
    cdef tuple __get_attributes(self):
        cdef long i
        deleted = [i for i in self.deleted]
        return (self.array, self.considered, self.size, self.space, 
                self.changeIndex, deleted, self.isStructured, self.isRecArray, 
                self.extentionFactor, self.aliases, self.zeroitem)
    
    def __reduce__(self):
        return (rebuild_FlexibleArray, self.__get_attributes())
    
    @staticmethod
    def new(np.ndarray array, np.ndarray considered,
            long size, long space, long changeIndex, list deleted,
            bint isStructured, bint isRecArray, double extentionFactor,
            list aliases, object zeroitem):
        cdef: 
            FlexibleArray flexArr = FlexibleArray.__new__(FlexibleArray)
        flexArr.__set_attributes_FA(array, considered, size, space, 
                                    changeIndex, deleted, isStructured, 
                                    isRecArray, extentionFactor, aliases, 
                                    zeroitem)
        return flexArr
    
    
cdef class FlexibleArrayDict(FlexibleArray):
    
    def __init__(self, nparray, fancyIndices=None, **flexibleArrayArgs):
        super().__init__(nparray, **flexibleArrayArgs)
        if hasattr(fancyIndices, "__iter__"):
            self.indexDict = {iD:index for index, iD 
                              in enumerate(fancyIndices)}
            if not len(self.indexDict) == self.size:
                raise ValueError("There must be as many distinct keys "
                                 + "as elements!")
        else:
            self.indexDict = {i:i for i in range(self.size)}
    
    def __delitem__(self, index): 
        self.delitem(index)
    
    cdef void delitem(self, long index):
        FlexibleArray.delitem(self, self.indexDict.at(index))
        self.indexDict.erase(index)
    
    def __getitem__(self, index):
        return self.getitem(index)
    
    cdef object getitem(self, long index):
        return FlexibleArray.getitem(self, self.indexDict.at(index))
    
    def __setitem__(self, index, value):
        self.setitem(index, value)
    
    def setitem_flexible(self, index, value=None, **keywordData):
        cdef long internalIndex
        
        if self.indexDict.count(index):
            internalIndex = self.indexDict[index]
            if value is None:
                value = keywordData
                keywordData = None
            self.__setitem_by_flexible_keywords(internalIndex, value, 
                                                keywordData)
            self.make_considered(internalIndex)
        else:
            internalIndex = FlexibleArray.add(self, value, **keywordData)
            self.indexDict[index] = internalIndex
        return internalIndex
    
    cpdef long setitem(self, long index, object value):
        cdef long internalIndex 
        if self.indexDict.count(index):
            internalIndex = self.indexDict[index]
            self.array[internalIndex] = value
        else:
            internalIndex = FlexibleArray.add_tuple(self, value)
        self.indexDict[index] = internalIndex
        return internalIndex
    
    def setitem_by_keywords(self, index, **keywordData):
        return self.setitem_by_dict(index, keywordData)
    
    cdef long setitem_by_dict(self, long index, dict keywordData):
        cdef long internalIndex 
        if self.indexDict.count(index):
            internalIndex = self.indexDict.at(index)
            FlexibleArray.setitem_by_dict(self, internalIndex, keywordData)
        else:
            internalIndex = FlexibleArray.add_by_dict(self, keywordData)
        self.indexDict[index] = internalIndex
        return internalIndex
    
    cpdef object get(self, long index, object default):
        cdef long internalIndex
        if self.indexDict.count(index):
            internalIndex = self.indexDict.at(index)
            return FlexibleArray.getitem(self, internalIndex)
        else:
            return None
    
    def __contains__(self, index):
        return self.indexDict.count(index) > 0 
            
    def exists(self, *indices):
        indexDict = self.indexDict
        return all(indexDict.count(index) for index in indices)
     
    def extend(self, newRows):
        raise NotImplementedError("This method is still to be implemented.")
        #FlexibleArray.extend(self, newRows)
    def getColumnView(self, column):
        return FlexibleArrayDictColumnView.new(self, column)
    
    cdef void __set_attributes_FAD(self, unordered_map[long, long] indexDict):
        self.indexDict = indexDict
        
    cdef tuple __get_attributes(self):
        return (*FlexibleArray.__get_attributes(self), self.indexDict)
    
    def __reduce__(self):
        return (rebuild_FlexibleArrayDict, self.__get_attributes())
    
    @staticmethod
    def new(np.ndarray array, np.ndarray considered,
            long size, long space, long changeIndex, list deleted,
            bint isStructured, bint isRecArray, double extentionFactor,
            list aliases, object zeroitem, unordered_map[long, long] indexDict,
            ):
        cdef: 
            FlexibleArrayDict flexArrDict = FlexibleArrayDict.__new__(
                                                            FlexibleArrayDict)
        flexArrDict.__set_attributes_FA(array, considered, size, space, 
                                        changeIndex, deleted, isStructured, 
                                        isRecArray, extentionFactor, aliases, 
                                        zeroitem)
        flexArrDict.__set_attributes_FAD(indexDict)
        return flexArrDict
    
    
    

class FlexibleArrayDictColumnView(FlexibleArrayDict):
    @staticmethod
    def new(flexibleArray, column):
        self = copy.copy(flexibleArray)
        self.__class__ = FlexibleArrayDictColumnView
        self.array = self.array[column]
        
        remainingFunctionality = set(("__iter__", "__len__", "__repr__",
                                      "__str__", "exists", "get_array",
                                      "get_array_indices", "getitem",
                                      "__getitem__", "get"))
        d = self.__dict__
        for key in d.keys():
            if not key in remainingFunctionality:
                del d[key]
        return self


class FlexibleArrayIterator(object):
    
    def __init__(self, flexibleArray):
        self.flexibleArray = flexibleArray
        self.relevantIndices = np.nonzero(flexibleArray.considered)[0]
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        flexibleArray = self.flexibleArray
        try:
            i = self.i
            self.i += 1
            return flexibleArray.array[i]
        except IndexError:
            raise StopIteration()


def get_chunks(arr, key=None, returnKeys=False):
    if key is None:
        key = arr 
    elif not (hasattr(key, "__iter__") and not type(key) == str):
        key = arr[key]
    
    splits = np.nonzero(key!=np.roll(key, 1))[0]
    if not len(splits):
        if returnKeys:  
            return [key[0]], [arr]
        else: 
            return [arr]
    splittedArr = np.split(arr, splits[1:])
    
    if returnKeys:
        return key[splits], splittedArr
    return splittedArr

cpdef np.ndarray[double] pointer_sum(np.ndarray[double] arr, np.ndarray[long long] indptr):
    if not arr.ndim == 1:
        raise ValueError("Value array must be 1D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    cdef: 
        #double[:] arr_c = arr
        #double* arr_cp = <double*> arr_c
        #long[:] indptr_c = indptr
        #    long* indptr_cp = <long*> indptr_c
        np.ndarray[double] result = np.empty(indptr.size-1)
        #double[:] result_c = result
    #    double* result_cp = <double*> result_c
    #sectionsum(&arr_c[0], &indptr_c[0], arr.size, indptr.size, &result_c[0])
    sectionsum(<double*> arr.data, <long long*> indptr.data, arr.size, indptr.size, <double*> result.data)
    return result

cpdef np.ndarray[double] pointer_prod(np.ndarray[double] arr, np.ndarray[long long] indptr):
    if not arr.ndim == 1:
        raise ValueError("Value array must be 1D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    cdef: 
        #double[:] arr_c = arr
        #double* arr_cp = <double*> arr_c
        #long[:] indptr_c = indptr
        #    long* indptr_cp = <long*> indptr_c
        np.ndarray[double] result = np.empty(indptr.size-1)
        #double[:] result_c = result
    #    double* result_cp = <double*> result_c
    #sectionsum(&arr_c[0], &indptr_c[0], arr.size, indptr.size, &result_c[0])
    sectionprod(<double*> arr.data, <long long*> indptr.data, indptr.size, <double*> result.data)
    return result

cpdef np.ndarray[double] pointer_sum_chosen(np.ndarray[double] arr, np.ndarray[long long] indptr, 
                                            np.ndarray[long long] considered, np.ndarray[long long] consideredindptr):
    if not arr.ndim == 1:
        raise ValueError("Value array must be 1D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    if not indptr.size == consideredindptr.size:
        raise ValueError("Data array and index array must have the same size.")
    cdef: 
        #double[:] arr_c = arr
        #double* arr_cp = <double*> arr_c
        #long[:] indptr_c = indptr
        #    long* indptr_cp = <long*> indptr_c
        np.ndarray[double] result = np.empty(indptr.size-1)
        #double[:] result_c = result
    #    double* result_cp = <double*> result_c
    #sectionsum(&arr_c[0], &indptr_c[0], arr.size, indptr.size, &result_c[0])
    sectionsum_chosen(<double*> arr.data, <long long*> indptr.data, 
                      <long long*> considered.data, <long long*> consideredindptr.data, 
                      indptr.size, <double*> result.data)
    return result

cpdef np.ndarray[double] pointer_sum_chosen_rows(np.ndarray[double] arr, 
                                                 np.ndarray[long long] indptr, 
                                                 np.ndarray[long long] considered, 
                                                 np.ndarray[long long] consideredindptr,
                                                 np.ndarray[long long] rows):
    if not arr.ndim == 1:
        raise ValueError("Value array must be 1D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    if not rows.size == consideredindptr.size-1:
        raise ValueError("Row array size must be index array size minus 1.")
    cdef: 
        #double[:] arr_c = arr
        #double* arr_cp = <double*> arr_c
        #long[:] indptr_c = indptr
        #    long* indptr_cp = <long*> indptr_c
        np.ndarray[double] result = np.empty(rows.size)
        #double[:] result_c = result
    #    double* result_cp = <double*> result_c
    #sectionsum(&arr_c[0], &indptr_c[0], arr.size, indptr.size, &result_c[0])
    sectionsum_chosen_rows(<double*> arr.data, <long long*> indptr.data, 
                      <long long*> considered.data, <long long*> consideredindptr.data, 
                      <long long*> rows.data,
                      rows.size, <double*> result.data)
    return result

cpdef np.ndarray[double] pointer_sum_chosen_rows_fact(
        np.ndarray[double] arr, np.ndarray[long long] indptr, 
        np.ndarray[long long] considered, np.ndarray[long long] consideredindptr,
        np.ndarray[long long] rows, np.ndarray[double] factor):
    if not arr.ndim == 1:
        raise ValueError("Value array must be 1D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    if not rows.size == consideredindptr.size-1:
        raise ValueError("Row array size must be index array size minus 1.")
    if not considered.size == factor.size:
        raise ValueError("Each considered item must have one factor. "+
                         "Therefore, considered and factor must have the same size.")
    cdef: 
        #double[:] arr_c = arr
        #double* arr_cp = <double*> arr_c
        #long[:] indptr_c = indptr
        #    long* indptr_cp = <long*> indptr_c
        np.ndarray[double] result = np.empty(rows.size)
        #double[:] result_c = result
    #    double* result_cp = <double*> result_c
    #sectionsum(&arr_c[0], &indptr_c[0], arr.size, indptr.size, &result_c[0])
    sectionsum_chosen_rows_fact(<double*> arr.data, <long long*> indptr.data, 
                      <long long*> considered.data, <long long*> consideredindptr.data, 
                      <long long*> rows.data, <double*> factor.data, 
                      rows.size, <double*> result.data)
    return result

cpdef np.ndarray[double] pointer_sum_row_prod(
        np.ndarray[double] arr1, np.ndarray[long long] arr1indptr, 
        np.ndarray[long long] columns1, np.ndarray[long long] columns1indptr,
        np.ndarray[long long] columns1rows,
        np.ndarray[long long] rows1, np.ndarray[double] arr2,
        np.ndarray[long long] arr2indptr, np.ndarray[long long] rows2):
    
    if not arr1.ndim == 1:
        raise ValueError("arr1 must be 1D")
    if not arr1indptr.ndim == 1:
        raise ValueError("arr1indptr must be 1D")
    if not columns1.ndim == 1:
        raise ValueError("columns1 must be 1D")
    if not columns1indptr.ndim == 1:
        raise ValueError("columns1indptr must be 1D")
    if not columns1rows.ndim == 1:
        raise ValueError("columns1rows must be 1D")
    if not rows1.ndim == 1:
        raise ValueError("rows1 must be 1D")
    if not arr2.ndim == 1:
        raise ValueError("arr2 must be 1D")
    if not arr2indptr.ndim == 1:
        raise ValueError("arr2indptr must be 1D")
    if not rows2.ndim == 1:
        raise ValueError("rows2 must be 1D")
    if not (columns1rows.size == rows1.size and rows1.size == rows2.size):
        raise ValueError("All three row arrays must have the same sizes.")
    cdef: 
        np.ndarray[double] result = np.empty(rows1.size)
       
    sectionsum_rowprod(<double*> arr1.data, <long long*> arr1indptr.data, 
                      <long long*> columns1.data, <long long*> columns1indptr.data, 
                      <long long*> columns1rows.data, 
                      <long long*> rows1.data, <double*> arr2.data, 
                      <long long*> arr2indptr.data, <long long*> rows2.data,
                      rows1.size, <double*> result.data)
    return result



@cython.boundscheck(False)
@cython.wraparound(False)
def pointer_sum3D(np.ndarray[double, ndim=3] arr, #np.ndarray[DOUBLE_t, ndim=3] arr, 
                  np.ndarray[long long, ndim=1] indptr):
    if not arr.ndim == 3:
        raise ValueError("Value array must be 3D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    cdef: 
        long i, j
        #np.ndarray[DOUBLE_t, ndim=3] result = np.empty((arr.shape[0], arr.shape[1],
        double[:,:,::1] arr_t = arr
        long long* indptr_t = <long long*> indptr.data
        np.ndarray[DOUBLE_t, ndim=3] result = np.empty((arr.shape[0], arr.shape[1],
                                              indptr.size-1), 
                                             dtype=np.double)
        double[:,:,:] result_t = result
        long Ni = arr.shape[0]
        long Nj = arr.shape[1]
        long Nk = arr.shape[2]
        long indptrsize = indptr.size
    #with nogil, parallel(num_threads=4):
    #    for i in prange(Ni, schedule='dynamic'):
    for i in range(Ni):
        for j in range(Nj):
            #print(pointer_sum(arr[i,j], indptr))
            #print(result[i, j])
            sectionsum(&arr_t[i,j,0], indptr_t, Nk, indptrsize, &result_t[i,j,0])
            #print(i, j, result)
            
        #print("ps3", i, result[:,:,i])
        #print(np.sum(arr[:,:,indptr[i]:indptr[i+1]], 2))
    return result

def pointer_sum3DY(np.ndarray[double, ndim=3] arr, #np.ndarray[DOUBLE_t, ndim=3] arr, 
                  np.ndarray[long long, ndim=1] indptr):
    if not arr.ndim == 3:
        raise ValueError("Value array must be 3D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    cdef: 
        int i, j
        np.ndarray[DOUBLE_t, ndim=3] result = np.empty((arr.shape[0], arr.shape[1],
        #double[:,:,:] result = np.empty((arr.shape[0], arr.shape[1],
                                              indptr.size-1), 
                                             dtype=np.double)
        long Ni = arr.shape[0]
        long Nj = arr.shape[1]
    for i in range(Ni):
        for j in range(Nj):
            #print(pointer_sum(arr[i,j], indptr))
            #print(result[i, j])
            result[i, j] = pointer_sum(arr[i,j], indptr)
            #print(i, j, pointer_sum(arr[i,j], indptr))
        #print("ps3", i, result[:,:,i])
        #print(np.sum(arr[:,:,indptr[i]:indptr[i+1]], 2))
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def pointer_sum3DX(double[:,:,:] arr, #np.ndarray[DOUBLE_t, ndim=3] arr, 
                  np.ndarray[long long, ndim=1] indptr):
    if not arr.ndim == 3:
        raise ValueError("Value array must be 3D")
    if not indptr.ndim == 1:
        raise ValueError("Index array must be 1D")
    cdef: 
        int i
        np.ndarray[DOUBLE_t, ndim=3] result = np.empty((arr.shape[0], arr.shape[1],
        #double[:,:,:] result = np.empty((arr.shape[0], arr.shape[1],
                                              indptr.size-1), 
                                             dtype=np.double)
    
    
    for i in range(indptr.size-1, schedule='static'):
        #print("ps3", i, result[:,:,i])
        #print(np.sum(arr[:,:,indptr[i]:indptr[i+1]], 2))
        result[:,:,i] = np.sum(arr[:,:,indptr[i]:indptr[i+1]], 2)
    return result

"""
def pointer_sum2(np.ndarray[double] arr, np.ndarray[long] indptr):
    cdef: 
        long i
        np.ndarray[double] result = np.empty((indptr.size, arr.shape[1]))
    
    for i in range(indptr.size-1):
        result[i] = np.sum(arr[indptr[i]:indptr[i+1]])
    return result
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def get_common_element2d(np.ndarray[double, ndim=2] arr1, 
                         np.ndarray[double, ndim=2] arr2):
    cdef np.ndarray[double, ndim=2] arrTmp 
    if arr1.shape[1] > arr2.shape[1]:
        arrTmp = arr1
        arr1 = arr2
        arr2 = arrTmp
    
    cdef np.ndarray[double, ndim=1] result = np.empty(arr1.shape[0])
    cdef int dim1 = arr1.shape[1]
    cdef int dim2 = arr2.shape[1]
    cdef int i, j
    cdef unordered_map[double, int] tmpset = unordered_map[double, int]()
    
    for i in range(arr1.shape[0]):
        for j in range(dim1):
            tmpset[arr1[i, j]]
        for j in range(dim2):
            if tmpset.count(arr2[i,j]):
                result[i] = arr2[i,j]
                break
        else:
            result[i] = NAN
        tmpset.clear()
        
    return result

"""
@cython.boundscheck(False)
@cython.wraparound(False)
def get_common_element(np.ndarray[double, ndim=2] arr1, 
                       np.ndarray[double, ndim=2] arr2):
    
    cdef np.ndarray[double, ndim=1] result = np.empty(arr1.shape[0])
    cdef int dim2 = arr2.shape[1]
    cdef int i, j
    cdef set tmpset = set()
    
    for i in range(arr1.shape[0]):
        tmpset.update(arr1[i])
        for j in range(dim2):
            if arr2[i,j] in tmpset:
                result[i] = arr2[i,j]
                break
        else:
            result[i] = np.nan
        tmpset.clear()
        
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def get_common_element2(np.ndarray[double, ndim=2] arr1, 
                        np.ndarray[double, ndim=2] arr2):
    
    cdef np.ndarray[double, ndim=1] result = np.empty(arr1.shape[0])
    cdef int dim2 = arr2.shape[1]
    cdef int i
    cdef set tmpset = set()
    
    for i in range(arr1.shape[0]):
        tmpset.update(arr1[i])
        tmpset.intersection_update(arr2[i])
        if tmpset:
            result[i] = tmpset.pop()
        else:
            result[i] = np.nan
        
    return result
"""

@cython.boundscheck(False)
@cython.wraparound(False)
def find_next_nonzero2d(np.ndarray[UINT8_t, ndim=2, cast=True] mask, 
                        np.ndarray[INT_t, ndim=1] ind1,
                        np.ndarray[INT_t, ndim=1] ind2,
                        long startIndex):
    cdef long i
    cdef long endIndex = ind1.size
    
    for i in range(startIndex, endIndex):
        if mask[ind1[i], ind2[i]]:
            return i
    else:
        return None

@cython.boundscheck(False)
@cython.wraparound(False)
def unique_tol(np.ndarray[DOUBLE_t, ndim=1] lower,
               np.ndarray[DOUBLE_t, ndim=1] higher,
               np.ndarray[DOUBLE_t, ndim=1] original):
    cdef long i, count
    cdef long endIndex = lower.size
    cdef unordered_map[double, short] vals = unordered_map[double, short]()
    cdef np.ndarray[DOUBLE_t, ndim=1] result_vals = np.empty_like(lower)
    cdef np.ndarray[INT_t, ndim=1] result_indices = np.empty_like(lower, 
                                                                  dtype=int)
    
    count = 0
    for i in range(endIndex): 
        if not vals.count(lower[i]) and not vals.count(higher[i]):
            
            # insert in result
            result_vals[count] = original[i]
            result_indices[count] = i
            
            # put lowerVal and higherVal in the hashMap
            vals[lower[i]]
            vals[higher[i]]
            
            # update the index in the result
            count += 1
            
    return result_vals[:count], result_indices[:count]

    

