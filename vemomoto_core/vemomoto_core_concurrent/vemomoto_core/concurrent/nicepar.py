'''
Created on 27.04.2016

@author: Samuel
'''

from functools import partial
import multiprocessing
import itertools
import os
from warnings import warn

import numpy as np
import sharedmem 


class SharableListArr():
    def __init__(self, arr, dtype):
        self.size = len(arr)
        self.pos = sharedmem.empty(self.size+1, int)
        index = 0
        for i, row in enumerate(arr):
            self.pos[i] = index
            if row:
                index += len(row)
        self.pos[-1] = index
        self.data = sharedmem.empty(index, dtype = dtype)
        for i, row in enumerate(arr):
            self.data[self.pos[i]:self.pos[i+1]] = row
    def __getitem__(self, pos):
        return self.data[self.pos[pos]:self.pos[pos+1]]
    def __str__(self):
        result = ""
        for i in range(self.size):
            result += str(self[i]) + "\n"
        return result[:-1]

class Locked():
    """
    Context manager ensuring that no dirty read/write 
    occurs. Use with the "with" statement:
    
        with Locked(self):
            # do something that requires the object to be 
            # accassible to this process only
            pass
    """
    def __init__(self, parent):
        self.parent = parent
    def __enter__(self):
        self.parent.lock.acquire()
        return self.parent
    def __exit__(self, type, value, traceback):
        self.parent.lock.release()

class DummyVar():
    def __init__(self, value):
        self.value = value
        
class DummyObj():
    def __init__(self, parent):
        self.parent = parent
        self.update()
    def update(self):
        parent = self.parent
        self.__dict__ = {key:DummyVar(key) for 
                         key in self.parent.__dict__.keys()}
        self.parent = parent

class Lockable(object):
    def __init__(self, lock=None):
        if lock is None:
            self.lock = multiprocessing.Lock()
        else:
            self.lock = lock
        

class ParallelClass(Lockable):
    '''
    Class to simplify the implementation of objects whose
    methods can be executed in parallel.
    '''
    def __init__(self, num_workers = None, make_sharable_functions = [], 
                 exclude = [], prohibit_sharing = False):
        """
        Constructor
        """
        super().__init__()
        self.num_workers = num_workers
        self.__to_do_list = []
        self.original = DummyObj(self)
        self.make_sharable_functions = make_sharable_functions
        self.exclude = exclude
        self.prohibit_sharing = prohibit_sharing
        self.running_parallel = False
    
    def __make_np_arrays_sharable(self):
        """
        Replaces all numpy array object variables with dimension 
        > 0 with a sharedmem array, which should have the same  
        behaviour / properties as the numpy array
        """
        varDict = self.__dict__
        for key, var in varDict.items():
            if type(var) is np.ndarray:
                if not key in self.exclude:
                    try:
                        varDict[key] = sharedmem.copy(var)
                    except AttributeError:
                        share_var = sharedmem.empty(1, type(var))
                        share_var[0] = var
                        varDict[key] = share_var
    
    def __parallel_wrapper(self, f, args):
        self.__to_do_list = []
        if hasattr(args, '__iter__'):
            return (f(*args), self.__to_do_list)
        else:
            return (f(args), self.__to_do_list)
    
    def __simple_wrapper(self, f, args):
        if hasattr(args, '__iter__'):
            return f(*args)
        else:
            return f(args)
    
    def __postpone_wrapper(self, f_args):
        f, args = f_args
        if hasattr(args, '__iter__'):
            args = list(args)
            for i, arg in enumerate(args):
                if type(arg) is DummyVar:
                    args[i] = self.__dict__[arg.value]
                return f(*args)
        else:
            if type(args) is DummyVar:
                args = self.__dict__[args.value]
            return f(args)
        
    def postpone_task(self, f, *args):
        self.__to_do_list.append((f, args))
        
    
    def parmap(self, f, argList):
        """
        Executes f(arg) for arg in argList in parallel
        returns a list of the results in the same order as the 
        arguments, invalid results (None) are ignored
        """
        if len(self.__to_do_list):
            warn("__postponed_task_list was not empty. However, I deleted its "
                 + "entries. The tasks are ignored.")
            self.__to_do_list = []
        
        self.original.update()
        
        for func in self.make_sharable_functions:
            func()
            
        if os.name == 'posix':
            if not self.prohibit_sharing:
                self.__make_np_arrays_sharable()
            
            self.running_parallel = True
            with sharedmem.MapReduce(np = self.num_workers) as pool:
                results, to_do_list = zip(*pool.map(partial(
                                           self.__parallel_wrapper, f), argList)
                                                            )
            self.running_parallel = False
        else:
            warn("Parallelization with shared memory is ony possible " +
                 "on Unix-based systems. Thus, the code will not be " +
                 "executed in parallel.")
            results = tuple(map(partial(self.__simple_wrapper, f), argList))
            to_do_list = (self.__to_do_list,)
        
        any(map(self.__postpone_wrapper, itertools.chain(*to_do_list)))
        
        self.__to_do_list = []
        return results
    
class ParallelCounter(Lockable):
    def __init__(self, size=1, interval=None, lock=None, manager=None):
        if lock is None and manager is not None:
            lock = manager.Lock()
        super().__init__(lock)
        lock = multiprocessing.Lock()
        
        self.interval = interval
        if manager is None:
            self.size = multiprocessing.Value("l", lock=lock)
            self.index = multiprocessing.Value("l", lock=lock) #sharedmem.full(1, 0, int)
            self.nextStep = multiprocessing.Value("d", lock=lock) #sharedmem.full(1, self.interval, float) 
            self.size.value = size
            self.index.value = 0
            self.nextStep.value = interval
        else:
            self.size = manager.Value("l", size)
            self.index = manager.Value("l", 0) #sharedmem.full(1, 0, int)
            self.nextStep = manager.Value("d", interval) #sharedmem.full(1, self.interval, float) 
        
    def next(self):
        with Locked(self):
            self.index.value += 1
            newValue = self.index.value / self.size.value
            if self.interval is not None:
                if newValue >= self.nextStep.value:
                    self.nextStep.value += self.interval
                    return newValue
            else:
                return newValue
    
    def reset(self):
        with Locked(self):
            self.index.value = 0
            self.nextStep.value = self.interval
    
    def __bool__(self):
        with Locked(self):
            return bool(self.index.value < self.size.value)
    
    def __repr__(self, *args, **kwargs):
        with Locked(self):
            return "ParallelCounter: {}, {}.".format(self.index.value, 
                                                     self.size.value)

class CircularParallelCounter(Lockable):
    def __init__(self, size=1, lock=None):
        super().__init__(lock)
        self.size = size
        self.index = sharedmem.full(1, 0, int)
        
    def next(self):
        with Locked(self):
            self.index[0] += 1
            if self.index[0] >= self.size:
                self.index[0] = 0
                return True
            else:
                return False
    
    def reset(self):
        with Locked(self):
            self.index[0] = 0
    
    def __repr__(self, *args, **kwargs):
        return "CircularParallelCounter: {:3.2%}.".format(self.index[0] 
                                                          / self.size)
    
        
class Counter():
    def __init__(self, size=1, interval=None):
        self.size = size
        self.interval = interval
        self.reset()
    def next(self):
        self.index += 1
        newValue = self.index / self.size
        if self.interval:
            if newValue >= self.nextStep:
                self.nextStep += self.interval
                return newValue
        else:
            return newValue
    def reset(self):
        self.nextStep = self.interval
        self.index = 0

def getCounter(size=1, interval=None):
    if os.name == 'posix': 
        return ParallelCounter(size, interval)
    else:
        return Counter(size, interval)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
