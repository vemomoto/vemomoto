'''
Created on 03.01.2017

@author: Samuel
'''
from concurrent.futures import ProcessPoolExecutor as conc_ProcessPoolExecutor
from concurrent.futures.process import _ExceptionWithTraceback, _get_chunks, _ResultItem
from functools import partial
import multiprocessing
import itertools
import os
import numpy as np
from multiprocessing import sharedctypes
CPU_COUNT = os.cpu_count() 


def get_cpu_chunk_counts(task_length, chunk_number=5, min_chunk_size=1):
    cpu_count = max(min(CPU_COUNT, 
                        task_length // (chunk_number*min_chunk_size)), 1)
    chunk_size = max(min_chunk_size, task_length // (cpu_count*chunk_number))
    return cpu_count, chunk_size

def _process_worker(call_queue, result_queue, const_args=[], shared_arrays=[]):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A multiprocessing.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A multiprocessing.Queue of _ResultItems that will written
            to by the worker.
        shutdown: A multiprocessing.Event that will be set as a signal to the
            worker that it should exit when call_queue is empty.
    """
    
    shared_arrays_np = [np.ctypeslib.as_array(arr).view(dtype).reshape(shape) 
                        for arr, dtype, shape in shared_arrays]
    
    
    while True:
        call_item = call_queue.get(block=True)
        if call_item is None:
            result_queue.put(os.getpid())
            return
        try:
            r = call_item.fn(*call_item.args, const_args=const_args,
                             shared_arrays=shared_arrays_np,
                             **call_item.kwargs)
        except BaseException as e:
            exc = _ExceptionWithTraceback(e, e.__traceback__) 
            result_queue.put(_ResultItem(call_item.work_id, exception=exc))
        else:
            result_queue.put(_ResultItem(call_item.work_id,
                                         result=r))


def _process_chunk(fn, chunk, const_args, shared_arrays):
    """ Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*const_args, *shared_arrays, *args) for args in chunk]



class ProcessPoolExecutor(conc_ProcessPoolExecutor):
    '''
    classdocs 
    '''

    def __init__(self, max_workers=None, const_args=[], shared_np_arrs=[]):
        '''
        Constructor
        '''
        super().__init__(max_workers)
        self._const_args = const_args
        shared_arrays_ctype = []
        shared_arrays_np = []
        
        # TODO do not create copy of shared array, if it already has a suitable 
        # data structure
        for arr in shared_np_arrs:
            dtype = arr.dtype
            arrShared = np.empty(arr.size*dtype.itemsize, np.int8)
            arrShared = np.ctypeslib.as_ctypes(arrShared)
            ctypes_arr = sharedctypes.RawArray(arrShared._type_, arrShared)
            shared_arrays_ctype.append((ctypes_arr, arr.dtype, arr.shape))
            view = np.ctypeslib.as_array(ctypes_arr).view(arr.dtype).reshape(
                                                                    arr.shape)
            view[:] = arr
            shared_arrays_np.append(view)
        self._shared_arrays_np = shared_arrays_np
        self._shared_arrays = shared_arrays_ctype
        
    def _adjust_process_count(self):
        for _ in range(len(self._processes), self._max_workers):
            p = multiprocessing.Process(
                    target=_process_worker,
                    args=(self._call_queue,
                          self._result_queue,
                          self._const_args,
                          self._shared_arrays))
            p.start()
            self._processes[p.pid] = p    
            
    def map(self, fn, *iterables, timeout=None, chunksize=None, 
            tasklength=None, chunknumber=5, min_chunksize=1):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a time.
            tasklength: length of the iterable. If provided, the cpu count
                and the chunksize will be adjusted approprietly, if they are not
                explicietely given.
        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        tmp_max_workers = self._max_workers
        if tasklength and tasklength > 0:
            cpu_count, chunksize_tmp = get_cpu_chunk_counts(tasklength, 
                                                            chunknumber,
                                                            min_chunksize)
            if not chunksize:
                chunksize = chunksize_tmp
                
            self._max_workers = cpu_count
        
        if not chunksize:
            chunksize = 1
        
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")
        
        results = super(conc_ProcessPoolExecutor, self).map(partial(_process_chunk, fn),
                              _get_chunks(*iterables, chunksize=chunksize),
                              timeout=timeout)
        
        self._max_workers = tmp_max_workers 
        
        return itertools.chain.from_iterable(results)
    
    
    def get_shared_arrays(self):
        return self._shared_arrays_np
    
