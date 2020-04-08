'''
Created on 25.11.2014

@author: Samuel
'''

import pstats, cProfile

def profile(func, globs, locs):
    """Profiles a given method.
    
    Usage:
    ``profile("myMethod(*args)", globals(), locals())``
    
    """
    
    print("Profiling " + func)
    cProfile.runctx(func, globs, locs, "Profile.prof")
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()  
