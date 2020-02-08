'''
Created on 29.11.2017

@author: Samuel
'''

from npextc import FlexibleArray as FlexibleArray_c, \
                    FlexibleArrayDict as FlexibleArrayDict_c

def rebuild_FlexibleArray(*args):
    return FlexibleArray_c.new(*args)
def rebuild_FlexibleArrayDict(*args):
    return FlexibleArrayDict_c.new(*args)