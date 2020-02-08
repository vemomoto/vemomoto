'''
Created on 04.01.2017

@author: Samuel
'''
import dill

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        dill.dump(obj, file, byref=True)

def load_object(filename):
    with open(filename, 'rb') as file:
        return dill.load(file)

"""
class T(object):
    def __init__(self, x):
        self.x = x
    def t(self, y):
        return self.x + y

fn = "test.txt"

test = T(5)
save_object(test, fn)
print(test.t(3))

test2 = load_object(fn)
print(test2.t(3))
"""