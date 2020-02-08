'''
Created on 26.07.2016

@author: Samuel
'''

class EmptyList():
    def __iter__(self):
        return self
    def __len__(self):
        return 0
    def __next__(self):
        raise StopIteration()

class Repeater(object):
    def __init__(self, value):
        self.value = value
    def __getitem__(self, index):
        return self.value
    def __setitem__(self, index, value):
        self.value = value
    def __next__(self):
        return self.value
    def __iter__(self):
        return self

class DictIterator(object):
    def __init__(self, dictionary, stopValue):
        self.dictionary = dictionary
        self.stopValue = stopValue
        self.count = 0
    def __next__(self):
        dictionary = self.dictionary
        if dictionary:
            key, value = dictionary.popitem()
            if not value == self.stopValue:
                self.count += 1
                return key
        raise StopIteration()
    def __iter__(self):
        self.count = 0
        return self


