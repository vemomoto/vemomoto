'''
Created on 18.06.2016

@author: Samuel
'''

import sys

class Writer():
    def __init__(self, file, out):
        self.file = file
        self.out = out
    def write(self, data):
        self.file.write(data)
        self.file.flush()  
        self.out.write(data)  
    def flush(self):
        self.file.flush()   
        self.out.flush()

class Tee(object):
    def __init__(self, file_name, stdout = True, sterr = True):
        self.file = open(file_name, "w")
        if stdout:
            self.stdout = sys.stdout
            sys.stdout = Writer(self.file, self.stdout)
        else: self.stdout = None
        if sterr:
            self.stderr = sys.stderr
            sys.stderr = Writer(self.file, self.stderr)
        else: self.sterr = None
    def __del__(self):
        if not self.stdout is None:
            self.stdout.flush()
            sys.stdout = self.stdout
        if not self.stderr is None:
            self.stderr.flush()
            sys.stderr = self.stderr
        self.file.close()
        
