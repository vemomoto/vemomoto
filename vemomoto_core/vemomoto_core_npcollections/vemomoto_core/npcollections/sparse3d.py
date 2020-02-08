'''
Created on 07.05.2016

@author: Samuel
'''

import scipy.sparse as sp
import numpy as np
import copy as cp

class Sparse3D():
    """
    Class to store and access 3 dimensional sparse matrices efficiently
    """
    def __init__(self, *sparseMatrices):
        """
        Constructor
        Takes a stack of sparse 2D matrices with the same dimensions
        """
        self.data = sp.vstack(sparseMatrices, "dok")
        self.shape = (len(sparseMatrices), *sparseMatrices[0].shape)
        self._dim1_jump = np.arange(0, self.shape[1]*self.shape[0], self.shape[1])
        self._dim1 = np.arange(self.shape[0])
        self._dim2 = np.arange(self.shape[1])
        
    def __getitem__(self, pos):
        if not type(pos) == tuple:
            if not hasattr(pos, "__iter__") and not type(pos) == slice: 
                return self.data[self._dim1_jump[pos] + self._dim2]
            else:
                return Sparse3D(*(self[self._dim1[i]] for i in self._dim1[pos]))
        elif len(pos) > 3:
            raise IndexError("too many indices for array")
        else:
            if (not hasattr(pos[0], "__iter__") and not type(pos[0]) == slice or
                not hasattr(pos[1], "__iter__") and not type(pos[1]) == slice):
                if len(pos) == 2:
                    result = self.data[self._dim1_jump[pos[0]] + self._dim2[pos[1]]]
                else:
                    result = self.data[self._dim1_jump[pos[0]] + self._dim2[pos[1]], pos[2]].T
                    if hasattr(pos[2], "__iter__") or type(pos[2]) == slice:
                        result = result.T
                return result
            else:
                if len(pos) == 2:
                    return Sparse3D(*(self[i, self._dim2[pos[1]]] for i in self._dim1[pos[0]]))
                else:
                    if not hasattr(pos[2], "__iter__") and not type(pos[2]) == slice:
                        return sp.vstack([self[self._dim1[pos[0]], i, pos[2]] #or hstack? 
                                          for i in self._dim2[pos[1]]]).T
                    else:
                        return Sparse3D(*(self[i, self._dim2[pos[1]], pos[2]] 
                                          for i in self._dim1[pos[0]]))
                        
    def toarray(self):
        return np.array([self[i].toarray() for i in range(self.shape[0])])
    
    def sum(self, axis = None):
        if axis is None:
            return self.data.sum()
        elif axis == 0:
            result = sp.dok_matrix(self.shape[1:])
            for i in range(self.shape[1]):
                result[i] = self.data[self._dim1_jump + i].sum(0)
        elif axis == 1:
            result = sp.dok_matrix((self.shape[0], self.shape[2]))
            for i in range(self.shape[0]):
                result[i] = self.data[self._dim2 + self._dim1_jump[i]].sum(0)
        elif axis == 2:
            result = sp.dok_matrix(self.data.sum(1).reshape(self.shape[:2]))
        else:
            raise IndexError("too many indices for array")
        return result
        
    def imultiply(self, other, axis = 0):
        if hasattr(other, "__iter__"):
            if axis == 0:
                if len(other.shape) == 1:
                    repOther = sp.csc_matrix(np.expand_dims(np.repeat(other, 
                                                                      self.shape[1]), 1))
                elif len(other.shape) == 2:
                    if other.shape == self.shape[1:]:
                        if type(other) == sp.dok_matrix:
                            other = other.toarray()
                        repOther = np.tile(other.flatten(), 
                                           self.shape[0]).reshape((self.shape[0] * 
                                                                   self.shape[1], 
                                                                   self.shape[2]))
                        repOther = sp.csc_matrix(repOther)
                    else:
                        raise IndexError("Factor must have same shape as array[0]")
                else:
                    raise NotImplementedError("Multiplication with factors with more "+
                                              "than two dimensions has not been implemented "+
                                              "yet.")
            elif axis == 2:
                if len(other.shape) == 1:
                    repOther = sp.csr_matrix(other.reshape((1, other.size)))
                else:
                    raise NotImplementedError("Multiplication w.r.t. the third dimension with " +
                                              "factors with more than one dimension " +
                                              "has not been implemented yet.")
                
            else:
                raise NotImplementedError("Multiplication with vectors w.r.t. to the" +
                                          "second dimension has not been implemented yet.")
        else: 
            repOther = other
        self.data = self.data.multiply(repOther).todok()
        return self
    
    def multiply(self, other, axis = 0):
        newObj = cp.deepcopy(self)
        return newObj.imultiply(other, axis)

def test1():
    d1 = sp.dok_matrix((4,5))
    d2 = sp.dok_matrix((4,5))
    d3 = sp.dok_matrix((4,5))
    d1[1,1] = 1
    d1[0,1] = 1.6
    d1[2,2] = 7
    d2[1,0] = 3
    d2[1,1] = 2
    d2[1,2] = 1
    d2[0,1] = 20
    d2[1,1] = 2
    d2[2,1] = .2
    d3[1,1] = 3
    d3[1,2] = 8
    d3[0,0] = 9
    d = Sparse3D(d1, d2, d3)
    c = d[:].toarray()
    print(np.sum(c)-d.sum())
    print(np.sum(c, 0)-d.sum(0).toarray())
    print(d.sum(0).toarray())
    print(c)
    print("..............................")
    print(d[1].toarray() - c[1])
    print(d[:1].toarray() - c[:1])
    print(d[:,1].toarray() - c[:,1])
    print("----------------")
    print(d[:,:,:].toarray() - c[:,:,:])
    print(d[:,:,1].toarray() - c[:,:,1])
    print(d[:,1,:].toarray() - c[:,1,:])
    print(d[:,1,1].toarray() - c[:,1,1])
    print(d[1,:,:].toarray() - c[1,:,:])
    print(d[1,:,1].toarray() - c[1,:,1])
    print(d[1,1,:].toarray() - c[1,1,:])
    print(d[1,1,1] - c[1,1,1])

def test2():
    
    a1 = np.ones((4,5))
    a2 = a1 + 1
    a3 = a2 + 1
    s1 = sp.dok_matrix(a1)
    s2 = sp.dok_matrix(a2)
    s3 = sp.dok_matrix(a3)
    d = Sparse3D(s1, s2, s3)
    c = d.toarray()
    
    t1 = np.arange(3)
    t2 = np.arange(20).reshape((4,5))
    
    tt1 = np.arange(5)
    
    m0 = d.multiply(2, 0)
    m1 = d.multiply(t1, 0)
    m2 = d.multiply(t2, 0)
    
    mm1 = d.multiply(tt1, 2)
    
    print(c)
    print(m0.toarray())
    print(c*2)
    print(m1.toarray())
    print(c*np.expand_dims(np.expand_dims(t1,1),1))
    print(m2.toarray())
    print(c*t2)
    print(mm1.toarray())
    print(c*np.expand_dims(np.expand_dims(tt1,0),0))

def test3():
    
    a1 = np.ones((4,5))
    a2 = a1 + 1
    a3 = a2 + 1
    s1 = sp.dok_matrix(a1)
    s2 = sp.dok_matrix(a2)
    s3 = sp.dok_matrix(a3)
    d = Sparse3D(s1, s2, s3)
    
    d1 = sp.dok_matrix((4,5))
    d2 = sp.dok_matrix((4,5))
    d3 = sp.dok_matrix((4,5))
    d1[1,1] = 1
    d1[0,1] = 1.6
    d1[2,2] = 7
    d2[1,0] = 3
    d2[1,1] = 2
    d2[1,2] = 1
    d2[0,1] = 20
    d2[1,1] = 2
    d2[2,1] = .2
    d3[1,1] = 3
    d3[1,2] = 8
    d3[0,0] = 9
    d = Sparse3D(d1, d2, d3)
    
    print(d[[0, 2]].toarray())
    
    c = d.toarray()
    
    
    m0 = d.sum(0)
    m1 = d.sum(1)
    m2 = d.sum(2)
    
    r0 = c.sum(0)
    r1 = c.sum(1)
    r2 = c.sum(2)
    
    print(c)
    print(m0.toarray())
    print(r0)
    print(m1.toarray())
    print(r1)
    print(m2.toarray())
    print(r2)

if __name__ == '__main__':
    test3()