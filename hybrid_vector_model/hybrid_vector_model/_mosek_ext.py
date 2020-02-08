'''
Created on 08.11.2017

@author: Samuel
'''
import mosek
from cvxopt import matrix, spmatrix, sparse

import sys

from cvxopt.msk import *


def lpX(c, G, h, A=None, b=None, taskfile=None):
    """
    Solves a pair of primal and dual LPs 

        minimize    c'*x             maximize    -h'*z - b'*y 
        subject to  G*x + s = h      subject to  G'*z + A'*y + c = 0
                    A*x = b                      z >= 0.
                    s >= 0
                    
    using MOSEK 8.0.

    (solsta, x, z, y) = lp(c, G, h, A=None, b=None).

    Input arguments 

        c is n x 1, G is m x n, h is m x 1, A is p x n, b is p x 1.  G and 
        A must be dense or sparse 'd' matrices.  c, h and b are dense 'd' 
        matrices with one column.  The default values for A and b are 
        empty matrices with zero rows.

        Optionally, the interface can write a .task file, required for
        support questions on the MOSEK solver.

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.optimal, then (x, y, z) contains the 
                primal-dual solution.
            If solsta is mosek.solsta.prim_infeas_cer, then (x, y, z) is a 
                certificate of primal infeasibility.
            If solsta is mosek.solsta.dual_infeas_cer, then (x, y, z) is a 
                certificate of dual infeasibility.
            If solsta is mosek.solsta.unknown, then (x, y, z) are all None.

            Other return values for solsta include:  
                mosek.solsta.dual_feas  
                mosek.solsta.near_dual_feas
                mosek.solsta.near_optimal
                mosek.solsta.near_prim_and_dual_feas
                mosek.solsta.near_prim_feas
                mosek.solsta.prim_and_dual_feas
                mosek.solsta.prim_feas
             in which case the (x,y,z) value may not be well-defined.
        
        x, y, z  the primal-dual solution.                    

    Options are passed to MOSEK solvers via the msk.options dictionary. 
    For example, the following turns off output from the MOSEK solvers
    
        >>> msk.options = {mosek.iparam.log: 0} 
    
    see the MOSEK Python API manual.                    
    """

    env = mosek.Env()

    if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1: 
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1: raise ValueError("number of variables must be at least 1")

    if (type(G) is not matrix and type(G) is not spmatrix) or \
        G.typecode != 'd' or G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    m = G.size[0]
    if m is 0: raise ValueError("m cannot be 0")

    if type(h) is not matrix or h.typecode != 'd' or h.size != (m,1):
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" %m)

    if A is None:  A = spmatrix([], [], [], (0,n), 'd')
    if (type(A) is not matrix and type(A) is not spmatrix) or \
        A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix "\
            "with %d columns" %n)
    p = A.size[0]
    if b is None: b = matrix(0.0, (0,1))
    if type(b) is not matrix or b.typecode != 'd' or b.size != (p,1): 
        raise TypeError("'b' must be a dense matrix of size (%d,1)" %p)
 
    bkc = m*[ mosek.boundkey.up ] + p*[ mosek.boundkey.fx ]
    blc = m*[ -inf ] + [ bi for bi in b ]
    buc = matrix([h, b])

    bkx = n*[mosek.boundkey.fr] 
    blx = n*[ -inf ] 
    bux = n*[ +inf ]

    colptr, asub, acof = sparse([G,A]).CCS
    aptrb, aptre = colptr[:-1], colptr[1:]

    task = env.Task(0,0) 
    task.set_Stream (mosek.streamtype.log, streamprinter) 

    # set MOSEK options 
    for (param, val) in options.items():
        if str(param)[:6] == "iparam":
            task.putintparam(param, val)
        elif str(param)[:6] == "dparam":
            task.putdouparam(param, val)
        elif str(param)[:6] == "sparam":
            task.putstrparam(param, val)
        else:
            raise ValueError("invalid MOSEK parameter: " + str(param))

    task.inputdata (m+p, # number of constraints
                    n,   # number of variables
                    list(c), # linear objective coefficients  
                    0.0, # objective fixed value  
                    list(aptrb), 
                    list(aptre), 
                    list(asub),
                    list(acof), 
                    bkc,
                    blc,
                    buc, 
                    bkx,
                    blx,
                    bux) 

    task.putobjsense(mosek.objsense.minimize)

    if taskfile:
        task.writetask(taskfile)

    task.optimize()

    task.solutionsummary (mosek.streamtype.msg); 

    solsta = task.getsolsta(mosek.soltype.bas)

    #x, z = n*[ 0.0 ], n*[ 0.0 ] BUG!
    x, z = n*[ 0.0 ], m*[ 0.0 ]
    task.getsolutionslice(mosek.soltype.bas, mosek.solitem.xx, 0, n, x) 
    task.getsolutionslice(mosek.soltype.bas, mosek.solitem.suc, 0, m, z) 
    x, z = matrix(x), matrix(z)
    
    if p is not 0:
        yu, yl = p*[0.0], p*[0.0]
        task.getsolutionslice(mosek.soltype.bas, mosek.solitem.suc, m, 
            m+p, yu) 
        task.getsolutionslice(mosek.soltype.bas, mosek.solitem.slc, m, 
            m+p, yl) 
        y = matrix(yu) - matrix(yl)
    else:
        y = matrix(0.0, (0,1))

    if (solsta is mosek.solsta.unknown):
        return (solsta, None, None, None)
    else:
        return (solsta, x, z, y)
    
mosek.msk.lp = lpX