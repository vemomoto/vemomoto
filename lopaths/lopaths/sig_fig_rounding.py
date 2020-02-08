'''
Retrieved on 08.07.2019
from https://github.com/odysseus9672/SELPythonLibs/blob/master/SigFigRounding.py

@author: odysseus9672

'''

#Library to implement significant figure rounding using numpy functions

import decimal as decim
__context = decim.getcontext()
__context.prec = 64
decim.setcontext(__context)

__logBase10of2_decim = decim.Decimal(2).log10()
__logBase10of2 = float(__logBase10of2_decim)
__logBase10ofe_decim = 1 / decim.Decimal(10).ln()
__logBase10ofe = float(__logBase10ofe_decim)


import numpy as np

# __res = decim.Decimal(str(np.finfo(float).resolution))

Land = np.logical_and
Lor = np.logical_or
Lnot = np.logical_not


def RoundToSigFigs_fp( x, sigfigs, upper=False):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.
    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not ( type(sigfigs) is int or type(sigfigs) is long or
             isinstance(sigfigs, np.integer) ):
        raise TypeError( "RoundToSigFigs_fp: sigfigs must be an integer." )

    if sigfigs <= 0:
        raise ValueError( "RoundToSigFigs_fp: sigfigs must be positive." )
    
    if not np.all(np.isreal( x )):
        raise TypeError( "RoundToSigFigs_fp: all x must be real." )

    #temporarily suppres floating point errors
    errhanddict = np.geterr()
    np.seterr(all="ignore")

    matrixflag = False
    if isinstance(x, np.matrix): #Convert matrices to arrays
        matrixflag = True
        x = np.asarray(x)
    
    xsgn = np.sign(x)
    absx = xsgn * x
    mantissas, binaryExponents = np.frexp( absx )
    
    decimalExponents = __logBase10of2 * binaryExponents
    omags = np.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - omags)
    
    if type(mantissas) is float or isinstance(mantissas, np.floating):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0
            
    else: #elif np.all(np.isreal( mantissas )):
        fixmsk = mantissas < 1.0, 
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0
    
    result = xsgn * 10.0**omags
    
    if upper:
        result *= np.around(mantissas+10**(1-sigfigs), decimals=sigfigs - 1)
    else:
        result *= np.around(mantissas, decimals=sigfigs - 1)
         
    if matrixflag:
        result = np.matrix(result, copy=False)

    np.seterr(**errhanddict)
    return result


def RoundToSigFigs_decim( x, sigfigs ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Uses logarithm function and Python's Decimal library, so will be slower.
    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a Python Decimal value. No arrays accepted.
    """
    if not ( type(sigfigs) is int or type(sigfigs) is long or
             isinstance(sigfigs, np.integer) ):
        raise TypeError( "RoundToSigFigs_Decim: sigfigs must be an integer." )

    if sigfigs <= 0:
        raise ValueError( "RoundToSigFigs_Decim: sigfigs must be positive." )
    
    if not (type(x) == type(decim.Decimal(1))):
        raise TypeError( "RoundToSigFigs_Decim: x must by a Python Decimal." )

    xsgn = decim.Decimal(1).copy_sign(x)
    absx = x * xsgn

    if absx.is_nan():
        result = decim.Decimal("NaN")
        
    elif absx > 0 and not absx.is_infinite():
        dec10 = decim.Decimal(10)
        
        log10x = absx.log10()
        omag = int(log10x.quantize(1, rounding=decim.ROUND_FLOOR))
        mantissas = absx * dec10**(-omag)
        
        result = xsgn * mantissas.quantize( dec10**(1 - sigfigs) ) * dec10**omag
        result = result.normalize()
        
    elif absx == 0:
        result = decim.Decimal(0)
        
    else: # absx.is_infinite():
        result = xsgn * decim.Decimal("inf")
    
    return result


def ValueWithUncsRounding( x, uncs, uncsigfigs=1 ):
    """
    Rounds all of the values in uncs (the uncertainties) to the number of
    significant figures in uncsigfigs. Then
    rounds the values in x to the same decimal pace as the values in uncs.
    Return value is a two element tuple each element of which has the same
    type as x and uncs, respectively.
    Restrictions:
    - uncsigfigs must be a positive integer. 
    
    - x must be a real value or an array like object containing only real
      values.
    - uncs must be a real value or an array like object containing only real
      values.
    """
    if not ( type(uncsigfigs) is int or type(uncsigfigs) is long or
             isinstance(uncsigfigs, np.integer) ):
        raise TypeError(
            "ValueWithUncsRounding: uncsigfigs must be an integer." )

    if uncsigfigs <= 0:
        raise ValueError(
            "ValueWithUncsRounding: uncsigfigs must be positive." )

    if not np.all(np.isreal( x )):
        raise TypeError(
            "ValueWithUncsRounding: all x must be real." )

    if not np.all(np.isreal( uncs )):
        raise TypeError(
            "ValueWithUncsRounding: all uncs must be real." )

    if np.any( uncs <= 0 ):
        raise ValueError(
            "ValueWithUncsRounding: uncs must all be positive." )

    #temporarily suppres floating point errors
    errhanddict = np.geterr()
    np.seterr(all="ignore")

    matrixflag = False
    if isinstance(x, np.matrix): #Convert matrices to arrays
        matrixflag = True
        x = np.asarray(x)

    #Pre-round unc to correctly handle cases where rounding alters the
    # most significant digit of unc.
    uncs = RoundToSigFigs_fp( uncs, uncsigfigs )
    
    mantissas, binaryExponents = np.frexp( uncs )
    
    decimalExponents = __logBase10of2 * binaryExponents
    omags = np.floor(decimalExponents)

    mantissas *= 10.0**(decimalExponents - omags)
    if type(mantissas) is float or np.issctype(np.dtype(mantissas)):
        if mantissas < 1.0:
            mantissas *= 10.0
            omags -= 1.0
            
    else: #elif np.all(np.isreal( mantissas )):
        fixmsk = mantissas < 1.0
        mantissas[fixmsk] *= 10.0
        omags[fixmsk] -= 1.0

    scales = 10.0**omags

    prec = uncsigfigs - 1
    result =  ( np.around( x / scales, decimals=prec ) * scales,
                np.around( mantissas, decimals=prec ) * scales )
    if matrixflag:
        result = np.matrix(result, copy=False)

    np.seterr(**errhanddict)
    return result


import math
# import decimal as decim


def FormatValToSigFigs( x, sigfigs, sciformat=True ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value is a string. The keyword argument
    sciformat will force scientific notation if true.
    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value or an array like object containing only real values.
    """
    if not ( type(sigfigs) is int or np.issubdtype(sigfigs, np.integer) ):
        raise TypeError(
            "FormatValToSigFigs: sigfigs must be an integer." )

    if sigfigs <= 0:
        raise ValueError(
            "FormatValToSigFigs: sigfigs must be positive." )

    if not np.isreal(x):
        raise TypeError(
            "FormatValToSigFigs: x must be real." )
    
    if sciformat:
        formstrn = "{:." + str(sigfigs-1) + "f}e{:+d}"
        def formatter( m, e ):
            return formstrn.format(m, int(e))
    else:
        formstrn = "{:." + str(sigfigs-1) + "g}"
        def formatter( m, e ):
            return formstrn.format(m*10.0**e)

    x = float(x)

    xsgn = np.sign(x)
    absx = xsgn*x
    mantissa, binaryExponent = np.frexp( absx )

    decimalExponent = __logBase10of2 * binaryExponent
    omag = np.floor(decimalExponent)

    mantissa *= 10.0**(decimalExponent - omag)
    if mantissa < 1.0:
        mantissa *= 10.0
        omag -= 1

    mantissa = xsgn * np.around( mantissa, decimals=sigfigs - 1 )
    
    if ( not math.isinf(absx) and not math.isinf(omag) and not math.isnan(x)
         and absx > 0.0 ):
        result = formatter( mantissa, omag )
    elif math.isinf(absx):
        mantissa = float(xsgn) * float("inf")
        result = "{:f}".format(mantissa)
    elif not math.isnan(x):
        result = formatter( 0.0, 0 )
    else:
        result = "nan"

    return result


def FormatValWithUncRounding( x, unc, uncsigfigs=1, sciformat=True ):
    """
    Rounds unc (the uncertainty) to the number of significant figures in
    uncsigfigs. Then rounds the value in x to the same decimal pace as the
    value in unc. Uses the decimal package for maximal accuracy.
    Return value is a tuple containing two strings. The keyword argument
    sciformat will force scientific notation if true.
    Restrictions:
    - uncsigfigs must be a positive integer.
    - x must be a real value or floating point.
    - unc must be a real value or floating point
    """
    if not ( type(uncsigfigs) is int or type(uncsigfigs) is long or
             isinstance(uncsigfigs, np.integer) ):
        raise TypeError(
            "FormatValWithUncRounding: uncsigfigs must be an integer." )

    if uncsigfigs <= 0:
        raise ValueError(
            "FormatValWithUncRounding: uncsigfigs must be positive." )

    if not np.isreal(x):
        raise TypeError(
            "FormatValWithUncRounding: x must be real." )

    if not np.isreal(unc) or unc <= 0.0:
        raise TypeError(
            "FormatValWithUncRounding: unc must be a positive real." )

    if isinstance(x, np.matrix): #Convert matrices to arrays
        x = np.asarray(x)

    #sys.stderr.write("Warning: FormatValWithUncRounding is untested.\n")

    #Pre-round unc to correctly handle cases where rounding alters the
    # most significant digit of unc.
    unc = RoundToSigFigs_fp( unc, uncsigfigs ) 

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissa, binaryExponent = np.frexp( absx )
    uncmant, uncbinExponent = np.frexp( unc )
    
    decimalExponent = __logBase10of2 * binaryExponent
    uncdecExponent = __logBase10of2 * uncbinExponent
    
    omag = np.floor(decimalExponent)
    uncomag = np.floor(uncdecExponent)

    uncmant *= 10**(uncdecExponent - uncomag)
    if not np.isnan(uncmant) and uncmant < 1.0:
        uncmant *= 10.0
        uncomag -= 1.0

    mantissa *= 10**(decimalExponent - omag)
    if not np.isnan(mantissa) and mantissa < 1.0:
        mantissa *= 10.0
        omag -= 1.0

    omagdiff = omag - uncomag
    prec = uncsigfigs - 1 + int(omagdiff)


    mantissa, uncOut = ( np.around( mantissa, decimals=prec ),
                         np.around( unc * 10**(-omag), decimals=prec ) )

    if sciformat:
        def formatter( m, u, e, prec ):
            s = "{:." + str(max(prec, 0)) + "f}e{:+d}"
            return (s.format(m, int(e)), s.format(u, int(e)))
    else:
        def formatter( m, u, e, prec ):
            s = "{:." + str(max(prec, 0)) + "g}"
            return (s.format(m*10.0**e), s.format(u*10**e))
    
    return formatter( mantissa, uncOut, omag, prec )

def SetDecimalPrecision( precision ):
    if not ( type(sigfigs) is int or type(sigfigs) is long or
             isinstance(sigfigs, np.integer) ):
        raise TypeError( "SetDecimalPrecision: prec must be an integer." )

    if precision < 0:
        raise ValueError( "SetDecimalPrecision: prec cannot be negative." )

    global __context
    __context.prec = precision
    decim.setcontext(__context)
    
    global __logBase10of2_decim, __logBase10ofe_decim
    __logBase10of2_decim = decim.Decimal(2).log10()
    __logBase10ofe_decim = 1 / decim.Decimal(10).ln()

    return None

if __name__ == '__main__':
    a = [0.9989, 1., 0.9999999999]
    print(RoundToSigFigs_fp(a, 3, True))
    print(RoundToSigFigs_fp(a, 3))
    
    
# __testsigfigs = (1, 3, 6)
# __finfo = np.finfo(float)
# __testnums = [  0.0, -1.2366e22, 1.2544444e-15, 0.001222, 0.0,
#                 float("nan"), float("inf"), float.fromhex("0x4.23p-1028"),
#                 0.5555, 1.5444, 1.72340, 1.256e-15, 10.5555555,
#                 __finfo.max, __finfo.min,
#                 0.5555, 0.5555*(1.0 + __finfo.eps), 0.5555*(1.0 - __finfo.epsneg) ]
# __testres = {1: [ 0.0, -1e22, 1e-15, 1e-3, 0.0,
#                   float("nan"), float("inf"), 1e-309,
#                   0.6, 2.0, 2.0, 1.0e-15, 10.0,
#                   float("inf"), -float("inf"),
#                   0.6, 0.6, 0.6 ],
#              3: [ 0.0, -1.24e22, 1.25e-15, 1.22e-3, 0.0,
#                   float("nan"), float("inf"), 1.44e-309,
#                   5.56e-1, 1.54, 1.72, 1.26e-15, 10.6,
#                   float("inf"), -float("inf"),
#                   0.556, 0.556, 0.555 ],
#              6: [ 0.0, -1.2366e22, 1.25444e-15, 0.001222, 0.0,
#                 float("nan"), float("inf"), 1.43820e-309,
#                 0.5555, 1.5444, 1.72340, 1.256e-15, 10.5556,
#                 1.79769e+308, -1.79769e+308,
#                 0.5555, 0.5555, 0.5555 ] }

# def TestFuncs():
#     finfo = np.finfo(float)
#     if finfo.bits != 64 or finfo.nexp != 11 or finfo.nmant != 52:
#         raise RuntimeError("Test only implemented for 64 bit floats.")

#     testarr = np.array(__testnums)
    
#     for sf in __testsigfigs:
#         out = RoundToSigFigs_fp( testarr, sf )
        
#         standard = np.array(__testres[sf])
#         infmsk = np.isinf(standard)
#         nanmsk = np.isnan(standard)
#         normmsk = Lnot(Lor( infmsk, nanmsk ))
        
#         absdiff = np.abs( out[normmsk] - standard[normmsk] )

#         if ( np.any(absdiff > 16.0 * __finfo.eps) or
#              np.any(out[infmsk] != standard[infmsk]) or
#              np.any(Lnot(np.isnan( out[nanmsk] ))) ):
#             raise RuntimeError(
#                 "RoundToSigFigs_fp test failed at sigfigs={:d}".format(sf) )

#         out = np.asarray([ float(RoundToSigFigs_decim( decim.Decimal(x), sf ))
#                            for x in testarr ])
        
#         standard = np.array(__testres[sf])
#         infmsk = np.isinf(standard)
#         nanmsk = np.isnan(standard)
#         normmsk = Lnot(Lor( infmsk, nanmsk ))
        
#         absdiff = np.abs( out[normmsk] - standard[normmsk] )

#         if ( np.any(absdiff > 16.0 * __finfo.eps) or
#              np.any(out[infmsk] != standard[infmsk]) or
#              np.any(Lnot(np.isnan( out[nanmsk] ))) ):
#             print(absdiff > 16.0 * __finfo.eps)
#             print(absdiff)
#             raise RuntimeError(
#                 "RoundToSigFigs_decim test failed at sigfigs={:d}".format(sf) )

#     return None