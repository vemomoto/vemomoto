'''
Created on 03.10.2019

@author: Samuel
'''
from itertools import cycle
import re

import numpy as np
import matplotlib.pyplot as plt

COMPARE_PSEUDOINVERSE = True
FOLDER = "plots\\"
METHODS = ['Wald', 'RVM', 'RVM_psI', 'bisection', 'mixed_min', 'constrained_max', 'binsearch', 'VM', 'gridsearch']
#PLOTORDER = [ 7   ,    0 ,      3     ,       5     ,          6      ,        4  ,      1 ,   2       ]
PLOTORDER = [ 7   ,    0 ,     0.5,        3     ,       5     ,          6      ,        4  ,      1 ,   2       ]
METHODNAMES = ['Wald', 'RVM', 'RVM-MPI', 'Bisection', 'Neale-Miller', 'Constr. max.', 'Binary search', 'VM', 'Grid search']



DATA = [
    ["3", 500,
"""
Wald           : success=0.00000, error=329.5709, errorReduced=  3.8879, errorTot= 29.6541, largeErrors=0.16750, largeErrorsTot=0.16750, evals=     0.0, evalsTot=     0.0
RVM            : success=0.98583, error=  1.1031, errorReduced=  0.0045, errorTot=  0.0050, largeErrors=0.00083, largeErrorsTot=0.00083, evals=   774.3, evalsTot=  1525.9
RVM_psI        : success=0.98667, error=  1.1019, errorReduced=  0.0045, errorTot=  0.0052, largeErrors=0.00083, largeErrorsTot=0.00083, evals=   673.6, evalsTot=  1451.4
bisection      : success=0.77750, error=  0.2663, errorReduced=  0.0691, errorTot= 52.2622, largeErrors=0.03833, largeErrorsTot=0.06250, evals=  1532.6, evalsTot=  7350.2
mixed_min      : success=0.41500, error=  1.6748, errorReduced=  0.1446, errorTot=106.9708, largeErrors=0.00083, largeErrorsTot=0.09167, evals=   739.8, evalsTot=  1374.4
constrained_max: success=0.54167, error= 11.7703, errorReduced=  0.1839, errorTot=  5.3583, largeErrors=0.01750, largeErrorsTot=0.02083, evals=   581.3, evalsTot=  1054.8
binsearch      : success=0.80500, error=  0.0002, errorReduced=  0.0002, errorTot= 23.3635, largeErrors=0.00000, largeErrorsTot=0.01250, evals=  2194.5, evalsTot= 13465.2
VM             : success=0.08583, error= 31.8943, errorReduced=  0.2506, errorTot=  0.7301, largeErrors=0.00583, largeErrorsTot=0.00583, evals=   552.0, evalsTot=   317.0
gridsearch     : success=0.86500, error=168.1895, errorReduced=  0.0005, errorTot=204.6465, largeErrors=0.05000, largeErrorsTot=0.10167, evals=  8649.8, evalsTot=  9596.3
"""],
    ["3", 1000,
"""
Wald           : success=0.00000, error=275.0677, errorReduced=  4.5724, errorTot= 22.1636, largeErrors=0.18417, largeErrorsTot=0.18417, evals=     0.0, evalsTot=     0.0
RVM            : success=0.92583, error=  0.0002, errorReduced=  0.0002, errorTot=  0.0002, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   404.0, evalsTot=  1350.9
RVM_psI        : success=0.91083, error=  0.0002, errorReduced=  0.0002, errorTot=  0.0002, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   313.8, evalsTot=  1285.7
bisection      : success=0.78417, error=  0.2019, errorReduced=  0.0578, errorTot=  6.5979, largeErrors=0.05500, largeErrorsTot=0.06250, evals=  1335.7, evalsTot=  5618.0
mixed_min      : success=0.44333, error=  0.1066, errorReduced=  0.1066, errorTot=167.4518, largeErrors=0.00000, largeErrorsTot=0.14500, evals=   776.3, evalsTot=  1216.5
constrained_max: success=0.62000, error= 11.6432, errorReduced=  0.1671, errorTot= 10.7497, largeErrors=0.01917, largeErrorsTot=0.02750, evals=   618.4, evalsTot=  1084.8
binsearch      : success=0.87833, error=  0.0002, errorReduced=  0.0002, errorTot=  4.4954, largeErrors=0.00000, largeErrorsTot=0.00333, evals=  1987.1, evalsTot= 10605.7
VM             : success=0.22583, error= 12.5513, errorReduced=  0.1346, errorTot=  2.7998, largeErrors=0.00750, largeErrorsTot=0.00750, evals=   437.6, evalsTot=   339.6
gridsearch     : success=0.89417, error= 31.0430, errorReduced=  0.0006, errorTot= 45.0764, largeErrors=0.01250, largeErrorsTot=0.05500, evals=  6909.5, evalsTot=  7986.1
"""],
    ["3", 3000,
"""
Wald           : success=0.00000, error= 59.3576, errorReduced=  4.6430, errorTot= 18.8396, largeErrors=0.11083, largeErrorsTot=0.11083, evals=     0.0, evalsTot=     0.0
RVM            : success=0.98417, error=  0.0042, errorReduced=  0.0042, errorTot=  0.0042, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   244.7, evalsTot=   390.5
RVM_psI        : success=0.98417, error=  0.0042, errorReduced=  0.0042, errorTot=  0.0042, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   244.9, evalsTot=   384.9
bisection      : success=0.91417, error=  0.3843, errorReduced=  0.0799, errorTot=  0.3784, largeErrors=0.00833, largeErrorsTot=0.00833, evals=   402.0, evalsTot=  1351.4
mixed_min      : success=0.59250, error=  0.3191, errorReduced=  0.2959, errorTot=221.8844, largeErrors=0.00167, largeErrorsTot=0.22917, evals=   757.0, evalsTot=  1057.9
constrained_max: success=0.79667, error=  2.1308, errorReduced=  0.2337, errorTot= 11.5054, largeErrors=0.01250, largeErrorsTot=0.02417, evals=   649.1, evalsTot=   960.0
binsearch      : success=0.98167, error=  0.0006, errorReduced=  0.0006, errorTot=  0.0006, largeErrors=0.00000, largeErrorsTot=0.00000, evals=  1207.5, evalsTot=  2418.7
VM             : success=0.62083, error=  1.5650, errorReduced=  0.0093, errorTot=  1.9579, largeErrors=0.01417, largeErrorsTot=0.03250, evals=   303.0, evalsTot=   432.2
gridsearch     : success=0.94667, error=  0.0007, errorReduced=  0.0007, errorTot=  5.5064, largeErrors=0.00167, largeErrorsTot=0.04000, evals=  3144.7, evalsTot=  3619.0
"""],
    ["3", 10000,
"""
Wald           : success=0.00000, error=  3.7752, errorReduced=  3.4170, errorTot=  3.7752, largeErrors=0.01083, largeErrorsTot=0.01083, evals=     0.0, evalsTot=     0.0
RVM            : success=1.00000, error=  0.0001, errorReduced=  0.0001, errorTot=  0.0001, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   169.5, evalsTot=   169.5
RVM_psI        : success=1.00000, error=  0.0001, errorReduced=  0.0001, errorTot=  0.0001, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   169.5, evalsTot=   169.5
bisection      : success=1.00000, error=  0.0103, errorReduced=  0.0103, errorTot=  0.0103, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   243.1, evalsTot=   243.1
mixed_min      : success=0.79917, error=  0.4284, errorReduced=  0.3742, errorTot= 10.0984, largeErrors=0.00500, largeErrorsTot=0.01500, evals=   751.5, evalsTot=   762.2
constrained_max: success=0.85833, error=  0.1481, errorReduced=  0.1481, errorTot=  3.4722, largeErrors=0.00000, largeErrorsTot=0.00333, evals=   554.8, evalsTot=   603.3
binsearch      : success=1.00000, error=  0.0002, errorReduced=  0.0002, errorTot=  0.0002, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   912.0, evalsTot=   912.0
VM             : success=0.87750, error=  0.0893, errorReduced=  0.0519, errorTot=  0.4128, largeErrors=0.00083, largeErrorsTot=0.00083, evals=   237.8, evalsTot=   580.7
gridsearch     : success=1.00000, error=  0.0007, errorReduced=  0.0007, errorTot=  0.0007, largeErrors=0.00000, largeErrorsTot=0.00000, evals=  1788.4, evalsTot=  1788.4
"""],
    ["11cx", 50,
"""
Wald           : success=0.41318, error= 94.9196, errorReduced=  0.4698, errorTot=177.6506, largeErrors=0.08727, largeErrorsTot=0.08727, evals=     0.0, evalsTot=     0.0
RVM            : success=0.91932, error=  0.0003, errorReduced=  0.0003, errorTot= 35.3982, largeErrors=0.00750, largeErrorsTot=0.06273, evals=  2273.4, evalsTot= 19212.6
RVM_psI        : success=0.97386, error=  0.0001, errorReduced=  0.0001, errorTot= 13.5498, largeErrors=0.01750, largeErrorsTot=0.02364, evals=  1649.5, evalsTot= 16011.4
bisection      : success=0.39114, error=  2.6273, errorReduced=  0.0016, errorTot=112.1667, largeErrors=0.35432, largeErrorsTot=0.45523, evals=  1905.1, evalsTot= 32364.8
mixed_min      : success=0.24386, error=  6.6671, errorReduced=  0.5488, errorTot= 15.7724, largeErrors=0.02841, largeErrorsTot=0.03432, evals=  2661.1, evalsTot=  4685.6
constrained_max: success=0.46227, error=  5.6586, errorReduced=  0.0058, errorTot=  8.4952, largeErrors=0.00341, largeErrorsTot=0.01000, evals=  1203.2, evalsTot=  1356.3
binsearch      : success=0.52500, error=  0.0003, errorReduced=  0.0003, errorTot= 32.2177, largeErrors=0.00000, largeErrorsTot=0.01318, evals=  7902.6, evalsTot= 39368.6
VM             : success=0.35045, error=  0.0711, errorReduced=  0.0001, errorTot=  0.0711, largeErrors=0.00114, largeErrorsTot=0.00114, evals=   850.0, evalsTot=   511.8
gridsearch     : success=0.49500, error= 24.3749, errorReduced=  0.0063, errorTot=120.1377, largeErrors=0.42909, largeErrorsTot=0.49364, evals= 37619.1, evalsTot= 64827.9
"""],
    ["11cx", 100,
"""
Wald           : success=0.21000, error= 13.6389, errorReduced=  0.2674, errorTot= 13.9721, largeErrors=0.01545, largeErrorsTot=0.01545, evals=     0.0, evalsTot=     0.0
RVM            : success=0.98477, error=  0.0001, errorReduced=  0.0001, errorTot=  1.3701, largeErrors=0.00068, largeErrorsTot=0.01409, evals=   664.2, evalsTot=  2195.0
RVM_psI        : success=0.99841, error=  0.0001, errorReduced=  0.0001, errorTot=  0.0176, largeErrors=0.00091, largeErrorsTot=0.00136, evals=   662.4, evalsTot=  1693.7
bisection      : success=0.94750, error=  0.2426, errorReduced=  0.0002, errorTot=  6.5341, largeErrors=0.02523, largeErrorsTot=0.04227, evals=   907.2, evalsTot=  3701.5
mixed_min      : success=0.76455, error=  0.8103, errorReduced=  0.1259, errorTot=  0.9261, largeErrors=0.00636, largeErrorsTot=0.00682, evals=  1536.9, evalsTot=  1779.9
constrained_max: success=0.95227, error=  0.3252, errorReduced=  0.0047, errorTot=  0.2704, largeErrors=0.00068, largeErrorsTot=0.00091, evals=   760.7, evalsTot=   797.6
binsearch      : success=0.96750, error=  0.0003, errorReduced=  0.0003, errorTot=  0.6173, largeErrors=0.00000, largeErrorsTot=0.00091, evals=  4369.1, evalsTot=  7323.5
VM             : success=0.94500, error=  0.0001, errorReduced=  0.0001, errorTot=  0.0001, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   739.3, evalsTot=   716.6
gridsearch     : success=0.94977, error=  2.3037, errorReduced=  0.0009, errorTot=  7.8223, largeErrors=0.03250, largeErrorsTot=0.04795, evals= 15137.6, evalsTot= 19290.7
"""],
    ["11cx", 300,
"""
Wald           : success=0.81295, error=  0.0244, errorReduced=  0.0244, errorTot=  0.0244, largeErrors=0.00000, largeErrorsTot=0.00000, evals=     0.0, evalsTot=     0.0
RVM            : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   451.0, evalsTot=   451.0
RVM_psI        : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   451.0, evalsTot=   451.0
bisection      : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   904.6, evalsTot=   904.6
mixed_min      : success=0.95023, error=  0.0077, errorReduced=  0.0077, errorTot=  0.0077, largeErrors=0.00000, largeErrorsTot=0.00000, evals=  1087.1, evalsTot=  1087.1
constrained_max: success=0.98886, error=  0.0018, errorReduced=  0.0018, errorTot=  0.0018, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   607.9, evalsTot=   607.9
binsearch      : success=1.00000, error=  0.0005, errorReduced=  0.0005, errorTot=  0.0005, largeErrors=0.00000, largeErrorsTot=0.00000, evals=  3258.9, evalsTot=  3258.9
VM             : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   602.8, evalsTot=   602.8
gridsearch     : success=0.99864, error=  0.0008, errorReduced=  0.0008, errorTot=  0.0008, largeErrors=0.00000, largeErrorsTot=0.00000, evals= 10721.5, evalsTot= 10721.5
"""],
    ["11cx", 1000,
"""
Wald           : success=0.98682, error=  0.0057, errorReduced=  0.0057, errorTot=  0.0057, largeErrors=0.00000, largeErrorsTot=0.00000, evals=     0.0, evalsTot=     0.0
RVM            : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   451.0, evalsTot=   451.0
RVM_psI        : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   451.0, evalsTot=   451.0
bisection      : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   942.0, evalsTot=   942.0
mixed_min      : success=0.99614, error=  0.0019, errorReduced=  0.0019, errorTot=  0.0019, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   928.2, evalsTot=   928.2
constrained_max: success=0.97818, error=  0.0030, errorReduced=  0.0030, errorTot=  0.0030, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   663.0, evalsTot=   663.0
binsearch      : success=1.00000, error=  0.0005, errorReduced=  0.0005, errorTot=  0.0005, largeErrors=0.00000, largeErrorsTot=0.00000, evals=  3298.6, evalsTot=  3298.6
VM             : success=1.00000, error=  0.0000, errorReduced=  0.0000, errorTot=  0.0000, largeErrors=0.00000, largeErrorsTot=0.00000, evals=   597.0, evalsTot=   597.0
gridsearch     : success=0.99864, error=  0.0008, errorReduced=  0.0008, errorTot=  0.0008, largeErrors=0.00000, largeErrorsTot=0.00000, evals=  9786.2, evalsTot=  9786.2
"""],
    ["11", 500,
"""
Wald           : success=0.02318, error=303.4153, errorReduced=  0.8097, errorTot= 61.7644, largeErrors=0.05477, largeErrorsTot=0.05477, evals=     0.0, evalsTot=     0.0
RVM            : success=0.89659, error=  4.8438, errorReduced=  0.0473, errorTot=  6.2265, largeErrors=0.00932, largeErrorsTot=0.01455, evals=  3961.5, evalsTot= 11800.9
RVM_psI        : success=0.82114, error=  4.0107, errorReduced=  0.0363, errorTot= 50.1538, largeErrors=0.02545, largeErrorsTot=0.04182, evals=  4656.8, evalsTot= 14374.8
bisection      : success=0.70886, error= 83.8969, errorReduced=  0.0592, errorTot=108.7771, largeErrors=0.13250, largeErrorsTot=0.16205, evals=  4822.9, evalsTot= 17381.9
mixed_min      : success=0.50659, error=  2.1420, errorReduced=  0.1782, errorTot=6771321335269173.0000, largeErrors=0.07977, largeErrorsTot=0.10659, evals=  1734.6, evalsTot=  2172.6
constrained_max: success=0.57091, error=4504648188.6076, errorReduced=  0.1868, errorTot=130.4789, largeErrors=0.03205, largeErrorsTot=0.04318, evals=  1220.8, evalsTot=  2703.0
binsearch      : success=0.71182, error= 14.1099, errorReduced=  0.1199, errorTot= 47.2506, largeErrors=0.01909, largeErrorsTot=0.04773, evals= 16281.7, evalsTot= 82755.7
VM             : success=0.24273, error= 10.8335, errorReduced=  0.0898, errorTot=  0.9376, largeErrors=0.00341, largeErrorsTot=0.00386, evals=  2049.4, evalsTot=  2942.8
gridsearch     : success=0.76841, error= 59.3983, errorReduced=  0.0433, errorTot=120.6592, largeErrors=0.14159, largeErrorsTot=0.17455, evals=147042.5, evalsTot=220170.5
"""],
    ["11", 1000,
"""
Wald           : success=0.03818, error=225.2044, errorReduced=  0.7528, errorTot= 37.6608, largeErrors=0.07614, largeErrorsTot=0.07614, evals=     0.0, evalsTot=     0.0
RVM            : success=0.90659, error=  3.8319, errorReduced=  0.0230, errorTot=  6.9243, largeErrors=0.00955, largeErrorsTot=0.01659, evals=  3170.7, evalsTot= 10291.7
RVM_psI        : success=0.87318, error=  3.9953, errorReduced=  0.0131, errorTot= 29.9446, largeErrors=0.01386, largeErrorsTot=0.02614, evals=  3090.3, evalsTot= 11212.8
bisection      : success=0.75773, error= 59.9321, errorReduced=  0.0340, errorTot= 51.4892, largeErrors=0.11636, largeErrorsTot=0.13545, evals=  3824.5, evalsTot= 15607.8
mixed_min      : success=0.57864, error=  3.2899, errorReduced=  0.1193, errorTot=3262584376525311.5000, largeErrors=0.05955, largeErrorsTot=0.08568, evals=  1574.6, evalsTot=  2034.5
constrained_max: success=0.63523, error=163.7362, errorReduced=  0.0899, errorTot= 66.6031, largeErrors=0.02932, largeErrorsTot=0.04068, evals=  1088.0, evalsTot=  2486.4
binsearch      : success=0.75273, error=  9.3412, errorReduced=  0.0976, errorTot= 32.5381, largeErrors=0.01091, largeErrorsTot=0.03591, evals= 15150.5, evalsTot= 73976.1
VM             : success=0.32068, error=  7.2177, errorReduced=  0.1128, errorTot=  0.4650, largeErrors=0.00386, largeErrorsTot=0.00432, evals=  1915.2, evalsTot=  3351.9
gridsearch     : success=0.80159, error= 31.3783, errorReduced=  0.0334, errorTot= 66.6911, largeErrors=0.13591, largeErrorsTot=0.15909, evals=180081.8, evalsTot=243021.1
"""],
    ["11", 3000,
"""
Wald           : success=0.08659, error=159.3901, errorReduced=  0.4562, errorTot= 17.1385, largeErrors=0.07205, largeErrorsTot=0.07205, evals=     0.0, evalsTot=     0.0
RVM            : success=0.93659, error=  3.7162, errorReduced=  0.0099, errorTot=  7.9023, largeErrors=0.00886, largeErrorsTot=0.01341, evals=  2190.8, evalsTot=  7451.9
RVM_psI        : success=0.90818, error=  3.8506, errorReduced=  0.0128, errorTot=  8.0271, largeErrors=0.00977, largeErrorsTot=0.01523, evals=  2229.1, evalsTot=  8263.6
bisection      : success=0.82432, error= 30.3407, errorReduced=  0.0485, errorTot= 16.4771, largeErrors=0.08705, largeErrorsTot=0.10091, evals=  3125.4, evalsTot= 10866.5
mixed_min      : success=0.69159, error=  2.3051, errorReduced=  0.0453, errorTot=15814584170206738.0000, largeErrors=0.02364, largeErrorsTot=0.05068, evals=  1401.5, evalsTot=  1877.6
constrained_max: success=0.72977, error= 30.9454, errorReduced=  0.0352, errorTot= 19.2405, largeErrors=0.02182, largeErrorsTot=0.02795, evals=  1051.8, evalsTot=  2059.0
binsearch      : success=0.82205, error=  4.0463, errorReduced=  0.0395, errorTot= 22.6494, largeErrors=0.00659, largeErrorsTot=0.02523, evals= 13667.2, evalsTot= 58230.7
VM             : success=0.46341, error=  2.5814, errorReduced=  0.0403, errorTot=  0.2392, largeErrors=0.00205, largeErrorsTot=0.00227, evals=  1472.1, evalsTot=  3929.4
gridsearch     : success=0.85000, error= 11.6432, errorReduced=  0.0216, errorTot= 31.7003, largeErrors=0.09636, largeErrorsTot=0.11568, evals=205155.5, evalsTot=407456.8
"""],
    ["11", 10000,
"""
Wald           : success=0.16909, error= 90.9859, errorReduced=  0.2847, errorTot=  8.7071, largeErrors=0.06205, largeErrorsTot=0.06205, evals=     0.0, evalsTot=     0.0
RVM            : success=0.98227, error=  3.1278, errorReduced=  0.0038, errorTot=  1.1732, largeErrors=0.00750, largeErrorsTot=0.00750, evals=  1229.2, evalsTot=  3773.7
RVM_psI        : success=0.95614, error=  3.2261, errorReduced=  0.0039, errorTot=  1.1757, largeErrors=0.00659, largeErrorsTot=0.00659, evals=  1101.4, evalsTot=  4441.4
bisection      : success=0.93273, error=  9.3139, errorReduced=  0.0211, errorTot=  1.6629, largeErrors=0.03886, largeErrorsTot=0.04114, evals=  2307.8, evalsTot=  6408.4
mixed_min      : success=0.81182, error=  1.5260, errorReduced=  0.0194, errorTot= 19.6726, largeErrors=0.00386, largeErrorsTot=0.02227, evals=  1300.8, evalsTot=  1550.9
constrained_max: success=0.85659, error=  9.8237, errorReduced=  0.0219, errorTot=  4.3433, largeErrors=0.01000, largeErrorsTot=0.01341, evals=  1036.5, evalsTot=  1537.3
binsearch      : success=0.91205, error=  0.9294, errorReduced=  0.0003, errorTot=  3.2472, largeErrors=0.00227, largeErrorsTot=0.00568, evals= 11049.7, evalsTot= 35509.1
VM             : success=0.69864, error= 11.3392, errorReduced=  0.1237, errorTot=  1.2125, largeErrors=0.01250, largeErrorsTot=0.01295, evals=  1181.5, evalsTot=  2684.7
gridsearch     : success=0.87932, error=  1.9279, errorReduced=  0.0044, errorTot=  6.8491, largeErrors=0.07273, largeErrorsTot=0.08568, evals= 96252.3, evalsTot=517540.5
"""],



]


dtype = [("success", float), ("error", float), 
         ("errorReduced", float), ("errorTot", float), ("largeErrors", float),
         ("largeErrorsTot", float), ("evals", float), ("evalsTot", float)]


DATA = [[*d[:2], np.genfromtxt(re.sub("[^\d\.\n\,]", "", d[2]).strip().split('\n'), 
                               delimiter=",", dtype=dtype)]
        for d in DATA]
METHODS = np.array(METHODS)
METHODNAMES = np.array(METHODNAMES)
PLOTORDER = np.array(PLOTORDER)


def get_feature(feature):
    data = np.array([d[feature] for _, _, d in DATA]).T
    print(feature, "========================")
    for fun in np.max, np.min, np.mean, np.median:
        print(fun, "----------------")
        for method, row in zip(METHODNAMES, data):
            print("{:<15} {:6.3f}".format(method, fun(row)))
    
def plot_feature(key, feature, getLegend=False):
    data = [d[1:] for d in DATA if d[0]==key]
        
    pltData = np.array([d[feature] for _, d in data]).T
    markers = ["o", "v", "s", "p", "D", ">", "*", "X", "."]
    x = [d[0] for d in data]
    figsize = (3.5, 3)
    rect = [0.15, 0.18, 0.75, 0.78]
    ax = plt.figure(figsize=figsize).add_axes(rect)
    ax.locator_params(nticks=3, nbins=3)
    order = np.argsort(PLOTORDER)
    
    if COMPARE_PSEUDOINVERSE:
        order = order[:2]
    else:
        order = np.delete(order, 1)
    
    colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    tmp = colors[0]
    colors[0] = colors[1]
    colors[1] = tmp
    linestyle_cycler = cycle(["-","--","-.",":"])
    
    for method, d, c, m, l, o in zip(METHODNAMES[order], pltData[order], colors,
                                     markers, linestyle_cycler, 
                                     range(order.size)[::-1]):
        if method=="Wald" and feature=="evals":
            continue
        plt.plot(x, d, label=method, color=c, marker=m, linestyle=l,
                 alpha=0.6, zorder=o)
    plt.xscale("log")
    if feature=="error":
        plt.yscale("symlog")
        plt.ylim((-0.5,1000))
    elif feature=="evals":
        plt.yscale("log")
        plt.ylim((100,1e6))
    elif feature=="success":
        if COMPARE_PSEUDOINVERSE:
            plt.ylim((0.75,1.05))
        else:
            plt.ylim((-0.05,1.05))
        
    if getLegend:
        plt.legend(loc='lower right', fontsize=11.5, #frameon=False,
                   fancybox=True, framealpha=0.5)
        #figLegend = plt.figure(figsize = (2,2))
        #plt.figlegend(*ax.get_legend_handles_labels(), loc='upper left',
        #              fancybox=True, frameon=False)
    
    if COMPARE_PSEUDOINVERSE:
        addition = "-psi"
    else:
        addition = ""
    plt.savefig(FOLDER + feature + key + addition + ".pdf")
    #plt.show()

def createPlotsKey(key):
    plot_feature(key, "success")
    plot_feature(key, "error")
    plot_feature(key, "evals")

def createPlots():
    createPlotsKey("11")
    createPlotsKey("3")
    createPlotsKey("11cx")

if __name__ == '__main__':
    #createPlots()
    #plot_feature("11cx", "success", True)
    get_feature("error")
    get_feature("errorReduced")
    get_feature("largeErrors")
    get_feature("success")
    plt.show()