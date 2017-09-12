"""
The choices for cov function.

Author: Zhiang Zhang

First Created: Sept 12th, 2017
Last Updated: Sept 12th, 2017
"""
from BayCab4BEM.covFunction import *

covFuncMapping = {'covFuncNumpyImp': getCovMat_numpyImp,
				  'covFuncPymcNat': getCovMat_pymcNat};