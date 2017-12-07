"""
Get native values from their normalized values.

Author: Zhiang Zhang
Date: Dec 7th, 2017
"""
import numpy as np

def getNatValuesFromMinMaxNorm(stdLHSSamples, paraRanges):
	"""
	The method return the LHS sampled random varaibles for the calibration parameters in their
	native value range. The conversion method is reverse min-max standarilization, i.e
		varInNativeRange = varMin + (varMax - varMin) * stdVar

	Args:
		stdLHSSamples: numpy.ndarray
			The LHS samples variables in the standalized way (0-1). Each row is a sample, each
			col is a feature.
		paraRanges: list
			A 2-D list, where each row corresponds to one parameter. For each row, index 0 is 
			the maximum limit, index 1 is the min limit. 
	Ret: numpy.ndarray
		An array where each row one sample, each col is one feature containing the LHS sampled
		random variables in their native range. 
	"""
	retSampleInputs = np.copy(stdLHSSamples);
	paraRangesArray = np.array(paraRanges);
	paraRangeLen = paraRangesArray[:, 0] - paraRangesArray[:, 1];

	for sampleRowIdx in range(stdLHSSamples.shape[0]):
		retSampleInputs[sampleRowIdx, :] = paraRangesArray[:, 1] + \
										np.multiply(paraRangeLen, retSampleInputs[sampleRowIdx, :]);
	return retSampleInputs;