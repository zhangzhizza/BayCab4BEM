"""
Run EnergyPlus with random calibration parameters.

Author: Zhiang Zhang

First Created: Sept 1st, 2017
Last Updated: Sept 5th, 2017
"""
from multiprocessing import Lock

import threading
import itertools
import numpy as np
import theano.tensor as tt
import time

class CovFunctionMultiThread(object):

	def __init__(self, etaKernelClass, deltaKernelFunction):
		self._EtaKernelClass = etaKernelClass;
		self._deltaKernelFunction = deltaKernelFunction;

	def getEtaCovMat(self, data, beta_eta, lambda_eta, xcols, threadingNum):
		"""
		The function returns the covariance matrix using the etaKernelFunction.

		Args:
			data: np.ndarray
				2-D array, the combined data, M rows, N cols.
			beta_eta: np.ndarray
				1-D array, the beta values, N cols.
			lambda_eta: float
				The lambda value.
			xcols: int
				The number of cols that are for x values. 

		Ret: np.ndarray
			The cov function mat, M * M.

		"""
		
		M = data.shape[0];
		# Set up shared result container, lock and others for multithreading
		globalCovMat = np.zeros((M, M));
		globalLock = Lock();
		totalRunsBeDone = M * M;
		segSize = int(totalRunsBeDone/threadingNum);
		threads = [];
		rowColPairs = [pair for pair in itertools.product(range(M), repeat = 2)];
		rowColParisSeg = [];
		for seg_i in range(0, totalRunsBeDone, segSize):
			thisSeg = [];
			if seg_i + segSize >= totalRunsBeDone:
				thisSeg.extend(rowColPairs[seg_i:]);
			else:
				thisSeg.extend(rowColPairs[seg_i: seg_i + segSize]);
			rowColParisSeg.append(thisSeg);
		# Start calculating cov mat
		timest = time.time();
		idnum = 0;
		for thisSeg in rowColParisSeg:
			threadWrapper = MultiThreadingCovMatWrapper();
			worker_work = lambda: threadWrapper.multiThreadingCovMatWrapper(
								idnum, globalLock, globalCovMat, thisSeg, self._EtaKernelClass(),
								data.copy(), beta_eta[0: xcols].copy(), beta_eta[xcols:].copy(), lambda_eta);
			thread = threading.Thread(target = (worker_work));
			thread.start();
			threads.append(thread);
			idnum += 1;
		for thread in threads:
			thread.join();
		print (time.time()- timest)
		return globalCovMat;


class MultiThreadingCovMatWrapper():

	def multiThreadingCovMatWrapper(self, idnum, globalLock, globalResultContainer, rowColPairs, kernelObj, data, *kernelArgs):
		"""
		This is a wrapper function for multi threading cov calculation.

		Args:
			globalLock: multiprocessing.Lock
				The lock.
			globalResultContainer: np.ndarray
				The cov mat to return.
			rowColPairs: list
				List of tuple, each tuple is the row and col that this function needs to work with. 
			kernelObj: gpKernel object.
			data: np.ndarray
				2-D, M*N data. 
			kernelArgs: variable length arguments for the kernelFunc.

		Ret: None.
		"""
		print ('thread %d start'%idnum);
		localRestList = [];
		for ijPair in rowColPairs:
			i, j = ijPair;
			kernelRet = kernelObj.getValue(data[i, :], data[j, :], *kernelArgs);
			localRestList.append([ijPair, kernelRet]);
		print ('thread %d finish'%idnum);
		globalLock.acquire() # will block if lock is already held
		for resPair in localRestList:
			globalResultContainer[resPair[0][0], resPair[0][1]] = resPair[1];
		globalLock.release()


def getCovMat(data, beta, lambdaVal, data_row, data_col):
	"""
	The function returns the covariance matrix using the etaKernelFunction.

	Args:
		data: np.ndarray
			2-D array, the combined data, M rows, N cols.
		beta: np.ndarray
			1-D array, the beta values, N cols.
		lambdaVal: float
			The lambda value.

	Ret: np.ndarray
		The cov function mat, M * M.

	"""
	timest = time.time()
	# Change the data from [[x11, x12, x13,...,x1m],...[xn1, xn2, xn3,...,xnm]] to 
	# [[[x11, x12, x13,...,x1m], [x11, x12, x13,...,x1m],...repeat n times],
	#   ....
	#  [[xn1, xn2, xn3,...,xnm], [xn1, xn2, xn3,...,xnm],...repeat n times]]
	# It changes the data from 2-D array to 3-D array by repeating each row n times. 
	# Final dimension is n * n * m
	dataRowRepeat = np.repeat(data, data_row, axis = 0) # Dim: (n*n) * m
	dataRowRepeatRow3D = tt.reshape(dataRowRepeat, (data_row, data_row, data_col), ndim = 3);
	# Transpose the dataRowRepeatRow3D for the 1st and 2nd dim
	dataRowRepeatCol3D = np.transpose(dataRowRepeatRow3D, (1, 0, 2));
	# Get the difference between dataRowRepeatRow3D and dataRowRepeatCol3D, 
	# This is equivelent to get differences of all possible row pairs in the data
	dataDiffAllRowPairs = dataRowRepeatRow3D - dataRowRepeatCol3D;
	# Get the square of all elements
	dataDiffAllRowPairsSqr = dataDiffAllRowPairs ** 2.0;
	# Multuply by the beta values
	dataDiffAllRowPairsSqrMulBeta = dataDiffAllRowPairsSqr * beta
	# Negative sum up the inner array
	negSum = -tt.sum(dataDiffAllRowPairsSqrMulBeta, axis = 2)
	# Exp and lambda
	ret = tt.exp(negSum)/lambdaVal;
	print ('time', time.time() - timest)
	return ret;