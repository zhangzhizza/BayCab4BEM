"""
Down sample a large dataset.

Author: Zhiang Zhang

First Created: Sept 5th, 2017
Last Updated: Sept 5th, 2017
"""
from numpy.random import randint

import numpy as np

class DownSampler(object):
	"""
	Implement the method of section 2.4.2 of Chong(2017) PhD thesis.
	The method use KL divergence as a metric to determine what is the sample size of 
	the down-sampled data.
	"""

	def __init__(self, orgData, bins, dirichlet_prior):
		"""
		Args:
			orgData: np.ndarray
				2-D data. Each row is a sample, each col is a feature. 
			bins: int
				The number of bins to discretize the data of one feature.
			dirichlet_prior: float
				The a_j in equation 2.15 of Chong(2017) PhD thesis. This value can be 0.5, 1, 
				1/p or sqrt(n)/p where n is total sample size and p is the feature number (not sure)
		"""
		self._orgData = orgData;
		self._bins = bins;
		self._dirichlet_prior = dirichlet_prior
		self._R = orgData.shape[1]; # Num of cols
		self._rangeInfo = [];
		self._orgDataDists = [];
		for coli in range(orgData.shape[1]):
			self._rangeInfo.append((orgData[:, coli].min(), orgData[:, coli].max()));
			self._orgDataDists.append(
				self._histToProbWithDirichletPrior(
					np.histogram(orgData[:, coli], bins = self._bins)[0], dirichlet_prior));

		
	def sample(self, stSampleSize, increRatio, qualityThres):
		"""
		Do the random sampling with increasing sample size until the quality metric reaches the
		threshold.  

		Args:
			stSampleSize: int
				The starting sample size.
			increRatio: float
				Larger than 1 value, the increment ratio of the sample size.
			qualityThres: float
				Smaller than 1 value, the threshold to stop sampling. 1 is the max value. 

		Ret: (np.ndarray, np.ndarray)
			The first ret is the down-sampled data, 2-D array.
			The second ret is the quality vs sample size history, 2-D array where col 0 is sample size. 
		"""

		sampleQuality = 0;
		sampleSize = stSampleSize;
		maxSampleNum = self._orgData.shape[0];
		qualityHist = [];
		thisSample = None;
		while sampleQuality < qualityThres and sampleSize < maxSampleNum:
			sampleSize = min(sampleSize, maxSampleNum);
			thisSample = self._orgData[randint(maxSampleNum, size = sampleSize)]
			sampleQuality = self._getQualityMetric(thisSample);
			qualityHist.append([sampleSize, sampleQuality]);
			sampleSize *= increRatio;
			sampleSize = int(sampleSize)
			print ('sample quality', sampleQuality);
			print ('sampleSize', sampleSize);

		return (thisSample, np.array(qualityHist));




	def _getQualityMetric(self, sampledData):
		"""
		Return the quality of the sampled data by equation 2.12, 2.13 2.14 of the Chong(2017) PhD thesis.

		Args:
			sampledData: np.ndarray
				2-D array, with each row a sample, each col a feature. 
				The col 0 of the data has zero mean, 1 std, all other cols have 
				min 0 max 1.

		Ret: float
			The quality metric value.  
		"""
		J = 0;
		for coli in range(sampledData.shape[1]):
			sampledHist = np.histogram(sampledData[:, coli], bins = self._bins,
											 range=self._rangeInfo[coli])[0];
			sampledDist = self._histToProbWithDirichletPrior(sampledHist, self._dirichlet_prior);
			J += self._klDivergence(sampledDist, self._orgDataDists[coli]);
		J /= self._R;
		Q = np.exp(-J);
		return Q;


	def _klDivergence(self, distSample, distOrg):
		"""
    	K-L divergence.

    	Args:
    		distSample: np.ndarray
    			1-D array, the distribution of each bin of the sampled data (one feature).
    		distOrg: np.ndarray
    			1-D array, the distribution of each bin of the original data (one feature).

    	Ret: float
    		The K-L divergence value. 
    	"""

		klDiv = np.multiply((distSample - distOrg), np.log(distSample / distOrg)).sum();

		return klDiv;


	def _histToProbWithDirichletPrior(self, histIn, dirichlet_prior):
		"""
		Convert the histogram to distribution with the dirichlet prior, as shown in 
		equation 2.15 of Chong(2017) PhD thesis. 

		Args: 
			histIn: np.ndarray
				1-D array of the histogram of occurrence of each bin for one feature of the data. 
			dirichlet_prior: float
				The a_j in equation 2.15 of Chong(2017) PhD thesis.
			
		Ret: np.ndarray
			1-D array, the distribution of each bin of the data (one feature).
		"""
		
		histSum = np.sum(histIn);
		binNum = histIn.shape[0];
		probDenom = dirichlet_prior * binNum + histSum;
		binProb = [];
		for hist_i in range(binNum):
			binProb.append((dirichlet_prior + histIn[hist_i])/probDenom);

		return np.array(binProb);






