"""
Run MCMC using PyMC3, some code snippets borrowed from Adrian Chong. 

Author: Zhiang Zhang

First Created: Sept 5th, 2017
Last Updated: Sept 11th, 2017
"""

import pystan as ps
import numpy as np
import pickle as pk
import os

class MCMC4Posterior_pystan(object):

	def __init__(self, z, xf, xc, t, logger):
		"""
		Args:
			z: np.ndarray
				1-D, n + m row
			xf: np.ndarray
				2-D, n * p
			xc: np.ndarray
				2-D, m * p
			t: np.ndarray
				2-D, m * q
			thetaPriorInfo, rho_etaPriorInfo, rho_deltaPriorInfo, lambda_etaPriorInfo, 
				lambda_deltaPriorInfo, lambda_epsiPriorInfo: 1-D list
				It contains the prior distribution info, like mean, std, 
				or max and min. For this impl, it is is [pymc3 dist function, *args for the dist func]
		"""
		
		self._n = xf.shape[0]; # Number of field observations
		self._p = xf.shape[1]; # Number of field x features
		self._q = t.shape[1]; # Number of calibration parameters
		self._m = xc.shape[0] # Number of simulation observations
		self._N = self._n + self._m;
		self._z = z;
		self._xf = xf;
		self._xc = xc;
		self._t = t;
		self._logger = logger;
		self._dataMap = self._prepareStanData(self._n, self._p, self._q, self._m, self._N, 
												self._z, self._xf, self._xc, self._t);

	def _prepareStanData(self, *args):
		"""
		Prepare the python dic for stan data.

		Args:
			args: list of arguments.

		Ret: dict
			Python dict for stan data. 
		"""
		dataMap = {};
		dataMap['n'] = self._n;
		dataMap['p'] = self._p;
		dataMap['q'] = self._q;
		dataMap['m'] = self._m;
		dataMap['N'] = self._N;
		dataMap['z'] = self._z;
		dataMap['xf'] = self._xf;
		dataMap['xc'] = self._xc;
		dataMap['t'] = self._t;
		print (dataMap['m'])

		return dataMap;

	def build(self, stanInFileName = './BayCab4BEM/pystan_models/stan_in/chong.stan', 
				stanModelFileName = None, 
				dftModelName = './BayCab4BEM/pystan_models/stan_compiled/chong.stan.pkl'):
		"""
		Build the MCMC model.

		Args:
			stanInFileName: str
				The stan input file name. If stanModelFileName is not none, the dafault model
				will be first searched; if the default model is not found, a new model will be
				compiled using the stan input file. 
			stanModelFileName: str
				The compiled stan model file name, default is None. If not none, the saved model
				will be loaded from file.  
			dftModelName: str
				The default stan model name to search. 
		"""
		sm = None;
		if stanModelFileName == None:
			if os.path.isfile(dftModelName):
				self._logger.info('Loading the default stan model %s ...'%(dftModelName));
				sm = pk.load(open(dftModelName, 'rb'));
			else:
				self._logger.info('Compiling the stan model from input file %s ...'%(stanInFileName));
				sm = ps.StanModel(file = stanInFileName);
				# Write the compiled model to file
				stanModelsDirIdx = stanInFileName[0: stanInFileName.rfind(os.sep)].rfind(os.sep);
				stanModelsDir = stanInFileName[0: stanModelsDirIdx];
				compiledModelName = stanInFileName[stanInFileName.rfind(os.sep) + 1: ] + '.pkl';
				stanCompiledDumpDir = stanModelsDir + os.sep +  'stan_compiled';
				if not os.path.isdir(stanCompiledDumpDir):
					os.makedirs(stanCompiledDumpDir);
				with open(stanCompiledDumpDir + os.sep + compiledModelName, 'wb') as f:
					pk.dump(sm, f);
					self._logger.info('Dumped the stan compiled model to %s', f.name);
		else:
			self._logger.info('Loading the saved stan model %s ...'%(stanModelFileName));
			sm = pk.load(open(stanModelFileName), 'rb');

		return sm;

	def run(self, pystanModel, iterations, sampler, chains, warmup, n_jobs = -1):
		"""
		Run MCMC sampling.

		Args:
			pystanModel: pystan.StanFit
				The pystan StanFit instance.
			iterations: int
				Iterations for MCMC sampling.
			warmup: int
				Warmup iterations.
			chains: int
				Independent chains.
			n_jobs: int
				Sample in parallel.

		Return:
			pystan.StanFit4Model instance containing the fitted results. 
		"""
		self._logger.info('Start sampling...');
		self._logger.info('Sampling configurations: \n Iterations: %s\n Sampler: %s\n '
							'Chains: %s\n Warmup: %s\n n_jobs: %s', iterations, sampler, 
							chains, warmup, n_jobs);
		fit = pystanModel.sampling(data = self._dataMap, chains = chains, iter = iterations, 
									algorithm = sampler, warmup = warmup, n_jobs = n_jobs);

		return fit;

