"""
Run MCMC using PyMC3, some code snippets borrowed from Adrian Chong. 

Author: Zhiang Zhang

First Created: Sept 5th, 2017
Last Updated: Sept 11th, 2017
"""

from pymc3 import Model, sample, forestplot, MvNormal, Metropolis

import theano.tensor as tt
import numpy as np
import pymc3 as pm

class MCMC4Posterior(object):

	def __init__(self, z, xf, xc, t, thetaPriorInfo, rho_etaPriorInfo, rho_deltaPriorInfo, lambda_etaPriorInfo, 
				lambda_deltaPriorInfo, lambda_epsiPriorInfo):
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
		self._z = z;
		self._xf = xf;
		self._xc = xc;
		self._t = t;
		self._thetaPriorInfo = thetaPriorInfo;
		self._rho_etaPriorInfo = rho_etaPriorInfo;
		self._rho_deltaPriorInfo = rho_deltaPriorInfo;
		self._lambda_etaPriorInfo = lambda_etaPriorInfo; 
		self._lambda_deltaPriorInfo = lambda_deltaPriorInfo;
		self._lambda_epsiPriorInfo = lambda_epsiPriorInfo;
		self._data_left = np.concatenate([self._xf, self._xc], axis = 0);



	def build(self, getCovMat):

		with Model() as mvNormalLikelihood:
			# Define priors
			# For theta
			theta_dist = self._thetaPriorInfo[0];
			theta = theta_dist('theta', *self._thetaPriorInfo[1:], shape=self._q);
			# For rho_eta
			rho_eta_dist = self._rho_etaPriorInfo[0];
			rho_eta = rho_eta_dist('rho_eta', *self._rho_etaPriorInfo[1:], shape = self._p + self._q);
			beta_eta = -4.0 * tt.log(rho_eta);
			# For rho_delta
			rho_delta_dist = self._rho_deltaPriorInfo[0];
			rho_delta = rho_delta_dist('rho_delta', *self._rho_deltaPriorInfo[1:], shape = self._p);
			beta_delta = -4.0 * tt.log(rho_delta);
			# For lambda_eta
			lambda_eta = self._lambda_etaPriorInfo[0]('lambda_eta', *self._lambda_etaPriorInfo[1:]);
			# For lambda delta
			lambda_delta = self._lambda_deltaPriorInfo[0]('lambda_delta', *self._lambda_deltaPriorInfo[1:]);
			# For lambda_epsi
			lambda_epsi = self._lambda_epsiPriorInfo[0]('lambda_epsi', *self._lambda_epsiPriorInfo[1:]);
			# Construct the data
			data_right = tt.concatenate([tt.tile(theta,(self._n, 1)), self._t], axis=0); 		
			data = tt.concatenate([self._data_left, data_right], axis = 1);
			# Calculate the cov function		
			sigma_eta = getCovMat(data, beta_eta, lambda_eta, self._m + self._n, self._p + self._q)
			sigma_delta = getCovMat(self._xf, beta_delta, lambda_delta, self._n, self._p)
			sigma_y = (1/lambda_epsi) * tt.eye(self._n);
			sigma_z = tt.set_subtensor(sigma_eta[0:self._n, 0:self._n], 
										sigma_eta[0:self._n, 0:self._n] + sigma_y + sigma_delta);
			
			# Add a small value to diagonal to ensure positive definite of sigma_z																																					
			#pertubation = 0.0;
			#sigma_z = sigma_z + tt.eye(self._n + self._m) * pertubation;								
			# Cholesky decomp																																
			#L = tt.slinalg.cholesky(sigma_z);
			# Model likelihood
			post_obs = MvNormal('post_obs', mu = np.zeros(self._m + self._n), cov = sigma_z, observed = self._z);

		return mvNormalLikelihood;

	def run(self, pymc3Model, draws, sampler = 'metropolis'):
		trace = None;
		if sampler.lower() == 'metropolis':
			with pymc3Model:
				trace = sample(draws, step = Metropolis(), tune = int(draws * 0.25));
		elif sampler.lower() == 'nuts':
			with pymc3Model:
				trace = sample(draws);
		return trace;

