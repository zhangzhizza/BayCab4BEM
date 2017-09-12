"""
Run MCMC using PyMC3, some code snippets borrowed from Adrian Chong. 

Author: Zhiang Zhang

First Created: Sept 5th, 2017
Last Updated: Sept 11th, 2017
"""

from pymc3 import Model, sample, forestplot
from pymc3.distributions.multivariate import MvNormal
from pymc3.step_methods.metropolis import Metropolis

import numpy as np
import theano.tensor as tt

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
	


	def build(self, getCovMat):
		print ('p', self._p);
		print ('q', self._q);
		print ('etaLen', len(self._rho_etaPriorInfo));
		print ('deltaLen', len(self._rho_deltaPriorInfo))
		with Model() as mvNormalLikelihood:
			# Define priors
			# For theta
			#theta = [];
			#for theta_i in range(self._q):
			theta_dist = self._thetaPriorInfo[0];
			theta = theta_dist('theta', *self._thetaPriorInfo[1:], shape=self._q);
				#theta.append(ith_theta);
			#theta = tt.stack(theta);
			# ?? theta = self._thetaPriorInfo[0]('theta', *self._thetaPriorInfo[1:], shape = self._q);
			# For rho_eta
			#rho_eta = [];
			#for rho_eta_i in range(self._p + self._q):
			rho_eta_dist = self._rho_etaPriorInfo[0];
			rho_eta = rho_eta_dist('rho_eta', *self._rho_etaPriorInfo[1:], shape = self._p + self._q);
				#rho_eta.append(ith_rho_eta);
			#rho_eta = tt.stack(rho_eta);
			# ?? rho_eta = self._rho_etaPriorInfo[0]('rho_eta', *self._rho_etaPriorInfo[1:], shape = self._m + self._n);
			beta_eta = -4.0 * tt.log(rho_eta);
			# For rho_delta
			#rho_delta = [];
			#for rho_delta_i in range(self._p):
			rho_delta_dist = self._rho_deltaPriorInfo[0];
			rho_delta = rho_delta_dist('rho_delta', *self._rho_deltaPriorInfo[1:], shape = self._p);
				#rho_delta.append(ith_rho_delta);
			#rho_delta = tt.stack(rho_delta)
			# ?? rho_delta = self._rho_deltaPriorInfo[0]('rho_delta', *self._rho_deltaPriorInfo[1:], shape = self._n);
			beta_delta = -4.0 * tt.log(rho_delta);
			# For lambda_eta
			lambda_eta = self._lambda_etaPriorInfo[0]('lambda_eta', *self._lambda_etaPriorInfo[1:]);
			# For lambda delta
			lambda_delta = self._lambda_deltaPriorInfo[0]('lambda_delta', *self._lambda_deltaPriorInfo[1:]);
			# For lambda_epsi
			lambda_epsi = self._lambda_epsiPriorInfo[0]('lambda_epsi', *self._lambda_epsiPriorInfo[1:]);
			
			# Construct the data
			data_right = tt.concatenate([tt.tile(theta,(self._n, 1)), self._t], axis=0); 
			data_left = tt.concatenate([self._xf, self._xc], axis = 0);
			data = tt.concatenate([data_left, data_right], axis = 1);
			print ('data eval', data.type);
			# Calculate the cov function			
			sigma_eta =  getCovMat(data, beta_eta, lambda_eta, 
									data_row = self._m + self._n, data_col = self._p + self._q);
			sigma_theta = getCovMat(self._xf, beta_delta, lambda_delta, 
									data_row = self._n, data_col = self._p);
			sigma_y = np.identity(self._n)/lambda_epsi;
			sigma_z = tt.set_subtensor(sigma_eta[0:self._n, 0:self._n], sigma_eta[0:self._n, 0:self._n] + sigma_y + sigma_theta);
			# Add a small value to diagonal to ensure positive definite of sigma_z																																					
			pertubation = 0.0;
			sigma_z = sigma_z + tt.eye(self._n + self._m) * pertubation;								

			# Cholesky decomp																																
			L = tt.slinalg.cholesky(sigma_z);

			# Model likelihood
			posterior = MvNormal('Posterior', mu = 0, chol = L, observed = self._z);

		return mvNormalLikelihood;

	def run(self, pymc3Model, draws):
		with pymc3Model:
			trace = sample(draws, step = Metropolis(), init = 'nuts', n_init = 250, njobs = 8);

		return trace;

