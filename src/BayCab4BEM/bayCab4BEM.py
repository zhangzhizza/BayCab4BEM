"""
The main class to combine all sub-tasks for Bayesian calibration for BEM. 

Author: Zhiang Zhang

First Created: Sept 6th, 2017
Last Updated: Sept 6th, 2017
"""
from BayCab4BEM.mcmc_pymc3 import MCMC4Posterior_pymc3
from BayCab4BEM.mcmc_pystan import MCMC4Posterior_pystan



import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt

class BC4BEM(object):

	def __init__(self, logger):
		self._logger = logger;
		self._MCMC_PACKAGE = {'pymc3': self._getMCMCModel_pymc3,
							  'pystan': self._getMCMCModel_pystan};

	def runWithSimulation(self, xf_filePath, y_filePath, caliParaConfigPath, simulatorName, 
						baseInputFilePath, runNumber, maxRunInParallel, cmbYMethodNArgs, 
						simulatorExeInfo, draws, resPath, sampler, chains, mcmcPackage, *args):
		"""
		
		"""
		##########################################
		############# MCMC Sampling ##############
		##########################################
		mcmcObj = self._MCMC_PACKAGE[mcmcPackage](z, xf, xc, t);
		# Build and run
		mcmcPymcModel = mcmcObj.build(*args);
		# Run the model
		mcmcTrace = mcmcObj.run(mcmcPymcModel, draws, sampler, chains);

		##########################################
		############### Return ###################
		##########################################
		return mcmcTrace;

	def _getMCMCModel_pymc3(self, z, xf, xc, t):
		# Construct the mcmc object using the pymc3 package
		mcmcObj = MCMC4Posterior_pymc3(z, xf, xc, t, thetaPriorInfo, rho_etaPriorInfo, rho_deltaPriorInfo, 
								lambda_etaPriorInfo, lambda_deltaPriorInfo, lambda_epsiPriorInfo, self._logger);
		return mcmcObj;

	def _getMCMCModel_pystan(self, z, xf, xc, t):

		# Construct the mcmc object using the pystan
		mcmcObj = MCMC4Posterior_pystan(z, xf, xc, t, self._logger);
		return mcmcObj;

	def runWithData(self, fieldDataFile, simDataFile, draws, resPath, sampler, 
					mcmcPackage, chains, *args):
		"""
		Most of the following code is taken from Adrian Chong
		"""
		
		mcmcObj = self._MCMC_PACKAGE[mcmcPackage](z, xf, xc, tc);
		# Build and run
		mcmcPymcModel = mcmcObj.build(*args);
		# Run the model
		self._logger.info('MCMC sampling starts to run...')
		mcmcTrace = mcmcObj.run(mcmcPymcModel, draws, sampler, chains);
		self._logger.info('MCMC sampling stopped.')
		# Save the results
		self._logger.info('Saving the sampling results...')
		# Save trace plot
		# axTrplt = pm.traceplot(mcmcTrace);
		# fig, axsShow = plt.subplots(*axTrplt.shape);
		# pm.traceplot(mcmcTrace, ax = axsShow);
		# fig.savefig(resPath + '/' + 'mcmcout_traceplot.png')
		# Save summary text
		pm.summary(mcmcTrace, to_file = resPath + '/' + 'mcmcout_summary.csv');


		return mcmcTrace;	


