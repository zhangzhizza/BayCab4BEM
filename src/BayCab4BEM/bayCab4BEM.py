"""
The main class to combine all sub-tasks for Bayesian calibration for BEM. 

Author: Zhiang Zhang

First Created: Sept 6th, 2017
Last Updated: Sept 6th, 2017
"""
from BayCab4BEM.runSimulator import RunSimulatorWithRandomCaliPara;
from BayCab4BEM.simulatorChoices import simulatorObjMapping
from BayCab4BEM.covFuncChoices import covFuncMapping
from BayCab4BEM.cmbYChoices import cmbYMtdMapping
from BayCab4BEM.downSampler import DownSampler
from BayCab4BEM.mcmc import MCMC4Posterior

from pymc3.distributions.continuous import Uniform
from pymc3.distributions.continuous import Beta
from pymc3.distributions.continuous import Gamma

import pandas as pd
import numpy as np

class BC4BEM(object):

	def run(self, xf_filePath, y_filePath, caliParaConfigPath, simulatorName, baseInputFilePath, 
			runNumber, maxRunInParallel, cmbYMethodNArgs, simulatorExeInfo, covFuncName):
		"""
		Args:
			xf_filePath, y_filePath: str
				The file path to the xf values and y values. The file must be csv file, with each 
				row corresponding to one timestep, each col corresponding to one feature. The first 
				row and first col of the csv file will be ignored. So the actual data starts from the
				second row and second col. 
		"""
		##########################################
		#### Read measurement data from files ####
		##########################################
		# Read xf values and y values from files
		xf = pd.read_csv(xf_filePath, header = 0, index_col = 0) # np.ndarray, 2D
		y = pd.read_csv(y_filePath, header = 0, index_col = 0) # np.ndarray, 2D
		# Delete cols with na, this may happen when using Excel like software to edit the csv file
		y.dropna(axis = 1, how = 'any', inplace = True)
		xf.dropna(axis = 1, how = 'any', inplace = True)
		# Delete rows with na, this may happen when using Excel like software to edit the csv file
		y.dropna(axis = 0, how = 'any', inplace = True)
		xf.dropna(axis = 0, how = 'any', inplace = True)
		# Get np.ndarry from the pandas dataframe
		y = y.values;
		xf = xf.values;
		# Combine multi output y together into single dimension
		if len(cmbYMethodNArgs) > 0:
			y = cmbYMethodNArgs[0](y, np.array(cmbYMethodNArgs[1:]));
		print ('y', y);
		##########################################
		########### Run Simulations ##############
		##########################################
		# Run simulations
		simulatorObj = simulatorObjMapping[simulatorName];
		runSimulatorObj = RunSimulatorWithRandomCaliPara(caliParaConfigPath, simulatorObj, 
							baseInputFilePath, simulatorExeInfo);
		simOrgResults = runSimulatorObj.getRunResults(runNumber, maxRunInParallel);
		##########################################
		#######Process Simulation Results ########
		##########################################
		# Construct t and eta
		oneRunLen = simOrgResults[0][1].shape[0];
		t = None;
		eta = None;
		for runi_res in simOrgResults:
			runi_t_org = runi_res[0];
			runi_eta_org = runi_res[1];
			runi_t_repeat = np.repeat(np.reshape(runi_t_org, (1, -1)), oneRunLen, axis = 0);
			if t is None:
				t = runi_t_repeat
			else:
				t = np.append(t, runi_t_repeat, axis = 0);
			if eta is None:
				eta = runi_eta_org;
			else:
				eta = np.append(eta, runi_eta_org, axis = 0);
		# Combine eta if eta is multi output
		if len(cmbYMethodNArgs) > 0:
			eta = cmbYMethodNArgs[0](eta, np.array(cmbYMethodNArgs[1:]));
		print ('t', t)
		print ('eta', eta);
		print ('xf', xf);
		##########################################
		############# Normalize Data #############
		##########################################
		# Standarlize xf, y, eta
		xf_norm = (xf - xf.min(axis = 0)) / xf.ptp(axis = 0); # Min max norm
		eta_mean = eta.mean();
		eta_std = eta.std();
		eta_norm = (eta - eta_mean) / eta_std; 
		y_norm = (y - eta_mean) / eta_std;
		print ('xf norm', xf_norm);
		print ('y_norm', y_norm);
		##########################################
		#### Construct Data to Required Format ###
		##########################################
		# Construct xc
		xc_norm = np.tile(xf_norm, (runNumber, 1));
		print ('xc_norm', xc_norm);
		# Construct D_sim
		d_sim = np.append(eta_norm.reshape(-1, 1), xc_norm, axis = 1);
		d_sim = np.append(d_sim, t, axis = 1);
		# Construct D_field
		d_field = np.append(y_norm.reshape(-1, 1), xf_norm, axis = 1);
		print ('d_sim', d_sim.shape);
		print ('d_field', d_field.shape);
		##########################################
		######### Down Sample the Data ###########
		##########################################
		print ('start down sample');
		downSampler_dSim = DownSampler(d_sim, bins = 50, dirichlet_prior = 0.5);
		(d_sim_down, d_sim_sp_hist) = downSampler_dSim.sample(stSampleSize = 50, increRatio = 1.1, qualityThres = 0.95);
		downSampler_dField = DownSampler(d_field, bins = 50, dirichlet_prior = 0.5);
		(d_field_down, d_field_sp_hist) = downSampler_dField.sample(stSampleSize = 50, increRatio = 1.1, qualityThres = 0.95);
		print ('d_sim_down hist', d_sim_sp_hist);
		print ('d_field_down hist', d_field_sp_hist);
		##########################################
		################# MCMC ###################
		##########################################
		# Set the input data
		z = np.append(d_field_down[:, 0], d_sim_down[:, 0]);
		xf = d_field_down[:, 1:];
		xc = d_sim_down[:, 1:1 + xf.shape[1]];
		t = d_sim_down[:, 1 + xf.shape[1]:];
		print ('z', z.shape);
		print ('xf', xf.shape);
		print ('xc', xc.shape);
		print ('t', t.shape);
		print ('d sim down', d_sim_down.shape);
		print ('d fd down', d_field_down.shape);
		mcmcObj = self._getMCMCModel(z, xf, xc, t);
		# Build and run
		mcmcPymcModel = mcmcObj.build(covFuncMapping[covFuncName]);
		# Run the model
		print ('run mcmc')
		mcmcTrace = mcmcObj.run(mcmcPymcModel, draws = 500);

		##########################################
		############### Return ###################
		##########################################
		return mcmcTrace;

	def _getMCMCModel(self, z, xf, xc, t):
		# Set the prior distributions, taken from Chong(2017) PhD thesis
		thetaPriorInfo = [Uniform, 0, 1];
		rho_etaPriorInfo = [Beta, 1, 0.5];
		rho_deltaPriorInfo = [Beta, 1, 0.4];
		lambda_etaPriorInfo = [Gamma, 10, 10];
		lambda_deltaPriorInfo = [Gamma, 10, 0.3];
		lambda_epsiPriorInfo = [Gamma, 10, 0.03];
		# Construct the mcmc object
		mcmcObj = MCMC4Posterior(z, xf, xc, t, thetaPriorInfo, rho_etaPriorInfo, rho_deltaPriorInfo, 
								lambda_etaPriorInfo, lambda_deltaPriorInfo, lambda_epsiPriorInfo);
		return mcmcObj;

	def runWithData(self, fieldDataFile, simDataFile, covFuncName):
		"""
		Most of the following code is taken from Adrian Chong
		"""
		# Read file
		D_COMP = np.genfromtxt(simDataFile, delimiter = ',')
		D_FIELD = np.genfromtxt(fieldDataFile, delimiter = ',')
		# Organize data
		y = D_FIELD[:,0]
		xf = D_FIELD[:,1:]
		(n,p) = xf.shape
		eta = D_COMP[:,0]
		xc = D_COMP[:,1:(p+1)]
		tc = D_COMP[:,(p+1):]
		(m,q) = tc.shape
		# Standarlization
		eta_mu = np.nanmean(eta)
		eta_sd = np.nanstd(eta)
		y = (y - eta_mu) / eta_sd
		eta = (eta - eta_mu) /eta_sd
		z = np.concatenate((y,eta), axis=0)
		x = np.concatenate((xf,xc), axis=0)
		x = (x - x.min(axis = 0)) / x.ptp(axis = 0); # Min max norm
		xf = x[0:n,:]
		xc = x[n:,:]
		tc = (tc - tc.min(axis = 0)) / tc.ptp(axis = 0); # Min max norm
		# MCMC
		mcmcObj = self._getMCMCModel(z, xf, xc, tc);
		# Build and run
		mcmcPymcModel = mcmcObj.build(covFuncMapping[covFuncName]);
		# Run the model
		print ('run mcmc')
		mcmcTrace = mcmcObj.run(mcmcPymcModel, draws = 500);
		return mcmcTrace;	


