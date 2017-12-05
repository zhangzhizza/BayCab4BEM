"""
The main class to combine all sub-tasks for Bayesian calibration for BEM. 

Author: Zhiang Zhang

First Created: Sept 6th, 2017
Last Updated: Sept 6th, 2017
"""
from BayCab4BEM.runSimulator import RunSimulatorWithRandomCaliPara;
from BayCab4BEM.simulatorChoices import simulatorObjMapping
from BayCab4BEM.cmbYChoices import cmbYMtdMapping
from BayCab4BEM.downSampler import DownSampler

import pandas as pd
import numpy as np
import pickle as pk
import os

class Preprocessor(object):

	def __init__(self, logger):
		self._logger = logger;

	def getDataFromSimulation(self, xf_filePath, y_filePath, caliParaConfigPath, simulatorName, 
							baseInputFilePath, runNumber, maxRunInParallel, cmbYMethodNArgs, 
							simulatorExeInfo, ydim, is_debug, outputPath, raw_output_process_func,
							downSampleBin = 25, downSampleThres = 0.9):
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
		self._logger.info('Reading xf and yf from files...')
		xf = pd.read_csv(xf_filePath, header = 0, index_col = 0) # np.ndarray, 2D
		y = pd.read_csv(y_filePath, header = 0, index_col = 0) # np.ndarray, 2D
		# Delete cols with na, this may happen when using Excel like software to edit the csv file
		y.dropna(axis = 1, how = 'any', inplace = True)
		xf.dropna(axis = 1, how = 'any', inplace = True)
		# Delete rows with na, this may happen when using Excel like software to edit the csv file
		y.dropna(axis = 0, how = 'any', inplace = True)
		xf.dropna(axis = 0, how = 'any', inplace = True)
		# Get xf and y headers
		y_header = list(y.columns.values);
		xf_header = list(xf.columns.values);
		# Get np.ndarry from the pandas dataframe
		y = y.values;
		xf = xf.values;
		##########################################
		########### Run Simulations ##############
		##########################################
		# Run simulations
		self._logger.info('Start to run %d %s simulations...', runNumber, simulatorName);
		simulatorObj = simulatorObjMapping[simulatorName];
		runSimulatorObj = RunSimulatorWithRandomCaliPara(caliParaConfigPath, simulatorObj, 
							baseInputFilePath, simulatorExeInfo, outputPath, self._logger);
		simOrgResults = runSimulatorObj.getRunResults(runNumber, maxRunInParallel, 
													  raw_output_process_func);
		etaHeader, tHeader = runSimulatorObj.getHeaders();
		if is_debug:
			self._logger.info('Dumping the simOrgResults to file for debugging...');
			with open(outputPath + os.sep + 'DEBUG_simOrgResults.pkl', 'wb') as f:
				pk.dump(simOrgResults, f);
		##########################################
		#######Process Simulation Results ########
		##########################################
		# Construct t and eta
		self._logger.info('Constructing t and eta from simulation outputs...')
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
		
		##########################################
		#### Construct Data to Required Format ###
		##########################################
		self._logger.info('Constructing xc, D_sim and D_field...')
		# Make the headers
		D_sim_header = ','.join(etaHeader + xf_header + tHeader);
		D_field_header = ','.join(y_header + xf_header);
		# Construct xc
		actualSucceedRunNum = len(simOrgResults);
		xc = np.tile(xf, (actualSucceedRunNum, 1));
		# Construct D_sim
		d_sim = np.append(eta, xc, axis = 1);
		d_sim = np.append(d_sim, t, axis = 1);	
		# Construct D_field
		d_field = np.append(y, xf, axis = 1);
		if is_debug:
			self._logger.info('Saving original D_sim and D_field to files...')
			np.savetxt(outputPath + os.sep + 'DEBUG_D_sim_org.csv', d_sim, delimiter=",", header = D_sim_header);
			np.savetxt(outputPath + os.sep + 'DEBUG_D_field_org.csv', d_field, delimiter=',', header = D_field_header);
		##########################################
		######### Down Sample the Data ###########
		##########################################
		self._logger.info('Downsampling the data...')
		downSampler_dSim = DownSampler(d_sim, bins = downSampleBin, dirichlet_prior = 0.5);
		(d_sim_down, d_sim_sp_hist) = downSampler_dSim.sample(stSampleSize = 50, increRatio = 1.1, qualityThres = downSampleThres);
		downSampler_dField = DownSampler(d_field, bins = downSampleBin, dirichlet_prior = 0.5);
		(d_field_down, d_field_sp_hist) = downSampler_dField.sample(stSampleSize = 50, increRatio = 1.1, qualityThres = downSampleThres);
		if is_debug:
			self._logger.info('Saving downsampled D_sim and D_field to files...')
			np.savetxt(outputPath + os.sep + 'DEBUG_D_sim_down.csv', d_sim_down, delimiter=",", header = D_sim_header);
			np.savetxt(outputPath + os.sep + 'DEBUG_D_field_down.csv', d_field_down, delimiter=',', header = D_field_header);

		##########################################
		######### MCMC Data Preparation ##########
		##########################################
		self._logger.info('Preparing data for MCMC sampling...')
		return self._prepareMCMCIn(d_sim_down, d_field_down, cmbYMethodNArgs, ydim);

	def _prepareMCMCIn(self, D_COMP, D_FIELD, cmbYMethodNArgs, ydim):
		"""
		Prepare data for MCMC.

		Args:

		Ret:

		"""
		# Get info when to comb y
		cmbYBeforeZStd = None;
		if len(cmbYMethodNArgs) > 0:
			if cmbYMethodNArgs[-1] == 'before':
				cmbYBeforeZStd = True;
			elif cmbYMethodNArgs[-1] == 'after':
				cmbYBeforeZStd = False;
		# Extract y and xf
		y = D_FIELD[:,0:ydim]
		xf = D_FIELD[:,ydim:]
		(n,p) = xf.shape
		# Extract eta, xc, and tc
		eta = D_COMP[:,0:ydim]
		xc = D_COMP[:,ydim:(ydim+p)]
		tc = D_COMP[:,(p+ydim):]
		x = np.concatenate((xf,xc), axis=0)
		(m,q) = tc.shape
		# Mix max normalization x eta y and tc
		self._logger.debug('Data shape before norm of y xf eta xc tc: %s %s %s %s %s'%(y.shape, 
							xf.shape, eta.shape, xc.shape, tc.shape));
		x = self._getMinMaxNormalized(x);
		tc = self._getMinMaxNormalized(tc);
		if ydim > 1:
			eta = self._getMinMaxNormalized(eta);
			y = self._getMinMaxNormalized(y); 
			self._logger.debug('Data shape after norm of y eta x: %s %s %s'%(y.shape, eta.shape, x.shape));
		# Reduce dimension of z to one, if not one
		z = np.concatenate((y,eta), axis=0);
		if cmbYBeforeZStd == True:
			self._logger.debug('Data shape before dim reduction before std of z: %s',z.shape);
			z = cmbYMtdMapping[cmbYMethodNArgs[0]](z, *cmbYMethodNArgs[1:-1]);
			if len(z.shape) > 1:
				z = np.reshape(z, (-2,)) # Make z to be one-dim array
		# Standardize the z
		self._logger.debug('z shape before standardization %s', z.shape);
		(z_y_stand, z_eta_stand) = self._getStandardizedByEta(z[0:n], z[n:]);
		z = np.append(z_y_stand, z_eta_stand, axis = 0);
		self._logger.debug('z shape after standardization %s', z.shape);
		# Reduce dimension of z to one, if not one
		if cmbYBeforeZStd == False:
			self._logger.debug('Data shape before dim reduction after std of z: %s',z.shape);
			z = cmbYMtdMapping[cmbYMethodNArgs[0]](z, *cmbYMethodNArgs[1:-1]);
			if len(z.shape) > 1:
				z = np.reshape(z, (-2,)) # Make z to be one-dim array
		self._logger.debug('Data shape after dim reduction after std of z: %s',z.shape);
		# Extract xf and xc
		xf = x[0:n,:]
		xc = x[n:,:]
		return (z, xf, xc, tc)

	def getDataFromFile(self, fieldDataFile, simDataFile, cmbYMethodNArgs, ydim):
		"""
		Most of the following code is taken from Adrian Chong
		"""
		self._logger.info('Reading dataset from files %s and %s...', simDataFile, fieldDataFile);
		# Read file
		D_COMP = np.genfromtxt(simDataFile, delimiter = ',', skip_header = 1)
		D_FIELD = np.genfromtxt(fieldDataFile, delimiter = ',', skip_header = 1)

		self._logger.info('Preparing data...')
		self._logger.info('Y dimension reduction method is %s.', cmbYMethodNArgs);
		return self._prepareMCMCIn(D_COMP, D_FIELD, cmbYMethodNArgs, ydim);

	def _getMinMaxNormalized(self, x):

		x_norm = (x - x.min(axis = 0)) / x.ptp(axis = 0);
		return x_norm;
	
	def _getStandardizedByEta(self, y, eta):
		eta_mean = eta.mean(axis = 0);
		eta_std = eta.std(axis = 0);
		eta_stand = (eta - eta_mean) / eta_std; 
		y_stand = (y - eta_mean) / eta_std;
		return (y_stand, eta_stand);



