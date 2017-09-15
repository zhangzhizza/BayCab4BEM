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

class Preprocessor(object):

	def __init__(self, logger):
		self._logger = logger;

	def getDataFromSimulation(self, xf_filePath, y_filePath, caliParaConfigPath, simulatorName, 
							baseInputFilePath, runNumber, maxRunInParallel, cmbYMethodNArgs, 
							simulatorExeInfo, ydim):
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
			eta = cmbYMethodNArgs[0](eta, *cmbYMethodNArgs[1:]);
		##########################################
		############# Normalize Data #############
		##########################################
		# Min Max normalize xf, y, eta
		xf_norm = self._getMinMaxNormalized(xf);
		eta_norm = self._getMinMaxNormalized(eta);
		y_norm = self._getMinMaxNormalized(y); 
		##########################################
		#### Construct Data to Required Format ###
		##########################################
		# Construct xc
		xc_norm = np.tile(xf_norm, (runNumber, 1));
		# Construct D_sim
		d_sim = np.append(eta_norm, xc_norm, axis = 1);
		d_sim = np.append(d_sim, t, axis = 1);	
		# Construct D_field
		d_field = np.append(y_norm, xf_norm, axis = 1);
		##########################################
		######### Down Sample the Data ###########
		##########################################
		downSampler_dSim = DownSampler(d_sim, bins = 50, dirichlet_prior = 0.5);
		(d_sim_down, d_sim_sp_hist) = downSampler_dSim.sample(stSampleSize = 50, increRatio = 1.1, qualityThres = 0.95);
		downSampler_dField = DownSampler(d_field, bins = 50, dirichlet_prior = 0.5);
		(d_field_down, d_field_sp_hist) = downSampler_dField.sample(stSampleSize = 50, increRatio = 1.1, qualityThres = 0.95);
		##########################################
		######### MCMC Data Preparation ##########
		##########################################
		# Set the input data
		z = np.append(d_field_down[:, 0:ydim], d_sim_down[:, 0:ydim]);
		# Combine multi output y together into single dimension
		if len(cmbYMethodNArgs) > 0:
			z = cmbYMtdMapping[cmbYMethodNArgs[0]](z, np.array(cmbYMethodNArgs[1:]));
		if len(z.shape) > 1:
			z = np.reshape(z, (-2,))
		# Standardize the z
		n = d_field_down.shape[0]; # Number of measured samples
		m = d_sim_down.shape[0]; # Number of simulation samples
		self._logger.debug('n = %d, m = %d', n, m);
		self._logger.debug('z shape before standardization %s', z.shape);
		(z_y_stand, z_eta_stand) = self._getStandardizedByEta(z[0:n], z[n:]);
		z = np.append(z_y_stand, z_eta_stand);
		self._logger.debug('z shape after standardization %s', z.shape);
		# Extract xf xc t 
		xf = d_field_down[:, 1:];
		xc = d_sim_down[:, 1:1 + xf.shape[1]];
		t = d_sim_down[:, 1 + xf.shape[1]:];
		# Return
		return (z, xf, xc, t);

	def getDataFromFile(self, fieldDataFile, simDataFile, cmbYMethodNArgs, ydim):
		"""
		Most of the following code is taken from Adrian Chong
		"""
		self._logger.info('Reading dataset from files...')
		# Read file
		D_COMP = np.genfromtxt(simDataFile, delimiter = ',')
		D_FIELD = np.genfromtxt(fieldDataFile, delimiter = ',')
		# Organize data
		self._logger.info('Preparing data...')
		y = D_FIELD[:,0:ydim]
		xf = D_FIELD[:,ydim:]
		(n,p) = xf.shape
		eta = D_COMP[:,0:ydim]
		xc = D_COMP[:,ydim:(ydim+p)]
		tc = D_COMP[:,(p+ydim):]
		(m,q) = tc.shape
		x = np.concatenate((xf,xc), axis=0)
		self._logger.debug('Data shape before norm of y xf eta xc tc: %s %s %s %s %s'%(y.shape, 
							xf.shape, eta.shape, xc.shape, tc.shape));
		# Min max normalization for x eta y
		x = self._getMinMaxNormalized(x);
		eta = self._getMinMaxNormalized(eta);
		y = self._getMinMaxNormalized(y); 
		self._logger.debug('Data shape after norm of y eta x: %s %s %s'%(y.shape, eta.shape, x.shape));
		z = np.concatenate((y,eta), axis=0);
		self._logger.debug('Data shape before dim reduction of z: %s',z.shape);
		# Combine multi output y together into single dimension
		if len(cmbYMethodNArgs) > 0:
			z = cmbYMtdMapping[cmbYMethodNArgs[0]](z, *cmbYMethodNArgs[1:]);
		# Make z to be one-dim
		if len(z.shape) > 1:
			z = np.reshape(z, (-2,))
		# Standardize the z
		self._logger.debug('z shape before standardization %s', z.shape);
		(z_y_stand, z_eta_stand) = self._getStandardizedByEta(z[0:n], z[n:]);
		z = np.append(z_y_stand, z_eta_stand);
		self._logger.debug('z shape after standardization %s', z.shape);
		# Extract xf xc tc
		xf = x[0:n,:]
		xc = x[n:,:]
		tc = (tc - tc.min(axis = 0)) / tc.ptp(axis = 0); # Min max norm
		self._logger.debug('Data shape before dim reduction of z xf xc tc: %s %s %s %s', z.shape,
							xf.shape, xc.shape, tc.shape);
		return (z, xf, xc, tc)

	def _getMinMaxNormalized(self, x):

		x_norm = (x - x.min(axis = 0)) / x.ptp(axis = 0);
		return x_norm;
	
	def _getStandardizedByEta(self, y, eta):
		eta_mean = eta.mean(axis = 0);
		eta_std = eta.std(axis = 0);
		eta_stand = (eta - eta_mean) / eta_std; 
		y_stand = (y - eta_mean) / eta_std;
		return (y_stand, eta_stand);



