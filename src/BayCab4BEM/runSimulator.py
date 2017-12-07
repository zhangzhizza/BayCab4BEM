"""
Run simulator with random calibration parameters.

Author: Zhiang Zhang

First Created: Sept 1st, 2017
Last Updated: Sept 1st, 2017
"""
from pyDOE import lhs;
from multiprocessing import Lock

import os
import shutil
import numpy as np
import threading

from BayCab4BEM.processConfigFile import processConfigFile;
from BayCab4BEM.dataDenormalize import getNatValuesFromMinMaxNorm

class RunSimulatorWithRandomCaliPara(object):

	def __init__(self, configFilePath, simulationWorkerObject, baseInputFilePath, 
					simulatorExeInfo, outputPath, logger):
		configFileContent = processConfigFile(configFilePath);
		self._calibParaConfig = configFileContent[0]; # The config info for calibration parameters
		self._outputConfig = configFileContent[1]; # The config info for target simulation outputs
		self._paraNum = len(self._calibParaConfig) # The number of calibration parameter
		self._simulationWorkerObject = simulationWorkerObject;
		self._baseInputFilePath = baseInputFilePath;
		self._simulatorExeInfo = simulatorExeInfo;
		self._outputPath = outputPath;
		self._logger = logger;

	def getHeaders(self):
		"""
		This function is for output display. It gets the headers of all columnes. 
		"""
		
		etaHeaders = [etaConfigDict['name'] for etaConfigDict in self._outputConfig];
		tHeaders = [tConfigDict['name'] for tConfigDict in self._calibParaConfig];

		return (etaHeaders, tHeaders);

	def getRunResults(self, runNumber, maxRunInParallel, raw_output_process_func, 
						deleteWorkingPathAfterRun = False):
		"""
		The method run simulator with random calibration parameters in multi-threading way, and
		return the simulation outputs. 

		Args:
			runNumber: int
				The total number of simulations to be run
			maxRunInParallel: int
				The maximum allowed number parallel simulations. 

		Ret: list
			A python 2D list with col 0 the nat calibration parameter inputs (a np.ndarray), 
			col 1 the outputs (a np.ndarray with each row corresponds to one timestep). 
		"""
		# Random values for the calibration parameters, standerlized
		stdCaliPara = self._getLHSSampling(runNumber, self._paraNum);
		# Random values for the calibration parameters, non-standerlized
		paraRanges = [thisDict['range'] for thisDict in self._calibParaConfig]; 
		natCaliPara = getNatValuesFromMinMaxNorm(stdCaliPara, paraRanges);
		# Target calibration parameter info, where each row corresponds to one parameter, 
		# the contents of each row describe how to locate the parameter.
		targetParaInfo = [thisDict['keys'] for thisDict in self._calibParaConfig]; # 2-D List
		# Target output info, a 2-D list, where each row corresponds to one output type, 
		# the contents of each row describe how to locate the output.
		targetOutputInfo = [thisDict['keys'] for thisDict in self._outputConfig]; # 2-D List
		# Set up shared result container, lock and others for multithreading
		globalLock = Lock();
		globalResList = [];
		totalRunsBeDone = runNumber;
		threads = [];
		# Set up the simulator working dir
		workingDir = self._outputPath# os.getcwd();
		simulatorWorkingDir = workingDir + os.sep + 'simulatorRuns';
		if os.path.isdir(simulatorWorkingDir):
			shutil.rmtree(simulatorWorkingDir);
		os.makedirs(simulatorWorkingDir);

		# Start run simulations
		jobCount = 0;
		while totalRunsBeDone > 0 or len(threads) > 0: 
			# If there are still jobs to be done, or some jobs have not been finished, 
			# enter this loop to hold the program
			if len(threads) < maxRunInParallel and totalRunsBeDone > 0:
				# If there are free threads available and there are still some jobs need to be done,
				# enter this branch
				natModifyValues = natCaliPara[jobCount, :];
				stdModifyValues = stdCaliPara[jobCount, :];
				thisWorker = self._simulationWorkerObject();
				worker_work = lambda: thisWorker.updateWithThisInstanceOutput(self._baseInputFilePath,
							 targetParaInfo, natModifyValues, targetOutputInfo, globalResList, globalLock,
							 stdModifyValues, jobCount, simulatorWorkingDir, self._simulatorExeInfo,
							 raw_output_process_func);
				thread = threading.Thread(target = (worker_work), name = 'Simulation_job_%d'%jobCount);
				thread.start();
				self._logger.info('Simulation job %d started.', jobCount);
				totalRunsBeDone -= 1;
				threads.append(thread);
				jobCount += 1;
			for thread in threads:
				if not thread.isAlive():
					thisThreadJobNum = int(thread.getName().split('_')[-1]);
					self._logger.info('Simulation job %d finished.'%thisThreadJobNum);
					threads.remove(thread);
		# Clear the tmp working path
		if deleteWorkingPathAfterRun:
			shutil.rmtree(simulatorWorkingDir);
		# Return results
		return globalResList;

	def _getLHSSampling(self, sampleNum, dimension):
		"""
		The method do the LHS sampling.

		Args:
			sampleNum: int
				The sample number.
			dimension: int
				The sample dimension.

		Ret: numpy.ndarray
			An array where each row is one sample, each col is one feature. All values are in the
			range of 0~1.
		"""
		return lhs(dimension, samples=sampleNum, criterion='maximin');



class SimulatorRunWorker(object):
	"""
	The class is responsible for running one instance of the simulator with  with a new set of values for the
	calibration parameters. 
	"""

	def updateWithThisInstanceOutput(self, baseInputFilePath, targetParaInfo, natModifyValues, 
									targetOutputInfo, globalDict, globalLock, stdModifyValues,
									jobID, workingDir, simulatorExeInfo):
		"""
		The method create a new simulation input file, run the simulation, extract relavent outputs 
		from the raw output files, and update the global results container globalDict with the outputs. 
		This method varies for different simulators. 

		Args:
		    baseInputFilePath: str
				The path to the base simulator input file.
			targetParaInfo: list
				A 2-D list, where each row corresponds to one parameter, the contents of each row
				describe how to locate the parameter. This may vary for different simulation 
				programs. 
			natModifyValues: 1-D np.ndarray
				The new values to the calibration parameters in their native range.
			targetOutputInfo: list
				A 2-D list, where each row corresponds to one output type, the contents of each row
				describe how to locate the output. This may vary for different simulatio programs. 
			globalDict: dict
				The shared result container, with key the std calibration parameter inputs 
				(argument stdCaliPara), value the outputs (a np.ndarray with each row corresponds 
				to one timestep). The ultimate goal of this method is to add simulation outputs 
				from this run to this globalDict. 
			globalLock: multiprocessing.Lock
				The shared lock for all threads.
			jobID: int
				An ID, mainly to avoid name conflicts. 
			workingDir: str
				The simulator base working dir.
			simulatorExeInfo: list
				A list of string containing the required info to run the simulator. 

		Ret: None
		"""
		raise NotImplementedError;
    
   

