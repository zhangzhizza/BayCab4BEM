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
import xml.etree.ElementTree
import numpy as np
import threading

class RunSimulatorWithRandomCaliPara(object):

	def __init__(self, configFilePath, simulationWorkerObject, baseInputFilePath, simulatorExeInfo):
		configFileContent = self._processConfigFile(configFilePath);
		self._calibParaConfig = configFileContent[0]; # The config info for calibration parameters
		self._outputConfig = configFileContent[1]; # The config info for target simulation outputs
		self._paraNum = len(self._calibParaConfig) # The number of calibration parameter
		self._simulationWorkerObject = simulationWorkerObject;
		self._baseInputFilePath = baseInputFilePath;
		self._simulatorExeInfo = simulatorExeInfo;
	

	def getRunResults(self, runNumber, maxRunInParallel, deleteWorkingPathAfterRun = False):
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
		natCaliPara = self._getLHSSampledRandnVars(stdCaliPara, paraRanges);
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
		currentDir = os.getcwd();
		simulatorWorkingDir = currentDir + '/.tmp';
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
							 stdModifyValues, jobCount, simulatorWorkingDir, self._simulatorExeInfo);
				thread = threading.Thread(target = (worker_work));
				thread.start();
				totalRunsBeDone -= 1;
				threads.append(thread);
				jobCount += 1;	
			for thread in threads:
				if not thread.isAlive():
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

	def _getLHSSampledRandnVars(self, stdLHSSamples, paraRanges):
		"""
		The method return the LHS sampled random varaibles for the calibration parameters in their
		native value range. The conversion method is reverse min-max standarilization, i.e
			varInNativeRange = varMin + (varMax - varMin) * stdVar

		Args:
			stdLHSSamples: numpy.ndarray
				The LHS samples variables in the standalized way (0-1). Each row is a sample, each
				col is a feature.
			paraRanges: list
				A 2-D list, where each row corresponds to one parameter. For each row, index 0 is 
				the maximum limit, index 1 is the min limit. 
		Ret: numpy.ndarray
			An array where each row one sample, each col is one feature containing the LHS sampled
			random variables in their native range. 
		"""
		retSampleInputs = np.copy(stdLHSSamples);
		paraRangesArray = np.array(paraRanges);
		paraRangeLen = paraRangesArray[:, 0] - paraRangesArray[:, 1];

		for sampleRowIdx in range(stdLHSSamples.shape[0]):
			retSampleInputs[sampleRowIdx, :] = paraRangesArray[:, 1] + \
											np.multiply(paraRangeLen, retSampleInputs[sampleRowIdx, :]);
		return retSampleInputs;


	def _processConfigFile(self, configFilePath):
		"""
		The method process the configuration file, extract the raw contents into list. 

		Args:
			configFilePath: str
				The configuration file path.

		Ret: (python list, python list)
			The first is a python list of dicts related to each calibration parameter info, 
			each dict has the following structure:
				{name: parameter name,
				 keys: [[key content1, key content2, ...], [key content1, key content2, ...], ...],
				 range: [max, min],
				 description: description string}
			The second is a python list of dicts related to each output info,
			each dict has the following structure:
				{name: output name,
				 keys: [key content1, key content2, ...],
				 description: description string}

		"""
		ret0 = [];
		ret1 = [];
		e =  xml.etree.ElementTree.parse(configFilePath).getroot();
		for entry in e:
			if entry.tag == 'calibration_parameter':
				thisDict = {};
				thisDict['keys'] = [];
				for item in entry:
					if item.tag == ('name' or 'description'):
						thisDict[item.tag] = item.text;
					elif item.tag == 'keys':
						thisKeys = [];
						keyNum = int(item.attrib['number']);
						for keyIndex in range(keyNum):
							thisKeys.append(item[keyIndex].text);
						thisDict['keys'].append(thisKeys);
					elif item.tag == 'range':
						thisRange = [None, None];
						for rangeIndex in range(2):
							if item[rangeIndex].tag == 'max':
								thisRange[0] = float(item[rangeIndex].text);
							elif item[rangeIndex].tag == 'min':
								thisRange[1] = float(item[rangeIndex].text);
						thisDict['range'] = thisRange;
				ret0.append(thisDict);
			if entry.tag == 'output':
				thisDict = {};
				for item in entry:
					if item.tag == 'name' or 'description':
						thisDict[item.tag] = item.text;
					if item.tag == 'keys':
						thisKeys = [];
						keyNum = int(item.attrib['number']);
						for keyIndex in range(keyNum):
							thisKeys.append(item[keyIndex].text);
						thisDict[item.tag] = thisKeys;
				ret1.append(thisDict);

		return (ret0, ret1);

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
    
   

