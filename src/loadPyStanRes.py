import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import sys
import csv

from Util.io import getFileDir
from BayCab4BEM.dataDenormalize import getNatValuesFromMinMaxNorm
from BayCab4BEM.processConfigFile import processConfigFile

def getTrace(traceDir, modelDir):
	model = pk.load(open(modelDir, 'rb'));
	trace = pk.load(open(traceDir, 'rb'));
	return trace;

def showPlot(traceDir, modelDir):
	trace = getTrace(traceDir, modelDir);
	theta = np.array(trace['theta']);
	# Plot for each trace
	colorlist = ['red', 'green', 'blue', 'm', 'orange', 'y', 'sienna']
	sns.set(font_scale=1.4) 
	bins = 100;
	for i in range(theta.shape[1]):
		kde_kws = {'label': r'$\theta_%d KDE$'%i,
			   'clip': (0.0, 1.0)};
		hist_kws = {'label': r'$\theta_%d Histogram$'%i}
		target = theta[:, i];
		filtered = target[(target > 0) & (target < 1.0)];
		if i < len(colorlist):
			sns.distplot(filtered, bins = bins, color = colorlist[i], kde_kws = kde_kws, hist_kws = hist_kws);
		else:
			sns.distplot(filtered, bins = bins, kde_kws = kde_kws, hist_kws = hist_kws);
	plt.xlabel('Normalized Calibration Parameter Value', fontsize = 20);
	plt.ylabel('Density', fontsize = 20)
	#plt.ylim(ymax=5.0);
	plt.show()

def caluculateMode(traceDir, modelDir, configFilePath, bins):
	trace = getTrace(traceDir, modelDir);
	theta = np.array(trace['theta']);
	res = [];
	# Calulate the mode for each trace
	mode = [];
	for i in range(theta.shape[1]):
		this_trace = theta[:, i];
		this_hist = np.histogram(this_trace, bins = bins, range = (0,1))
		this_mode = ((1/bins)/2.0) + (1/bins)* np.argmax(this_hist[0]);
		res.append('theta_%d mode is %0.04f.'%(i, this_mode));
		mode.append(this_mode);
	mode = np.array(mode).reshape((1, -1));
	# Calculate the parameter values in their native range
	configFileContent = processConfigFile(configFilePath);
	paraRange = [thisDict['range'] for thisDict in configFileContent[0]]; 
	print (paraRange)
	paraNat = getNatValuesFromMinMaxNorm(mode, paraRange);
	# Write res to string
	for j in range(theta.shape[1]):
		res[j] += ' native value is %0.04f.'%(paraNat[0, j]);
	# Write to file
	with open("%s/mode.csv"%getFileDir(traceDir, 1),'w') as resultFile:
		resultFile.write('\n'.join(res));
		resultFile.close();




if __name__ == "__main__":
	caluculateMode(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
	showPlot(sys.argv[1], sys.argv[2])
