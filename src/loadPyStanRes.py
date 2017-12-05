import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
import sys
import csv

def getTrace(resDir):
	model = pk.load(open(resDir + os.sep + 'model.pkl', 'rb'));
	trace = pk.load(open(resDir + os.sep + 'trace.pkl', 'rb'));
	return trace;

def showPlot(resDir):
	trace = getTrace(resDir);
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
	plt.ylim(ymax=5.0);
	plt.show()

def caluculateMode(resDir):
	trace = getTrace(resDir);
	theta = np.array(trace['theta']);
	res = [];
	# Calulate the mode for each trace
	for i in range(theta.shape[1]):
		this_trace = theta[:, i];
		bins = 100;
		this_hist = np.histogram(this_trace, bins = bins, range = (0,1))
		this_mode = ((1/bins)/2.0) + (1/bins)* np.argmax(this_hist[0]);
		res.append('theta_%d mode is %0.04f.'%(i, this_mode));
	print (res)
	with open("%s/mode.csv"%resDir,'w') as resultFile:
		resultFile.write('\n'.join(res));
		resultFile.close();




if __name__ == "__main__":
	caluculateMode(sys.argv[1])
	showPlot(sys.argv[1])
