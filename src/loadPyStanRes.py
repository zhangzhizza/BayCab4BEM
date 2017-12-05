import pickle as pk
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

def getTrace(resDir):
	model = pk.load(open(resDir + os.sep + 'model.pkl', 'rb'));
	trace = pk.load(open(resDir + os.sep + 'trace.pkl', 'rb'));
	return trace;

def showPlot(resDir):
	trace = getTrace(resDir);
	theta = np.array(trace['theta']);
	# Plot for each trace
	sns.distplot(theta[:, i], 100, False, label = r'$\theta_%d$'%i) for i in range(theta.shape[0])
	plt.xlabel('Normalized Calibration Parameter Value');
	plt.ylabel('Distribution')
	plt.show()

if __name__ == "__main__":
	showPlot(sys.argv[1])
