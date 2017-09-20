import pickle as pk
import matplotlib.pyplot as plt

import os
import sys

def getTrace(resDir):
	model = pk.load(open(resDir + os.sep + 'model.pkl', 'rb'));
	trace = pk.load(open(resDir + os.sep + 'trace.pkl', 'rb'));
	return trace;

def showPlot(resDir):
	trace = getTrace(resDir);
	trace.plot()
	plt.show()

if __name__ == "__main__":
	showPlot(sys.argv[1])
