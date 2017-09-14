"""
The methods to combine multi-output into single output

Author: Zhiang Zhang

First Created: Sept 6th, 2017
Last Updated: Sept 6th, 2017
"""
import numpy as np

from sklearn.decomposition import PCA

def linearCmbY(y_org, *fractionList):
	"""
	Linearly combine each col of the y_org by factions. 

	Args:
		y_org: np.ndarray
			2-D, each row corresponds to one time step, each col is one feature. 
		fractionList: np.ndarray
			1-D, the length is the same as cols of y_org. Each item corresponds to the fraction
			of the col in y_org in the linear combination. 
	Return: np.ndarray
		1-D. The linearly combined y. 
	"""
	fractionList = np.array(fractionList);
	singleOutY = np.sum(y_org * fractionList, axis = 1);

	return singleOutY;

def pcaCmbY(y_org):
	"""
	Reduce the dimension of y_org by PCA by taking the first principle components. 

	Args:
		y_org: np.ndarray
			2-D, each row corresponds to one time step, each col is one feature. 
	Return: np.ndarray
		1-D. The linearly combined y. 
	"""
	pcaObj = PCA(n_components = 1)
	pcaObj.fit(y_org)
	yftrans = pcaObj.transform(y_org)

	return yftrans;