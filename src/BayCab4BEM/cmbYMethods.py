"""
The methods to combine multi-output into single output

Author: Zhiang Zhang

First Created: Sept 6th, 2017
Last Updated: Sept 6th, 2017
"""

def linearCmbY(y_org, fractionList):
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

	singleOutY = np.sum(y_org * fractionList, axis = 1);

	return singleOutY;