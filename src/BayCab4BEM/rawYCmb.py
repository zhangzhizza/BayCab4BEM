"""
This file contains functions that combine multiple Ys from the raw data.

Author: Zhiang Zhang
Create Date: Nov 21st, 2017
"""
import numpy as np;

def randomCmb(fieldY, simY, is_debug):
	"""
	Args:
		fieldY: np.ndarray.
			2-D array with each row the entry, each col the y features.
		simY: np.ndarray.
			2-D array with each row the entry, each col the y features. The row number of simY is
			a multiple of row number of fieldY.

	Return:
		(np.ndarray, np.ndarray)
		2d array of the combined Y for filedY and simY, but with only 1 col. 
		The row number is the same from the raw after the combination.
	"""
	filedYRows, yCols = fieldY.shape;
	randomBase = np.random.randint(yCols, size = filedYRows);
	randomSelectMat = np.zeros((filedYRows, yCols));
	randomSelectMat[np.arange(filedYRows), randomBase] = 1;

	# Construct combined fieldY
	cmbedFiledY = fieldY * randomSelectMat;
	cmbedFiledY = np.sum(cmbedFiledY, 1, keepdims = True);

	# Construct combined simY
	repeatTime = int(simY.shape[0]/filedYRows);
	cmbedSimY = simY * np.tile(randomSelectMat, (repeatTime,1));
	cmbedSimY = np.sum(cmbedSimY, 1, keepdims = True);

	if is_debug:
		return (cmbedFiledY, cmbedSimY, randomSelectMat);
	else:
		return (cmbedFiledY, cmbedSimY);
