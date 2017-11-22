"""
IO related utility functions.
Author: Zhiang Zhang
Create Date: Nov 21st, 2017
"""
import os;

def getFileDir(filePath, upStreamLevel = 1):
	maxUpStream = filePath.count(os.sep);
	if upStreamLevel > maxUpStream:
		raise ValueError('upStreamLevel exceeds the maximum directory level (%d),'
						' contained in the filePath.'%maxUpStream);
	elif upStreamLevel < 1:
		raise ValueError('upStreamLevel must be >= 1.');
	for i in range(upStreamLevel):
		filePath = filePath[0: filePath.rfind(os.sep)];
	return filePath;

def getFileName(filePath, containExtension = False):
	nameWithExt = filePath[filePath.rfind(os.sep) + 1:];
	if containExtension:
		return nameWithExt;
	else:
		return nameWithExt[0: nameWithExt.rfind('.')];

