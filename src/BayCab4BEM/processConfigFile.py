"""
The function to extract info from the configuration file.

Author: Zhiang Zhang
Date: Dec 7th, 2017
"""
import xml.etree.ElementTree

def processConfigFile(configFilePath):
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