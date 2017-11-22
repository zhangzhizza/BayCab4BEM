from BayCab4BEM.downSampler import DownSampler
from Util.io import getFileDir, getFileName
import numpy as np
import os
import csv

simDataFile = './iwCabData/config_12/dataFromSim/raw/D_sim_org_cmbdY.csv'
fieldDataFile = './iwCabData/config_12/dataFromSim/raw/D_field_org_cmbdY.csv'
bins = 25;
qualityThres = 0.9;
outputPath = getFileDir(simDataFile, 2) + os.sep + 'down' + os.sep + 'b%d_t%d'%(bins, qualityThres * 100);
try:
	os.makedirs(outputPath);
except Exception as e:
	print (e)
dirichlet_prior = 0.5;

# Read file header
d_sim_head = None;
with open(simDataFile, 'r') as f:
	reader = csv.reader(f);
	d_sim_head = ','.join(next(reader));
d_field_head = None;
with open(fieldDataFile, 'r') as f:
	reader = csv.reader(f);
	d_field_head = ','.join(next(reader));
# Read data from file
d_sim = np.genfromtxt(simDataFile, delimiter = ',', skip_header = 1)
d_field = np.genfromtxt(fieldDataFile, delimiter = ',', skip_header = 1)
# Down sample
downSampler_dSim = DownSampler(d_sim, bins = bins, dirichlet_prior = dirichlet_prior);
(d_sim_down, d_sim_sp_hist) = downSampler_dSim.sample(stSampleSize = 50, increRatio = 1.05, qualityThres = qualityThres);
downSampler_dField = DownSampler(d_field, bins = bins, dirichlet_prior = dirichlet_prior);
(d_field_down, d_field_sp_hist) = downSampler_dField.sample(stSampleSize = 50, increRatio = 1.05, qualityThres = qualityThres);
# Save down sampled data to file
np.savetxt(outputPath + os.sep + getFileName(simDataFile, False) + '_down.csv', d_sim_down, delimiter=",", header = d_sim_head);
np.savetxt(outputPath + os.sep + getFileName(fieldDataFile, False) + '_down.csv', d_field_down, delimiter=',', header = d_field_head);