"""
This file combine multiple Ys from the raw data randomly

Author: Zhiang Zhang
Create Date: Nov 21st, 2017
"""
import numpy as np;
import csv;
import os;

from BayCab4BEM.rawYCmb import randomCmb;
from Util.io import getFileDir;

rawFieldDataPath = './iwCabData/config_14/dataFromSim/raw/DEBUG_D_field_org.csv';
rawSimDataPath = './iwCabData/config_14/dataFromSim/raw/DEBUG_D_sim_org.csv';
yDim = 2;
# Read file header
d_sim_head = None;
with open(rawSimDataPath, 'r') as f:
	reader = csv.reader(f);
	d_sim_head = ','.join(next(reader));
d_field_head = None;
with open(rawFieldDataPath, 'r') as f:
	reader = csv.reader(f);
	d_field_head = ','.join(next(reader));
# Modify file header
d_sim_head = d_sim_head.split(',')[yDim - 1:];
d_sim_head[0] = 'cmbedY';
d_field_head = d_field_head.split(',')[yDim - 1:];
d_field_head[0] = 'cmbedY';
d_sim_head = ','.join(d_sim_head);
d_field_head = ','.join(d_field_head);
# Read data from file
d_field = np.genfromtxt(rawFieldDataPath, delimiter = ',', skip_header = 1)
d_sim = np.genfromtxt(rawSimDataPath, delimiter = ',', skip_header = 1)
# Extract the Ys
d_field_y = d_field[:, 0:yDim];
d_sim_y = d_sim[:, 0:yDim];
# Randomly combine
d_field_y_cmbd, d_sim_y_cmbd, debug_out = randomCmb(d_field_y, d_sim_y, is_debug = True);
# Put cmbd y back to data
d_field_cmbd = np.delete(d_field, np.s_[:yDim], 1) # Delete old cols
d_field_cmbd = np.hstack((d_field_y_cmbd, d_field_cmbd));
d_sim_cmbd = np.delete(d_sim, np.s_[:yDim], 1) # Delete old cols
d_sim_cmbd = np.hstack((d_sim_y_cmbd, d_sim_cmbd));
# Write back to file
np.savetxt(getFileDir(rawSimDataPath) + os.sep + 'D_sim_org_cmbdY.csv', d_sim_cmbd, delimiter=",", header = d_sim_head);
np.savetxt(getFileDir(rawFieldDataPath) + os.sep + 'D_field_org_cmbdY.csv', d_field_cmbd, delimiter=",", header = d_field_head);
np.savetxt(getFileDir(rawFieldDataPath) + os.sep + 'cmbdY_debug_randomBase.csv', debug_out, delimiter=",");
