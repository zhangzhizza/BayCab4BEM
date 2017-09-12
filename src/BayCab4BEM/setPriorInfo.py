"""
Set prior information here.

Author: Zhiang Zhang

First Created: Sept 12th, 2017
Last Updated: Sept 12th, 2017
"""
from pymc3 import Uniform, Beta, Gamma

# Set the prior distributions, taken from Chong(2017) PhD thesis
thetaPriorInfo = [Uniform, 0, 1];
rho_etaPriorInfo = [Beta, 1, 0.5];
rho_deltaPriorInfo = [Beta, 1, 0.4];
lambda_etaPriorInfo = [Gamma, 5., 5.]#[Gamma, 10, 10];
lambda_deltaPriorInfo = [Gamma, 1., 0.00001]#[Gamma, 10, 0.3];
lambda_epsiPriorInfo = [Gamma, 1., 0.00001]#[Gamma, 10, 0.03];