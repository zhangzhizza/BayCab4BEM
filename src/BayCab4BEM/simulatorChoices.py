"""
Define the map between simulator name and SimulatorRunWorker object

Author: Zhiang Zhang

First Created: Sept 6th, 2017
Last Updated: Sept 6th, 2017
"""
from BayCab4BEM.runEplus import EnergyPlusRunWorker

simulatorObjMapping = {'energyplus': EnergyPlusRunWorker};