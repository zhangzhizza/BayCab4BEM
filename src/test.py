import BayCab4BEM.bayCab4BEM as bc4b 
xf = './BayCab4BEM/sample/x_hourly.csv'
yf = './BayCab4BEM/sample/y_hourly.csv'
calif = './BayCab4BEM/configSample.xml'
simName = 'energyplus'
baseIdf = './BayCab4BEM/sample/iw_base_hourly.idf'
runNum = 2;
maxRun = 5;
cmbY = [];
simExe = ['./BayCab4BEM/EnergyPlus-8-3-0/energyplus', './BayCab4BEM/sample/pittsburgh.epw']
bc4bObj = bc4b.BC4BEM();
bc4bObj.run(xf, yf, calif, simName, baseIdf, runNum, maxRun, cmbY, simExe)
