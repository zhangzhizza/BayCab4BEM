from BayCab4BEM.data_preprocessor import Preprocessor
from BayCab4BEM.mcmc_pymc3 import MCMC4Posterior_pymc3
from BayCab4BEM.mcmc_pystan import MCMC4Posterior_pystan
from Util.logger import Logger 

mcmcPackage = 'pystan';
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
logger = Logger().getLogger('BC4B_logger', LOG_LEVEL, LOG_FMT, log_file_path = None)

fieldDataFile = 'DATAFIELD_sample.csv'
simDataFile = 'DATACOMP_sample.csv'
cmbYArgs = ['linear', 0.5, 0.5];
ydim = 2;

prep = Preprocessor(logger);
(z, xf, xc, t) = prep.getDataFromFile(fieldDataFile, simDataFile, cmbYArgs, ydim);

trace = None;

if mcmcPackage == "pymc3":
	mcmcObj = MCMC4Posterior_pymc3(z, xf, xc, t, logger)
	model = mcmcObj.build(covFuncName = 'covFuncPymcNat');
	trace = mcmcObj.run(model, draws = 500, sampler = 'Metropolis', njobs = 1)

elif mcmcPackage == 'pystan':
	mcmcObj = MCMC4Posterior_pystan(z, xf, xc, t, logger);
	model = mcmcObj.build();
	trace = mcmcObj.run(model, iterations = 500, sampler = 'NUTS', chains = 4, warmup = 250, n_jobs = 6);