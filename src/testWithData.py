from BayCab4BEM.data_preprocessor import Preprocessor
from BayCab4BEM.mcmc_pymc3 import MCMC4Posterior_pymc3
from BayCab4BEM.mcmc_pystan import MCMC4Posterior_pystan
from Util.logger import Logger
import os 
import pickle as pk;

def get_output_folder(parent_dir, job_name):
    """
    The function give a string name of the folder that the output will be
    stored. It finds the existing folder in the parent_dir with the highest
    number of '-run#', and add 1 to the highest number of '-run#'.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    env_name: str
      The EnergyPlus environment name. 

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, job_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

mcmcPackage = 'pystan';
LOG_LEVEL = 'DEBUG';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";

fieldDataFile = './iwCabData/config_3/dataFromSim/down/b20_t90/D_field_down.csv'#'./iwCabData/adrian_data/DATAFIELD_sample.csv'
simDataFile = './iwCabData/config_3/dataFromSim/down/b20_t90/D_sim_down.csv'#'./iwCabData/adrian_data/DATACOMP_sample.csv'
cmbYArgs = [];
ydim = 1;

stanInFileName = './iwCabData/config_3/stan_in/theta_norm.stan'
dftModelName = './iwCabData/config_3/stan_compiled/theta_norm.stan.pkl'

save_base_dir = './mcmcRes/config_3/fromData' 
save_dir = get_output_folder(save_base_dir, 'IW_cab');
os.makedirs(save_dir)

logger = Logger().getLogger('BC4B_logger', LOG_LEVEL, LOG_FMT, save_dir + os.sep + 'log.log')

prep = Preprocessor(logger);
(z, xf, xc, t) = prep.getDataFromFile(fieldDataFile, simDataFile, cmbYArgs, ydim);

trace = None;

if mcmcPackage == "pymc3":
	mcmcObj = MCMC4Posterior_pymc3(z, xf, xc, t, logger)
	model = mcmcObj.build(covFuncName = 'covFuncPymcNat');
	trace = mcmcObj.run(model, draws = 500, sampler = 'Metropolis', njobs = 1)

elif mcmcPackage == 'pystan':
	
	mcmcObj = MCMC4Posterior_pystan(z, xf, xc, t, logger);
	model = mcmcObj.build(stanInFileName = stanInFileName, 
						  stanModelFileName = None, 
						  dftModelName = dftModelName);
	with open(save_dir + os.sep + 'model.pkl', 'wb') as modelfile:
		pk.dump(model, modelfile);

	trace = mcmcObj.run(model, iterations = 500, sampler = 'NUTS', chains = 4, warmup = 250, n_jobs = 6);
	with open(save_dir + os.sep + 'posteriors.txt', 'w') as txtfile:
		txtfile.write(str(trace));
	with open(save_dir + os.sep + 'trace.pkl', 'wb') as tracefile:
		pk.dump(trace, tracefile);
	

