import BayCab4BEM.bayCab4BEM as bc4b

from Util.logger import Logger 


LOG_LEVEL = 'INFO';
LOG_FMT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s";
logger = Logger().getLogger('BC4B_logger', LOG_LEVEL, LOG_FMT, log_file_path = None)

fieldDataFile = 'cal_example_field_withoutSummer.csv'
simDataFile = 'cal_example_com_withoutSummer.csv'
bc4bObj = bc4b.BC4BEM(logger);
bc4bObj.runWithData(fieldDataFile, simDataFile, 'covFuncPymcNat', draws = 500, sampler = 'nuts')
