import os, glob
os.chdir('.')

import tqdm
import numpy as np
import netCDF4
import joblib

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

SCENARIO = 119
SEASON = 4
DAY_COUNT = 40
TIME_STEP = 24
WIDTH = 67
HEIGHT = 82

conc_list = [
    'NO2','NO','O3P','O3','NO3','N2O5',
    'HNO3','O1D2','HO','HONO','HO2','CO','HNO4',
    'HO2H','SO2','SULF','SULRXN','C_O2','HCHO',
    'COOH','MEOH','RO2_R','ROOH','R2O2','RO2_N',
    'RNO3','MEK','PROD2','CCO_O2','PAN','CCO_OOH',
    'CCO_OH','RCO_O2','PAN2','CCHO','RCO_OOH',
    'RCO_OH','BZCO_O2','PBZN','BZ_O','MA_RCO3',
    'MA_PAN','TBU_O','ACET','NPHE','PHEN','BZNO2_O',
    'HOCOO','HCOOH','RCHO','GLY','MGLY','BACL',
    'CRES','BALD','METHACRO','MVK','ISOPROD','DCB1',
    'DCB2','DCB3','ETHENE','ISOPRENE','ISOPRXN',
    'TRP1','TRPRXN','ALK1','ALK2','ALK3','ALK4',
    'ALK5','ALK5RXN','ARO1','ARO1RO2','TOLNRXN',
    'TOLHRXN','ARO2','ARO2RO2','XYLNRXN','XYLHRXN',
    'BENZENE','BENZRO2','BNZNRXN','BNZHRXN','OLE1',
    'OLE2','SESQ','SESQRXN','ASO4J','ASO4I','ANH4J',
    'ANH4I','ANO3J','ANO3I','AALKJ','AXYL1J','AXYL2J',
    'AXYL3J','ATOL1J','ATOL2J','ATOL3J','ABNZ1J',
    'ABNZ2J','ABNZ3J','ATRP1J','ATRP2J','AISO1J',
    'AISO2J','ASQTJ','AORGCJ','AORGPAJ','AORGPAI',
    'AECJ','AECI','A25J','A25I','ACORS','ASOIL',
    'NUMATKN','NUMACC','NUMCOR','SRFATKN','SRFACC',
    'SRFCOR','AH2OJ','AH2OI','ANAJ','ANAI','ACLJ',
    'ACLI','ANAK','ACLK','ASO4K','ANH4K','ANO3K',
    'AH2OK','AISO3J','AOLGAJ','AOLGBJ','NH3','HCL',
    'SV_ALK','SV_XYL1','SV_XYL2','SV_TOL1','SV_TOL2',
    'SV_BNZ1','SV_BNZ2','SV_TRP1','SV_TRP2','SV_ISO1',
    'SV_ISO2','SV_SQT',
]

rsm_path = []
for path in glob.glob('/mnt/dsk0/bggo/CMAQ_dataset/*'):
    if os.path.isdir(path) and 'RSM' in path:
        rsm_path.append(path)
rsm_path.sort(key=lambda x: int(x.split('/')[-1].split('_')[1]))

def worker(root):
    j, root_path = root
    ncf_list = glob.glob(os.path.join(root_path, '*.ncf'))
    ncf_list.sort()
    desc = f'Load {str(j).zfill(3)} ncf datasets'

    ncfs = netCDF4.MFDataset(ncf_list)
    dataset = [ncfs[conc][:] for conc in tqdm.tqdm(conc_list, desc=desc, position=j)]
    dataset = np.array(dataset, dtype=np.float32).squeeze()
    np.save(f'datasets/hourly/all_conc_{j}.npy', dataset)

args = [(j, root_path) for j, root_path in enumerate(rsm_path)]
for arg in args:
    print(arg)
    worker(arg)
