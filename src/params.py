"""Parameters for CMAQNet & optimizer"""

SEASON = ['January', 'April', 'July', 'October']
CTRL_KEY_LIST = ['ALL_POW','ALL_IND','ALL_MOB','ALL_RES','NH3_AGR','ALL_SLV','ALL_OTH']
EMIS_KEY = ['SO2', 'NH3', 'VOCs', 'CO', 'PM2_5', 'NOx']
CONC_KEY = ['PM2.5_SO4', 'PM2.5_NH4', 'PM2.5_NO3', 'PM2.5_Total']#, 'O3']
REGION_CODE = {
    'A': 'Seoul City', 'B': 'Incheon City', 'C': 'Busan City', 'D': 'Daegu City',
    'E': 'Gwangju City', 'F': 'Gyeonggi-do', 'G': 'Gangwon-do', 'H': 'Chungbuk-do',
    'I': 'Chungnam-do', 'J': 'Gyeongbuk-do', 'K': 'Gyeongnam-do', 'L': 'Jeonbuk-do',
    'M': 'Jeonnam-do', 'N': 'Jeju-do', 'O': 'Daejeon City', 'P': 'Ulsan City', 'Q': 'Sejong City'
}

SCENARIO = 119
REGION = 17
SECTOR = len(CTRL_KEY_LIST)
PRECURSOR = len(EMIS_KEY)

HEIGHT = 82
WIDTH = 67
