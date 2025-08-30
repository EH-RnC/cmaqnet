import os

import numpy as np
import pandas as pd
from netCDF4 import Dataset
import geopandas as gpd
from shapely.geometry import Point
from cmaqnet.alloc import allocation
from cmaqnet.params import (
    REGION_CODE,
    SCENARIO,
    WIDTH,
    HEIGHT,
    REGION,
    PRECURSOR
)

def read_cdf_map_data(
        data_path:str,
        target:str,
        prefix:str,
        keys:list[str]) -> np.ndarray:
    """
    Read netCDF map data from path.

    Note:
    ----
        Scene No.1: base scenario map data
        Scene No.2 ~ No.119: simulation result

    Args:
    ----
        data_path (str): root path which has raw datas
        target (str): emission (SMOKE) or concentration (CMAQ)
        prefix (str): raw data prefix. EMIS_AVG or ACONC
        keys (list[str]): precursor symbols

    Returns:
    ----
        datasets (numpy.ndarray)
    """
    paths = [os.path.join(f'{data_path}/{target}/', f'{prefix}.{i+1}') for i in range(SCENARIO)]
    datasets = [[Dataset(path, 'r')[key][0, 0].data.tolist() for key in keys] for path in paths]
    return np.transpose(datasets, (0, 2, 3, 1))

def read_ctrl_matrix(data_path:str) -> np.ndarray:
    """
    Read control matrix value from csv file

    Note:
    ----
        Region: A ~ Q
        Sector: ALL_POW,ALL_IND,ALL_MOB,ALL_RES,NH3_AGR,ALL_SLV,ALL_OTH

    Args:
    ----
        data_path (str): root path which has raw datas

    Returns:
    ----
        datasets (numpy.ndarray)
    """
    path = f'{data_path}/control_matrix.csv'
    return pd.read_csv(path, index_col=0).values

def read_region_emission_matrix(data_path:str) -> np.ndarray:
    """
    Read region based emission matrix value from csv

    Note:
    ----
        Region: Cities of Korea
        Emission: SO2,PM2_5,NOx,VOCs,NH3,CO

    Args:
    ----
        data_path (str): root path which has raw datas

    Returns:
    ----
        datasets (numpy.ndarray)
    """
    path = f'{data_path}/r_base_sample.csv'
    return pd.read_csv(path, index_col=0, encoding='cp949')\
            .drop(index=3)\
            .reset_index(drop=True)\
            .values

def get_ctprvn_map(data_path:str) -> gpd.GeoDataFrame:
    """
    Get CTPRVN geopandas dataframe of Korea

    Note:
    ----
        C: City
        T: Town
        P: Province
        R: Region
        V: Village
        N: Neighborhood

    Args:
    ----
        data_path (str): root path which has raw datas

    Returns:
    ----
        ctprvn (geopandas.GeoDataFrame)
    """
    path = f'{data_path}/geoinfo/ctp_rvn.shp'
    ctprvn = gpd.GeoDataFrame.from_file(path, encoding='cp949')
    ctprvn.crs = 'EPSG:5179'
    return ctprvn

def get_base_raster(ctprvn:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    proj = '+proj=lcc +lat_1=30 +lat_2=60 +lon_1=126 +lat_0=38 +lon_0=126 +ellps=GRS80 +units=m'
    points = [Point(i, j)
                for i in range(-180000, -180000 + 9000 * WIDTH, 9000)
                for j in range(-585000, -585000 + 9000 * HEIGHT, 9000)]
    grid_data = gpd.GeoDataFrame(points, geometry='geometry', columns=['geometry'])
    grid_data.crs = ctprvn.to_crs(proj).crs
    grid_data.loc[:,'x_m'] = grid_data.geometry.x
    grid_data.loc[:,'y_m'] = grid_data.geometry.y
    grid_data.loc[:,'value'] = 0
    grid_data.loc[:,'index'] = grid_data.index
    return grid_data

def get_region_pixel_indices(data_path:str) -> list:
    """
    Get pixel indices of each region in Korea.
    
    Note:
    ----
        0: 강원도\n
        1: 경기도\n
        2: 경상남도\n
        3: 경상북도\n
        4: 광주광역시\n
        5: 대구광역시\n
        6: 대전광역시\n
        7: 부산광역시\n
        8: 서울특별시\n
        9: 세종특별자치시\n
        10: 울산광역시\n
        11: 인천광역시\n
        12: 전라남도\n
        13: 전라북도\n
        14: 제주특별자치도\n
        15: 충청남도\n
        16: 충청북도

    Args:
    ----
        data_path (str): root path which has raw datas

    Returns:
    ----
        pixel_indices (list)
    """
    ctprvn = get_ctprvn_map(data_path)
    grid_data = get_base_raster(ctprvn)

    cities = {
        0: '강원도', 1: '경기도', 2: '경상남도', 3: '경상북도',
        4: '광주광역시', 5: '대구광역시', 6: '대전광역시', 7: '부산광역시',
        8: '서울특별시', 9: '세종특별자치시', 10: '울산광역시', 11: '인천광역시',
        12: '전라남도', 13: '전라북도', 14: '제주특별자치도', 15: '충청남도',
        16: '충청북도'
    }

    gdf_joined_loc = ['CTPRVN_CD', 'CTP_ENG_NM', 'CTP_KOR_NM', 'index_right']
    gdf_joined = gpd.sjoin(ctprvn, grid_data.to_crs(5179), predicate='contains')

    indices = gpd.GeoDataFrame(pd.merge(
        left=grid_data, right=gdf_joined.loc[:,gdf_joined_loc], 
        how='left', left_on='index', right_on='index_right'
    ), geometry='geometry').dropna()
    pixel_indices = \
        [[(idx%82, idx//82) for idx in indices.loc[indices.CTP_KOR_NM==cities[region]].index.tolist()]
         for region, _ in cities.items()]
    return pixel_indices

def get_region_base_map(
        dataset:np.ndarray,
        base_grid:np.ndarray,
        indices:list) -> np.ndarray:
    """
    Allocate normalized emission ratio to base emission map using region based emission matrix.

    Args:
    ----
        dataset (numpy.ndarray): region based emission matrix
        base_grid (numpy.ndarray): emission map of base scenario
        indices (list): list of grid indices of each region in Korea

    Returns:
    ----
        region_base_map (numpy.ndarray)
    """
    region_base_map = np.zeros((SCENARIO, HEIGHT, WIDTH, REGION, PRECURSOR))
    emission_val = \
        dataset.copy().reshape(SCENARIO, REGION, PRECURSOR) \
        / dataset[0].reshape(1, REGION, PRECURSOR)
    for region in range(REGION):
        row_indices, col_indices = zip(*indices[region])
        region_base_map[:, row_indices, col_indices, region, :] = \
            emission_val[:, region, :].reshape(SCENARIO, 1, PRECURSOR)
    return np.sum(region_base_map, axis=3) * base_grid

def get_ctrl_map(X, grid_alloc):
    X = X.reshape(-1, 17, 7)
    ctrl_map = np.zeros((X.shape[0], 82, 67, 7))
    ctrl_map[:, :, :, 6] = 1.0
    for i, key in enumerate(REGION_CODE.keys()):
        index = grid_alloc.loc[grid_alloc.Region_Code==key, ['Row', 'Column']]
        index = index.drop_duplicates().values - 1
        row, col = zip(*index)
        ctrl_map[:, row, col, :] = X[:, i:i+1, :]
    return ctrl_map

def gridding(ctrl_mat:np.ndarray) -> np.ndarray:
    ctrl_map = np.zeros((119, 82, 67, 7))
    for i, key in enumerate(list(REGION_CODE.keys())):
        value = ctrl_mat[:, i:i+1, :]
        index = allocation.loc[allocation['Region_Code']==key, ['Row', 'Column']].values
        ratio = allocation.loc[allocation['Region_Code']==key, ['Ratio']].values / 100
        ratio = ratio.reshape(1, -1, 1)
        value = value * ratio
        row, col = zip(*index)
        ctrl_map[:, row, col, :] += value
    ctrl_map[:,:,:,6] = np.where(ctrl_map[:,:,:,6]>0, ctrl_map[:,:,:,6], 1)
    return ctrl_map