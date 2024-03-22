import numpy as np
import pandas as pd
import os.path as osp
import geopandas as gpd
import pyproj
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Add dummy sensors to AirQuality dataset')

parser.add_argument("--n_dummy_sensors", type=int, default=0, help="number of dummy sensors to include")
args = parser.parse_args()

dir = '../datasets/AirQuality'

df_city = pd.read_csv(osp.join(dir, 'city.csv')).query('cluster_id == 1')
xmin, ymin, xmax, ymax = gpd.GeoDataFrame(df_city, geometry=gpd.points_from_xy(df_city.longitude, df_city.latitude),
                       crs='EPSG:4326').total_bounds
df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))

crs_local = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=0.5*(ymin + ymax), lon_0=0.5*(xmin + xmax)).crs


if args.n_dummy_sensors > 0:
    dummy_lon = np.random.rand(args.n_dummy_sensors) * (xmax - xmin) + xmin
    dummy_lat = np.random.rand(args.n_dummy_sensors) * (ymax - ymin) + ymin

    n_sensors = df_sensors.node_idx.max() + 1
    print(df_sensors.node_idx)
    dummy_idx = np.arange(n_sensors, n_sensors + args.n_dummy_sensors)
    print(dummy_idx)

    df_dummy = pd.DataFrame(dict(longitude=dummy_lon, latitude=dummy_lat, node_idx=dummy_idx))

    gdf_dummy = gpd.GeoDataFrame(df_dummy, geometry=gpd.points_from_xy(df_dummy.longitude, df_dummy.latitude),
                           crs='EPSG:4326')
    gdf_dummy_local = gdf_dummy.to_crs(crs_local)

    df_dummy['x'] = gdf_dummy_local.geometry.x
    df_dummy['y'] = gdf_dummy_local.geometry.y

    df_sensors = pd.concat([df_sensors[['node_idx', 'latitude', 'longitude', 'x', 'y']], df_dummy])
    df_sensors.to_csv(osp.join(dir, f'sensors_ndummy={args.n_dummy_sensors}.csv'))


