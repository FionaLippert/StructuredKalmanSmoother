import numpy as np
import pandas as pd
import os.path as osp
import geopandas as gpd
import pyproj
from matplotlib import pyplot as plt
import argparse


dir = '../datasets/AirQuality'

df_city = pd.read_csv(osp.join(dir, 'city.csv')).query('cluster_id == 1')
xmin, ymin, xmax, ymax = gpd.GeoDataFrame(df_city, geometry=gpd.points_from_xy(df_city.longitude, df_city.latitude),
                       crs='EPSG:4326').total_bounds
df_sensors = pd.read_csv(osp.join(dir, 'station.csv'))

gdf = gpd.GeoDataFrame(df_sensors, geometry=gpd.points_from_xy(df_sensors.longitude, df_sensors.latitude),
                       crs='EPSG:4326')
gdf = gdf.cx[xmin:xmax, ymin:ymax]
id2idx = {id: idx for idx, id in enumerate(gdf.station_id)}

crs_local = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=0.5*(ymin + ymax), lon_0=0.5*(xmin + xmax)).crs
gdf_local = gdf.to_crs(crs_local)

df_sensors = df_sensors[df_sensors.station_id.isin(id2idx.keys())]
df_sensors['x'] = gdf_local.geometry.x
df_sensors['y'] = gdf_local.geometry.y
df_sensors['node_idx'] = df_sensors.station_id.apply(lambda id: id2idx[id])


shape = gpd.read_file(osp.join(dir, 'shapes', 'CHN_adm0.shp'))

fig, ax = plt.subplots(figsize=(10,10))
shape.plot(ax=ax, color='gray', edgecolor='white', alpha=0.3)
gdf.plot(ax=ax, color='red')
pad = 2
ax.set(xlim=(xmin - pad, xmax + pad), ylim=(ymin - pad, ymax + pad))
fig.savefig(osp.join(dir, 'sensors.png'), dpi=200)


df_measurements = pd.read_csv(osp.join(dir, 'airquality.csv'))
df_measurements = df_measurements[df_measurements.station_id.isin(id2idx.keys())]
df_measurements['node_idx'] = df_measurements.station_id.apply(lambda id: id2idx[id])

if args.n_dummy_sensors > 0:
    dummy_lon = np.random.rand(args.n_dummy_sensors) * (xmax - xmin) + xmin
    dummy_lat = np.random.rand(args.n_dummy_sensors) * (ymax - ymin) + ymin

    n_sensors = df_sensors.node_idx.max() + 1
    dummy_idx = np.arange(n_sensors, n_sensors + args.n_dummy_sensors)

    df_dummy = pd.DataFrame(dict(longitude=dummy_lon, latitude=dummy_lat, node_idx=dummy_idx))

    gdf_dummy = gpd.GeoDataFrame(df_dummy, geometry=gpd.points_from_xy(df_dummy.longitude, df_dummy.latitude),
                           crs='EPSG:4326')
    gdf_dummy_local = gdf_dummy.to_crs(crs_local)

    df_dummy['x'] = gdf_local.geometry.x
    df_dummy['y'] = gdf_local.geometry.y

    df_sensors = pd.concat([df_sensors[['node_idx', 'latitude', 'longitude', 'x', 'y']], df_dummy])
    df_sensors.to_csv(osp.join(dir, f'sensors_ndummy={args.n_dummy_sensors}.csv'))
else:
    df_sensors.to_csv(osp.join(dir, 'sensors.csv'))

df_measurements.to_csv(osp.join(dir, 'measurements.csv'))

