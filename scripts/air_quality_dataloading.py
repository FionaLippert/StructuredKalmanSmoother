import numpy as np
import pandas as pd
import os.path as osp
# import torch
# import torch_geometric as ptg
# import networkx as nx
import geopandas as gpd
import pyproj
from matplotlib import pyplot as plt
# import contextily as cx

# Equirectangular projection
def project_eqrect(lon, lat):
    # max_pos = long_lat.max(axis=0)
    # min_pos = long_lat.min(axis=0)

    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    lon_center = 0.5 * (lon_min + lon_max)
    lat_center = 0.5 * (lat_min + lat_max)

    # center_point = 0.5*(max_pos + min_pos)
    # centered_pos = long_lat - center_point

    centered_lon = lon - lon_center
    centered_lat = lat - lat_center

    # Projection will be maximally correct on center of the map
    x = centered_lon * np.cos(lat_center * np.pi / 180.)
    # centered_pos[:,0] *= np.cos(center_point[1]*(np.pi/180.))


    y = centered_lat / centered_lat.max()
    # Rescale to longitude in ~[-1,1]
    # pos = centered_pos / centered_pos[:,0].max()
    return x, y


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
# cx.add_basemap(ax, crs=gdf.crs)
fig.savefig(osp.join(dir, 'sensors.png'), dpi=200)


df_measurements = pd.read_csv(osp.join(dir, 'airquality.csv'))
df_measurements = df_measurements[df_measurements.station_id.isin(id2idx.keys())]
df_measurements['node_idx'] = df_measurements.station_id.apply(lambda id: id2idx[id])

df_measurements.to_csv(osp.join(dir, 'measurements.csv'))
df_sensors.to_csv(osp.join(dir, 'sensors.csv'))


# df_sensors['x'], df_sensors['y'] = project_eqrect(df_sensors.longitude.values, df_sensors.latitude.values)
# idx2id = {i: id for i, id in enumerate(df_sensors.district_id.values)}
#
# print(df_sensors.head())
#
# pos = np.stack([df_sensors.x.values, df_sensors.y.values]).T
# print(pos.shape)
# point_data = ptg.data.Data(pos=torch.tensor(pos))
#
# # construct voronoi tessellation
# graph_transforms = ptg.transforms.Compose((
#     ptg.transforms.Delaunay(),
#     ptg.transforms.FaceToEdge(),
# ))
# G = graph_transforms(point_data)
#

# G_nx = ptg.utils.convert.to_networkx(G)
# nx.draw_networkx_nodes(G_nx, pos, node_size=10, ax=ax)
# nx.draw_networkx_edges(G_nx, pos, ax=ax, width=2, arrowsize=2)

