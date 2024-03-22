import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx
import pickle
import os.path as osp
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import contextily as ctx
import matplotlib.pyplot as plt
import pyproj
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


NORTH = 37.450
SOUTH = 37.210
EAST = -121.80
WEST = -122.10

DATA_DATETIME = '2022_10_03'
META_DATETIME = '2022_10_22'

SIMPLIFY = True

def load_roads(dir):
    filename = 'road_graph.pkl' if SIMPLIFY else 'road_graph_raw.pkl'
    if not osp.isfile(osp.join(dir, filename)):
        cf = '["highway"~"motorway|motorway_link"]'  # road filter, don't use small ones
        # simplify=True only retaines nodes at junctions and dead ends
        G = osmnx.graph_from_bbox(north=NORTH, south=SOUTH, east=EAST, west=WEST, simplify=False, custom_filter=cf)
        with open(osp.join(dir, filename), 'wb') as f:  # frequent loading of maps leads to a temporal ban
            pickle.dump(G, f)
    else:
        with open(osp.join(dir, filename), 'rb') as f:
            G = pickle.load(f)

    return G

def clean_graph(G):
    # G = osmnx.get_undirected(G)
    for _ in range(2):
        out_degree = G.degree
        to_remove = [node for node in G.nodes if out_degree[node] == 1]
        G.remove_nodes_from(to_remove)
    G = nx.convert_node_labels_to_integers(G)

    lanes = {e : int(l) if len(l) == 1 else int(max(l)) for e, l in nx.get_edge_attributes(G, 'lanes').items()}

    # add lane info to motorway_links
    new_lanes = {(u, v, k): 1 for u, v, k, d in G.edges(keys=True, data=True) if not 'lanes' in d}
    nx.set_edge_attributes(G, values={**lanes, **new_lanes}, name='lanes')

    # add attributes 'speed_kph' and 'travel_time' to all edges (if speed info is not available it is imputed)
    G = osmnx.add_edge_speeds(G)
    G = osmnx.add_edge_travel_times(G)

    return G

def load_sensor_info(dir):
    # load station meta data
    df = pd.read_csv(osp.join(dir, f"d04_text_meta_{META_DATETIME}.txt"), sep="\t").dropna()

    # filter stations
    df = df[(df['Longitude'] < EAST) & (df['Longitude'] > WEST)]
    df = df[(df['Latitude'] < NORTH) & (df['Latitude'] > SOUTH)]
    df = df[df['ID'] != 401937] # remove double measurements
    df.reset_index(inplace=True)

    # coords
    coords = df[['Longitude', 'Latitude']].to_numpy()

    return df, coords

"""
Copied from https://github.com/spbu-math-cs/Graph-Gaussian-Processes/blob/main/examples/utils/utils.py
"""
def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [distance]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[:i+1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]

"""
Copied and adjusted from https://github.com/spbu-math-cs/Graph-Gaussian-Processes/blob/main/examples/utils/preprocessing.py
"""
def insert_stations(G, coords, crs=3857):
    G = osmnx.project_graph(G, to_crs=crs)
    idx_to_node = {}
    for idx, (lon, lat) in enumerate(coords):

        sensor_point = gpd.GeoSeries(Point(lon, lat), crs=4326).to_crs(crs)[0]
        u, v, key = osmnx.distance.nearest_edges(G, sensor_point.x, sensor_point.y)
        edge = G.edges[(u, v, key)]
        # if edge doesn't have geometry create line string: [LineString(coords_u]), LineString(coords_v)]
        if "geometry" in edge:
            geom = edge["geometry"]
        else:
            geom = LineString([(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])])

        cut_out = cut(geom, geom.project(sensor_point))

        if len(cut_out) == 2:
            G.remove_edge(u, v, key)
            idx_to_node[idx] = len(G)  # adding new vertex at the end
            l_ratio = geom.project(sensor_point, normalized=True)
            l_1, l_2 = l_ratio * edge['length'], (1 - l_ratio) * edge['length']
            new_vertex = nearest_points(geom, sensor_point)[0]
            new_vertex_lonlat = gpd.GeoSeries(new_vertex, crs=crs).to_crs(4326)[0]

            G.add_node(len(G), x=new_vertex.x, y=new_vertex.y,
                       lon=new_vertex_lonlat.x, lat=new_vertex_lonlat.y, street_count=2)
            # convert length [m] and speed [km/h] to travel time [s]
            speed_ms = edge['speed_kph'] * 5 / 18
            travel_time_1 = l_1 / speed_ms
            travel_time_2 = l_2 / speed_ms
            G.add_edge(u, len(G) - 1, length=l_1, geometry=cut_out[0], lanes=edge['lanes'],
                       speed_kph=edge['speed_kph'], travel_time=travel_time_1)
            G.add_edge(len(G) - 1, v, length=l_2, geometry=cut_out[1], lanes=edge['lanes'],
                       speed_kph=edge['speed_kph'], travel_time=travel_time_2)
        else:
            # sensor falls onto node u or v
            if (cut_out[0] == 0) and (u not in idx_to_node.values()):
                idx_to_node[idx] = u
            elif (cut_out[0] == geom.length) and (v not in idx_to_node.values()):
                idx_to_node[idx] = v
            else:
                print(f'ignore sensor idx {idx} with distance {cut_out[0]}')

    G = osmnx.project_graph(G, to_crs=4326)

    G_nx = nx.MultiDiGraph(G.edges())

    for edge_attr in ['length', 'lanes', 'speed_kph', 'travel_time']:
        nx.set_edge_attributes(G_nx, values=nx.get_edge_attributes(G, edge_attr), name=edge_attr)
    for node_attr in ['x', 'y', 'lon', 'lat', 'street_count']:
        values = nx.get_node_attributes(G, node_attr)
        print(node_attr, len(values))
        nx.set_node_attributes(G_nx, values=values, name=node_attr)

    # clean up edges
    remove_edges = [(u, v) for u, v, attr in G_nx.edges(data=True) if len(attr) < 1]
    G_nx.remove_edges_from(remove_edges)


    return G, G_nx, idx_to_node


def load_traffic_data(dir, id_to_idx, hours=range(5, 23), minutes=range(60)):

    # load station data
    df = pd.read_csv(osp.join(dir, f"d04_text_station_5min_{DATA_DATETIME}.txt"), sep=",", header=None)

    columns = ['Timestamp', 'ID', 'District', 'Freeway', 'Direction', 'Lane Type', 'Station Length', 'Samples',
               'Observed', 'Flow', 'Occupancy', 'Speed']
    df = df.iloc[:, :12].set_axis(columns, axis=1)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index(['Timestamp'], inplace=True)

    # filter time
    df = df[df.index.minute.isin(minutes)]
    df = df[df.index.hour.isin(hours)]

    # filter stations
    df = df[df.ID.isin(id_to_idx.keys())]

    print(f'df size before = {len(df)}')

    # filter out zero-flow measurements
    df.Flow[df.Flow == 0] = np.nan
    df.dropna(inplace=True)

    print(f'df size after = {len(df)}')

    return df

def plot_graph(dir, G, vals, vertex_id, normalization, ax, fig, cax, vmin=None, vmax=None, filename=None, bbox=None,
              nodes_to_label=[], node_size=20, alpha=0.6, edge_linewidth=0.4, label_nodes=False,
              cmap_name='viridis', cut_colormap=False,
              plot_title=None):
    n, s, e, w = bbox  # bounds of crossroads
    mean, std = normalization
    vals = vals*std + mean
    vertex_id_dict = {vertex_id[i]: i for i in range(len(vertex_id))}

    if vmin is None:
        vmin = np.min(vals)
    if vmax is None:
        if not cut_colormap:
            vmax = np.max(vals)
        else:
            vmax = np.sort(vals)[9*len(vals)//10]  # variance to high on distant points

    cmap = colormaps.get_cmap(cmap_name)
    norm = Normalize(vmin=vmin, vmax=vmax)

    colors = []
    for i in range(len(G)):
        if vertex_id_dict.get(i) is not None:
            val = vals[vertex_id_dict[i]]
            colors.append(cmap(norm(val)))
        else:
            colors.append((0, 0, 0, 1))  # black

    osmnx.plot_graph(G, show=False, close=False, bgcolor='w', node_color=colors, node_size=50,
                     edge_color='black', edge_linewidth=edge_linewidth, bbox=bbox, ax=ax)
    for idx in vertex_id:
        ax.text(G.nodes[idx]['x'], G.nodes[idx]['y'], str(idx), fontsize=8)

    if plot_title is not None:
        ax.set_title(plot_title)

    # adding realworld map to the background
    ctx.add_basemap(ax=ax, crs='epsg:4326')
    ax.set_axis_off()
    if filename is not None:
        plt.savefig(osp.join(dir, f'{filename}.png'), dpi=500)


    if cut_colormap:
        cbar = fig.colorbar(cm.ScalarMappable(norm, cmap), orientation='vertical', extend='max', cax=cax)
    else:
        cbar = fig.colorbar(cm.ScalarMappable(norm, cmap), orientation='vertical', cax=cax)

    if filename is not None:
        plt.savefig(osp.join(dir, f'colorbar_{filename}.png'), dpi=500, transparent=True)



if __name__ == "__main__":

    dir = '../datasets/pems'

    # load sensor meta data
    df_meta, coords = load_sensor_info(dir)

    # load road network
    G = load_roads(dir)
    G = clean_graph(G)
    G, G_nx, idx_to_node = insert_stations(G, coords)
    print(f'graph size = {len(G)}')

    # mapping from station ID to index
    df_meta = df_meta[df_meta.index.isin(idx_to_node.keys())]
    df_meta['Index'] = df_meta.index
    df_meta['Node'] = df_meta.Index.apply(lambda idx: idx_to_node[idx])
    print(f'number of sensors used = {len(df_meta)}')
    id_to_idx = dict(zip(df_meta.ID, df_meta.index))
    id_to_lanes = dict(zip(df_meta.ID, df_meta.Lanes))
    id_to_dir = dict(zip(df_meta.ID, df_meta.Dir))

    # load traffic data
    df_traffic = load_traffic_data(dir, id_to_idx)

    # add node info
    df_traffic['Index'] = df_traffic.ID.apply(lambda id: id_to_idx[id])
    df_traffic['Node'] = df_traffic.Index.apply(lambda idx: idx_to_node[idx])
    df_traffic['Lanes'] = df_traffic.ID.apply(lambda id: id_to_lanes[id])
    df_traffic['Dir'] = df_traffic.ID.apply(lambda id: id_to_dir[id])
    df_traffic['AvgFlow'] = df_traffic.Flow / df_traffic.Lanes

    with open(osp.join(dir, 'processed_osmnx_graph.pkl'), 'wb') as f:
        pickle.dump(G, f)
    with open(osp.join(dir, 'processed_nx_graph.pkl'), 'wb') as f:
        pickle.dump(G_nx, f)

    pd.DataFrame.from_dict(idx_to_node, orient='index').to_csv(osp.join(dir, 'idx_to_node.csv'), header=False)

    df_traffic.to_csv(osp.join(dir, 'traffic.csv'))
    df_meta.to_csv(osp.join(dir, 'sensors.csv'))

    example_traffic = df_traffic[(df_traffic.index.minute == 0) & (df_traffic.index.hour == 8)][['Node', 'Speed']]
    len_before = len(example_traffic)
    example_traffic = example_traffic.dropna()
    print(f'removed {len_before - len(example_traffic)} measurements due to missing data')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    lat = nx.get_node_attributes(G, 'lat').values()
    lat = nx.get_node_attributes(G, 'lat').values()
    plot_graph(dir, G, example_traffic.Speed.values, example_traffic.Node.values, (0, 1), ax, fig, cax,
               plot_title='hour=8', filename='hour=8', bbox=(NORTH, SOUTH, EAST, WEST))

    bbox_zoomed = (37.330741, 37.315718, -121.883005, -121.903327)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    plot_graph(dir, G, example_traffic.Speed.values, example_traffic.Node.values, (0, 1), ax, fig, cax,
               plot_title='crossing at hour=8', filename='crossing_hour=8', bbox=bbox_zoomed, node_size=250,
                    alpha=0.95, edge_linewidth=2, label_nodes=True)
