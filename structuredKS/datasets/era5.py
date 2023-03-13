import cdsapi
import os
import xarray as xr
import rioxarray
from shapely import geometry
import pyproj
import geopandas as gpd
import torch
import torch_geometric as ptg
import stripy

VARNAMES = {'2m_temperature' : 't2m'}

def lonlat2local(lon, lat):
        pts_lonlat = gpd.GeoSeries([geometry.Point(point) for point in zip(lon, lat)], crs=f'epsg:4326')

        # setup local "azimuthal equidistant" coordinate system
        lat_0 = pts_lonlat.y.mean()
        lon_0 = pts_lonlat.x.mean()
        crs_local = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=lat_0, lon_0=lon_0).crs
        pts_local = pts_lonlat.to_crs(crs_local)

        print(pts_local.x, pts_local.y)

def load(data_dir, variable='2m_temperature'):
    assert variable in VARNAMES

    os.makedirs(data_dir, exist_ok=True)

    fp = os.path.join(data_dir, f'era5_{variable}.nc')

    if not os.path.isfile(fp):
        cds = cdsapi.Client()

        config = {'variable': variable,
                  'format': 'netcdf',
                  'product_type': 'reanalysis'}

        # months = [f'{(m + 1):02}' for m in range(12)]
        months = ['01']
        # days = [f'{(d + 1):02}' for d in range(31)]
        days = [f'{(d + 1):02}' for d in range(7)]
        time = [f'{h:02}:00' for h in range(24)]
        bounds = [72, -25, 34, 45]
        grid_res = 1
        resolution = [grid_res, grid_res]

        info = { 'year' : 2020,
                 'month' : months,
                 'day' : days,
                 'area': bounds,
                 'grid' : resolution,
                 'time' : time }

        config.update(info)
        cds.retrieve('reanalysis-era5-single-levels', config, fp)

    # load .nc file and extract relevant information
    data = xr.open_dataset(fp)
    data = data.rio.write_crs('EPSG:4326')  # set crs to lat lon
    y = data[VARNAMES[variable]]
    lonlat2local(data.longitude, data.latitude)

    # create spherical mesh from given latlon points
    #vertices_lat = np.radians(latlondeg.T[0])
    #vertices_lon = np.radians(latlondeg.T[1])

    #spherical_triangulation = stripy.sTriangulation(lons=vertices_lon, lats=vertices_lat)

    # or first create spherical mesh and then use it to index dataset
    #spherical_triangulation = stripy.spherical_meshes.icosahedral_mesh(refinement_levels=0)



# TODO: in the end data set should provide
#  - pos
#  - y (measurements)
#  - features (relevant covariates)


# Turn everything into pytorch tensors
# pos = torch.tensor(pos, dtype=torch.float32)
# y = torch.tensor(y, dtype=torch.float32)
#
# # Generate graphs
# print("Generating graph ...")
# point_data = ptg.data.Data(pos=pos)
#
# if args.graph_alg == "delaunay":
#     graph_transforms = ptg.transforms.Compose((
#         ptg.transforms.Delaunay(),
#         ptg.transforms.FaceToEdge(),
#     ))
#     graph_y = graph_transforms(point_data)
#     full_ds_name = args.dataset + "_delaunay"
#
# if len(y.shape) == 1:
#     # Make sure y tensor has 2 dimensions
#     y = y.unsqueeze(1)
# graph_y.x = y
#
#
# # Check if graph is connected or contains isolated components
# nx_graph = ptg.utils.to_networkx(graph_y, to_undirected=True)
# n_components = nx.number_connected_components(nx_graph)
# contains_iso = ptg.utils.contains_isolated_nodes(graph_y.edge_index, graph_y.num_nodes)
# print("Graph connected: {}, n_components: {}".format((n_components == 1), n_components))
# print("Contains isolated components (1-degree nodes): {}".format(contains_iso))
#
# # Create Mask
# if args.random_mask:
#     n_mask = int(args.mask_fraction*graph_y.num_nodes)
#     unobs_indexes = torch.randperm(graph_y.num_nodes)[:n_mask]
#
#     unobs_mask = torch.zeros(graph_y.num_nodes).to(bool)
#     unobs_mask[unobs_indexes] = True
# else:
#     assert not utils.is_none(mask_limits), "No mask limits exists for dataset"
#     unobs_masks = torch.stack([torch.bitwise_and(
#         (graph_y.pos >= limits[0]),
#         (graph_y.pos < limits[1])
#         ) for limits in mask_limits], dim=0) # Shape (n_masks, n_nodes, 2)
#
#     unobs_mask = torch.any(torch.all(unobs_masks, dim=2), dim=0) # Shape (n_nodes,)
#
#     graph_y.mask_limits = mask_limits
#
# obs_mask = unobs_mask.bitwise_not()
#
# n_masked = torch.sum(unobs_mask)
# print("Masked {} / {} nodes".format(n_masked, graph_y.num_nodes))
#
# graph_y.mask = obs_mask
#
# if args.plot:
#     vis.plot_graph(graph_y, "y", show=True, title="y")
#
# # Additional computation if weighting by node distances
# if args.dist_weight:
#     utils.dist_weight_graph(graph_y, args.dist_weight_eps,
#             compute_eigvals=bool(args.compute_eigvals))
#
# # Save dataset
# print("Saving graphs ...")
# if args.random_mask:
#     full_ds_name += "_random_{}".format(args.mask_fraction)
#
# utils.save_graph_ds({"graph_y": graph_y}, args, full_ds_name)
#
