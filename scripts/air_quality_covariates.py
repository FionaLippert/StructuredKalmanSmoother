import numpy as np
import pandas as pd
import os.path as osp
import argparse
from pvlib import solarposition

# from stdgmrf import utils
from stdgmrf.datasets import era5
# import utils_dgmrf

dir = '../datasets/AirQuality'
era5_dir = '../era5_data'
covariates = ['2m_temperature',
              '2m_dewpoint_temperature',
              'surface_pressure',
              'total_precipitation',
              '10m_u_component_of_wind',
              '10m_v_component_of_wind']

parser = argparse.ArgumentParser(description='Generate AirQuality dataset')

parser.add_argument("--year", type=int, default=2015, help="year to select")
parser.add_argument("--month", type=int, default=3, help="month to select")
parser.add_argument("--n_dummy_sensors", type=int, default=0, help="number of dummy sensors to include")

args = parser.parse_args()

if args.n_dummy_sensors > 0:
    df_sensors = pd.read_csv(osp.join(dir, f'sensors_ndummy={args.n_dummy_sensors}.csv'))
else:
    df_sensors = pd.read_csv(osp.join(dir, 'sensors.csv'))

# bounds has order [north, west, south, east]
bounds = [df_sensors.latitude.max() + 1, df_sensors.longitude.min() - 1,
          df_sensors.latitude.min() - 1, df_sensors.longitude.max() + 1]


# get covariates at all sensor locations for the given year and month
ds = era5.load(era5_dir, bounds, covariates, args.month, args.year)

# compute relative humidity based on temperature and dewpoint temperature in degrees Celsius
ds['rh'] = np.exp((ds['d2m'] - 273.15) * 17.625 / ((ds['d2m'] - 273.15 + 243.04))) / \
           np.exp((ds['t2m'] - 273.15) * 17.625 / ((ds['t2m'] - 273.15 + 243.04)))

all_covariates = {var : [] for var in (list(ds.data_vars) + ['solarpos', 'solarpos_dt', 'dayofyear'])}
var_sensors = []
timestamps = []
all_idx = []
for idx, row in df_sensors.iterrows():
    ds_sensor = ds.interp(longitude=row.longitude, latitude=row.latitude, method='linear')
    # print(ds_sensor)
    for var in ds_sensor.data_vars:
        var_ts = ds_sensor[var].data
        # print(var_ts)
        if np.isnan(var_ts).sum() > 0:
            # print(f'found {np.isnan(var_ts).sum()}/{len(var_ts)} NaNs for {var} and node {idx}')
            print(f'{var} node {idx} at {row.longitude}, {row.latitude}')
        all_covariates[var].append(var_ts)

    # add solarposition as variable
    dti = pd.DatetimeIndex(ds_sensor.time)
    t_range = dti.insert(-1, dti[-1] + pd.Timedelta(dti.freq))
    solarpos = np.array(solarposition.get_solarposition(t_range, row.latitude, row.longitude).elevation)
    all_covariates['solarpos_dt'].append(solarpos[1:] - solarpos[:-1])
    all_covariates['solarpos'].append(solarpos[:-1])
    all_covariates['dayofyear'].append(dti.dayofyear.values)

    timestamps.append(ds_sensor.time)
    all_idx.append([row.node_idx] * len(ds_sensor.time))

df_covariates = pd.DataFrame(dict(timestamp=np.concatenate(timestamps),
                                  node_idx=np.concatenate(all_idx)))
for var, values in all_covariates.items():
    df_covariates[var] = np.concatenate(values)

# save data frame
if args.n_dummy_sensors > 0:
    df_covariates.to_csv(osp.join(dir, f'covariates_{args.year}_{args.month}_ndummy={args.n_dummy_sensors}.csv'))
else:
    df_covariates.to_csv(osp.join(dir, f'covariates_{args.year}_{args.month}.csv'))
