# This will run OGGM and obtain velocity data from the MEaSUREs Multi-year
# Greenland Ice Sheet Velocity Mosaic, Version 1.
# It will give you velocity averages along the main centrelines
# with the respective uncertainty from the MEaSUREs tiff files
# Python imports
from __future__ import division
import os
import sys
import numpy as np
import geopandas as gpd
import pandas as pd
from configobj import ConfigObj
import time
import salem
import math
import xarray as xr
# Imports oggm
import oggm.cfg as cfg
from oggm import workflow
from oggm import tasks
from oggm.workflow import execute_entity_task
from oggm import utils
# Module logger
import logging
log = logging.getLogger(__name__)
# Time
start = time.time()

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# velocity module
from k_tools import utils_velocity as utils_vel
from k_tools import misc

# Regions: Greenland
rgi_region = '05'

# Initialize OGGM and set up the run parameters
# ---------------------------------------------
cfg.initialize()
rgi_version = '61'

SLURM_WORKDIR = os.environ["WORKDIR"]

# Local paths (where to write output and where to download input)
WORKING_DIR = SLURM_WORKDIR
cfg.PATHS['working_dir'] = WORKING_DIR

# Use multiprocessing
cfg.PARAMS['use_multiprocessing'] = True

# We make the border 20 so we can use the Columbia itmix DEM
cfg.PARAMS['border'] = 20
# Set to True for operational runs
cfg.PARAMS['continue_on_error'] = True
cfg.PARAMS['min_mu_star'] = 0.0
cfg.PARAMS['inversion_fs'] = 5.7e-20
cfg.PARAMS['use_tar_shapefiles'] = False
cfg.PARAMS['use_intersects'] = True
cfg.PARAMS['use_compression'] = False
cfg.PARAMS['compress_climate_netcdf'] = False

# We use intersects
path = utils.get_rgi_intersects_region_file(rgi_region, version=rgi_version)
cfg.set_intersects_db(path)

# RGI file
rgidf = gpd.read_file(os.path.join(MAIN_PATH, config['RGI_FILE']))
rgidf = rgidf.sort_values('RGIId', ascending=True)

# Read Areas for the ice-cap computed in OGGM during
# the pre-processing runs
df_prepro_ic = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['ice_cap_prepro']))
df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

# Assign an area to the ice cap from OGGM to avoid errors
rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
          'Area'] = df_prepro_ic.rgi_area_km2.values

# Pre-download other files which will be needed later
_ = utils.get_cru_file(var='tmp')
p = utils.get_cru_file(var='pre')
print('CRU file: ' + p)

# Run only for Lake Terminating and Marine Terminating
glac_type = [0]
keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
rgidf = rgidf.iloc[keep_glactype]

# Run only glaciers that have a week connection or are
# not connected to the ice-sheet
connection = [2]
keep_connection = [(i not in connection) for i in rgidf.Connect]
rgidf = rgidf.iloc[keep_connection]

# Exclude glaciers with prepro erros
de = pd.read_csv(os.path.join(MAIN_PATH, config['prepro_err']))
ids = de.RGIId.values
keep_errors = [(i not in ids) for i in rgidf.RGIId]
rgidf = rgidf.iloc[keep_errors]

# Make OGGM use ARTICDEM
rgidf['DEM_SOURCE'] = 'ARCTICDEM'

# for some glaciers there is no artic DEM we use gimp instead
df_gimp = pd.read_csv(os.path.join(MAIN_PATH, config['glaciers_gimp']))
rgidf.loc[rgidf['RGIId'].isin(df_gimp.RGIId.values), 'DEM_SOURCE'] = 'GIMP'

## Run a single id for testing
# glacier = ['RGI60-05.00304', 'RGI60-05.08443']
# keep_indexes = [(i in glacier) for i in rgidf.RGIId]
# rgidf = rgidf.iloc[keep_indexes]

# Sort for more efficient parallel computing
rgidf = rgidf.sort_values('Area', ascending=False)

log.info('Starting run for RGI reg: ' + rgi_region)
log.info('Number of glaciers: {}'.format(len(rgidf)))

# Go - initialize working directories
# -----------------------------------
gdirs = workflow.init_glacier_regions(rgidf)

execute_entity_task(tasks.glacier_masks, gdirs)

# Prepro tasks
task_list = [
    tasks.compute_centerlines,
    tasks.initialize_flowlines,
    tasks.catchment_area,
    tasks.catchment_intersections,
    tasks.catchment_width_geom,
    tasks.catchment_width_correction,
]
for task in task_list:
    execute_entity_task(task, gdirs)

# Log
m, s = divmod(time.time() - start, 60)
h, m = divmod(m, 60)
log.info("OGGM preprocessing finished! Time needed: %02d:%02d:%02d" %
         (h, m, s))

ids = []
vel_fls_avg = []
err_fls_avg = []
vel_calving_front = []
err_calving_front = []
rel_tol_fls = []
rel_tol_calving_front = []
length_fls = []

files_no_data = []

# We need the components of vel
fx = os.path.join(MAIN_PATH, config['vel_x_path'])
fy = os.path.join(MAIN_PATH, config['vel_y_path'])
# And Error
ex = os.path.join(MAIN_PATH, config['err_x_path'])
ey = os.path.join(MAIN_PATH, config['err_y_path'])

# Open the files with salem
dsx = salem.GeoTiff(fx)
dsy = salem.GeoTiff(fy)

dex = salem.GeoTiff(ex)
dey = salem.GeoTiff(ey)

for gdir in gdirs:

    # first we compute the centerlines as shapefile to crop the satellite
    # data
    misc.write_flowlines_to_shape(gdir, path=gdir.dir)
    shp_path = os.path.join(gdir.dir, 'RGI60-05.shp')
    shp = gpd.read_file(shp_path)

    utils_vel.its_live_to_gdir(gdir,
                               dsx=dsx, dsy=dsy,
                               dex=dex, dey=dey,
                               fx=fx)

    file_vel = xr.open_dataset(gdir.get_filepath('gridded_data'))
    proj = file_vel.attrs['proj_srs']

    vx = file_vel.obs_icevel_x
    vy = file_vel.obs_icevel_y
    dvel = np.sqrt(vx**2 + vy**2)

    dvel.attrs['pyproj_srs'] = proj

    ex = file_vel.obs_icevel_x_error
    ey = file_vel.obs_icevel_y_error
    derr = np.sqrt(ex**2 + ey**2)
    derr.attrs['pyproj_srs'] = proj

    # we crop the satellite data to the centerline shape file
    dvel_fls, derr_fls = utils_vel.crop_vel_data_to_flowline(dvel, derr, shp)

    out = utils_vel.calculate_itslive_vel(gdir, dvel_fls, derr_fls)

    if math.isfinite(out[2]) and math.isfinite(out[0]):
        ids = np.append(ids, gdir.rgi_id)
        vel_fls_avg = np.append(vel_fls_avg, out[0])
        err_fls_avg = np.append(err_fls_avg, out[1])

        rel_tol_fls = np.append(rel_tol_fls, out[1] / out[0])
        vel_calving_front = np.append(vel_calving_front, out[2])
        err_calving_front = np.append(err_calving_front, out[3])
        rel_tol_calving_front = np.append(rel_tol_calving_front,
                                          out[3] / out[2])
        length_fls = np.append(length_fls, out[4])
    else:
        print('There is no velocity data for this glacier')
        files_no_data = np.append(files_no_data, gdir.rgi_id)

d = {'RGIId': files_no_data}
df = pd.DataFrame(data=d)

df.to_csv(cfg.PATHS['working_dir'] + 'glaciers_with_no_velocity_data.csv')

dr = {'RGI_ID': ids,
      'vel_fls': vel_fls_avg,
      'error_fls': err_fls_avg,
      'rel_tol_fls': rel_tol_fls,
      'vel_calving_front': vel_calving_front,
      'error_calving_front': err_calving_front,
      'rel_tol_calving_front': rel_tol_calving_front}

df_r = pd.DataFrame(data=dr)
df_r.to_csv(cfg.PATHS['working_dir'] + '/velocity_observations.csv')
