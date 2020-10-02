import numpy as np
import logging
import pyproj
import rasterio
import salem
from affine import Affine
from salem import wgs84
from collections import defaultdict
from k_tools.misc import _get_flowline_lonlat
# Module logger
log = logging.getLogger(__name__)


def open_vel_raster(tiff_path):
    """
    Opens a tiff file from Greenland velocity observations
    and calculates a raster of velocities or uncertainties with the
    corresponding color bar
    :param
        tiff_path: path to the data
    :return
        ds: xarray object with data already scaled
    """

    # Processing vel data
    src = rasterio.open(tiff_path)

    # Retrieve the affine transformation
    if isinstance(src.transform, Affine):
        transform = src.transform
    else:
        transform = src.affine

    dy = transform.e

    ds = salem.open_xr_dataset(tiff_path)

    data = ds.data.values

    # Read the image data, flip upside down if necessary
    data_in = data
    if dy < 0:
        data_in = np.flip(data_in, 0)

    # Scale the velocities by the log of the data.
    d = np.log(np.clip(data_in, 1, 3000))
    data_scale = (255 * (d - np.amin(d)) / np.ptp(d)).astype(np.uint8)

    ds.data.values = np.flip(data_scale, 0)

    return ds


def crop_vel_data_to_glacier_grid(gdir, vel, error):
    """
    Crop velocity data and uncertainty to the glacier grid
    for plotting only!
    :param
        gdir: Glacier Directory
        vel: xarray data containing vel or vel errors from
                the whole Greenland
        error: xarray data containing the errors from
                the whole Greenland
    :return
        ds_array: an array of velocity croped to the glacier grid
        dr_array: an array of velocity erros croped to the glacier grid
    """

    # Crop to glacier grid
    ds_glacier = vel.salem.subset(grid=gdir.grid, margin=2)
    dr_glacier = error.salem.subset(grid=gdir.grid, margin=2)

    return ds_glacier, dr_glacier


def crop_vel_data_to_flowline(vel, error, shp):
    """
    Crop velocity data and uncertainty to the glacier flowlines
    :param
        vel: xarray data containing vel or vel errors from
             the whole Greenland
        error: xarray data containing the errors from
               the whole Greenland
        shp: Shape file containing the glacier flowlines
    :return
        ds_array: an array of velocity croped to the glacier main flowline .
        dr_array: an array of velocity erros croped to the glacier
                  main flowline.
    """

    # Crop to flowline
    ds_fls = vel.salem.roi(shape=shp.iloc[[0]])
    dr_fls = error.salem.roi(shape=shp.iloc[[0]])

    return ds_fls, dr_fls


def calculate_observation_vel(gdir, ds_fls, dr_fls):
    """
    Calculates the mean velocity and error at the end of the flowline
    exactly 5 pixels upstream of the last part of the glacier that contains
    a velocity measurements
    :param
        gdir: Glacier directory
        ds_flowline: xarray data containing vel observations from the main
                     lowline
        dr_flowline: xarray data containing errors in vel observations from
                     the main flowline
    :return
        ds_mean: a mean velocity value over the last parts of the flowline.
        dr_mean: a mean error of the velocity over the last parts of the
                 main flowline.
    """

    coords = _get_flowline_lonlat(gdir)

    x, y = coords[0].geometry[3].coords.xy

    # We only want one third of the main centerline! kind of the end of the
    # glacier. For very long glaciers this might not be that ideal

    x_2 = x[-np.int(len(x) / 3):]
    y_2 = y[-np.int(len(x) / 3):]

    raster_proj = pyproj.Proj(ds_fls.attrs['pyproj_srs'])

    # For the entire flowline
    x_all, y_all = salem.gis.transform_proj(wgs84, raster_proj, x, y)
    vel_fls = ds_fls.interp(x=x_all, y=y_all, method='nearest')
    err_fls = dr_fls.interp(x=x_all, y=y_all, method='nearest')
    # Calculating means
    ds_mean = vel_fls.mean(skipna=True).data.values
    dr_mean = err_fls.mean(skipna=True).data.values

    # For the end of the glacier
    x_end, y_end = salem.gis.transform_proj(wgs84, raster_proj, x_2, y_2)
    vel_end = ds_fls.interp(x=x_end, y=y_end, method='nearest')
    err_end = dr_fls.interp(x=x_end, y=y_end, method='nearest')
    # Calculating means
    ds_mean_end = vel_end.mean(skipna=True).data.values
    dr_mean_end = err_end.mean(skipna=True).data.values

    vel_fls_all = np.around(ds_mean, decimals=2)
    err_fls_all = np.around(dr_mean, decimals=2)
    vel_fls_end = np.around(ds_mean_end, decimals=2)
    err_fls_end = np.around(dr_mean_end, decimals=2)

    return vel_fls_all, err_fls_all, vel_fls_end, err_fls_end, len(x)


def calculate_model_vel(gdir, filesuffix=''):
    """ Calculates the average velocity along the main flowline
    in different parts and at the last one third region upstream
    of the calving front
    :param
        gdir: Glacier directory
        filesuffix: any string to be added to the file name
    :return
        surf_fls_vel: surface velocity velocity along all the flowline (m/yr)
        cross_fls_vel: velocity along all the flowline (m/yr)
        surf_calving_front: surface velocity at the calving front (m/yr)
        cross_final: cross-section velocity at the calving front (m/yr)
    """

    if filesuffix is None:
        vel = gdir.read_pickle('inversion_output')[-1]
    else:
        vel = gdir.read_pickle('inversion_output', filesuffix=filesuffix)[-1]

    vel_surf_data = vel['u_surface']
    vel_cross_data = vel['u_integrated']

    length_fls = len(vel_surf_data)/3

    surf_fls_vel = np.nanmean(vel_surf_data)
    cross_fls_vel = np.nanmean(vel_cross_data)

    surf_calving_front = np.nanmean(vel_surf_data[-np.int(length_fls):])
    cross_final = np.nanmean(vel_cross_data[-np.int(length_fls):])

    return surf_fls_vel, cross_fls_vel, surf_calving_front, cross_final


def find_k_values_within_vel_range(df_oggm, df_vel):
    """
    Finds all k values and OGGM velocity data that is within range of the
    velocity observation and its error. In the case that no OGGM vel is within
    range flags if OGGM overestimates or underestimates velocities.
    :param
        df_oggm: OGGM data from k sensitivity experiment
        df_vel: observations from MEaSUREs v.1.0
    :return
        out: dictionary with the OGGM data frame crop to observations values or
             with a flag in case there is over estimation or under estimation
    """

    obs_vel = df_vel.vel_calving_front.values
    error_vel = df_vel.error_calving_front.values

    r_tol = df_vel.rel_tol_calving_front.values

    first_oggm_value = df_oggm.iloc[0].velocity_surf
    last_oggm_value = df_oggm.iloc[-1].velocity_surf

    low_lim = obs_vel - error_vel
    up_lim = obs_vel + error_vel

    index = df_oggm.index[np.isclose(df_oggm.velocity_surf,
                                     obs_vel,
                                     rtol=r_tol,
                                     atol=0)].tolist()
    if not index and (last_oggm_value < low_lim):
        df_oggm_new = df_oggm.iloc[-1]
        message = 'OGGM underestimates velocity'
    elif not index and (first_oggm_value > up_lim):
        df_oggm_new = df_oggm.iloc[0]
        message = 'OGGM overestimates velocity'
    else:
        df_oggm_new = df_oggm.loc[index]
        mu_stars = df_oggm_new.mu_star
        if mu_stars.iloc[-1] == 0:
            df_oggm_new = df_oggm_new.iloc[-2]
            message = 'OGGM is within range but mu_star does not allows more calving'
        else:
            df_oggm_new = df_oggm_new
            message = 'OGGM is within range'

    out = defaultdict(list)
    out['oggm_vel'].append(df_oggm_new)
    out['vel_message'].append(message)
    out['obs_vel'].append(df_vel)
    out['low_lim_vel'].append(low_lim)
    out['up_lim_vel'].append(up_lim)

    return out