import numpy as np
import logging
import pyproj
import rasterio
import salem
from affine import Affine
from salem import wgs84
from k_tools.misc import _get_flowline_lonlat
# Module logger
log = logging.getLogger(__name__)


def open_vel_raster(tiff_path):
    """Opens a tiff file from Greenland velocity observations
     and calculates a raster of velocities or uncertainties with the
     corresponding color bar

     Parameters:
    ------------
    tiff_path: path to the data
    :returns
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

        :param:
            gdir: Glacier Directory
            vel: xarray data containing vel or vel errors from
                the whole Greenland
            error: xarray data containing the errors from
                the whole Greenland
        :return:
            ds_array: an array of velocity croped to the glacier grid
            dr_array: an array of velocity erros croped to the glacier
            grid
        """

    # Crop to glacier grid
    ds_glacier = vel.salem.subset(grid=gdir.grid, margin=2)
    dr_glacier = error.salem.subset(grid=gdir.grid, margin=2)

    return ds_glacier, dr_glacier


def crop_vel_data_to_flowline(vel, error, shp):
    """
    Crop velocity data and uncertainty to the glacier flowlines

    :param:
        vel: xarray data containing vel or vel errors from
            the whole Greenland
        error: xarray data containing the errors from
            the whole Greenland
        shp: Shape file containing the glacier flowlines
    :return:
        ds_array: an array of velocity croped to the glacier main flowline .
        dr_array: an array of velocity erros croped to the glacier
        main flowline.
    """

    # Crop to flowline
    ds_fls = vel.salem.roi(shape=shp.iloc[[0]])
    dr_fls = error.salem.roi(shape=shp.iloc[[0]])

    return ds_fls, dr_fls


def calculate_observation_vel_at_the_main_flowline(gdir,
                                                   ds_fls,
                                                   dr_fls):
    """
    Calculates the mean velocity and error at the end of the flowline
    exactly 5 pixels upstream of the last part of the glacier that contains
    a velocity measurements
    :param:
        gdir: Glacier directory
        ds_flowline: xarray data containing vel observations from the main
        flowline
        dr_flowline: xarray data containing errors in vel observations from
        the main flowline
    :return:
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
