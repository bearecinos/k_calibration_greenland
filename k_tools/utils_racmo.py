#  Tools to procress with RACMO
import pandas as pd
import numpy as np
import logging
import geopandas as gpd
import math
import os
import pyproj
import pickle
import xarray as xr
import rasterio
import matplotlib.colors as colors
import salem
from affine import Affine
from salem import wgs84
from shapely.ops import transform as shp_trafo
from functools import partial, wraps
from collections import OrderedDict
from oggm import cfg, entity_task, workflow
from oggm.utils._workflow import ncDataset, _get_centerline_lonlat

# Module logger
log = logging.getLogger(__name__)


def calving_flux_km3yr(gdir, smb):
    """Calving mass-loss from specific MB equivalent.
    to Km^3/yr. This is necessary to use RACMO to find k values.
    :param
    gdir: Glacier Directory
    smb: Surface Mass balance from RACMO in  mm. w.e a-1
    :returns
    q_calving: smb converted to calving flux
    """
    if not gdir.is_tidewater:
        return 0.
    # Original units: mm. w.e a-1, to change to km3 a-1 (units of specific MB)
    rho = cfg.PARAMS['ice_density']
    q_calving = (smb * gdir.rgi_area_m2) / (1e9 * rho)

    return q_calving


def open_racmo(netcdf_path, netcdf_mask_path=None):
    """Opens a netcdf from RACMO with a format PROJ (x, y, projection)
    and DATUM (lon, lat, time)

     Parameters:
    ------------
    netcdf_path: path to the data
    netcdf_mask_path: Must be given when openning SMB data else needs to be
    None.
    :returns
        out: xarray object with projection and coordinates in order
     """

    # RACMO open varaible file
    ds = xr.open_dataset(netcdf_path, decode_times=False)

    if netcdf_mask_path is not None:
        # open RACMO mask
        ds_geo = xr.open_dataset(netcdf_mask_path, decode_times=False)

        try:
            ds['x'] = ds_geo['x']
            ds['y'] = ds_geo['y']
            ds_geo.close()
        except KeyError as e:
            pass

    # Add the proj info to all variables
    proj = pyproj.Proj('EPSG:3413')
    ds.attrs['pyproj_srs'] = proj.srs
    for v in ds.variables:
        ds[v].attrs['pyproj_srs'] = proj.srs

    # Fix the time stamp
    ds['time'] = np.append(
        pd.period_range(start='2018.01.01', end='2018.12.01',
                        freq='M').to_timestamp(),
        pd.period_range(start='1958.01.01', end='2017.12.01',
                        freq='M').to_timestamp())

    out = ds
    ds.close()

    return out


def crop_racmo_to_glacier_grid(gdir, ds):
    """ Crops the RACMO data to the glacier grid

    Parameters
    -----------
    gdir: :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    ds: xarray object
    :returns
        ds_sel_roi: xarray with the data cropt to the glacier outline """
    try:
        ds_sel = ds.salem.subset(grid=gdir.grid, margin=2)
    except ValueError:
        ds_sel = None

    if ds_sel is None:
        ds_sel_roi = None
    else:
        ds_sel = ds_sel.load().sortby('time')
        ds_sel_roi = ds_sel.salem.roi(shape=gdir.read_shapefile('outlines'))

    return ds_sel_roi


def get_racmo_time_series(ds_sel_roi, var_name,
                          dim_one, dim_two, dim_three,
                          time_start=None, time_end=None):
    """ Generates RACMO time series for a 31 year period centered
    in a t-star year, with the data already cropped to the glacier outline
    Parameters
    ------------
    ds_sel_roi: xarray obj already cropped to the glacier outline
    var_name: ndarray the variable name to extract the time series
    dim_one : 'x' or 'lon'
    dim_two: 'y' or 'lat'
    dim_three: 'time'
    time_start: a year where the RACMO time series should begin
    time_end: a year where the RACMO time series should end

    :returns
    ts_31yr: xarray object with a time series of the RACMO variable, monthly
    data for a 31 reference period according to X. Fettweis et al. 2017 and
    Eric Rignot and Pannir Kanagaratnam, 2006.
    """
    if ds_sel_roi is None:
        ts_31 = None
    elif ds_sel_roi[var_name].isnull().all():
        #print('the glacier has no RACMO data')
        ts_31 = None
    else:
        ts = ds_sel_roi[var_name].mean(dim=[dim_one, dim_two],
                        skipna=True).resample(time='AS').mean(dim=dim_three,
                                                              skipna=True)

        if time_start is None:
            ts_31 = ts
        else:
            ts_31 = ts.sel(time=slice(str(time_start), str(time_end)))

    return ts_31


def get_racmo_std_from_moving_avg(ds_sel_roi,
                                  var_name,
                                  dim_one,
                                  dim_two):
    """ Generates RACMO time series for yearly averages and computes
    the glaciers std for RACMO smb.
    Parameters
    ------------
    ds_sel_roi: xarray obj already cropped to the glacier outline
    var_name: ndarray the variable name to extract the time series
    dim_one : 'x' or 'lon'
    dim_two: 'y' or 'lat'

    :returns
    std: standard deviation of smb data cropped to the glacier outline
    """
    if ds_sel_roi is None:
        std = None
    elif ds_sel_roi[var_name].isnull().all():
        #print('the glacier has no RACMO data')
        std = None
    else:
        ts = ds_sel_roi[var_name].mean(dim=[dim_one, dim_two], skipna=True)
        mean_yr = ts.rolling(time=12).mean()

        std = mean_yr.std()

    return std


def process_racmo_data(gdir, racmo_path,
                       time_start=None, time_end=None):
    """Processes and writes RACMO data in each glacier directory.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    racmo_path: the main path to the RACMO data

    :returns
    writes an nc file in each glacier directory with the RACMO data
    time series of SMB, precipitation, run off and melt for a 31 reference
    period according to X. Fettweis et al. 2017,
     Eric Rignot and Pannir Kanagaratnam, 2006.
    """
    mask_file = os.path.join(racmo_path,
                             'Icemask_Topo_Iceclasses_lon_lat_average_1km.nc')

    smb_file = os.path.join(racmo_path,
                    'smb_rec.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    prcp_file = os.path.join(racmo_path,
                        'precip.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    run_off_file = os.path.join(racmo_path,
                        'runoff.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    melt_file = os.path.join(racmo_path,
                        'snowmelt.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    fall_file = os.path.join(racmo_path,
                        'snowfall.1958-2018.BN_RACMO2.3p2_FGRN055_GrIS.MM.nc')

    # Get files as xarray all units mm. w.e.
    # Surface Mass Balance
    ds_smb = open_racmo(smb_file, mask_file)
    # Total precipitation: solid + Liquid
    ds_prcp = open_racmo(prcp_file)
    # Run off
    ds_run_off = open_racmo(run_off_file)
    # water that result from snow and ice melting
    ds_melt = open_racmo(melt_file)
    # Solid precipitation
    ds_fall = open_racmo(fall_file)

    # crop the data to glacier outline
    smb_sel = crop_racmo_to_glacier_grid(gdir, ds_smb)
    prcp_sel = crop_racmo_to_glacier_grid(gdir, ds_prcp)
    run_off_sel = crop_racmo_to_glacier_grid(gdir, ds_run_off)
    melt_sel = crop_racmo_to_glacier_grid(gdir, ds_melt)
    fall_sel = crop_racmo_to_glacier_grid(gdir, ds_fall)

    # get RACMO time series in 31 year period centered in t*
    smb_31 = get_racmo_time_series(smb_sel,
                                   var_name='SMB_rec',
                                   dim_one='x',
                                   dim_two='y',
                                   dim_three='time',
                                   time_start=time_start, time_end=time_end)

    smb_std = get_racmo_std_from_moving_avg(smb_sel,
                                            var_name='SMB_rec',
                                            dim_one='x',
                                            dim_two='y')

    prcp_31 = get_racmo_time_series(prcp_sel,
                                    var_name='precipcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end)

    run_off_31 = get_racmo_time_series(run_off_sel,
                                       var_name='runoffcorr',
                                       dim_one='lon',
                                       dim_two='lat',
                                       dim_three='time',
                                       time_start=time_start, time_end=time_end)

    melt_31 = get_racmo_time_series(melt_sel,
                                    var_name='snowmeltcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end)

    fall_31 = get_racmo_time_series(fall_sel,
                                    var_name='snowfallcorr',
                                    dim_one='lon',
                                    dim_two='lat',
                                    dim_three='time',
                                    time_start=time_start, time_end=time_end)

    fpath = gdir.dir + '/racmo_data.nc'
    if os.path.exists(fpath):
        os.remove(fpath)

    if smb_31 is None:
        return print('There is no RACMO file for this glacier ' + gdir.rgi_id)
    else:
        with ncDataset(fpath,
                       'w', format='NETCDF4') as nc:

            nc.createDimension('time', None)

            nc.author = 'B.M Recinos'
            nc.author_info = 'Open Global Glacier Model'

            timev = nc.createVariable('time', 'i4', ('time',))

            tatts = {'units': 'year'}

            calendar = 'standard'

            tatts['calendar'] = calendar

            timev.setncatts(tatts)
            timev[:] = smb_31.time

            v = nc.createVariable('smb', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'surface mass balance'
            v[:] = smb_31

            v = nc.createVariable('std', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'std surface mass balance'
            v[:] = smb_std

            v = nc.createVariable('prcp', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly precipitation amount'
            v[:] = prcp_31

            v = nc.createVariable('run_off', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly run off amount'
            v[:] = run_off_31

            v = nc.createVariable('snow_melt', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly snowmelt amount'
            v[:] = melt_31

            v = nc.createVariable('snow_fall', 'f4', ('time',))
            v.units = 'mm w.e.'
            v.long_name = 'total yearly snowfall amount'
            v[:] = fall_31


def get_smb31_from_glacier(gdir):
    """ Reads RACMO data and takes a mean over 31 year period of the
        surface mass balance

    Parameters
    ------------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    :returns
    smb_31 mean value in km^3/yr
    """
    fpath = gdir.dir + '/racmo_data.nc'

    if os.path.exists(fpath):
        with ncDataset(fpath, mode='r') as nc:
            smb = nc.variables['smb'][:]
            std = nc.variables['std'][:]
            if smb.all() == 0:
                smb_mean = None
                smb_std = None
                smb_cum = None
                smb_calving_mean = None
                smb_calving_std = None
                smb_calving_cum = None
                print('This glacier has no racmo data ' + gdir.rgi_id)
            else:
                smb_mean = np.nanmean(smb)
                smb_std = np.nanmean(std)
                smb_cum = np.nansum(smb)
                smb_calving_mean = calving_flux_km3yr(gdir, smb_mean)
                smb_calving_std = calving_flux_km3yr(gdir, smb_std)
                smb_calving_cum = calving_flux_km3yr(gdir, smb_cum)
    else:
        print('This glacier has no racmo data ' + gdir.rgi_id)
        smb_mean = None
        smb_std = None
        smb_cum = None
        smb_calving_mean = None
        smb_calving_std = None
        smb_calving_cum = None

    out_dic = dict(smb_mean = smb_mean,
                   smb_std = smb_std,
                   smb_cum = smb_cum,
                   smb_calving_mean = smb_calving_mean,
                   smb_calving_std = smb_calving_std,
                   smb_calving_cum = smb_calving_cum)

    return out_dic


def get_mu_star_from_glacier(gdir):
    """ Reads RACMO data and calculates the mean temperature sensitivity
    from RACMO SMB data and snow melt estimates. In a glacier wide average and
    a mean value of the entire RACMO time series. Based on the method described
    in Oerlemans, J., and Reichert, B. (2000).

        Parameters
        ------------
        gdir : :py:class:`oggm.GlacierDirectory`
            the glacier directory to process
        :returns
        Mu_star_racmo mean value in mm.w.e /K-1
        """
    fpath = gdir.dir + '/racmo_data.nc'

    if os.path.exists(fpath):
        with ncDataset(fpath, mode='r') as nc:
            smb = nc.variables['smb'][:]
            melt = nc.variables['snow_melt'][:]
            mu = smb / melt
            mean_mu = np.average(mu, weights=np.repeat(gdir.rgi_area_km2,
                                                       len(mu)))
    else:
        print('This glacier has no racmo data ' + gdir.rgi_id)
        mean_mu = None

    return mean_mu


# TODO: this function needs a lot of work still! we need to be able to tell
# the code what to do with different outcomes
def k_calibration_with_racmo(df_oggm,
                             df_racmo):

    rtol = df_racmo['q_calving_RACMO_mean_std'] / df_racmo['q_calving_RACMO_mean']
    racmo_flux = df_racmo['q_calving_RACMO_mean'].values
    racmo_flux_std = df_racmo['q_calving_RACMO_mean_std'].values

    if rtol is None:
        tol = 0.001
    else:
        tol = rtol

    if racmo_flux-racmo_flux_std <= 0:
        k_value = 0
        mu_star = 0
        u_cross = 0
        u_surf = 0
        calving_flux = 0
        racmo_flux = racmo_flux
        racmo_flux_std = racmo_flux_std
        rtol = tol
    else:
        oggm_values = df_oggm['calving_flux'].values

        if oggm_values[0] > racmo_flux+racmo_flux_std:
            index = 0
            df_oggm_new = df_oggm.loc[index]
            k_value = df_oggm_new['k_values']
            mu_star = df_oggm_new['mu_star']
            u_cross = df_oggm_new['velocity_cross']
            u_surf = df_oggm_new['velocity_surf']
            calving_flux = df_oggm_new['calving_flux']
            racmo_flux = racmo_flux
            racmo_flux_std = racmo_flux_std
            rtol = tol
            message = 'smallest k possible'
            print(message, df_racmo['RGI_ID'])
        elif oggm_values[-1] < racmo_flux-racmo_flux_std:
            index = -1
            df_oggm_new = df_oggm.loc[index]
            k_value = df_oggm_new['k_values']
            mu_star = df_oggm_new['mu_star']
            u_cross = df_oggm_new['velocity_cross']
            u_surf = df_oggm_new['velocity_surf']
            calving_flux = df_oggm_new['calving_flux']
            racmo_flux = racmo_flux
            racmo_flux_std = racmo_flux_std
            rtol = tol
            message = 'largest k possible'
            print(message, df_racmo['RGI_ID'])
        else:
            index = df_oggm.index[np.isclose(df_oggm['calving_flux'],
                                             racmo_flux,
                                             rtol=tol, atol=0)].tolist()
            # print(index)
            df_oggm_new = df_oggm.loc[index]

            k_value = np.mean(df_oggm_new['k_values'])
            mu_star = np.mean(df_oggm_new['mu_star'])
            u_cross = np.mean(df_oggm_new['velocity_cross'])
            u_surf = np.mean(df_oggm_new['velocity_surf'])
            calving_flux = np.mean(df_oggm_new['calving_flux'])
            racmo_flux = racmo_flux
            racmo_flux_std = racmo_flux_std
            rtol = tol
            message = 'closest k to racmo data within a tolerance'
            print(message, df_racmo['RGI_ID'])

    out_dic = dict(k_value=k_value,
                   mu_star=mu_star,
                   u_cross=u_cross,
                   u_surf=u_surf,
                   calving_flux=np.around(calving_flux, decimals=5),
                   racmo_flux=np.around(racmo_flux, decimals=5),
                   racmo_flux_std=np.around(racmo_flux_std, decimals=5),
                   rtol=rtol,
                   message=message)

    return out_dic


def calculate_PDM(gdir):
    """Calculates the Positive degree month sum; is the total sum,
    of monthly averages temperatures above 0°C in a 31 yr period
    centered in t_star year and with a reference height at the free board.

    Parameters
    ----------
    gidr : Glacier directory
    """
    # First we get the years to analise
    # Parameters
    temp_all_solid = cfg.PARAMS['temp_all_solid']
    temp_all_liq = cfg.PARAMS['temp_all_liq']
    temp_melt = cfg.PARAMS['temp_melt']
    prcp_fac = cfg.PARAMS['prcp_scaling_factor']
    default_grad = cfg.PARAMS['temp_default_gradient']

    df = gdir.read_json('local_mustar')
    tstar = df['t_star']
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    # Year range
    yr = [tstar - mu_hp, tstar + mu_hp]

    #Then the heights
    heights = gdir.get_inversion_flowline_hw()[0]

    #Then the climate data
    ds = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    # We only select the years tha we need
    new_ds = ds.sel(time=slice(str(yr[0])+'-01-01',
                               str(yr[1])+'-12-31'))
    # we make it a data frame
    df = new_ds.to_dataframe()

    # We create the new matrix
    igrad = df.temp * 0 + cfg.PARAMS['temp_default_gradient']
    iprcp = df.prcp
    iprcp *= prcp_fac

    npix = len(heights)

    # We now estimate the temperature gradient
    grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
    grad_temp *= (heights.repeat(len(df.index)).reshape(
        grad_temp.shape) - new_ds.ref_hgt)
    temp2d = np.atleast_2d(df.temp).repeat(npix, 0) + grad_temp

    # Precipitation
    prcpsol = np.atleast_2d(iprcp).repeat(npix, 0)
    fac = 1 - (temp2d - temp_all_solid) / (temp_all_liq - temp_all_solid)
    fac = np.clip(fac, 0, 1)
    prcpsol = prcpsol * fac

    data_temp = pd.DataFrame(temp2d,
                             columns=[df.index],
                             index=heights)
    data_prcp = pd.DataFrame(prcpsol,
                             columns=[df.index],
                             index=heights)

    temp_free_board = data_temp.iloc[-1]
    solid_prcp_top = data_prcp.iloc[0].sum()

    PDM_temp = temp_free_board[temp_free_board > 0].sum()
    PDM_number = temp_free_board[temp_free_board > 0].count()

    return PDM_temp, PDM_number, solid_prcp_top