import numpy as np
import logging
import geopandas as gpd
import pandas as pd
import os
import xarray as xr
import pickle
from salem import wgs84
from shapely.ops import transform as shp_trafo
from functools import partial
from collections import OrderedDict
from collections import defaultdict
from oggm import cfg
from scipy.stats import linregress

# Module logger
log = logging.getLogger(__name__)


def get_study_area(rgi, main_path, ice_cap_prepro_path):
    """
    Get study area sum
    :param
    RGI: RGI as a geopandas
    MAIN_PATH: repository path
    ice_cap_prepro_path: ice cap pre-processing to get ica cap areas
    :return
    study_area: Study area
    """
    rgidf = rgi.sort_values('RGIId', ascending=True)

    # Read Areas for the ice-cap computed in OGGM during
    # the pre-processing runs
    df_prepro_ic = pd.read_csv(os.path.join(main_path,
                                            ice_cap_prepro_path))
    df_prepro_ic = df_prepro_ic.sort_values('rgi_id', ascending=True)

    # Assign an area to the ice cap from OGGM to avoid errors
    rgidf.loc[rgidf['RGIId'].str.match('RGI60-05.10315'),
              'Area'] = df_prepro_ic.rgi_area_km2.values

    # Get rgi only for Lake Terminating and Marine Terminating
    glac_type = [0]
    keep_glactype = [(i not in glac_type) for i in rgidf.TermType]
    rgidf = rgidf.iloc[keep_glactype]

    # Get rgi only for glaciers that have a week connection or are
    # not connected to the ice-sheet
    connection = [2]
    keep_connection = [(i not in connection) for i in rgidf.Connect]
    rgidf = rgidf.iloc[keep_connection]

    study_area = rgidf.Area.sum()
    return study_area


def normalised(value):
    """Normalised value
        :params
        value : value to normalise
        :returns
        n_value: value normalised
        """
    value_min = min(value)
    value_max = max(value)
    n_value = (value - value_min) / (value_max - value_min)

    return n_value


def num_of_zeros(n):
  s = '{:.16f}'.format(n).split('.')[1]
  return len(s) - len(s.lstrip('0'))


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def calculate_volume_percentage(volume_one, volume_two):
    return np.around((volume_two * 100) / volume_one, 2) - 100


def calculate_sea_level_equivalent(value):
    """
    Calculates sea level equivalent of a volume
    of ice in km^3
    taken from: http://www.antarcticglaciers.org
    :param value: glacier volume
    :return: glacier volume in s.l.e
    """
    # Convert density of ice to Gt/km^3
    rho_ice = 900 * 1e-3  # Gt/km^3

    area_ocean = 3.618e8  # km^2
    height_ocean = 1e-6  # km (1 mm)

    # Volume of water required to raise global sea levels by 1 mm
    vol_water = area_ocean * height_ocean  # km^3 of water

    mass_ice = value * rho_ice  # Gt
    return mass_ice * (1 / vol_water)


def write_pickle_file(gdir, var, filename, filesuffix=''):
    """ Writes a variable to a pickle on disk.
    Parameters
    ----------
    gdir: Glacier directory
    var : object the variable to write to disk
    filename : str file name (must be listed in cfg.BASENAME)
    filesuffix : str append a suffix to the filename.
    """
    if filesuffix:
        filename = filename.split('.')
        assert len(filename) == 2
        filename = filename[0] + filesuffix + '.' + filename[1]

    fp = os.path.join(gdir.dir, filename)

    with open(fp, 'wb') as f:
        pickle.dump(var, f, protocol=-1)


def read_pickle_file(gdir, filename, filesuffix=''):
    """ Reads a variable to a pickle on disk.
    Parameters
    ----------
    gdir: Glacier directory
    filename : str file name
    filesuffix : str append a suffix to the filename.
    """
    if filesuffix:
        filename = filename.split('.')
        assert len(filename) == 2
        filename = filename[0] + filesuffix + '.' + filename[1]

    fp = os.path.join(gdir.dir, filename)

    with open(fp, 'rb') as f:
        out = pickle.load(f)

    return out


def area_percentage(gdir):
    """ Calculates the lowest 5% of the glacier area from the rgi area
    (this is used in the velocity estimation)
    :param gdir: Glacier directory
    :return: area percentage and the index along the main flowline array
    where that lowest 5% is located.
    """
    rgi_area = gdir.rgi_area_m2
    area_percent = 0.05 * rgi_area

    inv = gdir.read_pickle('inversion_output')[-1]

    # volume in m3 and dx in m
    section = inv['volume'] / inv['dx']

    # Find the index where the lowest 5% percent of the rgi area is located
    index = (np.cumsum(section) <= area_percent).argmin()

    return area_percent, index


def _get_flowline_lonlat(gdir):
    """Quick n dirty solution to write the flowlines as a shapefile"""

    cls = gdir.read_pickle('inversion_flowlines')
    olist = []
    for j, cl in enumerate(cls[::-1]):
        mm = 1 if j == 0 else 0
        gs = gpd.GeoSeries()
        gs['RGIID'] = gdir.rgi_id
        gs['LE_SEGMENT'] = np.rint(np.max(cl.dis_on_line) * gdir.grid.dx)
        gs['MAIN'] = mm
        tra_func = partial(gdir.grid.ij_to_crs, crs=wgs84)
        gs['geometry'] = shp_trafo(tra_func, cl.line)
        olist.append(gs)

    return olist


def write_flowlines_to_shape(gdir, filesuffix='', path=True):
    """Write the centerlines in a shapefile.

    Parameters
    ----------
    gdir: Glacier directory
    filesuffix : str add suffix to output file
    path:
        Set to "True" in order  to store the info in the working directory
        Set to a path to store the file to your chosen location
    """

    if path is True:
        path = os.path.join(cfg.PATHS['working_dir'],
                            'glacier_centerlines' + filesuffix + '.shp')

    olist = []

    olist.extend(_get_flowline_lonlat(gdir))

    odf = gpd.GeoDataFrame(olist)

    shema = dict()
    props = OrderedDict()
    props['RGIID'] = 'str:14'
    props['LE_SEGMENT'] = 'int:9'
    props['MAIN'] = 'int:9'
    shema['geometry'] = 'LineString'
    shema['properties'] = props

    crs = {'init': 'epsg:4326'}

    # some writing function from geopandas rep
    from shapely.geometry import mapping
    import fiona

    def feature(i, row):
        return {
            'id': str(i),
            'type': 'Feature',
            'properties':
                dict((k, v) for k, v in row.items() if k != 'geometry'),
            'geometry': mapping(row['geometry'])}

    with fiona.open(path, 'w', driver='ESRI Shapefile',
                    crs=crs, schema=shema) as c:
        for i, row in odf.iterrows():
            c.write(feature(i, row))


def calculate_PDM(gdir):
    """Calculates the Positive degree month sum; is the total sum,
    of monthly averages temperatures above 0°C in a 31 yr period
    centered in t_star year and with a reference height at the free board.

    Parameters
    ----------
    gdir: Glacier directory
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

    # Then the heights
    heights = gdir.get_inversion_flowline_hw()[0]

    # Then the climate data
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


def solve_linear_equation(a1, b1, a2, b2):
    """
    Solve linear equation

    Parameters
    ----------
        a1: Observation slope (either from the
                lower bound, value, upper bound)
        b1: Observation intercept. (either from the
                lower bound, value, upper bound)
        a2: Linear fit slope to the model data.
        b2: Linear fit intercept to the model data

    :returns
        Z: Intercepts (x, y) x will be k and y velocity.
    """

    A = np.array([[-a1, 1], [-a2, 1]], dtype='float')
    b = np.array([b1, b2], dtype='float')

    z = np.linalg.solve(A, b)
    return z
