import numpy as np
import logging
import geopandas as gpd
import os
import pickle
from salem import wgs84
from shapely.ops import transform as shp_trafo
from functools import partial
from collections import OrderedDict
from oggm import cfg

# Module logger
log = logging.getLogger(__name__)


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
