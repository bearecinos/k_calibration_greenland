# Finds 3 k values per glacier that produce
# model surface velocities, within
# MEaSUREs data range.
# K values are found by finding the intercepts between linear equations
# fitted to model and observations values
import os
import sys
import numpy as np
from configobj import ConfigObj
import glob
import pickle
from scipy.stats import linregress
from scipy.optimize import fsolve


def f(xy, a1, b1, a2, b2):
    x, y = xy
    z = np.array([y - (a1*x) - b1, y - (a2*x) - b2])
    return z


MAIN_PATH = os.path.expanduser('~/k_calibration_greenland/')
sys.path.append(MAIN_PATH)

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Velocity module
from k_tools import utils_velocity as utils_vel
from k_tools import misc as misc

# Sort files
filenames = sorted(glob.glob(os.path.join(MAIN_PATH,
                                          config['vel_calibration_results'],
                                          "*.pkl")))

for file in filenames:
    with open(file, 'rb') as handle:
        base = os.path.basename(file)
        rgi_id = os.path.splitext(base)[0]
        g = pickle.load(handle)

        # Observations slope, intercept. Slope here is always zero
        slope_obs, intercept_obs = [0,
                                    g[rgi_id]['obs_vel'][
                                        0].vel_calving_front.iloc[0]]
        slope_lwl, intercept_lwl = [0, g[rgi_id]['low_lim_vel'][0][0]]
        slope_upl, intercept_upl = [0, g[rgi_id]['up_lim_vel'][0][0]]

        # Get linear fit for OGGM model data
        # If there is only one model value (k, vel) we add (0,0) intercept to
        # the line. Zero k and zero velocity is a valid solution
        # else we take all the points found in the first calibration step
        df_oggm = g[rgi_id]['oggm_vel'][0]
        if len(df_oggm.shape) < 2:
            df_oggm = df_oggm.to_frame().T
            df_oggm.loc[len(df_oggm)+1] = 0
            df_oggm = df_oggm.sort_values(by=['k_values'])
            df_oggm = df_oggm.reset_index(drop=True)
        else:
            df_oggm = df_oggm

        k_values = df_oggm.k_values.values
        velocities = df_oggm.velocity_surf.values

        slope, intercept, r_value, p_value, std_err = linregress(k_values,
                                                                 velocities)

        # TODO :complete this function fitting function!!!!

