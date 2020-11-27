import os
import sys
import numpy as np
import glob
import pandas as pd
from configobj import ConfigObj

MAIN_PATH = os.path.expanduser('~/k_calibration_greenland/')
sys.path.append(MAIN_PATH)

from k_tools import utils_velocity as utils_vel
from k_tools import utils_racmo as utils_racmo
from k_tools import misc

config = ConfigObj(os.path.join(MAIN_PATH, 'config.ini'))

# Were to store merged data
output_path= os.path.join(MAIN_PATH, 'output_data/13_Merged_data')
if not os.path.exists(output_path):
    os.makedirs(output_path)


# Reading glacier directories per experiment
exp_dir_path = os.path.join(MAIN_PATH, config['volume_results'])

config_paths = pd.read_csv(os.path.join(MAIN_PATH,
                                        config['configuration_names']))

full_config_paths = []
for configuration in config_paths.config_path:
    full_config_paths.append(os.path.join(MAIN_PATH, exp_dir_path,
                                          configuration + '/'))

# Merge velocity data with calibration results
for path, results in zip(full_config_paths[0:6],
                         config_paths.results_output[0:6]):

    experimet_name = misc.splitall(path)[-2]

    oggm_file_path = os.path.join(path,
                                  'glacier_statisticscalving_' +
                                  experimet_name +
                                  '.csv')

    calibration_path = os.path.join(MAIN_PATH, config['linear_fit_to_data'],
                                    results + '.csv')

    df_merge = utils_vel.merge_vel_calibration_results_with_glac_stats(
        calibration_path,
        oggm_file_path,
        experimet_name)

    df_merge.to_csv(os.path.join(output_path,
                                 experimet_name+'_merge_results.csv'))

# Merge racmo data with calibration results
for path, results in zip(full_config_paths[-3:],
                         config_paths.results_output[-3:]):
    experimet_name = misc.splitall(path)[-2]

    oggm_file_path = os.path.join(path,
                                  'glacier_statisticscalving_' +
                                  experimet_name +
                                  '.csv')

    calibration_path = os.path.join(MAIN_PATH, config['linear_fit_to_data'],
                                    results + '.csv')

    df_merge = utils_racmo.merge_racmo_calibration_results_with_glac_stats(
        calibration_path,
        oggm_file_path,
        experimet_name)

    df_merge.to_csv(os.path.join(output_path,
                                 experimet_name + '_merge_results.csv'))


# Merge racmo data with velocity estimates after RACMO calibration
