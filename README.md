# OGGM k calibration

This repository contains the scripts used to calibrate a frontal ablation parameterization in [OGGM](https://docs.oggm.org/en/latest/) applied to Greenland peripheral calving glaciers and produce the results of the paper submitted to Journal of Glaciology. *Recinos, B. et al (submitted)*

This repository uses OGGMv1.2.0 pinned to the following [commit](https://github.com/OGGM/oggm/commit/d13b4438c6f0be2266cafb1ba21aa526eef93c14).

The content of the repository is the following:

**I. cryo_cluster_scripts** (scripts used in a cluster environment)

**II. plotting_sripts** (scripts used for plotting results)

**Please read the top of the scripts** to know more about the output of each model run and look inside each function for its description. The documentation and calibration method is still under constant development. 

## External libraries that are not included with OGGM conda env   
**III. velocity_tools** (calibration toolbox)
The velocity_tools is a collection of scripts to read in velocity observations from [MEaSUREs v1.0](https://nsidc.org/data/NSIDC-0670/versions/1) and [RACMO2.3p2 statistically downscaled to 1 km resolution](https://tc.copernicus.org/articles/10/2361/2016/), re-project the data in OGGM and calibrate the frontal ablation parametrization in the model.

[![DOI](https://zenodo.org/badge/249556625.svg)](https://zenodo.org/badge/latestdoi/249556625)



