<<<<<<< HEAD
---
title: "README"
format: html
---

## Overview

This repo contains the implementation of using a Spatio-Temporal Graph Neural Network (ST-GNN) framework for prediction in wildfire occurrence in British Columbia, Canada. This study has integrated multiple environmental data sources, including ERA5 reanalysis data, terrain characteristics, MODIS NDVI indices, and land cover data, along with the historical fire points and official BC boundary from 2019 to 2024. All implementation was done in Python, and the paper was produced in LaTeX via Overleaf.


## File Structure

The repo was originally structured as:
`Data` contains the raw data, including:

- `Data/CNFDB` contains the raw historical fire points data as obtained from the Canadian National Fire Database (CNFDB).
- `Data/NDVI` contains the monthly NDVI indices for BC from 2019 to 2024 as obtained from NASA’s Land Processes Distributed Active Archive Center (LP DAAC) via Google Earth Engine.
- `Data/DEM` contains DEM data of the slope, aspect, and elevation characteristics of BC as obtained from NASA’s Jet Propulsion Laboratory via Google Earth Engine.
- `Data/LandCover` contains the 2022 land cover data for BC as obtained from NASA’s Land Processes Distributed Active Archive Center (LP DAAC) via Google Earth Engine.

`Processed Data` contains the processed data, which includes:

- `Processed Data/Fire Points` contains the historical fire points shape data explicitly for BC from 2019 to 2024, and fire occurrence at the grid-day level.
- `Processed Data/Weather` contains the ERA5 reanalysis data from 2019 to 2024 in BC on a 0.25-degree scale, was originally obtained from the European Centre for Medium-Range Weather Forecasts (ECMWF) and accessed via Google Earth Engine.
- `Processed Data/Boundary` contains the official administrative boundary shape file of British Columbia from the 2021 Census Boundary File provided by Statistics Canada.
- `Processed Data/Grid` contains the partitioned grids in BC on a 0.25-degree scale.
- `Processed Data/Rasters` contains the parquet files of processed ERA5 reanalysis data, MODIS NDVI indices, DEM data, and land cover data on a grid level.
- `Processed Data/grid_all.parquet` is the parquet file that combines all fire points occurrence, ERA5, DEM, NDVI, and land cover data on the grid-day level.
- `Processed Data/grid_all_features.parquet` is the parquet file that combines `grid_all.parquet` along with engineered features on the grid-day level.

`Scripts` contains the scripts for data processing, model implementation, and data visualization that include:

- `Scripts/01_PreProcess` contains the script and Quarto files for downloading and processing the raw data.
- `Scripts/02_PreParation` contains the script and Quarto files for data check and clean, extraction and labelling of the pre-processed data.
- `Scripts/03_Exploration` contains the Quarto file for performing EDA and integration on the processed data.
- `Scripts/04_Models` contains the scripts and Quartio files that perform ST-GNN modelling on the processed data.
- `Scripts/05_Results` contains the Quarto files that analyze and visualize the prediction results from ST-GNN modelling.

`Output` contains the output files, including figures and results that are produced by `Scripts`:

- `Output/variable` contains the pickle file of the neighbourhood dictionary of grid cells on the 0.25-degree scale.
- `Output/Grid` contains the visualization of the grid on the map and the summary stats of ERA5 data on each grid.
- `Output/Sample Run` contains the testing results of the ST-GNN model on the data.
- `Output/Model Run` contains the ST-GNN model prediction results across five model configurations.
- `Output/Figures` contains the output figures that are used in the final paper.


`Paper` contains the Final written paper and correspnding citation bibtex files.

Note: The `Data` and `Processed_Data` directories are not included in this repository, as several files exceed GitHub’s 100 MB file size limit.



=======
The Data and Processed Data folders are not included in this repository as their contained files are too large to be pushed to GitHub.
>>>>>>> e692800ce0875d205c299c2b4c1373c37a4a99af
