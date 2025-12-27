"""
Extracting Features From TIF to CSV

We want to extract features from the four data files (TIF) to CSV for further analysis.
    - ERA5: Weather Data
    - NDVI: Plant Health and Density (Satellite measured)
    - DEM:  Graphical Representation of Terrain in BC
    - Landcover: Physical Surface Characteristicsvof BC
"""

## package
import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom
from scipy import stats
from tqdm import tqdm
from rasterio.windows import Window


# ================
# Set up Configuration
## ====


class Config:

    THIS_FILE = Path(__file__).resolve()
    PROJECT_ROOT = THIS_FILE.parents[2]

    ## Grid File
    GRID_FILE = PROJECT_ROOT/'Processed Data'/'Grid'/'STATCAN0.25'/'BC_grids_boundary0.25.geojson'
     
    ## Data File
    ERA5_DIR = PROJECT_ROOT/'Processed Data'/'Weather'/'ERA5'
    NDVI_DIR = PROJECT_ROOT/'Data'/'NDVI'/'Monthly'
    DEM_DIR = PROJECT_ROOT/'Data'/'DEM'
    LANDCOVER_FILE = PROJECT_ROOT/'Data'/'LandCover'/'bc_landcover_2022.tif'
    
    ## Output
    OUTPUT_DIR = PROJECT_ROOT/'Processed Data'/'Rasters'
    
    ## set standard crs
    GRID_CRS = "EPSG:4326" 
    
    ## sampling method
    SAMPLE_METHOD = "centroid"  # 'centroid' or 'zonal_stats'
    
    # parameters
    CHUNK_SIZE = 5000  # grid number handle each process
    FILL_NODATA = True  # fill in NAs
    
    # time range
    START_YEAR = 2019
    END_YEAR = 2024


# ================
# Extractor
## ====

class RasterExtractor:
    """Extract Raster"""
    
    def __init__(self, grids: gpd.GeoDataFrame):
        """
        
        Parameters:
        -----------
        grids : GeoDataFrame
            Grid geometry with centroids
        """
        self.grids = grids
        self.n_grids = len(grids)
        
        # make sure centroids exists
        if 'centroid_lon' not in grids.columns:
            print("Missing centroids, Computing...")
            self.grids['centroid_lon'] = grids.geometry.centroid.x
            self.grids['centroid_lat'] = grids.geometry.centroid.y
    
    def extract_point_values(
        self, 
        raster_path: str, 
        band: int = 1,
        nodata_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract centroids from each grids
        
        Parameters
        ----------
        raster_path : str
        band : int
        nodata_value : float, optional
            
        Returns
        -------
        values : ndarray
            extracted point values (len = # grid)
        """
        with rasterio.open(raster_path) as src:

            coords = list(zip(
                self.grids['centroid_lon'], 
                self.grids['centroid_lat']
            ))
            
            # extracting specified band
            values = np.array([x[0] for x in src.sample(coords, indexes=band)])
            
            # handling NAs
            if nodata_value is None:
                nodata_value = src.nodata
            
            if nodata_value is not None:
                values = np.where(values == nodata_value, np.nan, values)
            
            return values
    


    def extract_zonal_stats(
        self,
        raster_path: str,
        stat_list: List[str] = ['mean', 'max', 'min', 'std'],
        band: int = 1,
        verbose: bool = True  
    ) -> pd.DataFrame:
        """
        Extract summary statistics based on grid
        """
        results = {stat: [] for stat in stat_list}

        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            grid_crs = self.grids.crs

            if grid_crs is None:
                raise ValueError("self.grids has no CRS.")

            # transform grids CRS to rasters CRS
            # if CRS are not the same
            if raster_crs != grid_crs:
                grids_for_calc = self.grids.to_crs(raster_crs)
            else:
                grids_for_calc = self.grids

            # handle NAs
            if src.nodatavals and src.nodatavals[band - 1] is not None:
                nodata = src.nodatavals[band - 1]
            else:
                nodata = src.nodata

            # showing the band info currently executing
            desc = f"Zonal stats (band {band})" if verbose else "Processing"
            
            for idx, geom in tqdm(
                enumerate(grids_for_calc.geometry),
                total=len(grids_for_calc),
                desc=desc,
                disable=not verbose 
            ):
                try:
                    out_image, out_transform = mask(src, [geom], crop=True, filled=True)
                    data = out_image[band - 1]

                    # filter NAs
                    if nodata is not None:
                        valid_mask = (data != nodata) & (~np.isnan(data))
                    else:
                        valid_mask = ~np.isnan(data)

                    valid_data = data[valid_mask]

                    if valid_data.size == 0:
                        for stat in stat_list:
                            results[stat].append(np.nan)
                        continue

                    # compute stats
                    for stat in stat_list:
                        if stat == 'mean':
                            results[stat].append(float(np.mean(valid_data)))
                        elif stat == 'max':
                            results[stat].append(float(np.max(valid_data)))
                        elif stat == 'min':
                            results[stat].append(float(np.min(valid_data)))
                        elif stat == 'std':
                            results[stat].append(float(np.std(valid_data)))
                        elif stat == 'median':
                            results[stat].append(float(np.median(valid_data)))
                        elif stat == 'mode':
                            vals, counts = np.unique(valid_data, return_counts=True)
                            mode_val = vals[counts.argmax()]
                            results[stat].append(float(mode_val))
                        else:
                            raise ValueError(f"Not supported: {stat}")

                except Exception as e:
                    if verbose:
                        print(f"Extraction Failed: {e}")
                    for stat in stat_list:
                        results[stat].append(np.nan)

        return pd.DataFrame(results)




# ================
# Extractor For Each Data
## ====

class ERA5Extractor(RasterExtractor):

    
    def extract_timeseries(
        self,
        era_dir: str,
        variables: list,
        method: str = "point",       # 'point' or 'zonal'
        start_year: int = 2019,
        end_year: int = 2024,
        temporal: str = "daily",     # 'daily' or 'monthly'
    ) -> pd.DataFrame:
        """
        Extract timeseries from multi variables and days ERA5 tiff

        Assume:
        - Every .tif = one time step (one month), file name: BC_ERA5_Daily_2019_01.tif
        - Every .tif includes n_days * len(ERA5 VARIABLES) bands:
            Day 1: All Varibles
            DAy 2: All variables
            ...
            Day Last

        temporal:
        - 'daily'   -> row for each dat: var_YYYY_MM_DD
        - 'monthly' -> average of each day: var_YYYY_MM
        """
        temporal = temporal.lower()
        assert temporal in ("daily", "monthly")

        print(f"\nExtracting ERA5 time series ({temporal}, {method}) ...")

        # Check if variables in ERA5_VARIABLES 
        for v in variables:
            if v not in variables:
                raise ValueError(f"Var {v} not in  ERA5_VARIABLES")

        # look for files
        era_dir = Path(era_dir)
        pattern = str(era_dir / "**" / "*.tif")
        tif_files = sorted(glob.glob(pattern, recursive=True))

        if not tif_files:
            print("Cannot find TIF files.")
            return pd.DataFrame()

        print(f"   Found {len(tif_files)} files")

        result = self.grids[["grid_id"]].copy()
        num_vars_total = len(variables)

        for tif_file in tqdm(tif_files, desc="ERA5"):
            tif_path = Path(tif_file)
            filename = tif_path.stem

            # Extract month and date from files
            # # BC_ERA5_Daily_2019_01
            m = re.search(r"(\d{4})_(\d{2})", filename)
            if not m:
                print(f"Cannot identify date: {filename}")
                continue

            year, month = int(m.group(1)), int(m.group(2))
            if year < start_year or year > end_year:
                continue

            with rasterio.open(tif_path) as src:
                n_bands = src.count
                # compute days in this month
                n_days = n_bands // num_vars_total
                if n_bands % num_vars_total != 0:
                    print(
                        f"Warning: # of bands {n_bands} in file {filename} "
                        f"are not the  multiples of variables {num_vars_total}."
                        f"Varibles lists might differ from bands."
                    )

                if temporal == "daily":
                    # Extract per variable per day
                    for day in range(1, n_days + 1):
                        day_tag = f"{year}_{month:02d}_{day:02d}"

                        for var in variables:
                            global_idx = variables.index(var)  # 0-based
                            band = (day - 1) * num_vars_total + (global_idx + 1)

                            col_name = f"{var}_{day_tag}"

                            try:
                                if method == "point":
                                    vals = self.extract_point_values(
                                        tif_path, band=band
                                    )
                                    result[col_name] = vals
                                elif method == "zonal":
                                    zs = self.extract_zonal_stats(
                                        raster_path=tif_path,
                                        stat_list=["mean"],
                                        band=band,
                                    )
                                    result[col_name] = zs["mean"]
                                else:
                                    raise ValueError(f"Unknown method: {method}")
                            except Exception as e:
                                print(f"Extract {var} @ {day_tag} Failed: {e}")
                                result[col_name] = np.nan

                elif temporal == "monthly":
                    # extract per day and average
                    for var in variables:
                        global_idx = variables.index(var)
                        daily_list = []

                        for day in range(1, n_days + 1):
                            band = (day - 1) * num_vars_total + (global_idx + 1)

                            try:
                                if method == "point":
                                    vals = self.extract_point_values(
                                        tif_path, band=band
                                    )
                                elif method == "zonal":
                                    zs = self.extract_zonal_stats(
                                        raster_path=tif_path,
                                        stat_list=["mean"],
                                        band=band,
                                    )
                                    vals = zs["mean"].values
                                else:
                                    raise ValueError(f"Unknown method: {method}")

                                daily_list.append(vals)
                            except Exception as e:
                                print(f"Extract {var} day={day} @ {filename} Failed: {e}")
                                # fill with NAN
                                daily_list.append(np.full(len(self.grids), np.nan))

                        # (n_days, n_grids) -> (n_grids,)
                        daily_arr = np.stack(daily_list, axis=0)  # shape: (n_days, n_grids)
                        monthly_vals = np.nanmean(daily_arr, axis=0)

                        col_name = f"{var}_{year}_{month:02d}"
                        result[col_name] = monthly_vals

        n_feat = len(result.columns) - 1
        print(f"Complete! Extracted {n_feat} ERA5 {temporal} features")
        return result
    

    


class NDVIExtractor(RasterExtractor):
    
    
    def extract_timeseries(
        self,
        ndvi_dir: str,
        temporal_resolution: str = 'monthly',
        start_year: int = 2019,
        end_year: int = 2024,
        method: str = 'centroid',
        stat_list: List[str] = ['mean'],
        scale_factor: float = 10000.0,  
        auto_scale: bool = True,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Extract NDVI time series
        
        Parameters
        ----------
        scale_factor :
            Transform original value to range -1 to 1
            scale_factor default: 10000 (MODIS NDVI standard)
        auto_scale : bool
        """
        import re
        import glob
        from pathlib import Path

        print(f"\nExtracting NDVI ({temporal_resolution})...")
        print(f"   Method: {method}")
        if method == 'zonal':
            print(f"   Stats: {stat_list}")
        if auto_scale:
            print(f"   Auto scale: Divide by {scale_factor}")

        pattern = f"{ndvi_dir}/**/*.tif"
        tif_files = sorted(glob.glob(pattern, recursive=True))

        if not tif_files:
            print("Cannot find NDVI files")
            return pd.DataFrame()

        print(f"   Found {len(tif_files)} files")

        result = self.grids[['grid_id']].copy()

        for tif_file in tqdm(tif_files, desc="Processing NDVI.tif", disable=not verbose):
            filename = Path(tif_file).stem
            m = re.search(r'(\d{4})', filename)
            if not m:
                if verbose:
                    print(f"Cannot analyze: {filename}")
                continue

            year = int(m.group(1))
            if year < start_year or year > end_year:
                continue

            if verbose:
                print(f"\n   Processing {year}: {filename}")

            with rasterio.open(tif_file) as src:
                n_bands = src.count

            if temporal_resolution == 'monthly':
                if method == 'centroid':
                    for band in range(1, n_bands + 1):
                        month = band
                        col_name = f"ndvi_{year}_{month:02d}"
                        values = self.extract_point_values(tif_file, band=band)
                        
                        
                        if auto_scale:
                            values = values / scale_factor
                        
                        result[col_name] = values
                
                elif method == 'zonal':
                    for band in range(1, n_bands + 1):
                        month = band
                        
                        zonal_df = self.extract_zonal_stats(
                            raster_path=tif_file,
                            stat_list=stat_list,
                            band=band,
                            verbose=False
                        )
                        
                        for stat in stat_list:
                            if len(stat_list) == 1:
                                col_name = f"ndvi_{year}_{month:02d}"
                            else:
                                col_name = f"ndvi_{stat}_{year}_{month:02d}"
                            
                            
                            values = zonal_df[stat]
                            if auto_scale:
                                values = values / scale_factor
                            
                            result[col_name] = values
                        
                        if verbose:
                            print(f" {year}-{month:02d}")

            else:
                # yearly
                if method == 'centroid':
                    all_vals = []
                    for band in range(1, n_bands + 1):
                        vals = self.extract_point_values(tif_file, band=band)
                        if auto_scale:
                            vals = vals / scale_factor
                        all_vals.append(vals)

                    all_vals = np.stack(all_vals, axis=1)
                    yearly_mean = np.nanmean(all_vals, axis=1)
                    col_name = f"ndvi_{year}"
                    result[col_name] = yearly_mean
                
                elif method == 'zonal':
                    all_stats = {stat: [] for stat in stat_list}
                    
                    for band in range(1, n_bands + 1):
                        zonal_df = self.extract_zonal_stats(
                            raster_path=tif_file,
                            stat_list=stat_list,
                            band=band,
                            verbose=False
                        )
                        
                        for stat in stat_list:
                            vals = zonal_df[stat].values
                            if auto_scale:
                                vals = vals / scale_factor
                            all_stats[stat].append(vals)
                    
                    for stat in stat_list:
                        stat_array = np.stack(all_stats[stat], axis=1)
                        yearly_stat = np.nanmean(stat_array, axis=1)
                        
                        if len(stat_list) == 1:
                            col_name = f"ndvi_{year}"
                        else:
                            col_name = f"ndvi_{stat}_{year}"
                        
                        result[col_name] = yearly_stat
                    
                    if verbose:
                        print(f"      {year} (Average by year)")

        print(f"\nComplete! Extracted{len(result.columns) - 1} time steps")
        
        # show stats
        if auto_scale:
            ndvi_cols = [col for col in result.columns if col.startswith('ndvi_')]
            if ndvi_cols:
                print(f"\nStatistics of NDVI after scaled:")
                print(f"   Min: {result[ndvi_cols].min().min():.4f}")
                print(f"   Max: {result[ndvi_cols].max().max():.4f}")
                print(f"   Mean: {result[ndvi_cols].mean().mean():.4f}")
        
        return result


class DEMExtractor(RasterExtractor):
    
    def extract_terrain(self, dem_dir: str) -> pd.DataFrame:
        """
            
        Returns:
        --------
        df : DataFrame
            include [grid_id, elevation, slope, aspect] DataFrame
        """
        print(f"\nWxtracting DEM data...")
        
        result = self.grids[['grid_id']].copy()
        
        # Define the terrain layer to be extracted
        terrain_layers = {
            'elevation': ['bc_elevation.tif'],
            'slope': ['bc_slope.tif'],
            'aspect': ['bc_aspect.tif']
        }
        
        for layer_name, possible_names in terrain_layers.items():
            found = False
            
            for filename in possible_names:
                filepath = os.path.join(dem_dir, filename)
                
                if os.path.exists(filepath):
                    print(f"   Extacting {layer_name} from {filename}...")
                    values = self.extract_point_values(filepath)
                    result[layer_name] = values
                    found = True
                    break
            
            if not found:
                print(f"Cannot find {layer_name} file")
                result[layer_name] = np.nan
        
        # Compute features
        # if 'slope' in result.columns:
        #     # slope calss
        #     result['slope_class'] = pd.cut(result['slope'], 
        #                                   bins=[0, 5, 15, 25, 90],
        #                                   labels=['flat', 'gentle', 'moderate', 'steep'])
        
        # if 'aspect' in result.columns:
        #     # aspect class
        #     result['aspect_class'] = pd.cut(result['aspect'],
        #                                    bins=[0, 45, 90, 135, 180, 
        #                                          225, 270, 315, 360],
        #                                    labels=['N', 'NE', 'E', 'SE', 
        #                                           'S', 'SW', 'W', 'NW'])

        if 'slope' in result.columns:
            result['slope_code'] = pd.cut(
                result['slope'],
                bins=[0, 5, 15, 25, 90],
                labels=[0, 1, 2, 3],
                include_lowest=True
            ).astype('Int64')

        # aspect: sin / cos
        if 'aspect' in result.columns:
            rad = np.deg2rad(result['aspect'])
            result['aspect_sin'] = np.sin(rad)
            result['aspect_cos'] = np.cos(rad)
        
        print(f"Complete {len(result.columns)-1} terrain features")
        return result


class LandcoverExtractor(RasterExtractor):
    

    def extract_landcover(self, landcover_file: str) -> pd.DataFrame:
        """
        Returns
        -------
        DataFrame
            include grid_id, landcover_class, landcover_name
        """

        import numpy as np
        import rasterio
        import os

        print("\nExtracting Landcover data...")

        result = self.grids[['grid_id']].copy()

        if not os.path.exists(landcover_file):
            print(f"Cannot find file: {landcover_file}")
            result['landcover_class'] = pd.NA
            result['landcover_name'] = pd.NA
            return result

        # identify landcover class
        print("Identifying landcover class...")

        with rasterio.open(landcover_file) as src:
            arr = src.read(1)
            nodata = src.nodata
            unique_vals = np.unique(arr[arr != nodata])

        print(f"   Identified class number: {unique_vals[:20]} ...")

        # identify if MODIS LC_Type1(0–17)
        is_modis = (unique_vals.min() == 0) and (unique_vals.max() == 17)

        # identify if BC LULC(1–16)
        is_bc = (unique_vals.min() >= 1) and (unique_vals.max() <= 16)

        ## MODIS standard
        if is_modis:
            print("Identified as MODIS MCD12Q1 LC_Type1 Class (0-17)")
            name_map = {
                0: "Water",
                1: "Evergreen Needleleaf Forest",
                2: "Evergreen Broadleaf Forest",
                3: "Deciduous Needleleaf Forest",
                4: "Deciduous Broadleaf Forest",
                5: "Mixed Forests",
                6: "Closed Shrublands",
                7: "Open Shrublands",
                8: "Woody Savannas",
                9: "Savannas",
                10: "Grasslands",
                11: "Permanent Wetlands",
                12: "Croplands",
                13: "Urban/Built-Up",
                14: "Cropland/Natural Veg Mosaic",
                15: "Snow/Ice",
                16: "Barren/Sparse Veg",
                17: "Unclassified"
            }

        ## BC class
        elif is_bc:
            print("Identified asBC Landcover Class (1-16)")
            name_map = {
                1: 'Water',
                2: 'Snow/Ice',
                3: 'Rock/Rubble',
                4: 'Exposed/Barren',
                5: 'Bryoids',
                6: 'Shrubs',
                7: 'Wetland',
                8: 'Wetland-treed',
                9: 'Herbs',
                10: 'Coniferous',
                11: 'Broadleaf',
                12: 'Mixedwood',
                13: 'Grassland',
                14: 'Agriculture',
                15: 'Developed',
                16: 'Undifferentiated'
            }

        else:
            print("Cannot identify:")
            name_map = {}
            result['landcover_class'] = np.nan
            result['landcover_name'] = "Unknown"
            return result

     
        ## use zonal stats
        print("Extract using zonal_stats...")
        zonal_df = self.extract_zonal_stats(
            landcover_file,
            stat_list=['mode'],
            band=1
        )

        result['landcover_class'] = zonal_df['mode'].round().astype('Int64')


        ## mapping class
        # result['landcover_name'] = (
        #     result['landcover_class']
        #     .map(name_map)
        #     .fillna("Unknown")
        # )

        ## output
        print(f"Complete!")
        print(f"   Class: {'MODIS LC_Type1' if is_modis else 'BC Landcover'}")
        print("\n   Distribution:")
        print(result['landcover_class'].value_counts())

        return result



# =============================================================================
# Extraction
# =============================================================================

def main():
    
    print("="*70)
    print("Rasters Extraction...")
    print("="*70)
    
    
    grids = gpd.read_file(Config.GRID_FILE)
    print(f"   Loaded {len(grids)} grid")
    print(f"   CRS: {grids.crs}")
    
    # ensure grid_ids
    if 'grid_id' not in grids.columns:
        grids['grid_id'] = range(len(grids))
    
    # ERA5
    era5_variables = ['temperature', 'precipitation', 'wind_speed', 'relative_humidity']
    
    for variable in era5_variables:
        var_dir = os.path.join(Config.ERA5_DIR, variable)
        if os.path.exists(var_dir):
            extractor = ERA5Extractor(grids)
            df = extractor.extract_timeseries(
                var_dir, 
                variable,
                Config.START_YEAR,
                Config.END_YEAR
            )
            
            if not df.empty:
                output_file = os.path.join(Config.OUTPUT_DIR, 
                                         f"grid_era5_{variable}.csv")
                df.to_csv(output_file, index=False)
                print(f"   Saved to: {output_file}\n")
        else:
            print(f"Skipping {variable}: Cannot find files\n")
    
    # NDVI
    if os.path.exists(Config.NDVI_DIR):
        extractor = NDVIExtractor(grids)
        df_ndvi = extractor.extract_timeseries(
            Config.NDVI_DIR,
            temporal_resolution='monthly',
            start_year=Config.START_YEAR,
            end_year=Config.END_YEAR
        )
        
        if not df_ndvi.empty:
            output_file = os.path.join(Config.OUTPUT_DIR, "grid_ndvi_monthly.csv")
            df_ndvi.to_csv(output_file, index=False)
            print(f"   Saved to: {output_file}\n")
    else:
        print(f"Skipping NDVI: Cannot find files\n")
    
    # DEM
    if os.path.exists(Config.DEM_DIR):
        extractor = DEMExtractor(grids)
        df_dem = extractor.extract_terrain(Config.DEM_DIR)
        
        output_file = os.path.join(Config.OUTPUT_DIR, "grid_terrain.csv")
        df_dem.to_csv(output_file, index=False)
        print(f"   Saved to: {output_file}\n")
    else:
        print(f"Skipping DEM: Cannot find files\n")
    
    # Landcover
    if os.path.exists(Config.LANDCOVER_FILE):
        extractor = LandcoverExtractor(grids)
        df_landcover = extractor.extract_landcover(Config.LANDCOVER_FILE)
        
        output_file = os.path.join(Config.OUTPUT_DIR, "grid_landcover.csv")
        df_landcover.to_csv(output_file, index=False)
        print(f"   Saved to: {output_file}\n")
    else:
        print(f"Skipping Landcover: Cannot find files\n")



if __name__ == "__main__":
    main()
















