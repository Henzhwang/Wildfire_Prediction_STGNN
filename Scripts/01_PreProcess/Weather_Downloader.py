import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from pathlib import Path
import json

"""
1. Official historial weather data: https://climate.weather.gc.ca/
2. Datamart: https://dd.weather.gc.ca/
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from pathlib import Path
import json

class ECCCWeatherDownloader:
    
    def __init__(self):
        self.base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

    
    def download_station_data(self, station_id, climate_id, start_year, end_year, 
                             output_dir='weather_data'):
        

        Path(output_dir).mkdir(exist_ok=True)
        
        all_data = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # download URL
                url = f"{self.base_url}?format=csv&stationID={station_id}&Year={year}&Month={month}&timeframe=2&submit=Download+Data"
                
                try:
                    print(f"  Download {year}-{month:02d}...", end=' ')
                    
                    response = requests.get(url, timeout=30)
                    
                    if response.status_code == 200:
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text), encoding='utf-8')
                        
                        if len(df) > 0:
                            all_data.append(df)
                            print(f"({len(df)} record)")
                        else:
                            print("(No Data)")
                    else:
                        print(f"(HTTP {response.status_code})")
                    
                    # slower request
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"Fail: {str(e)[:50]}")
                    continue
        
        if all_data:
            # merge all data
            combined = pd.concat(all_data, ignore_index=True)
            
            # saved to csv
            output_file = f"{output_dir}/station_{station_id}_{climate_id}_{start_year}_{end_year}.csv"
            combined.to_csv(output_file, index=False)
            
            print(f"Station Data Saved: {output_file} ({len(combined)} record)")
            
            return combined
        else:
            print(f"Station {station_id} has no data")
            return None
    
    def download_all_bc_stations(self, 
                                stations,
                                start_year=2019, end_year=2024, 
                                output_dir='bc_weather_data'):
        
        print("=" * 70)
        print("Bulk Download of Weather Station Data")
        print(f"Time Range: {start_year} - {end_year}")
        print("=" * 70)
        
        if len(stations) == 0:
            print("No Station Found")
            return
        
        print(f"\nPreparing Download {len(stations)} Weather Station Data...")
        
        successful_downloads = []
        failed_stations = []
        
        for idx, station in stations.iterrows():

            station_id = station.get('Station ID', station.get('STATION_ID'))
            climate_id = station.get('Climate ID', station.get('CLIMATE_ID', 'unknown'))
            station_name = station.get('Name', station.get('STATION_NAME', 'Unknown'))
            
            print(f"\n[{idx+1}/{len(stations)}] {station_name} (ID: {station_id})")
            
            try:
                data = self.download_station_data(
                    station_id=station_id,
                    climate_id=climate_id,
                    start_year=start_year,
                    end_year=end_year,
                    output_dir=output_dir
                )
                
                if data is not None:
                    successful_downloads.append({
                        'station_id': station_id,
                        'station_name': station_name,
                        'climate_id': climate_id,
                        'records': len(data)
                    })
                else:
                    failed_stations.append(station_name)
                    
            except Exception as e:
                print(f"Station Fail: {e}")
                failed_stations.append(station_name)
                continue
        
        
        if successful_downloads:
            summary_df = pd.DataFrame(successful_downloads)
            summary_file = f"{output_dir}/download_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            print(f"\nDownload CSV saved: {summary_file}")
            
            total_records = summary_df['records'].sum()
            print(f"Total Record: {total_records:,}")
        
        return successful_downloads
    
    def merge_all_stations(self, input_dir='bc_weather_data', 
                          output_file='bc_all_stations_2019_2024.csv'):



        print("\nMerge all stations data...")
        
        csv_files = list(Path(input_dir).glob('station_*.csv'))
        
        if not csv_files:
            print("Cannot find download File")
            return
        
        print(f"Found {len(csv_files)} data files")
        
        all_data = []
        
        for file in csv_files:
            df = pd.read_csv(file)
            all_data.append(df)
            print(f" {file.name}: {len(df)} record")
        
        # merge
        combined = pd.concat(all_data, ignore_index=True)
        
        combined.to_csv(output_file, index=False)
        
        print(f"\n Merge Complete!")
        print(f"  File: {output_file}")
        print(f"  Total record: {len(combined):,}")
        print(f"  # Stations: {combined['Station Name'].nunique()}")
        print(f"  Time range: {combined['Date/Time'].min()} to {combined['Date/Time'].max()}")
        
        return combined
    
    def export_to_geojson(self, csv_file, output_geojson='bc_stations.geojson'):


        print(f"\nExport tp GeoJSON...")
        
        df = pd.read_csv(csv_file)
        
        # summary by stations
        stations_summary = df.groupby(['Station Name', 'Latitude (y)', 'Longitude (x)']).agg({
            'Date/Time': ['min', 'max', 'count']
        }).reset_index()
        
        stations_summary.columns = ['station_name', 'latitude', 'longitude', 
                                   'date_start', 'date_end', 'record_count']
        
        # construct geojson
        features = []
        
        for _, station in stations_summary.iterrows():
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [station['longitude'], station['latitude']]
                },
                'properties': {
                    'name': station['station_name'],
                    'date_start': str(station['date_start']),
                    'date_end': str(station['date_end']),
                    'record_count': int(station['record_count'])
                }
            }
            features.append(feature)
        
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        

        with open(output_geojson, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"GeoJSON Saved: {output_geojson}")
        print(f"  Include {len(features)} station")