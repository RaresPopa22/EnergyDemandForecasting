import argparse
import logging
import time
import pandas as pd
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry


from pathlib import Path
from io import StringIO

from src.utils import read_config

logger = logging.getLogger(__name__)


def fetch_energy_data(config):
    start_year = 2019
    end_year = 2025
    preprocessed_data_config = config['data_paths']['preprocessed']
    energy_csv_name = preprocessed_data_config['energy']
    output_file_path = Path(__file__).parent.parent / energy_csv_name

    semicolon_years = [2021, 2022]

    logger.info(f'Starting collecting the data for years: {start_year}-{end_year}')
    data = []

    for year in range(start_year, end_year + 1):
        logger.info(f'Downloading data for year {year}...')
        link = f'https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Publications/Statistics/{year}/monthly_hourly_load_values_{year}.csv'
        response = requests.get(link)

        if response.status_code == 200:
            logger.info(f'Got successful response for year={year}')
            decoded_content = response.content.decode('utf-8')
            csv_delimiter = ';' if year in semicolon_years else '\t'
            df = pd.read_csv(StringIO(decoded_content), delimiter=csv_delimiter, parse_dates=['DateUTC'], dayfirst=True)
            df = df[df['CountryCode'] == preprocessed_data_config['country']]
            df = df.drop(preprocessed_data_config['columns_to_drop'], axis=1, errors='ignore')
            df['DateUTC'] = df['DateUTC'].dt.strftime("%d-%m-%Y %H:%M")
            data.append(df)
            logger.info('Done.')
        else:
            logger.info(f'Something went wrong, {response.status_code}')

        time.sleep(1)

    df = pd.concat(data, axis=0)
    
    if df.empty:
        raise ValueError(f'Something went wrong. Empty dataframe.')    

    df.to_csv(output_file_path, index=False)


def fetch_temp_data(config):
    temp_csv_name = config['data_paths']['preprocessed']['temp']
    output_file_path = Path(__file__).parent.parent / temp_csv_name
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        "latitude": 44.4323,
        "longitude": 26.1063,
        "start_date": "2019-01-01",
        "end_date": "2025-12-31",
        "hourly": "temperature_2m",
    }

    responses = openmeteo.weather_api(url, params = params)
    response = responses[0]
    logger.info(f'Coordinates: {response.Latitude()}°N {response.Longitude()}°E')
    logger.info(f'Elevation: {response.Elevation()} m asl')
    logger.info(f'Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s')

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_data = {'DateUTC': pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit='s', utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit='s', utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive='left'
    )}

    hourly_data['temperature_2m'] = hourly_temperature_2m
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.to_csv(output_file_path, index=False)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset type to download: 'energy' or 'temperature'")
    parser.add_argument('--type', required=True, help='Types of dataset to download')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    config = read_config(base_path)

    if args.type == 'energy':
        fetch_energy_data(config)
    elif args.type == 'temp':
        fetch_temp_data(config)
    else:
        raise ValueError(f'Unknown required type. Available: energy, temp. Requested: {args.type}')