from io import StringIO
import logging
from pathlib import Path
import time

import pandas as pd
import requests

from src.utils import read_config

logger = logging.getLogger(__name__)

def fetch_data(config):
    start_year = 2019
    end_year = 2025
    raw_data_config = config['data_paths']['raw_data']
    output_file_path = Path(raw_data_config['dir']) / f"{raw_data_config['country']}_power_statistic.csv"

    semicolon_years = [2021, 2022]

    logger.info(f'Starting collecting the data for years: {start_year}-{end_year}')
    data = []

    for year in range(start_year, end_year + 1):
        logger.info(f'Downloading data for year {year}...')
        link = f'https://eepublicdownloads.blob.core.windows.net/public-cdn-container/clean-documents/Publications/Statistics/{year}/monthly_hourly_load_values_{year}.csv'
        response = requests.get(link)

        if response.status_code == 200:
            logger.info(f'Got successful response. Saving the data as {year}.csv')
            decoded_content = response.content.decode('utf-8')
            csv_delimiter = ';' if year in semicolon_years else '\t'
            df = pd.read_csv(StringIO(decoded_content), delimiter=csv_delimiter, parse_dates=['DateUTC'], dayfirst=True)
            df = df[df['CountryCode'] == raw_data_config['country']]
            df = df.drop(raw_data_config['columns_to_drop'], axis=1, errors='ignore')
            df['DateUTC'] = df['DateUTC'].dt.strftime("%d-%m-%Y %H:%M")
            data.append(df)
        else:
            logger.info(f'Something went wrong, {response.status_code}')

        logger.info('Done.')
        time.sleep(1)

    df = pd.concat(data, axis=0)
    df.to_csv(output_file_path, index=False)
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    config = read_config(base_path)
    fetch_data(config)