import math
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch

from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, features, targets, lookback, segment_lengths, forecast_horizon):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(targets, dtype=torch.float32)
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon

        start_offsets = np.concatenate([[0], np.cumsum(segment_lengths)[:-1]])
        self.valid_starts = []

        window_total = self.lookback + forecast_horizon

        for offset, length in zip(start_offsets, segment_lengths):
            self.valid_starts.extend(range(offset, offset + length - window_total + 1))


    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        X = self.X[start:start + self.lookback]
        y = self.y[start + self.lookback:start + self.lookback + self.forecast_horizon]

        return X, y


def get_df(config):
    preprocessed_data_config = config['data_paths']['preprocessed']
    energy_csv_name = preprocessed_data_config['energy']
    energy_csv_file = Path(__file__).parent.parent / energy_csv_name
    temp_csv_name = preprocessed_data_config['temp']
    temp_csv_file = Path(__file__).parent.parent / temp_csv_name

    energy_df = pd.read_csv(energy_csv_file, parse_dates=['DateUTC'], dayfirst=True)
    energy_df['DateUTC'] = energy_df['DateUTC'].dt.tz_localize('UTC')
    energy_df = energy_df.sort_values(by='DateUTC')

    temp_df = pd.read_csv(temp_csv_file, parse_dates=['DateUTC'])
    temp_df = temp_df.sort_values(by='DateUTC')

    return pd.merge(energy_df, temp_df, how='left', on='DateUTC')


def feature_engineer(df):
    hour_data = df['DateUTC'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour_data / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour_data / 24.0)

    day_data = df['DateUTC'].dt.day_of_week 
    df['day_sin'] = np.sin(2 * np.pi * day_data / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * day_data / 7.0)

    month_data = df['DateUTC'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_data / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * month_data / 12.0)

    df['is_weekend'] = (day_data > 4).astype(int)

    ro_holidays = holidays.country_holidays('RO', years=range(2019, 2026))
    date_only = df['DateUTC'].dt.date
    df['is_holiday'] = date_only.isin(ro_holidays).astype(int)
    
    date_min = df['DateUTC'].min()
    date_max_value = (df['DateUTC'].max() - date_min).total_seconds()

    df['trend'] = (df['DateUTC'] - date_min).dt.total_seconds() / date_max_value
    

def segment_dataframe(config, df):
    delta = df['DateUTC'] - df['DateUTC'].shift(1)
    gap_mask = delta.dt.total_seconds() > 3600
    gap_idxs = df.index[gap_mask].tolist()
    gap_idxs.insert(0, 0)
    gap_idxs.append(len(df))

    list_of_dfs = [ df.iloc[gap_idxs[n]:gap_idxs[n+1]] for n in range(len(gap_idxs)-1) ]
    min_size = config['hyperparams']['lookback'] + config['hyperparams']['forecast_horizon']
    list_of_dfs = list(filter(lambda x: len(x) >= min_size, list_of_dfs))
    segment_lengths = [len(l) for l in list_of_dfs]
    df = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    
    return (df, segment_lengths)


def process_df(config):
    df = get_df(config)
    df, segment_lengths = segment_dataframe(config, df)
    feature_engineer(df)
    df= df.drop('DateUTC', axis=1)

    return df, segment_lengths


def get_train_eval_test_df(config):
    df, segment_lengths = process_df(config)
    X = df.drop('Value', axis=1)
    y= df[['Value']]

    test_size = config['data']['split']['test_size']
    eval_size = config['data']['split']['eval_size']
    train_target = math.floor(len(X) * (1 - eval_size - test_size))
    eval_target = math.floor(len(X) * (1 - test_size))

    train_slices = []
    eval_slices = []
    test_slices = []

    exclusive_end = 0
    
    for segment in segment_lengths:
        start = exclusive_end
        exclusive_end += segment
        
        if start < train_target:
            train_slices.append((start, exclusive_end))
        elif start < eval_target:
            eval_slices.append((start, exclusive_end))
        else:
            test_slices.append((start, exclusive_end))
        

    train_slice_start, train_slice_end = train_slices[0][0], train_slices[-1][1]
    X_train, y_train = X.iloc[train_slice_start: train_slice_end], y.iloc[train_slice_start: train_slice_end]
    train_segment_lengths = [stop - start for start, stop in train_slices]

    eval_slice_start, eval_slice_end = eval_slices[0][0], eval_slices[-1][1]
    X_eval, y_eval = X.iloc[eval_slice_start: eval_slice_end], y.iloc[eval_slice_start: eval_slice_end]
    eval_segment_lengths = [stop - start for start, stop in eval_slices]

    test_slice_start, test_slice_end = test_slices[0][0], test_slices[-1][1]
    X_test, y_test = X.iloc[test_slice_start: test_slice_end], y.iloc[test_slice_start: test_slice_end]
    test_segment_lengths = [stop - start for start, stop in test_slices]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_eval = scaler.transform(X_eval)

    target_scaler = StandardScaler()
    y_train = target_scaler.fit_transform(y_train)
    y_test = target_scaler.transform(y_test)
    y_eval = target_scaler.transform(y_eval)
    
    X_train = np.column_stack([X_train, y_train])
    X_test = np.column_stack([X_test, y_test])
    X_eval = np.column_stack([X_eval, y_eval])

    train_tuple = (X_train, y_train.squeeze(), train_segment_lengths)
    eval_tuple = (X_eval, y_eval.squeeze(), eval_segment_lengths)
    test_tuple = (X_test, y_test.squeeze(), test_segment_lengths)

    return train_tuple, eval_tuple, test_tuple, scaler, target_scaler


def get_data_loader(config, data_tuple):
    X, y, segment_lengths = data_tuple
    batch_size = config['hyperparams']['batch_size']
    lookback = config['hyperparams']['lookback']
    forecast_horizon = config['hyperparams']['forecast_horizon']

    custom_dataset = CustomDataset(X, y, lookback, segment_lengths, forecast_horizon)
    return DataLoader(custom_dataset, batch_size, num_workers=config['hyperparams']['num_workers'])
