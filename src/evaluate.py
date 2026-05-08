import argparse
from datetime import timedelta
import logging
from pathlib import Path

import holidays
import joblib
import matplotlib.dates as mdates
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch

from matplotlib import pyplot as plt

from src.baselines import NaiveSeasonalForecast
from src.data_processing import get_data_loader
from src.seq2seq import Decoder, Encoder, Seq2SeqLSTM
from src.utils import read_configs, setup_device


logger = logging.getLogger(__name__)
device = setup_device()


def get_skill_score(hours, y_test, valid_segments, model_loss):
    diffs_per_segment = []
    start_offsets = np.concatenate([[0], np.cumsum(valid_segments)[:-1]])

    for offset, length in zip(start_offsets, valid_segments):
        y_t_idxs = range(offset + hours, offset + length)
        y_t_1_idxs = range(offset, offset + length-hours)
        diffs_per_segment.append(y_test[y_t_idxs] - y_test[y_t_1_idxs])

    diffs = np.concatenate(diffs_per_segment).squeeze()
    persistence_rmse = np.sqrt(np.mean(diffs ** 2))
    return 1 - model_loss / persistence_rmse


def log_metrics(y_test, y_pred, test_segments, mwh_rmse_loss):
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs(y_test - y_pred) / np.abs(y_test)) * 100
    r2 = 1 - np.mean((y_test - y_pred) ** 2) / np.var(y_test)
    lag1h = get_skill_score(1, y_test, test_segments, mwh_rmse_loss)
    lag24h = get_skill_score(24, y_test, test_segments, mwh_rmse_loss)
    
    logger.info(f'RMSE error={mwh_rmse_loss.item():.3f}MWh')
    logger.info(f'MAE:{mae:.3f}MWh')
    logger.info(f'MAPE:{mape:.3f}%')
    logger.info(f'R2:{r2:.3f}')
    logger.info(f'Skill score for 1h is: {lag1h.item():.3f}')
    logger.info(f'Skill score for 24h is: {lag24h.item():.3f}')


def plot_overall_result(config, dates, y_test, y_pred, y_naive):
    fig, axs = plt.subplots(3, figsize=(8, 8))

    lim = np.percentile(np.abs(y_test - y_pred), 99)
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[0].set_ylim(-lim, lim)
    axs[0].plot(dates, y_test - y_pred)
    axs[0].set_title('Residual')
    
    axs[1].scatter(y_test, y_pred, c='b', alpha=.1)
    axs[1].set_xlabel('y_test')
    axs[1].set_ylabel('y_pred')
    axs[1].axline((0, 0 ), slope=1)
    axs[1].set_title('Calibration')

    errors = np.abs(y_test - y_pred).ravel()
    rolling_mae = pd.Series(errors).rolling(window=config['plot']['window']).mean()
    
    naive_errors = np.abs(y_test - y_naive).ravel()
    naive_rolling_mae = pd.Series(naive_errors).rolling(window=config['plot']['window']).mean()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[2].plot(dates, rolling_mae, label='LSTM')
    axs[2].plot(dates, naive_rolling_mae, label='Naive')
    axs[2].set_title('Rolling MAE')
    axs[2].legend()

    fig.tight_layout()
    plt.savefig('outputs/evaluation_overall.png')
    plt.close()


def get_middle_segment_week(valid_segments):
    start = 0
    start_offsets = np.concatenate([[0], np.cumsum(valid_segments)[:-1]])
    
    for segment_start, segment in zip(start_offsets, valid_segments):
        if segment > 2 * 168:
            start = segment_start + segment // 2
            break

    end = start + 168

    return start, end


def get_worst_week_start(y_test, y_pred, valid_segments):
    start_offsets = np.concatenate([[0], np.cumsum(valid_segments)[:-1]])
    weekly_data = []

    for offset, segment in zip(start_offsets, valid_segments):
        weeks = segment // 168
        length = weeks * 168
        for i in range(offset, offset + length, 168):
            mae = np.mean(np.abs(y_test[i:i+168] - y_pred[i:i+168]))
            weekly_data.append((mae, i))

    max_week = max(weekly_data)
    max_week_idx = weekly_data.index(max_week)
    _, worst_start = weekly_data[max_week_idx]

    return worst_start


def get_holiday_week_mask(dates):
    year = pd.DatetimeIndex(dates).year.unique()
    ro_holidays = holidays.country_holidays('RO', years=year)
    holiday = ro_holidays.get_named('Christmas')[0]
    start = holiday - timedelta(days=3)
    end = holiday + timedelta(days=3)
    dates_mask = (dates.dt.date >= start) & (dates.dt.date <= end)

    return dates_mask
    

def plot_week_data(dates, y_test, y_pred, y_naive, valid_segments):
    fig, axs = plt.subplots(3, figsize=(8, 8))
    fig.tight_layout()

    start, end = get_middle_segment_week(valid_segments)
    worst_start = get_worst_week_start(y_test, y_pred, valid_segments)
    holidays_mask = get_holiday_week_mask(dates)

    y_test_week = y_test[start:end]
    y_pred_week = y_pred[start:end]
    y_naive_week = y_naive[start:end]
    dates_week = dates[start:end]

    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[0].plot(dates_week, y_test_week, label='Expected')
    axs[0].plot(dates_week, y_pred_week, label='LSTM')
    axs[0].plot(dates_week, y_naive_week, label='Naive')
    axs[0].set_title('Weekly data')
    axs[0].legend()

    y_test_worst_week = y_test[worst_start: worst_start + 168]
    y_pred_worst_week = y_pred[worst_start: worst_start + 168]
    dates_worst_week = dates[worst_start: worst_start + 168]

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[1].plot(dates_worst_week, y_test_worst_week, label='Expected')
    axs[1].plot(dates_worst_week, y_pred_worst_week, label='Predicted - LSTM')
    axs[1].set_title('Worst Week')
    axs[1].legend()

    y_test_holiday = y_test[holidays_mask]
    y_pred_holiday = y_pred[holidays_mask]
    dates_holiday = dates[holidays_mask]

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[2].plot(dates_holiday, y_test_holiday, label='Expected')
    axs[2].plot(dates_holiday, y_pred_holiday, label='Predicted - LSTM')
    axs[2].set_title('Holiday Week')
    axs[2].legend()

    plt.savefig('outputs/evaluation_weekly.png')
    plt.close()


def get_valid_counts(config, y, segments):
    valid_counts = []

    window_total = config['hyperparams']['lookback'] + config['hyperparams']['forecast_horizon']
    window_lengths = [segment - window_total + 1 for segment in segments]
    start_offsets = np.concatenate([[0], np.cumsum(window_lengths)[:-1]])
    for start, length in zip(start_offsets, window_lengths):
        chunk = y[start:start+length, 0]
        valid_counts.append(np.count_nonzero(~np.isnan(chunk)))

    return valid_counts

def evaluate(config):
    test_data = joblib.load(config['data_paths']['test_data'])
    target_scaler = joblib.load(config['data_paths']['target_scaler'])
    test_loader = get_data_loader(config, test_data)

    model_path = config['data_paths']['model']

    input_size = test_data[0].shape[1]
    decoder_input = test_data[1].shape[1] + 1
    encoder = Encoder(config, input_size)
    decoder = Decoder(config, decoder_input)
    model = Seq2SeqLSTM(config, encoder, decoder).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()


    predictions = []
    labels_y = []

    with torch.no_grad():
        for X_batch, X_future_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            X_future_batch = X_future_batch.to(device)
            y_batch = y_batch.to(device)
            hypothesis = model(X_batch, X_future_batch)

            predictions.append(hypothesis)
            labels_y.append(y_batch)

    y_test = torch.cat(labels_y).cpu().numpy()
    y_pred = torch.cat(predictions).cpu().numpy()

    test_segments = test_data[3]
    naive_model = NaiveSeasonalForecast(config, test_segments)
    y_naive = naive_model.predict(test_data[2])

    y_naive = target_scaler.inverse_transform(y_naive)

    valid_mask = np.isnan(y_naive)
    dates_valid_mask = ~np.isnan(y_naive[:, 0])
    valid_counts = get_valid_counts(config, y_naive, test_segments)

    y_naive = ma.masked_array(y_naive, mask=valid_mask)

    y_test = ma.masked_array(y_test, mask=valid_mask)
    y_pred = ma.masked_array(y_pred, mask=valid_mask)

    scaled_rmse_loss = np.sqrt(np.mean((y_pred - y_test) ** 2))
    logger.info(f'RMSE(scaled)={scaled_rmse_loss:.3f}')

    y_test = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(y_pred)

    mwh_rmse_loss = target_scaler.scale_ * scaled_rmse_loss
    mwh_rmse_loss_naive = np.sqrt(np.mean((y_naive - y_test) ** 2))

    log_metrics(y_test[:, 0], y_pred[:, 0], valid_counts, mwh_rmse_loss)
    logger.info('Naive model')
    log_metrics(y_test[:, 0], y_naive[:, 0], valid_counts, mwh_rmse_loss_naive)
    
    dates = test_data[4]
    dates = dates[dates_valid_mask]
    
    plot_overall_result(config, dates, y_test[:, 0], y_pred[:, 0], y_naive[:, 0])
    plot_week_data(dates, y_test[:, 0], y_pred[:, 0], y_naive[:, 0], valid_counts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model evaluation stage")
    parser.add_argument('--config', required=True, help='Path to the model`s configuration file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    config = read_configs(base_path, args.config)
    evaluate(config)