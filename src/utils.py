import joblib
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.dates as mdates
from datetime import timedelta
import holidays

from matplotlib import pyplot as plt

def read_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config if config is not None else {}


def deep_merge(base, override):
    result = {**base}

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def read_configs(base, override):
    base_config = read_config(base)
    override_config = read_config(override)

    return deep_merge(base_config, override_config)


def setup_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    

def set_random_seeds(config):
    seed = config['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)



def get_skill_score(hours, y_test, valid_segments, model_loss):
    print(f'y_test={len(y_test)}')
    print(f'valid_segments={valid_segments}')
    diffs_per_segment = []
    start_offsets = np.concatenate([[0], np.cumsum(valid_segments)[:-1]])

    for offset, length in zip(start_offsets, valid_segments):
        y_t_idxs = range(offset + hours, offset + length)
        y_t_1_idxs = range(offset, offset + length-hours)
        diffs_per_segment.append(y_test[y_t_idxs] - y_test[y_t_1_idxs])

    diffs = np.concatenate(diffs_per_segment).squeeze()
    persistence_rmse = np.sqrt(np.mean(diffs ** 2))
    return 1 - model_loss / persistence_rmse


def plot_overall_result(config, dates, y_test, y_pred, y_naive, horizon):
    fig, axs = plt.subplots(3, figsize=(8, 8))

    lim = np.percentile(np.abs(y_test[:, -1] - y_pred[:, -1]), 99)
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[0].set_ylim(-lim, lim)
    axs[0].plot(dates, y_test[:, -1] - y_pred[:, -1])
    axs[0].set_title('Residual')
    
    axs[1].scatter(y_test[:, -1], y_pred[:, -1], c='b', alpha=.1)
    axs[1].set_xlabel('y_test')
    axs[1].set_ylabel('y_pred')
    axs[1].axline((0, 0 ), slope=1)
    axs[1].set_title('Calibration')

    errors = np.abs(y_test[:, 0] - y_pred[:, 0]).ravel()
    rolling_mae = pd.Series(errors).rolling(window=config['plot']['window']).mean()
    
    naive_errors = np.abs(y_test[:, 0] - y_naive[:, 0]).ravel()
    naive_rolling_mae = pd.Series(naive_errors).rolling(window=config['plot']['window']).mean()
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[2].plot(dates, rolling_mae, label='Seq2Seq')
    axs[2].plot(dates, naive_rolling_mae, label='Naive')
    axs[2].set_title('Rolling MAE')
    axs[2].legend()

    fig.tight_layout()
    plt.savefig(f'outputs/evaluation_overall_{horizon}h.png')
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
    

def plot_week_data(dates, y_preds, y_test, valid_segments, horizon):
    fig, axs = plt.subplots(4, figsize=(8, 8))
    fig.tight_layout()
    start, end = get_middle_segment_week(valid_segments)

    fan_out_start = range(start, start + 7*24 , 24)
    y_pred = y_preds['seq2seq']
    y_test_day = y_test[fan_out_start, :]
    y_pred_day = y_pred[fan_out_start, :]
    dates_day = dates[start:start+24*7]

    y_test = y_test[:, 0]
    y_pred = y_pred[:, 0]
    y_naive = y_preds['naive']
    y_naive = y_naive[:, 0]

    worst_start = get_worst_week_start(y_test, y_pred, valid_segments)
    holidays_mask = get_holiday_week_mask(dates)

    y_test_week = y_test[start:end]
    dates_week = dates[start:end]

    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[0].plot(dates_week, y_test_week, label='Expected')

    for model_name, pred in y_preds.items():
        y_week = pred[:, 0][start:end]
        axs[0].plot(dates_week, y_week, label=f'{model_name}')
    axs[0].set_title('Weekly data')
    axs[0].legend()

    y_test_worst_week = y_test[worst_start: worst_start + 168]
    y_pred_worst_week = y_pred[worst_start: worst_start + 168]
    dates_worst_week = dates[worst_start: worst_start + 168]

    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[1].plot(dates_worst_week, y_test_worst_week, label='Expected')
    axs[1].plot(dates_worst_week, y_pred_worst_week, label='Predicted - Seq2Seq')
    axs[1].set_title('Worst Week')
    axs[1].legend()

    y_test_holiday = y_test[holidays_mask]
    y_pred_holiday = y_pred[holidays_mask]
    dates_holiday = dates[holidays_mask]

    axs[2].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    axs[2].plot(dates_holiday, y_test_holiday, label='Expected')
    axs[2].plot(dates_holiday, y_pred_holiday, label='Predicted - Seq2Seq')
    axs[2].set_title('Holiday Week')
    axs[2].legend()

    axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

    for i in range(len(y_test_day)):
        axs[3].plot(dates_day[24*i:24*i+24], y_test_day[i, :], c='b', linewidth=2)
        axs[3].plot(dates_day[24*i:24*i+24], y_pred_day[i, :], c='r')
        axs[3].axvline(dates_day.iloc[24*i+23], c='orange')
    
    axs[3].legend(['Expected', 'Predicted - Seq2Seq'])
    axs[3].set_title('Fan out')

    plt.savefig(f'outputs/evaluation_weekly_{horizon}h.png')
    plt.close()


def get_valid_counts(y, segments):
    valid_counts = []

    start_offsets = np.concatenate([[0], np.cumsum(segments)[:-1]])
    for start, length in zip(start_offsets, segments):
        chunk = y[start:start+length, 0]
        valid_counts.append(np.count_nonzero(~np.isnan(chunk)))

    return valid_counts


def get_common_timestamps(configs):
    timestamps = []
    for config in configs:
        test_data = joblib.load(config['data_paths']['test_data'])
        timestamps.append(test_data[-1])

    common_timestamps = sorted(set.intersection(*map(set, timestamps)))
    return common_timestamps


def slice_targets(y_pred, y_test, dates, common_timestamps):
    common_idxs = pd.DatetimeIndex(dates).get_indexer(common_timestamps)
    return y_pred[common_idxs, :], y_test[common_idxs, :]


def compute_segments_for_naive(dates, common_timestamps):
    common_idxs = pd.DatetimeIndex(dates).get_indexer(common_timestamps)
    dates = dates.iloc[common_idxs]

    delta = dates.diff(1)
    gap_mask = delta.dt.total_seconds() > 3600
    gap_idxs = np.where(gap_mask.values)[0].tolist()
    gap_idxs.insert(0, 0)
    gap_idxs.append(len(dates))

    list_of_dfs = [dates.iloc[gap_idxs[n]:gap_idxs[n+1]] for n in range(len(gap_idxs) - 1)]
    segment_lengths = [len(l) for l in list_of_dfs]

    return segment_lengths
