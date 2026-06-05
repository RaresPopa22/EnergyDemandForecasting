import argparse
import logging

import joblib
import numpy as np
import numpy.ma as ma
import pandas as pd
import torch
import copy

from pathlib import Path

from src.baselines import NaiveSeasonalForecast
from src.data_processing import get_data_loader
from src.model import EnergyModel
from src.seq2seq import Decoder, Encoder, Seq2SeqLSTM
from src.utils import compute_segments_for_naive, deep_merge, get_common_timestamps, get_skill_score, plot_overall_result, plot_week_data, read_config, setup_device, slice_dates, slice_targets


logger = logging.getLogger(__name__)
device = setup_device()


def log_metrics(y_test, y_pred, test_segments, model_name):
    mae = np.mean(np.abs(y_test - y_pred))
    mape = np.mean(np.abs(y_test - y_pred) / np.abs(y_test)) * 100
    r2 = 1 - np.mean((y_test - y_pred) ** 2) / np.var(y_test)

    hour_rmse_loss = np.sqrt(np.mean((y_pred[:, 0] - y_test[:, 0]) ** 2))
    day_rmse_loss = np.sqrt(np.mean((y_pred[:, -1] - y_test[:, -1]) ** 2))

    lag1h = get_skill_score(1, y_test, test_segments, hour_rmse_loss)
    lag24h = get_skill_score(24, y_test, test_segments, day_rmse_loss)

    rmse_loss = np.sqrt(np.mean((y_pred - y_test) ** 2))
    
    logger.info(f'\nLogging metrics for {model_name}')
    logger.info(f'RMSE error={rmse_loss.item():.3f}MWh')
    logger.info(f'MAE:{mae:.3f}MWh')
    logger.info(f'MAPE:{mape:.3f}%')
    logger.info(f'R2:{r2:.3f}')
    logger.info(f'Skill score for 1h is: {lag1h.item():.3f}')
    logger.info(f'Skill score for 24h is: {lag24h.item():.3f}')


def evaluate_sequence_models(config):
    model_name = config['model']['name']
    test_data = joblib.load(config['data_paths']['test_data'])
    test_loader = get_data_loader(config, test_data)
    input_size = test_data.X.shape[1]

    if model_name == 'lstm':
        model = EnergyModel(config, input_size).to(device)
    elif model_name == 'seq2seq':
        decoder_input = test_data.X_future.shape[1] + 1
        encoder = Encoder(config, input_size)
        decoder = Decoder(config, decoder_input)
        model = Seq2SeqLSTM(config, encoder, decoder).to(device)
    else:
        raise ValueError(f'Unknown sequence model. Available models=[lstm, seq2seq]. Requested={model_name}')    
    
    model_path = config['data_paths']['model']
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    predictions = []
    labels_y = []

    with torch.no_grad():
        for X_batch, X_future_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if model_name == 'lstm':
                hypothesis = model(X_batch)
            elif model_name == 'seq2seq':
                X_future_batch = X_future_batch.to(device)
                hypothesis = model(X_batch, X_future_batch)
            else:
                raise ValueError(f'Unknown model: {model_name}')

            predictions.append(hypothesis)
            labels_y.append(y_batch)

    y_test = torch.cat(labels_y).cpu().numpy()
    y_pred = torch.cat(predictions).cpu().numpy()

    return y_pred, y_test, test_data


def evaluate_tree_models(config):
    test_data = joblib.load(config['data_paths']['test_data'])
    model = joblib.load(config['data_paths']['model'])
    y_pred = model.predict(test_data.X)
    
    return y_pred, np.asarray(test_data.y), test_data.dates


def evaluate_naive(config, test_data, common_timestamps, target_scaler):
    common_idxs = pd.DatetimeIndex(test_data.dates).get_indexer(common_timestamps)
    naive_model = NaiveSeasonalForecast(config, test_data.segment_lengths)
    y_naive = naive_model.predict(test_data.y)
    y_naive = y_naive[common_idxs]
    y_naive = target_scaler.inverse_transform(y_naive)

    valid_mask = np.isnan(y_naive)
    segments = compute_segments_for_naive(test_data.dates, common_timestamps)

    return y_naive, valid_mask, segments


def process_ys(predict_dict, y_pred, y_test, target_scaler, model_name):
    valid_mask = predict_dict['naive']['valid_mask']

    if model_name in ['lstm', 'seq2seq']:
        y_diff = ma.masked_array(y_pred - y_test, valid_mask)
        scaled_rmse_loss = np.sqrt(np.mean(y_diff ** 2))
        logger.info(f'{model_name}: RMSE(scaled)={scaled_rmse_loss:.3f}')
        y_pred = target_scaler.inverse_transform(y_pred)
        y_test = target_scaler.inverse_transform(y_test)
    
    y_pred = ma.masked_array(y_pred, valid_mask)
    y_test = ma.masked_array(y_test, valid_mask)

    return y_pred, y_test


def predict_naive(config, target_scaler, common_timestamps):
    test_data = joblib.load(config['data_paths']['test_data'])
    y_naive, valid_mask, segments = evaluate_naive(config, test_data, common_timestamps, target_scaler)
    
    return y_naive, valid_mask, segments


def predict(configs):
    predict_dict = {}
    base_config = {}
    common_timestamps = get_common_timestamps(configs)

    for config in configs:
        model_name = config['model']['name']
        if model_name == 'seq2seq':
            base_config = copy.deepcopy(config)

    target_scaler = joblib.load(base_config['data_paths']['target_scaler'])

    if len(common_timestamps) == 0:
        raise ValueError(f'No common timestamps were found.')

    
    y_naive, valid_mask, segments = predict_naive(base_config, target_scaler, common_timestamps)
    predict_dict['naive'] = {
            'y_pred': ma.masked_array(y_naive, valid_mask),
            'valid_mask': valid_mask,
            'segments': segments
        }
    
    for config in configs:
        model_name = config['model']['name']
        if model_name == 'xgboost':
            y_pred, y_test, dates = evaluate_tree_models(config)
            y_pred, y_test = slice_targets(y_pred, y_test, dates, common_timestamps)
            y_pred, y_test = process_ys(predict_dict, y_pred, y_test, target_scaler, model_name)

            predict_dict[model_name] = {
                'y_pred': y_pred,
                'y_test': y_test,
                'segments': segments
            }
            
        elif model_name == 'seq2seq':
            y_pred, y_test, test_data = evaluate_sequence_models(config)
            y_pred, y_test = slice_targets(y_pred, y_test, test_data.dates, common_timestamps)
            y_pred, y_test = process_ys(predict_dict, y_pred, y_test, target_scaler, model_name)
            
            dates = slice_dates(test_data.dates, common_timestamps)
            predict_dict[model_name] = {
                'y_pred': y_pred,
                'y_test': y_test,
                'segments': segments,
                'dates': dates,
                'forecast_horizon': config['hyperparams']['forecast_horizon']
            }

            if 'y_test' not in predict_dict['naive']:
                predict_dict['naive']['y_test'] = y_test

        elif model_name == 'lstm':
            y_pred, y_test, test_data = evaluate_sequence_models(config)
            y_pred, y_test = slice_targets(y_pred, y_test, test_data.dates, common_timestamps)
            y_pred, y_test = process_ys(predict_dict, y_pred, y_test, target_scaler, model_name)

            predict_dict[model_name] = {
                'y_pred': y_pred,
                'y_test': y_test,
                'segments': segments
            }

            if 'y_test' not in predict_dict['naive']:
                predict_dict['naive']['y_test'] = y_test
        else:
            raise ValueError(f'Unknown model. Available sequence models=[lstm, seq2seq, xgboost]. Requested={model_name}')


    for model, values in predict_dict.items():
        log_metrics(values['y_test'], values['y_pred'], values['segments'], model)
    

    plot_overall_result(
        base_config, 
        predict_dict['seq2seq']['dates'], 
        predict_dict['seq2seq']['y_test'], 
        predict_dict['seq2seq']['y_pred'], 
        predict_dict['naive']['y_pred'],
        base_config['hyperparams']['forecast_horizon']
        )

    y_preds = {model_name:model_dict['y_pred'] for model_name, model_dict in predict_dict.items() if model_name not in ('lstm', 'xgboost')}
    plot_week_data(
        predict_dict['seq2seq']['dates'], 
        y_preds,
        predict_dict['seq2seq']['y_test'], 
        predict_dict['seq2seq']['segments'], 
        base_config['hyperparams']['forecast_horizon']
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Model evaluation stage")
    parser.add_argument('--configs', nargs='+', required=True, help='Path to the model`s configuration file')
    args = parser.parse_args()

    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    base_config = read_config(base_path)
    configs = [deep_merge(base_config, read_config(c)) for c in args.configs]

    predict(configs)
