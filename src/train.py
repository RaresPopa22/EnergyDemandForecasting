import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'

import argparse
import logging
import joblib
import numpy as np
import torch
import torch.optim as optim
import xgboost as xgb

from src.seq2seq import Decoder, Encoder, Seq2SeqLSTM
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from sklearn.multioutput import MultiOutputRegressor

from pathlib import Path
from torch import nn

from sklearn.metrics import root_mean_squared_error
from src.data_processing import get_data_for_tree_model, get_data_loader, get_train_eval_test_df
from src.model import EnergyModel
from src.utils import read_configs, set_random_seeds, setup_device


logger = logging.getLogger(__name__)


def plot_learning_curve(training_loss, eval_loss, best_eval_idx, model_name, horizon):
    plt.figure(figsize=(8, 8))
    plt.plot(training_loss, c='b', label='training')
    plt.plot(eval_loss, c='orange', label='eval')
    plt.axvline(best_eval_idx, c='r', label='saved checkpoint')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend()
    plt.savefig(f'outputs/learning_curve_{model_name}_{horizon}h.png')


def train_tree_model(config):
    model_config = config['hyperparams']['model']
    train, eval, test = get_data_for_tree_model(config)

    model = MultiOutputRegressor(xgb.XGBRegressor(**model_config, random_state=1, eval_metric='rmse'))

    X_train, y_train, _ = train
    X_eval, y_eval, _ = eval
    
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)

    error = root_mean_squared_error(y_eval, pred)
    logger.info(f'XGBoost RMSE={error}') # target_scaler=[1052.60945979]

    joblib.dump(model, config['data_paths']['model'])
    
    X_test, y_test, test_segment_lengths = test
    X_test.to_csv(config['data_paths']['X_test'], index=False)
    y_test.to_csv(config['data_paths']['y_test'], index=False)
    joblib.dump(test_segment_lengths, config['data_paths']['test_segments'])


def train_sequence_model(config):
    model_name = config['model']['name']
    hparam_config = config['hyperparams']
    train_tuple, eval_tuple, test_tuple, scaler, target_scaler = get_train_eval_test_df(config)
    train_loader = get_data_loader(config, train_tuple)
    eval_loader = get_data_loader(config, eval_tuple)

    input_size = train_tuple[0].shape[1]

    decoder_input = train_tuple[1].shape[1] + 1

    if model_name == 'lstm':
        model = EnergyModel(config, input_size).to(device)
    elif model_name == 'seq2seq':
        encoder = Encoder(config, input_size)
        decoder = Decoder(config, decoder_input)
        model = Seq2SeqLSTM(config, encoder, decoder).to(device)
    else:
        raise ValueError(f'Unknown model. Available sequence models=[lstm, seq2seq]. Requested={model_name}')

    optimizer = optim.AdamW(model.parameters(), lr=hparam_config['lr'], weight_decay=hparam_config['weight_decay'])
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=hparam_config['T_max'], eta_min=hparam_config['eta_min'])
    loss_fn = nn.MSELoss()

    training_avg_loss = []
    eval_avg_loss = []

    patience = 0
    best_eval = float('inf')
    max_norm = hparam_config['max_norm']


    for epoch in range(hparam_config['epochs']):
        if patience > hparam_config['patience']:
            break

        model.train()
        training_loss = 0
        for X_batch, X_future_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            if model_name == 'lstm':
                y_pred = model(X_batch)
            elif model_name == 'seq2seq':
                X_future_batch = X_future_batch.to(device)
                y_pred = model(X_batch, X_future_batch, y_batch, hparam_config['teacher_forcing_ratio'])
            else:
                raise ValueError(f'Unknown model: {model_name}')

            loss = loss_fn(y_pred, y_batch)
            training_loss += loss.data * X_batch.shape[0]
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        avg_rmse_loss = np.sqrt(training_loss.item()/len(train_loader.dataset))

        training_avg_loss.append(avg_rmse_loss)
        
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for X_batch, X_future_batch, y_batch in eval_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                if model_name == 'lstm':
                    y_pred = model(X_batch)
                elif model_name == 'seq2seq':
                    X_future_batch = X_future_batch.to(device)
                    y_pred = model(X_batch, X_future_batch)
                else:
                    raise ValueError(f'Unknown model: {model_name}')

                loss = loss_fn(y_pred, y_batch)
                eval_loss += loss.data * X_batch.shape[0]

        avg_rmse_loss = np.sqrt(eval_loss.item()/len(eval_loader.dataset))
        eval_avg_loss.append(avg_rmse_loss)

        if best_eval > (avg_rmse_loss + 1e-5):
            logger.info(f'Best eval so far={best_eval} - epoch={epoch}')
            best_eval = avg_rmse_loss
            patience=0
            torch.save(model.state_dict(), config['data_paths']['model'])
        else:
            patience += 1
        
        lr_scheduler.step()

        if epoch % 50 == 0:
            logger.info(f'Epoch={epoch}. Training loss {training_avg_loss[-1]:.3f}. Eval loss {eval_avg_loss[-1]:.3f}')

    
    joblib.dump(test_tuple, config['data_paths']['test_data'])
    joblib.dump(scaler, config['data_paths']['scaler'])
    joblib.dump(target_scaler, config['data_paths']['target_scaler'])


    best_eval_idx = np.argmin(eval_avg_loss)
    plot_learning_curve(training_avg_loss, eval_avg_loss, best_eval_idx, model_name, hparam_config['forecast_horizon'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training stage')
    parser.add_argument('--config', required=True, help='Path to the model`s configuration file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    config = read_configs(base_path, args.config)
    set_random_seeds(config)
    device = setup_device()

    model_name = config['model']['name']
    if model_name == 'xgboost':
        train_tree_model(config)
    elif model_name in ['lstm', 'seq2seq']:
        train_sequence_model(config)
    else:
        raise ValueError(f'Unknown model. Available sequence models=[lstm, seq2seq, xgboost]. Requested={model_name}')