import argparse
import logging
from pathlib import Path

import joblib
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data_processing import get_data_loader
from src.model import EnergyModel
from src.utils import read_configs, setup_device


logger = logging.getLogger(__name__)
device = setup_device()


def evaluate(config):
    model_path = config['data_paths']['model']
    model = EnergyModel(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.eval()

    loss_fn = nn.MSELoss()

    test_data = joblib.load(config['data_paths']['test_data'])
    target_scaler = joblib.load(config['data_paths']['target_scaler'])
    test_loader = get_data_loader(config, test_data)

    predictions = []
    labels_y = []
    test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            hypothesis = model(X_batch)
            loss = loss_fn(hypothesis, y_batch)
            test_loss += loss.data * X_batch.shape[0]

            predictions.append(hypothesis)
            labels_y.append(y_batch)

    scaled_rmse_loss = np.sqrt(test_loss.item() / len(test_loader.dataset))
    logger.info(f'Test loss={scaled_rmse_loss:.3f}')

    y_test = torch.cat(labels_y).cpu().numpy()
    y_pred = torch.cat(predictions).cpu().numpy()

    y_test = target_scaler.inverse_transform(y_test)
    y_pred = target_scaler.inverse_transform(y_pred)
    mwh_rmse_loss = target_scaler.scale_ * scaled_rmse_loss
    logger.info(f'Test loss, re-scaled={mwh_rmse_loss.item():.3f}MWh')

    labels_y_hat_t = []
    window_total = config['hyperparams']['lookback'] + config['hyperparams']['forecast_horizon']
    entry_counts = [segment - window_total + 1 for segment in test_data[2]]
    start_offsets = np.concatenate([[0], np.cumsum(entry_counts)[:-1]])

    for offset, length in zip(start_offsets, entry_counts):
        y_t_idxs = range(offset + 1, offset + length)
        y_t_1_idxs = range(offset, offset + length-1)
        labels_y_hat_t.append(y_test[y_t_idxs] - y_test[y_t_1_idxs])

    diffs = np.concatenate(labels_y_hat_t).squeeze()
    persistence_rmse = np.sqrt(np.mean(diffs ** 2))
    skill_score = mwh_rmse_loss / persistence_rmse
    logger.info(f'Skill score is: {skill_score.item()}')
    
    plt.figure(figsize=(8, 8))
    plt.plot(y_test, label='expected', c='g')
    plt.plot(y_pred, label='predicted', c='r')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Expected vs Predicted')
    plt.legend()
    plt.savefig('outputs/evaluation.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model evaluation stage")
    parser.add_argument('--config', required=True, help='Path to the model`s configuration file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    config = read_configs(base_path, args.config)
    evaluate(config)