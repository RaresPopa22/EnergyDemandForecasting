import argparse
import logging
import joblib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from pathlib import Path
from torch import nn

from src.data_processing import get_data_loader, get_train_eval_test_df
from src.model import EnergyModel
from src.utils import read_configs, setup_device


logger = logging.getLogger(__name__)
device = setup_device()


def train(config):
    hparam_config = config['hyperparams']
    train_tuple, eval_tuple, test_tuple, scaler, target_scaler = get_train_eval_test_df(config)
    train_loader = get_data_loader(config, train_tuple)
    eval_loader = get_data_loader(config, eval_tuple)
    model = EnergyModel(config).to(device)
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
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
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
            for X_batch, y_batch in eval_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                y_pred = model(X_batch)
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

    plt.figure(figsize=(8, 8))
    plt.plot(training_avg_loss, label='training')
    plt.plot(eval_avg_loss, label='eval')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve')
    plt.legend(['training', 'eval'])
    plt.savefig(f'outputs/learning_curve.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model training stage')
    parser.add_argument('--config', required=True, help='Path to the model`s configuration file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    base_path = Path(__file__).parent.parent / 'config' / 'base.yaml'
    config = read_configs(base_path, args.config)
    train(config)