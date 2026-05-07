import random

import numpy as np
from torch import nn
import torch


class Encoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        hyperparam_config = config['hyperparams']
        self.lstm = nn.LSTM(
            input_size= input_size,
            hidden_size=hyperparam_config['hidden_size'],
            num_layers=hyperparam_config['num_layers'],
            batch_first=hyperparam_config['batch_first']
        )

    def forward(self, input_seq):
        outputs, (hidden, cell) = self.lstm(input_seq)
        return outputs, hidden, cell
    

class Decoder(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        hyperparam_config = config['hyperparams']
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hyperparam_config['hidden_size'],
            num_layers=hyperparam_config['num_layers'],
            batch_first=hyperparam_config['batch_first']
        )
        self.out = nn.Linear(hyperparam_config['hidden_size'], 1)

    def forward(self, input_seq, hidden, cell):
        output, (hidden, cell) = self.lstm(input_seq, (hidden, cell))
        prediction = self.out(output)
        return prediction, hidden, cell
    

class Seq2SeqLSTM(nn.Module):
    def __init__(self, config, encoder, decoder):
        super().__init__()
        self.config = config
        hyperparam_config = self.config['hyperparams']
        self.encoder = encoder
        self.decoder = decoder
        self.forecast_horizon = hyperparam_config['forecast_horizon']

    # [64, 168, 11] - input.shape = [batch_size, lookback, features]
    # [64, 24] - output.shape = [batch_size, forecast_horizon]
    def forward(self, input_seq, future_input_seq, target_seq=None, teacher_forcing_ratio=0):
        target_len = self.config['hyperparams']['forecast_horizon']
        outputs = []
        _, hidden, cell = self.encoder(input_seq)

        dec_in = input_seq[:, -1, -1]

        for t in range(target_len):
            future_seq = future_input_seq[:, t, :]
            future_seq = future_seq[:, None, :]
            dec_in = dec_in[:, None, None]
            dec_in = torch.cat((future_seq, dec_in), dim=2)

            pred, hidden, cell = self.decoder(dec_in, hidden, cell)
            outputs.append(pred[:, 0, 0])
            teacher_force = random.random() < teacher_forcing_ratio
            dec_in = target_seq[:, t] if teacher_force and target_seq is not None else pred[:, 0, 0]

        outputs = torch.stack(outputs, dim=1)

        return outputs