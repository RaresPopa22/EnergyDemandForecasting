from torch import nn


class EnergyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        hyperparam_config = config['hyperparams']
        self.lstm = nn.LSTM(
            input_size=hyperparam_config['input_size'],
            hidden_size=hyperparam_config['hidden_size'],
            num_layers=hyperparam_config['num_layers'],
            batch_first=hyperparam_config['batch_first']
        )
        self.dropout = nn.Dropout(hyperparam_config['dropout'])
        self.linear = nn.Linear(hyperparam_config['hidden_size'], hyperparam_config['forecast_horizon'])

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x[:, -1])
        return x