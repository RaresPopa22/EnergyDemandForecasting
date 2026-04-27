import numpy as np

class NaiveSeasonalForecast():
    def __init__(self, config, segments):
        self.lookback = config['hyperparams']['lookback']
        self.forecast = config['hyperparams']['forecast_horizon']
        self.segments = segments
        self.start_offsets = np.concatenate([[0], np.cumsum(self.segments)[:-1]])
        self.m = config['naive']['m']

    def predict(self, y):
        y_pred = []
        
        for start, length in zip(self.start_offsets, self.segments):
            if self.m > length:
                y_temp = np.array([[np.nan] * self.forecast] * (length - self.forecast + 1))
                y_pred.extend(y_temp[self.lookback:])
                continue

            y_temp = []
            starting_length = length
            for i in range(self.lookback, length - self.forecast + 1):
                if i - self.m < 0:
                    y_temp.append(np.array([np.nan] * self.forecast))
                    starting_length -= 1
                    continue

                y_temp.append(y[start + i - self.m : start + i - self.m + self.forecast])
            
            y_pred.extend(y_temp)

        return np.asarray(y_pred)

        

        

        
        
