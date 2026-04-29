import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

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
            
            view_end = start + length - self.m
            if self.lookback < self.m:
                view_start = start
                warm_up = np.full((self.m - self.lookback, self.forecast), np.nan)
                y_view = np.concat((warm_up, sliding_window_view(y[view_start:view_end], self.forecast)), axis=0)
            else:
                view_start = start + self.lookback - self.m
                y_view = sliding_window_view(y[view_start:view_end], self.forecast)

            y_pred.extend(y_view)
            

        return np.asarray(y_pred)

        

        

        
        
