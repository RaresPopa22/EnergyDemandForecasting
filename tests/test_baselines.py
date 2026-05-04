import numpy as np

from src.baselines import NaiveSeasonalForecast


class TestNaiveSeasonalForecast():

    def test_predict_normal_flow(self, sample_y_data):
        config = {
            'hyperparams': {
                'lookback': 2,
                'forecast_horizon': 1
            },
            'naive': {'m': 5}
        }
        model = NaiveSeasonalForecast(config, [7, 4, 6])
        y_pred = model.predict(sample_y_data)

        expected = np.array([np.nan, np.nan, np.nan, 12, 10, np.nan, np.nan, np.nan, np.nan, np.nan, 5]).reshape(-1, 1)
        
        assert y_pred.shape == expected.shape
        assert np.array_equal(expected, y_pred, equal_nan=True)

    def test_predict_lookback_bigger_than_m(self, sample_y_data):
        config = {
            'hyperparams': {
                'lookback': 5,
                'forecast_horizon': 1
            },
            'naive': {'m': 1}
        }
        model = NaiveSeasonalForecast(config, [7, 4, 6])
        y_pred = model.predict(sample_y_data)


        expected = np.array([15, 16, 9]).reshape(-1, 1)
        assert y_pred.shape == expected.shape
        assert np.array_equal(expected, y_pred, equal_nan=True)

    def test_predict_forecast_2h(self, sample_y_data):
        config = {
            'hyperparams': {
                'lookback': 2,
                'forecast_horizon': 2
            },
            'naive': {'m': 3}
        }

        model = NaiveSeasonalForecast(config, [7, 4, 6])
        y_pred = model.predict(sample_y_data)


        expected = np.array([[np.nan, np.nan], [12, 10], [10, 11], [11, 13], [np.nan, np.nan], [np.nan, np.nan], [5, 6], [6, 7]])
        assert y_pred.shape == expected.shape
        assert np.array_equal(expected, y_pred, equal_nan=True)


    def test_predict_forecast_larger_than_length_minus_lookback(self, sample_y_data):
        config = {
            'hyperparams': {
                'lookback': 4,
                'forecast_horizon': 4
            },
            'naive': {'m': 1}
        }

        model = NaiveSeasonalForecast(config, [7, 4, 6])
        y_pred = model.predict(sample_y_data)

        expected = np.array([])
        assert y_pred.shape == expected.shape
        assert np.array_equal(expected, y_pred, equal_nan=True)



        
        


    