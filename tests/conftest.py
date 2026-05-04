import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_base_config():
    return {'A': 1}


@pytest.fixture
def sample_raw_data():
    dt = pd.date_range(start='2025-01-01 00:00:00', end='2025-12-31 23:00:00', freq='h')
    return pd.DataFrame({'DateUTC': dt})


@pytest.fixture
def sample_gap_data():
    s_1 = pd.date_range(start='2025-01-01 00:00:00', end='2025-01-31 23:00:00', freq='h')
    s_2 = pd.date_range(start='2025-02-01 02:00:00', end='2025-05-01 23:00:00', freq='h')
    s_3 = pd.date_range(start='2025-05-02 03:00:00', end='2025-05-02 23:00:00', freq='h') #this should be dropped as it's smaller than min size
    s_4 = pd.date_range(start='2025-05-03 03:00:00', end='2025-12-01 23:00:00', freq='h')
    s_5 = pd.date_range(start='2025-12-05 03:00:00', end='2025-12-31 23:00:00', freq='h')

    dt = s_1.union(s_2).union(s_3).union(s_4).union(s_5)
    return pd.DataFrame({'DateUTC': dt})


@pytest.fixture
def sample_config():
    return {
        'hyperparams': {
            'lookback': 24,
            'forecast_horizon': 1
        }
    }


@pytest.fixture
def sample_y_data():
    return np.array([12, 10, 11, 13, 15, 16, 17, 999, 998, 997, 996, 5, 6, 7, 8, 9, 10])

