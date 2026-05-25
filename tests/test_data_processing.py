import pandas as pd

from src.data_processing import feature_engineer, segment_dataframe


class TestDataProcessing:

    def test_feature_engineer(self, sample_raw_data):
        df = feature_engineer(sample_raw_data)

        assert 'hour_sin' in df.columns
        assert 'hour_cos' in df.columns
        assert df['hour_sin'].between(-1, 1).all()
        assert df['hour_cos'].between(-1, 1).all()
        assert df[['hour_sin', 'hour_cos']].round(6).drop_duplicates().shape[0] == 24

        assert 'month_sin' in df.columns
        assert 'month_cos' in df.columns
        assert df['month_sin'].between(-1, 1).all()
        assert df['month_cos'].between(-1, 1).all()
        assert df[['month_sin', 'month_cos']].round(6).drop_duplicates().shape[0] == 12

        actual = df['is_weekend'] == 1
        expected = df['DateUTC'].dt.dayofweek > 4
        assert (actual == expected).all()
        
        easter = df.loc[df['DateUTC'].dt.normalize() == "2025-04-20"]
        assert (easter['is_holiday'] == 1).all()


    def test_segment_dataframe(self, sample_config, sample_gap_data):
        df, segment_lengths = segment_dataframe(sample_config, sample_gap_data)

        assert len(segment_lengths) == 4
        assert segment_lengths[0] == 744
        assert segment_lengths[1] == 2158
        assert segment_lengths[2] == 5109
        assert segment_lengths[3] == 645
        assert len(df) == sum(segment_lengths)

    
