import numpy as np

from src.evaluate import get_skill_score


class TestEvaluate:

    def test_get_skill_score_1h(self):
        segments = [3, 2]
        y_mean = np.array([3, 4, 5, 66, 67])
        
        assert get_skill_score(1, y_mean,segments, .3) == .7

    def test_get_skill_score_2h(self):
        segments = [3, 3]
        y_mean = np.array([3, 4, 7, 66, 67, 70])
        
        assert get_skill_score(2, y_mean,segments, .4) == .9

    def test_get_skill_score_semantics(self):
        segments = [3, 2]
        y_mean = np.array([3, 4, 5, 66, 67])
        
        assert get_skill_score(1, y_mean,segments, 0) == 1
        assert get_skill_score(1, y_mean,segments, 1) == 0
        assert get_skill_score(1, y_mean,segments, 2) < 0
    


