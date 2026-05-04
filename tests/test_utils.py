from src.utils import deep_merge

class TestUtils:

    def test_deep_merge_flat_key_override(self, sample_base_config):
        override_config = {'A': 2}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 2

    def test_deep_merge_addin_new_keys(self, sample_base_config):
        override_config = {'B': 2}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 1
        assert config.get('B') == 2

    def test_deep_merge_nested_merge(self):
        base_config = {'A': {'B': {'C': 2, 'D': 3}}, 'E': 3}
        override_config = {'A': {'B': {'D': 9}}}
        config = deep_merge(base_config, override_config)
        assert config == {'A': {'B': {'C': 2, 'D': 9}}, 'E': 3}

    def test_deep_merge_empty(self, sample_base_config):
        override_config = {}
        config = deep_merge(sample_base_config, override_config)
        assert config.get('A') == 1

    def test_deep_merge_not_mutated(self, sample_base_config):
        override_config = {'B': 2}
        deep_merge(sample_base_config, override_config)
        assert sample_base_config == {'A': 1}