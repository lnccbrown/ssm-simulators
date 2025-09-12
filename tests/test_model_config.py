import random
import ssms
from ssms.config import model_config


class TestModelConfig:
    def test_model_config_dict_type(self):
        assert isinstance(model_config, ssms.config.CopyOnAccessDict)

    def test_model_config_copy_on_access(self):
        model_name = random.choice(list(model_config))
        selected_model = model_config[model_name]
        list_params = selected_model["params"]
        list_params.append("p_outlier")
        assert "p_outlier" not in model_config[model_name]["params"]
