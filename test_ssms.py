import ssms
# from pprint import pp
# mc = sorted(list(ssms.config.model_config.keys()))
# pp(mc)

# mc = ssms.config._modelconfig.get_model_config()
# pp(mc)

mc = ssms.config.model_config

for model_name, config in mc.items():
    print(f"Checking config: {model_name}")
    assert isinstance(config, dict), f"Config is not a dict; it is {type(config)}"
