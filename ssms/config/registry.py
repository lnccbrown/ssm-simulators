_model_registry = {}


def register_model(name: str):
    def decorator(func):
        if name in _model_registry:
            raise ValueError(f"Model '{name}' already registered")
        _model_registry[name] = func
        return func

    return decorator


def get_model_config(name: str):
    try:
        return _model_registry[name]()
    except KeyError:
        raise ValueError(f"No model config found for '{name}'")


def get_all_model_configs():
    return dict((name, func()) for name, func in _model_registry.items())
