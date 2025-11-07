class DotDict(dict):
    """A dictionary that supports dot notation as well as dictionary access notation."""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            if isinstance(value, list):
                self[key] = [
                    DotDict(item) if isinstance(item, dict) else item
                    for item in value
                ]
    
    def __getattr__(self, attr):
        return self.get(attr)

def convert_value(value):
    """Convert string value to appropriate type."""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if "," in value:
        return [convert_value(v) for v in value.split(",")]
    return value


def merge_args_to_config(cfg, unknown):

    for arg in unknown:
        if "=" not in arg:
            continue
        if arg.startswith("--"):
            arg = arg.lstrip("--")
        key, value = arg.split("=", 1)
        value = convert_value(value)
        keys = key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return cfg

def merge_dicts(dict1, dict2):
    """Recursively merge two dictionaries."""
    for key, value in dict2.items():
        if (
            key in dict1
            and isinstance(dict1[key], dict)
            and isinstance(value, dict)
        ):
            merge_dicts(dict1[key], value)
        else:
            dict1[key] = value
    return dict1