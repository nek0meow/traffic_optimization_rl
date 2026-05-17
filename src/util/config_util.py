from dataclasses import fields, is_dataclass

from src.util.config_dataclasses import TrainConfig, EnvConfig


def update_dataclass_from_dict(cls, data: dict):
    if not is_dataclass(cls):
        raise ValueError(f"{cls} is not a dataclass")

    parsed = {}

    for field in fields(cls):

        if field.name not in data:
            continue

        raw_value = data[field.name]

        parsed[field.name] = to_type(
            raw_value,
            field.type
        )

    return cls(**parsed)


def transform_dict_configs(configs: dict) -> tuple[TrainConfig, EnvConfig]:
    train_cfg = update_dataclass_from_dict(
        TrainConfig,
        configs.get("train", {})
    )

    env_cfg = update_dataclass_from_dict(
        EnvConfig,
        configs.get("env", {})
    )

    return train_cfg, env_cfg

def get_params(cls) -> list[dict]:
    if not is_dataclass(cls):
        raise ValueError("Expected dataclass type")

    result = []

    for f in fields(cls):
        result.append({
            "name": f.name,
            "type": f.type.__name__,
            "default": f.default
        })

    return result

def to_type(value, target_type):
    if value is None:
        return None

    # already correct
    if isinstance(value, target_type):
        return value

    try:
        if target_type == int:
            return int(value)

        if target_type == float:
            return float(value)

        if target_type == bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)

        if target_type == str:
            return str(value)

        return value

    except Exception:
        raise ValueError(
            f"Cannot cast value '{value}' to {target_type}"
        )
