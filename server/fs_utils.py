import os

from src.config import PROJ_DIR, DATA_DIR

def available_maps():
    maps_dir = os.path.join(PROJ_DIR, "maps")
    result = []

    for name in os.listdir(maps_dir):
        full = os.path.join(maps_dir, name)
        if os.path.isdir(full):
            result.append(name)
    return result

def available_histories():
    data_dir = os.path.join(PROJ_DIR, DATA_DIR)
    result = []

    if not os.path.exists(data_dir):
        return result

    for model_name in os.listdir(data_dir):
        history_path = os.path.join(
            data_dir,
            model_name,
            "history",
            f"{model_name}.npz"
        )
        if os.path.exists(history_path):
            result.append(model_name)

    result.sort()
    return result

def available_models():
    result = []

    data_dir = os.path.join(
        PROJ_DIR,
        DATA_DIR
    )
    if not os.path.exists(data_dir):
        return result

    for model_name in os.listdir(data_dir):

        models_dir = os.path.join(
            data_dir,
            model_name,
            "models"
        )
        if not os.path.exists(models_dir):
            continue
        for file in os.listdir(models_dir):
            if file.endswith(".pt"):
                result.append({
                    "name": f"{model_name}/{file}",
                    "path": os.path.join(
                        models_dir,
                        file
                    )
                })
    return result