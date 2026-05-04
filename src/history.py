import matplotlib.pyplot as plt
import numpy as np

class History:
    def __init__(self, keys: list[str]):
        self.keys = keys
        self.history: dict[str, list[float]] = { key: [] for key in keys}


    def __getitem__(self, key) -> list[float]:
        return self.history[key]
    

    def update(self, vals: list[float]) -> None:
        assert(len(self.keys) == len(vals))

        for key, val in zip(self.keys, vals):
            self.history[key].append(val)
    

    def clear(self) -> None:
        self.history = { key: [] for key in self.keys}
