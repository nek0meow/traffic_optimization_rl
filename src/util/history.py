from util.plot_utils import plots_over_time

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


class OverallHistory():
    def __init__(self):
        self.episode: list[int] = []
        self.avg_speed: list[float] = []
        self.avg_vehicles: list[float] = []
        self.avg_reward: list[float] = []
        self.labels = ['episode', 'avg_speed', 'avg_vehicles', 'avg_reward']
    
    def add_info(self,
                 episode: int,
                 avg_speed: float,
                 avg_vehicles: float,
                 avg_reward: float):
        self.episode.append(episode)
        self.avg_speed.append(avg_speed)
        self.avg_vehicles.append(avg_vehicles)
        self.avg_reward.append(avg_reward)
    
    def make_plots(self, save_to: str, show: bool=False, figsize: tuple=(14,7), avg_window=25):
        plots_over_time(
            time_arr=self.episode, 
            y_arrs=[self.avg_speed, self.avg_vehicles, self.avg_reward],
            labels=self.labels[1:],
            save_to=save_to,
            show=show,
            figsize=figsize,
            avg_window=avg_window
        )
    
    def to_dict(self) -> dict[str, list[int|float]]:
        arrs = [self.episode, self.avg_speed, self.avg_vehicles, self.avg_reward]
        return {
            label: arr for label, arr in zip(self.labels, arrs)
        }
        



