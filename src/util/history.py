from util.plot_utils import plots_over_time
from util.stat_dataclasses import StepStat, OverallStat
from dataclasses import fields
class OverallHistory():
    def __init__(self):
        self.stats: list[OverallStat] = []
        self.labels = ['episode', 'avg_speed', 'avg_vehicles', 'avg_wait', 'avg_reward']
    
    def add_info(self, stat: OverallStat):
        self.stats.append(stat)
    
    def make_plots(self, save_to: str, show: bool=False, figsize: tuple=(14,7), avg_window=25):
        episodes = [el.episode for el in self.stats]
        speeds = [el.avg_speed for el in self.stats]
        vehicles = [el.avg_vehicles for el in self.stats]
        waits = [el.avg_wait for el in self.stats]
        rewards = [el.avg_reward for el in self.stats]
        plots_over_time(
            time_arr=episodes, 
            y_arrs=[speeds, vehicles, waits, rewards],
            labels=self.labels[1:],
            save_to=save_to,
            show=show,
            figsize=figsize,
            avg_window=avg_window
        )
    
    def to_dict(self) -> dict[str, list[int | float]]:
        return {
            field.name: [
                getattr(stat, field.name) for stat in self.stats
            ]
            for field in fields(OverallStat)
        }
    



