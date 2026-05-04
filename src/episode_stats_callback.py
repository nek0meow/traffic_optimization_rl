from stable_baselines3.common.callbacks import BaseCallback

class EpisodeStatsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.avg_speeds:    list[float] = []
        self.avg_vehicles:  list[float] = []
        self.reward_sums:       list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode_info" in info:
                ep_data: dict[str, float] = info["episode_info"]

                self.avg_speeds.append(ep_data["avg_speed"])
                self.avg_vehicles.append(ep_data["avg_n_vehicles"])
                self.reward_sums.append(ep_data["reward_sum"])

        return True