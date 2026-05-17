import matplotlib.pyplot as plt
import numpy as np


def moving_average(y, window=50):
    return np.convolve(y, np.ones(window)/window, mode='valid')

def plots_over_time(
            time_arr: list,
            y_arrs: list[list[float | int]],
            labels: list[str],
            avg_window: int = 25,
            save_to=None, 
            show=True, 
            figsize=(14, 7)) -> None:

        fig, axes = plt.subplots(1, len(y_arrs), figsize=figsize)

        if len(y_arrs) == 1:
            axes = [axes]

        for i, (y_col, label) in enumerate(zip(y_arrs, labels)):
            if avg_window < len(y_col):
                y_smooth = moving_average(y_col, window=avg_window)

                axes[i].plot(time_arr, y_col, alpha=0.15)
                axes[i].plot(time_arr[avg_window-1:], y_smooth)
            else:
                 axes[i].plot(time_arr, y_col)
            axes[i].set_title(label)
            axes[i].grid(
                alpha=0.9
            )
        
        fig.tight_layout()

        if save_to is not None:
            fig.savefig(save_to)
        
        if show:
            plt.show()
        
        plt.close(fig)