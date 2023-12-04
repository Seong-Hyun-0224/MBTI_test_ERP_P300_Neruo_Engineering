import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class PlotEEG:
    def __init__(
        self,
        channels: List,
        result_dir: str,
        is_show: bool,
        is_save: bool,
        eeg,
        eeg_times,
        eeg_filename: str,
    ):
        self.channels = channels
        self.result_dir = result_dir
        self.is_show = is_show
        self.is_save = is_save
        self.eeg = eeg
        self.eeg_times = eeg_times
        self.eeg_filename = eeg_filename

    def init_plots(self, n_plots, width=6, height=2.5, is_line=False):
        fig, axarr = plt.subplots(
            n_plots, sharex=True, figsize=(width, height * n_plots)
        )
        plt.tight_layout()
        if is_line:
            plt.axhline(y=0, color="black", linewidth=1)
            plt.axvline(x=0, color="r", linestyle="--", linewidth=1)
        return fig, axarr

    def plot_data(
        self,
        axis,
        x,
        y,
        title="",
        fontsize=10,
        labelsize=8,
        color="salmon",
        xlabel="",
        ylabel="",
        linewidth=0.5,
        minima_indices=[]
    ):
        axis.set_title(title, fontsize=fontsize)
        axis.grid(which="both", axis="both", linestyle="--")
        axis.tick_params(labelsize=labelsize)
        axis.set_xlabel(xlabel, fontsize=labelsize + 1)
        axis.set_ylabel(ylabel, fontsize=labelsize + 1)
        axis.autoscale(tight=True)
        axis.plot(x, y, color=color, zorder=1, linewidth=linewidth)
        
        for i in minima_indices:
            axis.axvline(x=x[i], color="b", linestyle="--", linewidth=1)

    def plot_points(self, axis, values, indices):
        axis.scatter(x=indices, y=values[indices], c="black", s=50, zorder=2)

    def show_plot(self):
        plt.show()
        plt.clf()
        plt.close()

    def save_plot(self, fig, filename: str):
        fig.savefig(filename)

    def clean_plot(self):
        plt.clf()
        plt.close()

    def plot_eeg(self):
        fig, axarr = self.init_plots(1)
        for i in range(len(self.eeg)):  # For electrode
            self.plot_data(
                axis=axarr,
                x=self.eeg_times,
                y=self.eeg[i],
                color="black",
                xlabel="Time (s)",
                ylabel="uV",
                title="EEG (" + str(len(self.eeg)) + " channels)",
            )
        if self.is_save:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.save_plot(fig, filename=f"{self.result_dir}/{self.eeg_filename}.png")
        if self.is_show:
            self.show_plot()
        self.clean_plot()

    def plot_electrode(
        self,
        avg_evoked: List,
        times: List,
        filename: str,
    ):
        for i in range(len(avg_evoked)):  # For electrode
            
            times = np.array(times)
            
            # Find the indices where times are between 0 and 1
            valid_indices = np.where((times >= 0) & (times <= 0.4))[0]
            print('valid_indices:')
            print(valid_indices)

            evoked = avg_evoked[i]
            
            min_idx=np.argmin(evoked[valid_indices])
            print('min_idx :')
            print(min_idx)
            max_idx=np.argmax(evoked[valid_indices])
            print('max_idx :')
            print(max_idx)
            #min_idx = valid_indices[np.argmin(evoked[valid_indices])]
            #max_idx = valid_indices[np.argmax(evoked[valid_indices])]
            min_idx = np.argmin(evoked[valid_indices])
            max_idx = np.argmax(evoked[valid_indices])
            
            #min_idx = np.array(min_idx)
            #max_idx = np.array(max_idx)
            
            avg_idx = np.sum(min_idx,max_idx)//2
            print(avg_idx)
            
            
            
            fig, axarr = self.init_plots(1, is_line=True)
            self.plot_data(
                axis=axarr,
                x=times,
                y=avg_evoked[i],
                color="black",
                xlabel="Time (s)",
                ylabel="uV",
                title="EEG ("
                + str(len(avg_evoked))
                + " channels) - "
                + self.channels[i]
                + " Electrode",
                minima_indices=avg_idx
            )
            
            
            if self.is_save:
                if not os.path.exists(self.result_dir):
                    os.makedirs(self.result_dir)
                self.save_plot(
                    fig,
                    filename=f"{self.result_dir}/{filename}_average_{self.channels[i]}.png",
                )
            if self.is_show:
                self.show_plot()
            self.clean_plot()
