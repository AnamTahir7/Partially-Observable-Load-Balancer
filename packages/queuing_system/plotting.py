""" This file has all the dedicated functions that plots the results of different queueing systems """
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.ticker as ticker
from itertools import cycle
from typing import Iterable
from pathlib import Path
import numpy as np
from packages.utils import utils
import seaborn as sns
import pandas as pd
import os
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

@ticker.FuncFormatter
def log_formatter(x, pos):
    return '$10^{{{}}}$'.format(int(np.log10(x)))


class QueuingResultPlotting:
    def __init__(self, data: dict, output: Path, pomcp_compare: bool, ignore: list = None ):
        self._data = data
        self.pomcp_compare = pomcp_compare
        model_order = []
        if self.pomcp_compare:
            # model_order = ['pos', 'jmo', 'ejmo', 'sed', 'jsq']
            model_order = ['10','30','50','70','90']
        else:
            model_order = ['pol', 'jmo', 'ejmo', 'sed', 'jsq', 'djsq']

        ignore = [] if ignore is None else ignore

        # filter out models from all data
        self.models = {k.lower(): v for k, v in data.items() if isinstance(v, dict)}
        # drop the models in ignore list
        self.models = {k: v for k, v in self.models.items() if k not in ignore}
        # Sort the models in a pre-defined order
        if model_order:
            self.models = {k: self.models[k] for k in model_order if k in self.models}

        self.bins = []
        self.output = output
        self.cols = 1 if len(self.models) <= 3 else 2
        self.rows = int(np.ceil(len(self.models) / self.cols))
        self.label_map = {
            'pol': 'POL',
            'jmo': 'JMO',
            'djsq': 'DJSQ-FI',
            'ejmo': 'JMO-E',
            'jsq': 'JSQ-FI',
            'sed': 'SED-FI',
            'exponential': 'Exponential \n Reward',
            'linear': 'Linear \n Reward',
            'entropy': 'Entropy \n Reward',
            'self_clk': 'Self Clock \n Reward',
            'prop_alloc': 'Proportional \n Allocation \n Reward',
            'dsed': 'DSED',
        }


    def plot_box(self, ylm, tck, label_map, ax: Axes, data: dict, ylabel: str = '', title: str = '', log_scale=False):

        plt.margins(y=0.05)
        labels = list(map(lambda x: label_map.get(x, x), data.keys()))
        y = list(data.values())
        print(labels)
        ax.boxplot(y, labels=labels, whis=([5,95]), showfliers=True)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes', labelsize=14)

        if log_scale:
            ax.set_yscale('log')
        else:
            if ylm:
                plt.ylim(ylm)
        min_ylim, max_ylim = plt.ylim()
        if log_scale:
            plt.ylim(min_ylim, max_ylim + 1)
        min_xlim, max_xlim = plt.xlim()
        if not self.pomcp_compare:
            plt.text(x=max_xlim/8, y=max_ylim*0.75, s='Limited \n Information', fontsize=12)
            plt.text(x=max_xlim/1.6, y=max_ylim*0.75, s='Full \n Information', fontsize=12)
            plt.axvline(x=max_xlim/1.80, color='k', linestyle='dashed', linewidth=1)
        ax.set_ylabel(ylabel)

    def plot_line_ppr(self,label_map, ax: Axes, data: dict, linetype: Iterable, xlabel: str = '', ylabel: str = '', title: str = '',
                  logscale: bool = False):

        for model_name, model_data in data.items():
            ax.plot(np.arange(1, len(model_data) + 1), model_data, next(linetype),
                    label=label_map.get(model_name, model_name), alpha=0.7)
        if logscale:
            ax.set_yscale('symlog')

        ax.set_ylim(top=0)
        ax.set_xlim(left=0)
        plt.setp(ax.get_yaxis().get_offset_text(), visible=False)
        ax.set_ylabel('Cumulative reward x$10^{8}$', size=12)
        ax.set_xlabel(xlabel, size=14)
        ax.set_title(title)
        ax.legend()

    @staticmethod
    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n + 1) / n
        return x, y

    def plot_ccdf(self, label_map, ax: Axes, data: dict, linetype: Iterable, xlabel: str = '', ylabel: str = '', title: str = '',
                  logscale: bool = False):

        for model_name, model_data in data.items():
            x, y = self.ecdf(model_data)
            c = 1 - y
            ax.plot(x, c, next(linetype), label=label_map.get(model_name, model_name), alpha=0.7)

        if logscale:
            ax.set_yscale('symlog')
        plt.yscale('log')
        ax.set_xlim(left=0)
        ax.set_ylabel(ylabel, size=14)
        ax.set_xlabel(xlabel, size=14)
        ax.legend()
        plt.legend(loc='upper right')


    def draw_plots(self):
        line_types = ["-", ":", "--", "-.", "-"]  # , "*", "o"]
        utils.figure_configuration_ieee_standard()

        task = 'pkt_drp_rate'
        fig, ax = plt.subplots(constrained_layout=True)
        task_data = {m: v[task] for m, v in self.models.items()}
        tck = []
        m = 0
        for val in task_data.values():
            _max = max(val)
            m = _max if _max > m else m
        ylm = [-0.005, m+0.01]
        self.plot_box(ylm, tck, self.label_map, ax, task_data, 'Job Drop Rate', False)
        fig.savefig(self.output.joinpath('{}.pdf'.format(task)))
        plt.close(fig)

        task = 'reward_cumsum'
        task_data = {m: v[task] for m, v in self.models.items()}
        fig, ax = plt.subplots(constrained_layout=True)
        self.plot_line_ppr(self.label_map, ax, task_data, cycle(line_types),
                       'Total number of jobs', 'Cumulative reward')
        fig.savefig(self.output.joinpath('{}_ppr.pdf'.format(task)))
        plt.close(fig)

        # ccdf
        task = 'delay_org'
        fig, ax = plt.subplots(constrained_layout=True)
        task_data = {m: v[task] for m, v in self.models.items()}
        # Remove zero values
        task_data = {m: p[p > 0] for m, p in task_data.items()}
        self.plot_ccdf(self.label_map, ax, task_data, cycle(line_types),
                       'Response time [s]', 'CCDF')
        fig.savefig(self.output.joinpath('{}_ccdf.pdf'.format(task)))
        plt.close(fig)

        # # scatter plot
        # task_del = 'delay_org'
        # task_data_del = {m: v[task_del] for m, v in self.models.items()}
        # task_data_del = {m: p[p > 0] for m, p in task_data_del.items()}
        # task_pd = 'pkt_drp_rate'
        # task_data_pd = {m: v[task_pd] for m, v in self.models.items()}
        # task_data_pd = {m: p[p > 0] for m, p in task_data_pd.items()}
        # model_order = ['pol', 'jmo', 'ejmo', 'sed', 'jsq','djsq']
        # avg_del = {}
        # avg_pd = {}
        # for i in range(len(model_order)):
        #     avg_del[model_order[i]] = np.mean(task_data_del[model_order[i]])
        #     avg_pd[model_order[i]] = np.mean(task_data_pd[model_order[i]])
        #
        # mean_del = list(avg_del.values())
        # mean_pd = list(avg_pd.values())
        # # fig, ax = plt.subplots(constrained_layout=True)
        # for i in range(len(model_order)):
        #     plt.scatter(mean_pd[i], mean_del[i], label=model_order[i], s=16)
        # plt.xlim(left=0.6)
        # plt.xlim(right=0.75)
        # plt.ylim(bottom=0)
        # plt.ylim(top=85)
        # plt.xlabel('Average packet drop')
        # plt.ylabel('Average response time [s]')
        # plt.legend(loc='lower right')
        # plt.savefig(self.output.joinpath('scatter.pdf'))
        # plt.close()
