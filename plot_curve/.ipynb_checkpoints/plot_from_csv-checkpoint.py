"""
Helper script used for plotting the learning curves for existing experiments.

Usage:
```bash
python -m playground.plot_from_csv --load_paths runs/*/*/  --columns mean_rew max_rew  --smooth 2

# to group the results based on the name
python -m playground.plot_from_csv --load_paths runs/*/*/  --columns mean_rew max_rew  --name_regex ".*__([^_\/])*" --group 1
```
"""
from __future__ import absolute_import, division, print_function
from builtins import bytes, str, open, super, range, zip, round, input, int, pow, object

import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import argparse
import warnings
import math
import csv
import re
import os
import sys

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import warnings
from matplotlib import cm

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import time

plt.ion()
# matplotlib.use("TkAgg", warn=False, force=True)


class Plot(object):
    def __init__(self, title="", nrows=1, ncols=1, parent=None):
        self.fig = None
        self.nrows = nrows
        self.ncols = ncols

        if parent is None:
            self.parent = self
            # TODO: make up a name
            self.fig = plt.figure(figsize=(6.5 * nrows, 6.5 * ncols))
            if title:
                self.fig.canvas.set_window_title(title)
            self.subplot_cnt = 0
            # self.fig, self.subplots = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows))
            # self.subplots = np.array(self.subplots).reshape(-1)[::-1].tolist()
        else:
            self.parent = parent

    def _get_subplot(self, projection=None):
        self.subplot_cnt += 1
        return self.fig.add_subplot(
            self.ncols, self.nrows, self.subplot_cnt, projection=projection
        )

    def _redraw(self):
        if self.fig:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.parent._redraw()

    # Originally from: http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    def get_image(self):
        """
        @brief Convert its own figure to a 3D numpy array with RGB channels and return it
        @return a numpy 3D array of RGB values
        """
        if self.fig is None:
            return self.parent.get_image()
        # draw the renderer  .. TODO: remove?
        self.fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = self.fig.canvas.get_width_height()
        buf = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(
            (h, w, 3)
        )

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = numpy.roll ( buf, 3, axis = 2 )
        return buf / 256  # .transpose((2, 0, 1))


class LinePlot(Plot):
    COLORS = [
        "#ff0000",
        "#00035b",
        "#feb308",
        "#1B5E20",
        "#017b92",
        "#a2cffe",
        "#ff028d",
        "#8b2e16",
        "#916e99",
        "#b9ff66",
        "#000000",
        "#7bb274",
        "#ff000d",
    ]

    def __init__(
        self,
        xlim=[np.inf, -np.inf],
        ylim=[np.inf, -np.inf],
        num_scatters=1,
        plot_type="-",
        title="",
        xlabel="",
        ylabel="",
        ylog_scale=False,
        xlog_scale=False,
        alpha=1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.subplot = self.parent._get_subplot()
        self.subplot.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        if num_scatters > len(self.COLORS):
            warnings.warn(
                "Not enough colors for plotting: the same color may be re-used"
            )
        if ylog_scale:
            self.subplot.set_yscale("log")
        if xlog_scale:
            self.subplot.set_xscale("log")
        self.sc = [
            self.subplot.plot(
                [], [], plot_type, color=self.COLORS[i % len(self.COLORS)], alpha=alpha
            )[0]
            for i in range(num_scatters)
        ]
        # self.subplot.set_xlim(*xlim)
        # self.subplot.set_ylim(*ylim)
        self.subplot.set_title("Learning Curve of Experiments")
        self.subplot.set_xlabel("Number of Examples")
        self.subplot.set_ylabel("Est. Dev Accuracy")
        self.xlim = xlim
        self.ylim = [np.inf, -np.inf]
        self._redraw()

    def add_point(self, x, y, line_num=0, redraw=True):
        xs = np.append(self.sc[line_num].get_xdata(), [x])
        ys = np.append(self.sc[line_num].get_ydata(), [y])
        # xs = self.sc[line_num].get_xdata()
        # ys = self.sc[line_num].get_ydata()
        self.sc[line_num].set_xdata(xs)
        self.sc[line_num].set_ydata(ys)
        self.xlim = [min(self.xlim[0], x), max(self.xlim[1], x)]
        self.ylim = [min(self.ylim[0], y), max(self.ylim[1], y)]
        self.subplot.set_xlim(*self.xlim)
        self.subplot.set_ylim(*self.ylim)
        if redraw:
            self._redraw()

    def update(self, points, line_num=0):
        """
            points: should have the shape (N*2) and consist of x,y coordinates
        """
        self.sc[line_num].set_data(points[:, 0], points[:, 1])
        self.xlim = [
            min(self.xlim[0], points[:, 0].min()),
            max(self.xlim[1], points[:, 0].max()),
        ]
        self.ylim = [
            min(self.ylim[0], points[:, 1].min()),
            max(self.ylim[1], points[:, 1].max()),
        ]
        self.subplot.set_xlim(*self.xlim)
        self.subplot.set_ylim(*self.ylim)
        self._redraw()

    def fill_between(self, x, ymin, ymax, alpha=0.1, line_num=0):
        color = self.COLORS[line_num % len(self.COLORS)]
        self.subplot.fill_between(x, ymin, ymax, alpha=alpha, color=color)
        self.xlim = [min(self.xlim[0], x.min()), max(self.xlim[1], x.max())]
        self.ylim = [min(self.ylim[0], ymin.min()), max(self.ylim[1], ymax.max())]
        self.subplot.set_xlim(*self.xlim)
        self.subplot.set_ylim(*self.ylim)


class ScatterPlot(Plot):
    def __init__(
        self,
        value_range=[-1, 1],
        xlim=[np.inf, -np.inf],
        ylim=[np.inf, -np.inf],
        palette="seismic",
        title="",
        xlabel="",
        ylabel="",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        cmap = plt.get_cmap(palette)
        norm = matplotlib.colors.Normalize(*value_range)
        # FIXME: ignoring s=scale,
        self.subplot = self.parent._get_subplot()
        self.sc = self.subplot.scatter(
            x=[], y=[], c=[], norm=norm, cmap=cmap, alpha=0.8, edgecolors="none"
        )
        self.xlim = xlim
        self.ylim = ylim
        self.subplot.set_title(title)
        self.subplot.set_xlabel(xlabel)
        self.subplot.set_ylabel(ylabel)
        # self.subplot.set_xlim(*xlim)
        # self.subplot.set_ylim(*ylim)
        self._redraw()

    def update(self, points, values):
        """
            points: should have the shape (N*2) and consist of x,y coordinates
            values: should have the shape (N) and consist of the values at these coordinates (i.e. points)
        """
        # self.sc.set_offsets(np.c_[x,y])
        self.sc.set_offsets(points)
        self.sc.set_array(values)
        self._redraw()

    def add_point(self, point, value=1):
        self.sc.set_offsets(
            np.concatenate([self.sc.get_offsets(), np.array(point, ndmin=2)])
        )
        self.sc.set_array(
            np.concatenate([self.sc.get_array(), np.array(value, ndmin=1)])
        )
        self.xlim = [min(self.xlim[0], point[0]), max(self.xlim[1], point[0])]
        self.ylim = [min(self.ylim[0], point[1]), max(self.ylim[1], point[1])]
        xextra = max(2, self.xlim[1] - self.xlim[0]) / 10
        yextra = max(2, self.ylim[1] - self.ylim[0]) / 10
        # self.subplot.set_xlim(self.xlim[0] - xextra, self.xlim[1] + xextra)
        # self.subplot.set_ylim(self.ylim[0] - yextra, self.ylim[1] + yextra)
        self._redraw()


class SurfacePlot(Plot):
    def __init__(
        self,
        value_range=[-1, 1],
        xlim=[-1, 1],
        ylim=[-1, 1],
        palette="seismic",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        # cmap = plt.get_cmap(palette)
        # norm = matplotlib.colors.Normalize(*value_range)
        # FIXME: ignoring s=scale,
        self.subplot = self.parent._get_subplot(projection="3d")
        self.subplot.set_xlabel("X")
        self.subplot.set_ylabel("Y")
        empty = np.zeros((0, 0))
        self.sc = self.subplot.plot_surface(
            empty, empty, empty, cmap=cm.coolwarm, alpha=0.8, edgecolors="none"
        )
        self.subplot.set_xlim(*xlim)
        self.subplot.set_ylim(*ylim)
        self._redraw()

    def update(self, X, Y, Z):
        """
            points: should have the shape (N*2) and consist of x,y coordinates
            values: should have the shape (N) and consist of the values at these coordinates (i.e. points)
        """
        # self.sc.set_offsets(np.c_[x,y])
        self.sc.remove()
        self.sc = self.subplot.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.8)
        self.subplot.view_init(azim=0, elev=90)
        # self.sc.set_offsets(points)
        # self.sc.set_array(values)
        self._redraw()


class QuiverPlot(Plot):
    def __init__(self, xlim=[-1, 1], ylim=[-1, 1], palette="seismic", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.subplot = self.parent._get_subplot()
        norm = np.linalg.norm([xlim[1] - xlim[0], ylim[1] - ylim[0]])
        self.sc = self.subplot.quiver(
            [], [], [], [], units="xy", pivot="middle", scale=40 / norm
        )
        self.subplot.set_xlim(*xlim)
        self.subplot.set_ylim(*ylim)
        self._redraw()

    def update(self, points, dirs):
        """
            points: should have the shape (N*2) and consist of x,y coordinates
            values: should have the shape (N*2) and consist of the vector values in these coordinates (i.e. points)
        """
        self.sc.set_offsets(points)
        U, V = np.array(dirs).transpose()
        M = np.hypot(U, V)
        self.sc.set_UVC(U / M, V / M, M)
        self._redraw()

def str2bool(v):
    """
    Argument Parse helper function for boolean values
    """
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_paths", type=str, nargs="+")
    parser.add_argument("--columns", type=str, nargs="+")
    parser.add_argument("--row", type=str, default="total_num_steps")
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--smoothing", type=float, default=0)
    parser.add_argument("--group", type=str2bool, default=False)
    parser.add_argument("--log_scale", type=str2bool, default=False)
    parser.add_argument("--xlog_scale", type=str2bool, default=False)
    parser.add_argument("--legend", type=str2bool, default=True)
    parser.add_argument("--name_regex", type=str, default="")
    parser.add_argument("--final", type=str2bool, default=False)
    parser.add_argument("--save", type=str, default="figure.png")
    args = parser.parse_args()

    N = len(args.columns)
    nrows = math.floor(math.sqrt(N))
    title = "Results with smoothing %.1f" % args.smoothing
    if args.log_scale:
        title += " (Log Scale)"
    plot = Plot(nrows=nrows, ncols=math.ceil(N / nrows), title=title)
    plots = []
    for column in args.columns:
        if args.final:
            plots.append(ScatterPlot(parent=plot, ylabel=column, xlabel=args.row))
        else:
            plots.append(
                LinePlot(
                    parent=plot,
                    ylabel=column,
                    xlabel=args.row,
                    ylog_scale=args.log_scale,
                    xlog_scale=args.xlog_scale,
                    alpha=args.alpha,
                    num_scatters=len(args.load_paths),
                )
            )

    smoothing_method = lambda x: x
    if args.smoothing > 0.1:
        smoothing_method = lambda x: gaussian_filter1d(x, sigma=args.smoothing)

    if args.name_regex:
        legends = [re.findall(args.name_regex, path)[0] for path in args.load_paths]
        legend_paths = sorted(zip(legends, args.load_paths))
        legends = [x[0] for x in legend_paths]
        args.load_paths = [x[1] for x in legend_paths]
    else:
        common_prefix = os.path.commonprefix(args.load_paths)
        warnings.warn("Ignoring the prefix (%s) in the legend" % common_prefix)
        legends = [path[len(common_prefix) :] for path in args.load_paths]

    data = [load_path(path, args) for path in args.load_paths]

    # getting rid of
    legends = [l for (l, d) in zip(legends, data) if d is not None]
    data = [d for d in data if d is not None]

    if args.group:
        data, legends = compute_group_data(data, legends, args.row, args.columns)

    if args.legend:
        plot.fig.legend(legends, loc=(0.6, 0.3))

    for i, df in enumerate(data):
        for j, column in enumerate(args.columns):
            if args.final:
                y = df[column][-1:].item()
                x = float(legends[i])
                plots[j].add_point([x, y])
            else:
                df[column] = smoothing_method(df[column])
                plots[j].update(df[[args.row, column]].values, line_num=i)
                if args.group:
                    plots[j].fill_between(
                        df[args.row],
                        smoothing_method(df[column + "_min"]),
                        smoothing_method(df[column + "_max"]),
                        line_num=i,
                    )

    plt.savefig(args.save)
    plt.ioff()
    plt.show()


def load_path(path, args):
    print("Loading ... ", path)
    filename = os.path.join(path, "evaluate.csv" if args.final else "progress.csv")
    try:
        return pd.read_csv(filename)
    except:
        warnings.warn("Could not load %s" % filename)
        return None


def compute_group_data(data, group_names, row, columns):
    groups = {}
    for df, gname in zip(data, group_names):
        if gname not in groups:
            groups[gname] = []
        groups[gname].append(df)

    for key, dfs in groups.items():
        # Find shortest dataframe in each group
        # and truncate to that dataframe
        min_ = np.min([len(df[columns[0]]) for df in dfs])
        for i, _ in enumerate(dfs):
            dfs[i] = dfs[i].truncate(after=min_ - 1)

    stat_funcs = {"": np.mean, "_min": np.min, "_max": np.max}

    for key, dfs in groups.items():
        df_column_names = [row] + [
            c + fname for c in columns for fname in stat_funcs.keys()
        ]
        stats = np.concatenate(
            [
                [dfs[0][row]],
                [
                    func([df[c] for df in dfs], axis=0)
                    for c in columns
                    for func in stat_funcs.values()
                ],
            ]
        )
        groups[key] = pd.DataFrame(stats.transpose(), columns=df_column_names)

    return list(groups.values()), list(groups.keys())


if __name__ == "__main__":
    main()