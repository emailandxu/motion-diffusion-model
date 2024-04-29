import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm

class TrajViz:
    def __init__(self):
        self.x_index = 0
        self.z_index = 1
        self.y_index = 2
    
    def add_arrow(self, line, position=None, direction='right', size=15, color=None):
        if color is None:
            color = line.get_color()
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        if position is None:
            position = xdata.mean()
        start_ind = np.argmin(np.abs(xdata - position))
        if direction == 'right':
            end_ind = start_ind + 1
        else:
            end_ind = start_ind - 1
        line.axes.add_patch(FancyArrowPatch((xdata[start_ind], ydata[start_ind]),
                                            (xdata[end_ind], ydata[end_ind]),
                                            arrowstyle='-|>', mutation_scale=size, color=color))
    
    def plot_xz(self, ax, title, curve):
        line = ax.plot(curve[:, self.x_index], curve[:, self.z_index], marker='o', linestyle='-', color='b')[0]
        ax.set_title(f'{title}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Z Coordinate')
        ax.grid(True)
        for i in range(len(curve) - 1):
            self.add_arrow(line, position=curve[i, self.x_index], direction='right')
    
    def plot_y(self, ax, title, curve):
        y = curve[:, self.y_index]
        t = np.arange(0, len(y))
        line = ax.plot(t, y, marker='o', linestyle='-', color='b')[0]
        ax.set_title(f'{title}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True)
        for i in range(len(y) - 1):
            self.add_arrow(line, position=t[i], direction='right')
    
    def add_arrow_to_line(self, ax, x_start, y_start, x_end, y_end, color):
        arrow = FancyArrowPatch((x_start, y_start), (x_end, y_end),
                                arrowstyle='-|>', mutation_scale=10, color=color, linewidth=2)
        ax.add_patch(arrow)
    
    def plot_xz_colored_by_y(self, ax, title, curve):
        x = curve[:, self.x_index]
        z = curve[:, self.z_index]
        y = curve[:, self.y_index]
        points = np.array([x, z]).T.reshape(-1, 1, 2)
        print(points.mean(axis=0))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = Normalize(vmin=y.min(), vmax=y.max())
        lc = LineCollection(segments, cmap='viridis', norm=norm)
        lc.set_array(y[:-1])
        ax.add_collection(lc)
        ax.autoscale()
        for i in range(len(x) - 1):
            color = cm.viridis(norm(y[i]))
            self.add_arrow_to_line(ax, x[i], z[i], x[i+1], z[i+1], color)
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Y value')
        ax.set_title(f'{title}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Z Coordinate')
        ax.grid(True)


if __name__ == "__main__":
    import sys
    npypath = sys.argv[1]    

    results = np.load(npypath, allow_pickle=True).item()

    # samples = results['samples']
    samples = results["model_inputs"]

    samples = samples.squeeze(1)
    sample = samples[0]

    print(sample.shape)

    sample_slice = slice(6,9)

    trajviz = TrajViz()
    fig, axs = plt.subplots(1, 1, figsize=(10, 10))
    axs = [axs] if type(axs) is not list else axs

    trajviz.plot_xz_colored_by_y(axs[0], "input", sample[:256, sample_slice])
    plt.show()
    # fig, axs = plt.subplots(1, 1, figsize=(10, 10)) 
    # trajviz.plot_y(axs[0], "input", sample[:256, sample_slice])
    # plt.show()