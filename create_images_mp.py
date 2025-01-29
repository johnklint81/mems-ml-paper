from data_loaders import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
from queue import Empty

params_path = os.getcwd() + "/parameters/"
holes_path = os.getcwd() + "/holes/"


DPI = 72
WIDTH = 5
HEIGHT = 5


def make_img(jobs):
    while True:
        try:
            parameters = jobs.get(block=False)
        except Empty:
            break
            
        i = parameters[0]
        width = parameters[1][0] * 1e3
        depth = parameters[1][1] * 1e3
        holes_x = parameters[2]
        holes_y = parameters[3]
        holes_z = parameters[4]
        holes_radius = parameters[5]

        fig, ax = plt.subplots(1, figsize=(WIDTH, HEIGHT))
        rect = patches.Rectangle(xy=(0, 0), width=width, height=depth, color="grey")
        ax.add_patch(rect)
        ax.axis("scaled")
        ax.axis("off")
        for j, hole in enumerate(holes_x):
            hole_x = holes_x[j]
            hole_y = holes_y[j]
            hole_radius = holes_radius[j]
            circle = patches.Circle(xy=(hole_x, hole_y), radius=hole_radius, color="white")
            ax.add_patch(circle)
        fig.tight_layout()
        img_name = "img_" + str(i) + ".png"
        save_path = "images/" + img_name
        fig.savefig(save_path, dpi=DPI)
        plt.close(fig)


def boss(params, holes):
    jobs = Queue()
    holes_x = holes.loc[:, 'x_0':'x_9'].to_numpy() * 1e3
    holes_y = holes.loc[:, 'y_0':'y_9'].to_numpy() * 1e3
    holes_z = holes.loc[:, 'z_0':'z_9'].to_numpy() * 1e3
    holes_radius = holes.loc[:, 'radius_0':'radius_9'].to_numpy() * 1e3
    for i in range(len(params)):
        jobs.put([i, params[i], holes_x[i], holes_y[i], holes_z[i], holes_radius[i]])
    processes = []
    workers = 8
    for n in range(workers):
        process = Process(target=make_img, args=(jobs,))
        processes.append(process)
        process.start()


if __name__ == "__main__":
    # LOAD DATA
    params = read_params(params_path).to_numpy()
    holes = read_holes(holes_path)
    boss(params, holes)
