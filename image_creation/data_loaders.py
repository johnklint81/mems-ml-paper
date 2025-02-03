import os
import pandas as pd


def read_params(params_path):
    params = {}
    for file in os.listdir(params_path):
        filename = os.fsdecode(file)
        filepath = params_path + filename
        with open(filepath) as f:
            item = []
            for row in f.readlines():
                if row[0] != "#":
                    row = row.split(",")
                    index = row[0]
                    item = [row[1], row[2], row[3]]
            params[index] = item
    params = pd.DataFrame.from_dict(params, orient='index')
    params.index = params.index.astype(int)
    params = params.astype(float)
    params = params.sort_index()
    params = params.rename(columns={0: "width", 1: "depth", 2: "height"})
    return params


def read_holes(data_path):
    holes = {}
    for file in os.listdir(data_path):
        filename = os.fsdecode(file)
        filepath = data_path + filename
        with open(filepath) as f:
            xs = []
            ys = []
            zs = []
            radii = []
            index = int(filename.split("_")[1][:-4])
            for row in f.readlines():
                r = row.split(",")
                if row[0] != "#" and row[0] != "\n":
                    xs.append(float(r[1]))
                    ys.append(float(r[2]))
                    zs.append(float(r[2]))
                    radii.append(float(r[3]))
            item = xs + ys + zs + radii
            holes[index] = item
    holes = pd.DataFrame.from_dict(holes, orient='index')
    holes = holes.sort_index(axis=0)
    n_holes = int(len(item) / 4)
    for i in range(n_holes):
        x_col = "x_" + str(i)
        y_col = "y_" + str(i)
        z_col = "z_" + str(i)
        radius_col = "radius_" + str(i)
        holes = holes.rename(columns={i: x_col})
        holes = holes.rename(columns={i + n_holes: y_col})
        holes = holes.rename(columns={i + 2 * n_holes: z_col})
        holes = holes.rename(columns={i + 3 * n_holes: radius_col})

    return holes
