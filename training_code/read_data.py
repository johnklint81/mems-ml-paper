import os
import pandas as pd
import numpy as np


def read_data(data_path):
    data = {}
    for file in os.listdir(data_path):
        filename = os.fsdecode(file)
        filepath = data_path + filename
        with open(filepath) as f:
            eigs = []
            qs = []
            index = int(filename.split(".")[0][5:])
            for row in f.readlines():
                r = row.split(",")
                if row[0] != "#" and row[0] != "\n":
                    eigs.append(float(r[0]))
                    qs.append(float(r[1]))
            item = eigs + qs
            data[index] = item
    data = pd.DataFrame.from_dict(data, orient='index')
    data = data.sort_index()
    for i in range(len(eigs)):
        ef_name = "eigenfrequency" + str(i)
        qf_name = "quality_factor" + str(i)
        data = data.rename(columns={i: ef_name, i + len(eigs): qf_name})
    return data


def read_params(params_path):
    params = {}
    for file in os.listdir(params_path):
        filename = os.fsdecode(file)
        filepath = params_path + filename
        with open(filepath) as f:
            item = []
            index = filename.split(".")[0][11:]
            for row in f.readlines():
                if row[0] != "#":
                    row = row.split(",")
                    item = [row[0], row[1], row[2], row[3]]
            params[index] = item
    params = pd.DataFrame.from_dict(params, orient='index')
    params.index = params.index.astype(int)
    params = params.astype(float)
    params = params.sort_index()
    params = params.rename(columns={0: "width", 1: "depth", 2: "height", 3: "stress"})
    return params


def read_holes(holes_path, max_holes=10):
    holes = {}
    for file in os.listdir(holes_path):
        filename = os.fsdecode(file)
        filepath = holes_path + filename
        with open(filepath) as f:
            x = np.zeros([max_holes])
            y = np.zeros([max_holes])
            x_axis = np.zeros([max_holes])
            y_axis = np.zeros([max_holes])
            angle = np.zeros([max_holes])
            index = int(filename.split(".")[0][6:])
            counter = 0
            for row in f.readlines():
                r = row.split(",")
                if row[0] != "#" and row[0] != "\n":
                    x[counter] = r[1]
                    y[counter] = r[2]
                    x_axis[counter] = r[3]
                    y_axis[counter] = r[4]
                    angle[counter] = r[5]
                    counter += 1
            item = list(x) + list(y) + list(x_axis) + list(y_axis) + list(angle)
            holes[index] = item
    holes = pd.DataFrame.from_dict(holes, orient='index')
    holes = holes.sort_index()
    for i in range(max_holes):
        x_name = "x_" + str(i)
        y_name = "y_" + str(i)
        x_axis_name = "x_axis_" + str(i)
        y_axis_name = "y_axis_" + str(i)
        angle_name = "angle_" + str(i)
        holes = holes.rename(columns={i: x_name, i + max_holes: y_name, i + 2 * max_holes: x_axis_name,
                                    i + 3 * max_holes: y_axis_name, i + 4 * max_holes: angle_name})
    return holes
