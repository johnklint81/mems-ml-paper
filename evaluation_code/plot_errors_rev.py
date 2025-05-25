import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})


def load_all_data(models, folder_name):
  all_eigs_err = {}
  all_eigs_std_err = {}
  all_qfs_err = {}
  all_qfs_std_err = {}
  for model in models:
    eigs_err, eigs_std_err, qfs_err, qfs_std_err = load_data(model, folder_name)
    all_eigs_err[model] = eigs_err[model]
    all_eigs_std_err[model] = eigs_std_err.get(model, None)
    all_qfs_err[model] = qfs_err[model]
    all_qfs_std_err[model] = qfs_std_err.get(model, None)
  return all_eigs_err, all_eigs_std_err, all_qfs_err, all_qfs_std_err


def load_data(model_name, folder_name):
  files = [f for f in os.listdir(folder_name) if model_name in f]
  eigs_err = {}
  eigs_std_err = {}
  qfs_err = {}
  qfs_std_err = {}
  for file in files:
    key = model_name
    file_path = folder_name + file
    if "eigs" in file:
      if "std" in file:
        std_err_value = pd.read_csv(file_path, header=None)[1].to_numpy()
        eigs_std_err[key] = std_err_value
      else:
        err_value = pd.read_csv(file_path, header=None)[1].to_numpy()
        eigs_err[key] = err_value
    elif "qfs" in file:
      if "std" in file:
        std_err_value = pd.read_csv(file_path, header=None)[1].to_numpy()
        qfs_std_err[key] = std_err_value
      else:
        err_value = pd.read_csv(file_path, header=None)[1].to_numpy()
        qfs_err[key] = err_value
  return eigs_err, eigs_std_err, qfs_err, qfs_std_err


def plot_grouped_bar(data_dict, ylabel, ax, colors, dataset_name,
                     ylim=None, yticks=None, text_on_right=True):
  models = list(data_dict.keys())
  num_modes = len(next(iter(data_dict.values())))
  x = np.arange(num_modes)
  bar_width = 0.18
  hatch_patterns = ['/', '', '\\', '']

  for i, model in enumerate(models):
    offset = (i - len(models) / 2) * bar_width + bar_width / 2
    ax.bar(x + offset, data_dict[model], width=bar_width, label=model,
           color=colors[i], edgecolor='black', hatch=hatch_patterns[i % len(hatch_patterns)])

  ax.set_xticks(x)
  # Set x-limits with slight padding around the outermost bars
  total_width = len(models) * bar_width
  padding = bar_width * 1.1
  ax.set_xlim(x[0] - total_width / 2 - padding, x[-1] + total_width / 2 + padding)

  ax.set_xlabel("Eigenmode")

  ax.set_ylabel(ylabel)
  if ylim: ax.set_ylim(ylim)
  if yticks: ax.set_yticks(yticks)
  ax.grid(True, axis='y', linestyle='--', alpha=0.6)

  # Configurable placement
  if text_on_right:
    ax.legend(fontsize=16, loc="upper left")
    ax.text(0.98, 0.96, dataset_name, transform=ax.transAxes,
            fontsize=18, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='lightgrey', facecolor='white', alpha=0.8))
  else:
    ax.legend(fontsize=16, loc="upper right")
    ax.text(0.02, 0.96, dataset_name, transform=ax.transAxes,
            fontsize=18, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='lightgrey', facecolor='white', alpha=0.8))


if __name__ == "__main__":
  models = ["VGG19", "VisionTransformer16", "ResNet152", "FNN"]
  dataset_names = ["th10", "th20", "ada", "full"]
  colors = [
    "#009E73",  # green
    "#E69F00",  # orange
    "#0072B2",  # blue
    "#D55E00",  # vermillion
  ]

  dataset_name = dataset_names[2]
  folder_name = f"../df_saved/{dataset_name}/"
  figure_name = dataset_name + "_errors_bar.pdf"

  all_eigs_err, all_eigs_std_err, all_qfs_err, all_qfs_std_err = load_all_data(models, folder_name)

  fig, ax = plt.subplots(2, 1, figsize=(16, 10))
  plot_grouped_bar(all_eigs_err, "Relative $f$-error", ax[0], colors, dataset_name,
                   yticks=[0, 0.02, 0.04, 0.06], text_on_right=False)

  plot_grouped_bar(all_qfs_err, "Relative $D_Q$-error", ax[1], colors, dataset_name,
                   ylim=[0, 0.46], yticks=[0, 0.2, 0.4, 0.6], text_on_right=True)

  plt.tight_layout()
  fig.savefig("../figures/" + figure_name, bbox_inches='tight', pad_inches=0)
  plt.show()
