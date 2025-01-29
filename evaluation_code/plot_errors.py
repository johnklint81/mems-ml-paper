import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

def load_all_data(models, folder_name):
    all_eigs_err = {}
    all_eigs_std_err = {}
    all_qfs_err = {}
    all_qfs_std_err = {}
    for model in models:
        eigs_err, eigs_std_err, qfs_err, qfs_std_err = load_data(model, folder_name)
        all_eigs_err.update(eigs_err)
        all_eigs_std_err.update(eigs_std_err)
        all_qfs_err.update(qfs_err)
        all_qfs_std_err.update(qfs_std_err)
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


if __name__ == "__main__":
    models = ["VGG19", "VisionTransformer16", "ResNet152"]
    dataset_names = ["th10", "th20", "ada", "full"]
    colors = ["blue", "red", "green", "black"]
    markers = ["o", "*", "s", "^"]
    dataset_name = dataset_names[2]
    folder_name = f"df_saved/{dataset_name}/"
    figure_name = dataset_name + "_errors.pdf"


    mode_vec = np.arange(0, 15)
    xtick_vec = np.arange(0, 15, 2)
    alpha = 0.7
    alpha1 = 0.4

    all_eigs_err, all_eigs_std_err, all_qfs_err, all_qfs_std_err = load_all_data(models, folder_name)
    print(all_eigs_err)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    c = 0
    for key, value in all_eigs_err.items():
        ax[0].plot(mode_vec, value, label=key, color=colors[c], marker=markers[c], zorder=1)
        c += 1
    ax[0].legend(fontsize=12, loc="lower right")
    ax[0].set_xticks(xtick_vec)
    ax[0].set_yticks([0, 0.01, 0.02, 0.03, 0.04])
    # ax.set_xticks(eigen_labels)
    # ax.tick_params(axis='x', labelrotation=45)
    ax[0].set_xlabel("Eigenmode")
    ax[0].set_ylabel("Relative $f$-error")
    # fig.suptitle("Quality factors")
    ax[0].grid()
    text_x = 0.98
    text_y = 0.96
    ax[0].text(text_x, text_y, dataset_name, transform=ax[0].transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='lightgrey', facecolor='white',
                          alpha=0.8))
    plt.tight_layout()

    c = 0
    for key, value in all_qfs_err.items():
        ax[1].plot(mode_vec, value, label=key, color=colors[c], marker=markers[c], zorder=1)
        c += 1
    ax[1].legend(fontsize=12, loc="lower right")
    ax[1].set_xticks(xtick_vec)
    ax[1].set_yticks([0, 0.1, 0.2, 0.3])
    # ax.set_xticks(eigen_labels)
    # ax.tick_params(axis='x', labelrotation=45)
    ax[1].set_xlabel("Eigenmode")
    ax[1].set_ylabel("Relative $D_Q$-error")
    # fig.suptitle("Quality factors")
    ax[1].grid()
    text_x = 0.98
    text_y = 0.96
    ax[1].text(text_x, text_y, dataset_name, transform=ax[1].transAxes,
                fontsize=12, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor='lightgrey', facecolor='white',
                          alpha=0.8))
    plt.tight_layout()
    fig.savefig("figures/" + figure_name)
    plt.show()