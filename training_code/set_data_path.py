def set_data_path(dataset):
    image_path = []
    data_path = []

    train_path = []
    train_data_path = []

    val_path = []
    val_data_path = []
    # !!! REMOVE klintj FROM FILE PATHS BEFORE SUBMISSION !!!
    if dataset == "th20":
        image_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated20/imgs/"
        data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated20/data/"

        train_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated20/imgs/train/high_qf/"
        train_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated20/data/train/high_qf/"

        val_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated20/imgs/val/high_qf/"
        val_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated20/data/val/high_qf/"
    elif dataset == "th10":
        image_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated/imgs/"
        data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated/data/"

        train_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated/imgs/train/high_qf/"
        train_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated/data/train/high_qf/"

        val_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated/imgs/val/high_qf/"
        val_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_curated/data/val/high_qf/"

    elif dataset == "ada":
        image_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_adaptive/imgs/"
        data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_adaptive/data/"

        train_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_adaptive/imgs/train/high_qf/"
        train_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_adaptive/data/train/high_qf/"

        val_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_adaptive/imgs/val/high_qf/"
        val_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/all_modes_adaptive/data/val/high_qf/"

    elif dataset == "full":
        image_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/images_split/"
        data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/data_split/"

        train_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/images_split/train/real/"
        train_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/data_split/train/real/"

        val_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/images_split/val/real/"
        val_data_path = "/mimer/NOBACKUP/groups/nanophotonics_ml/klintj/data_split/val/real/"
    return image_path, data_path, train_path, train_data_path, val_path, val_data_path


def set_save_paths():
    figure_save_path = "/cephyr/users/klintj/Alvis/Figures/"
    model_save_path = "/cephyr/users/klintj/Alvis/Models/"
    return figure_save_path, model_save_path
