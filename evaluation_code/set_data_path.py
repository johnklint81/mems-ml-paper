def set_data_path(dataset):
    image_path = []
    data_path = []

    train_path = []
    train_data_path = []

    val_path = []
    val_data_path = []
    if dataset == "th20":
        image_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold20/imgs/test/high_qf/"
        data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold20/data/test/high_qf/"

        train_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold20/imgs/train/high_qf/"
        train_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold20/data/train/high_qf/"

        val_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold20/imgs/test/high_qf/"
        val_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold20/data/test/high_qf/"
    elif dataset == "th10":
        image_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold10/imgs/test/high_qf/"
        data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold10/data/test/high_qf/"

        train_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold10/imgs/train/high_qf/"
        train_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold10/data/train/high_qf/"

        val_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold10/imgs/test/high_qf/"
        val_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/threshold10/data/test/high_qf/"
    elif dataset == "ada":
        image_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/adaptive/imgs/test/high_qf/"
        data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/adaptive/data/test/high_qf/"

        train_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/adaptive/imgs/train/high_qf/"
        train_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/adaptive/data/train/high_qf/"

        val_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/adaptive/imgs/val/high_qf/"
        val_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/all_modes_curated/adaptive/data/val/high_qf/"
    elif dataset == "full":
        image_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/images_split/test/real/"
        data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/data_split/test/real/"

        train_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/images_split/train/real/"
        train_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/data_split/train/real/"

        val_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/images_split/val/real/"
        val_data_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/data_split/val/real/"
    return image_path, data_path, train_path, train_data_path, val_path, val_data_path
