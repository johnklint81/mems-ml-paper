import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from set_data_path import set_data_path
import matplotlib.pyplot as plt
import torchvision
from read_data import *
torchvision.disable_beta_transforms_warning()
from model_cnn import *
from torch.utils.data import DataLoader as DataLoader
from torchvision.transforms import v2, ToTensor
from torch import nn
from regression_dataloaders import Ellipses
from torchvision import models
from sklearn.metrics import *
import scipy.stats as st

plt.rcParams.update({'font.size': 15})


def create_regressor(n_inputs, n_hidden, n_outputs):
    new_regressor = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_outputs))
    return new_regressor


def create_transforms():
    t0 = v2.Compose([ToTensor()])
    transforms = [t0]
    return transforms


def predict(model, test_dataloader):
    # INITIALISE LISTS TO SAVE PREDICTIONS AND LABELS
    predictions = np.empty([len(test_dataloader), n_eigenmodes * 2], dtype=object)
    comsol_values = np.empty([len(test_dataloader), n_eigenmodes * 2], dtype=object)
    # PREDICT
    for batch_index, (x, y) in enumerate(test_dataloader):
        inputs, data = x.to(device), y.to(device)
        comsol_values[batch_index, :] = np.squeeze(data.cpu().numpy())
        inputs = inputs[:, :3, :, :]
        z = model.forward(inputs)
        if not z.shape:
            z = z.expand(1)
        predictions[batch_index, :] = np.squeeze(z.cpu().detach().numpy())
    return predictions, comsol_values


def evaluate_trained_vgg19(checkpoint, n_hidden):
    # LOAD VGG16, replace classifier
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    n_inputs = model.classifier[0].in_features
    regressor = create_regressor(n_inputs, n_hidden, n_outputs)
    model.classifier = regressor
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    vgg19_test_predictions, comsol_values = predict(model, test_dataloader)
    return vgg19_test_predictions, comsol_values


def evaluate_trained_resnet152(checkpoint, n_hidden):
    # LOAD ResNet50, replace fc
    model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    n_inputs = model.fc.in_features
    regressor = create_regressor(n_inputs, n_hidden, n_outputs)
    model.fc = regressor
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    resnet152_test_predictions, comsol_values = predict(model, test_dataloader)
    return resnet152_test_predictions, comsol_values


def evaluate_trained_vt(checkpoint, n_hidden):
    # LOAD VisionTransformer replace heads
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
    print(f"Total trainable parameters: {count_parameters(model):,}")
    n_inputs = 768
    regressor = create_regressor(n_inputs, n_hidden, n_outputs)
    model.heads = regressor
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    vgg16_test_predictions, comsol_values = predict(model, test_dataloader)
    return vgg16_test_predictions, comsol_values


def compute_metrics(y_pred, y_true, n_eigenmodes=15):
    eigs_features_error = []
    qfs_features_error = []
    tot_rel_eig_error = 0
    tot_rel_qf_error = 0
    for i in range(n_eigenmodes):
        rel_error_eig = (y_pred[:, i] - y_true[:, i]) / y_true[:, i]
        rel_error_qf = (y_pred[:, n_eigenmodes + i] - y_true[:, n_eigenmodes + i]) \
                       / y_true[:, n_eigenmodes + i]
        eigs_features_error.append(rel_error_eig)
        qfs_features_error.append(rel_error_qf)
        tot_rel_eig_error += np.sum(np.abs(rel_error_eig)) / n_eigenmodes
        tot_rel_qf_error += np.sum(np.abs(rel_error_qf)) / n_eigenmodes

    tot_rel_eig_error = np.sum(tot_rel_eig_error) / n_test_samples
    tot_rel_qf_error = np.sum(tot_rel_qf_error) / n_test_samples

    rmse_sklearn = root_mean_squared_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return eigs_features_error, qfs_features_error, tot_rel_eig_error, tot_rel_qf_error, rmse_sklearn, mape

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = 'cuda' if tc.cuda.is_available() else 'cpu'
    num_workers = 16
    print(f"Inference on device: {device}")
    # saved_model_name = "ft_es_ada_pre_VisionTransformer16l" # n_hidden = 1024

    saved_model_name = "ft_100e_th10_pre_VGG19"
    # saved_model_name = "ft_100e_th10_pre_VisionTransformer16"
    # saved_model_name = "ft_es_ada_pre_ResNet152"
    model_name_list = saved_model_name.split("_")
    model_name = model_name_list[-1]
    dataset = model_name_list[2]
    print(f"Loading model: {saved_model_name}")
    print(f"Inference on dataset: {dataset}")
    (image_path, data_path, train_path,
     train_data_path, val_path, val_data_path) = set_data_path(dataset)
    n_eigenmodes = 15
    n_hidden = 256
    n_outputs = 2 * n_eigenmodes
    img_size = 224
    n_qs = 15
    save_folder = "th10"
    eigen_labels = np.arange(0, n_qs, dtype=int)
    data_train = read_data(train_data_path)
    used_eigenmodes = np.arange(0, n_eigenmodes)
    column_names = []

    for i in used_eigenmodes:
        name1 = "eigenfrequency" + str(i)
        name2 = "quality_factor" + str(i)
        column_names.append(name1)
        column_names.append(name2)
    column_names.sort()
    y_train_max = data_train.max()

    # PREPARE TRANSFORMS AND LOAD TEST SET
    transforms = create_transforms()
    ell_test = Ellipses(image_path, data_path, transforms, y_train_max)
    test_dataloader = DataLoader(ell_test, batch_size=1, num_workers=num_workers)
    n_test_samples = len(ell_test)
    print(f"Number of test samples: {n_test_samples}\n")
    # LOAD CHECKPOINT
    checkpoint = tc.load("/home/j/PycharmProjects/Master_ADS/dardel/master/saved_models/" + saved_model_name + ".ckpt")
    # INFERENCE
    if "VGG19" in model_name:
        y_pred, y_true = evaluate_trained_vgg19(checkpoint, n_hidden=n_hidden)
    elif "ResNet152" in model_name:
        y_pred, y_true = evaluate_trained_resnet152(checkpoint, n_hidden=n_hidden)
    elif "VisionTransformer16" in model_name:
        y_pred, y_true = evaluate_trained_vt(checkpoint, n_hidden=n_hidden)

    for index, max_val in enumerate(y_train_max):
        y_pred[:, index] = np.multiply(y_pred[:, index], max_val)
        y_true[:, index] = np.multiply(y_true[:, index], max_val)

    tot_eigs_features_error, tot_qfs_features_error, avg_rel_eig_error, avg_rel_qf_error, \
        rmse_sklearn, mape = compute_metrics(y_pred, y_true, n_eigenmodes)

    # PRINT AND SAVE METRICS
    header = "----------" + str(saved_model_name) + "----------"
    print(header)
    print(f"rmse: {rmse_sklearn:.6e}")
    print(f"mape: {mape:.6f}")
    print(f"Mean relative eigenfrequency error: {avg_rel_eig_error:.6f}")
    print(f"Mean relative dilution coefficient error: {avg_rel_qf_error:.6f}")

    eigs_error_df = pd.DataFrame(tot_eigs_features_error).transpose()
    qfs_error_df = pd.DataFrame(tot_qfs_features_error).transpose()

    print(f"Mean relative eigenfrequency std: {eigs_error_df.abs().std().mean():.6f}")
    print(f"Mean relative dilution coefficient std: {qfs_error_df.abs().std().mean():.6f}")

    eigs_error_df.std().to_csv(f"df_saved/{save_folder}/{saved_model_name}_eigs_err_std.csv", header=False)
    qfs_error_df.std().to_csv(f"df_saved/{save_folder}/{saved_model_name}_qfs_err_std.csv", header=False)
    eigs_error_df.abs().mean().to_csv(f"df_saved/{save_folder}/{saved_model_name}_eigs_err.csv", header=False)
    qfs_error_df.abs().mean().to_csv(f"df_saved/{save_folder}/{saved_model_name}_qfs_err.csv", header=False)
