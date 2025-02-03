import torchvision
torchvision.disable_beta_transforms_warning()
from model_cnn import *
from torch.utils.data import DataLoader as DataLoader
from torch import nn
from main_ellipse_classification_vgg19 import Ellipses, create_transforms, output_to_label
from torchvision import models
from sklearn.metrics import *
from data_distribution import *

plt.rcParams.update({'font.size': 14})


# def create_new_classifier1(n_inputs, n_hidden):
#     new_classifier = nn.Sequential(
#         nn.Linear(n_inputs, n_hidden),
#         nn.ReLU(),
#         nn.Linear(n_hidden, 1),
#         nn.Flatten(start_dim=0),
#         nn.Sigmoid())
#     return new_classifier


def create_new_classifier2(n_inputs, n_hidden):
    new_classifier = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.Linear(n_hidden, 1),
        nn.Flatten(start_dim=0),
        nn.Sigmoid())
    return new_classifier


def create_new_classifier3(n_inputs, n_hidden):
    new_classifier = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, 1),
        nn.Flatten(start_dim=0),
        nn.Sigmoid())
    return new_classifier


def predict(model, test_dataloader):
    # INITIALISE LISTS TO SAVE PREDICTIONS AND LABELS
    test_predictions = np.array([])
    test_labels = []
    test_probs = np.array([])
    # PREDICT
    for batch_index, (x, y) in enumerate(test_dataloader, 1):
        inputs, labels = x.to(device), y.to(device)
        test_labels = np.append(test_labels, labels.cpu().numpy()[0])
        inputs = inputs[:, :3, :, :]
        z = model.forward(inputs)
        probs = z.detach().cpu().numpy()
        test_probs = np.append(test_probs, probs)
        if not z.shape:
            z = z.expand(1)
        hard_preds = output_to_label(z)
        test_predictions = np.append(test_predictions, hard_preds.cpu().numpy()[0])
    return test_predictions, test_labels, test_probs


def evaluate_trained_vgg19(checkpoint, n_hidden):
    # LOAD VGG19, REPLACE HEAD
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    n_inputs = model.classifier[0].in_features
    new_classifier = create_new_classifier2(n_inputs, n_hidden)
    model.classifier = new_classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    vgg19_test_predictions, test_labels, test_probs = predict(model, test_dataloader)
    return vgg19_test_predictions, test_labels, test_probs


def evaluate_trained_resnet(checkpoint, n_hidden):
    # LOAD ResNet, REPLACE HEAD
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    n_inputs = model.fc.in_features
    new_classifier = create_new_classifier2(n_inputs, n_hidden)
    model.fc = new_classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    resnet_test_predictions, test_labels, test_probs = predict(model, test_dataloader)
    return resnet_test_predictions, test_labels, test_probs


def evaluate_trained_vt(checkpoint, n_hidden):
    # LOAD ResNet, REPLACE HEAD
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    n_inputs = 768
    new_classifier = create_new_classifier3(n_inputs, n_hidden)
    model.heads = new_classifier
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    resnet_test_predictions, test_labels, test_probs = predict(model, test_dataloader)
    return resnet_test_predictions, test_labels, test_probs


def compute_metrics(test_predictions, test_labels, model_name):
    # COMPUTE RESULTS
    tot = len(test_predictions)
    tp = np.dot(test_predictions, test_labels)
    tn = np.dot(1 - test_predictions, 1 - test_labels)
    fp_list = [1 if (test_predictions[i] == 1) and (test_labels[i] == 0) else 0 for i in range(tot)]
    fp = sum(fp_list)
    fn_list = [1 if (test_predictions[i] == 0) and (test_labels[i] == 1) else 0 for i in range(tot)]
    fn = sum(fn_list)

    # COMPUTE METRICS
    acc = (tp + tn) / (fp + fn + tp + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)

    # PRINT METRICS
    header = "----------" + str(model_name) + "----------"
    print(header)
    print(f"accuracy: {acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f1 score: {f1}\n")

    return fp_list, fn_list


def plot_cm(y_true, y_pred, ax, palette, model_name):
    class_labels = ["Buckling", "No buckling"]
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            # annot[i, j] = '%.1f%%\n%d' % (p, c)
            annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    hm = sns.heatmap(cm,
                cmap=palette,
                annot=annot,
                fmt='',
                cbar=False,
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax,
                annot_kws={'size': 22})
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=18)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=18)
    ax.set_ylabel(cm.index.name, fontsize=20, weight="semibold")
    ax.set_xlabel(cm.columns.name, fontsize=20, weight="semibold")
    ax.text(0.98, 0.98, model_name, fontsize=22, fontweight="roman", ha="right", va="top", transform=ax.transAxes)


def plot_cm_example(y_true, y_pred, ax, palette):
    class_labels = ["Positive", "Negative"]
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    annot = np.empty_like(cm).astype(str)

    annot[0, 0] = 'TPR\nTP'
    annot[0, 1] = 'FNR\nFN'
    annot[1, 0] = 'FPR\nFP'
    annot[1, 1] = 'TNR\nTN'

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    hm = sns.heatmap(cm,
                cmap=palette,
                annot=annot,
                fmt='',
                cbar=False,
                xticklabels=class_labels,
                yticklabels=class_labels,
                ax=ax,
                annot_kws={'size': 20})
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=16)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=16)
    ax.set_ylabel(cm.index.name, size=18, weight="semibold")
    ax.set_xlabel(cm.columns.name, size=18, weight="semibold")


def get_misclassified_data(data, holes, fx_list, ell):
    fx_indices = []
    for fx in fx_list:
        fx_indices.append([ell.get_id(index) for index, i in enumerate(fx) if i == 1])
    fx_indices = list(set.intersection(*[set(i) for i in fx_indices]))
    fx_data = data.query(f"index == {fx_indices}")

    fx_holes = holes.query(f"index == {fx_indices}")
    return fx_data, fx_holes


def plot_fp_data(data, holes, ax, palette, model_name="", colourbar=True):
    plot_data = data.copy(deep=True)
    plot_data.loc[:, "eigenfrequency0":"eigenfrequency14"] = plot_data.loc[:, "eigenfrequency0":"eigenfrequency14"] * 1e-6
    plot_eig_q_hole_area(plot_data, holes, ax, palette, colourbar=colourbar)
    ax.text(0.98, 0.98, model_name, fontsize=20, fontweight="roman", ha="right", va="top", transform=ax.transAxes)


if __name__ == "__main__":
    device = 'cuda' if tc.cuda.is_available() else 'cpu'
    num_workers = 16

    path = ""  # Add full path if data is stored elsewhere
    data_path = path + "data/test_small/"
    hole_path = path + "holes/test_small/"
    test_path = path + "images/test_small/"

    # PREPARE TRANSFORMS AND LOAD TEST SET
    transforms = create_transforms()
    ell_test = Ellipses(test_path, transforms)
    test_dataloader = DataLoader(ell_test, batch_size=1, num_workers=num_workers)
    n_test_samples = len(ell_test)
    n_hidden = 64
    print(f"Number of test samples: {n_test_samples}\n")

    real_data = read_data(data_path + "real/")
    real_holes = read_holes(hole_path + "real/")
    imag_data = read_data(data_path + "complex/")
    imag_holes = read_holes(hole_path + "complex/")

    fp_all = []
    fn_all = []

    fig_all_cm, ax_all_cm = plt.subplots(2, 2, figsize=(14, 12))
    fig_all_fp, ax_all_fp = plt.subplots(2, 2, figsize=(14, 12))

    # VGG19 - Finetuned
    checkpoint = tc.load(path + "/saved_models/vgg19_64_300e_finetuned.ckpt")
    vgg19_test_preds, vgg19_test_labels, vgg19_test_probs = evaluate_trained_vgg19(checkpoint, n_hidden=n_hidden)
    fp_vgg19, fn_vgg19 = compute_metrics(vgg19_test_preds, vgg19_test_labels, "VGG19 - Finetuned")
    fp_data_vgg19, fp_holes_vgg19 = get_misclassified_data(real_data, real_holes, [fp_vgg19], ell_test)
    fp_all.append(fp_vgg19)
    fn_all.append(fn_vgg19)

    plot_fp_data(fp_data_vgg19, fp_holes_vgg19, ax_all_fp[0, 1], sns.color_palette("Reds", as_cmap=True), "VGG19")
    plot_cm(vgg19_test_labels, vgg19_test_preds, ax_all_cm[0, 1], sns.color_palette("Reds", as_cmap=True), "VGG19")

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_fp_data(fp_data_vgg19, fp_holes_vgg19, ax, sns.color_palette("Reds", as_cmap=True), "VGG19")
    fig.tight_layout()
    fig.savefig("figures/fp_vgg19_small.pdf", dpi=300)

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_cm(vgg19_test_labels, vgg19_test_preds, ax, sns.color_palette("Reds", as_cmap=True), "VGG19")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrix_vgg19_small.pdf", dpi=300)

    # ResNet - Finetuned
    checkpoint = tc.load(path + "/saved_models/resnet50_64_300e_finetuned.ckpt")
    resnet_test_preds, resnet_test_labels, resnet_test_probs = evaluate_trained_resnet(checkpoint, n_hidden=n_hidden)
    fp_resnet, fn_resnet = compute_metrics(resnet_test_preds, resnet_test_labels, "ResNet - Finetuned")
    fp_data_resnet, fp_holes_resnet = get_misclassified_data(real_data, real_holes, [fp_resnet], ell_test)
    fp_all.append(fp_resnet)
    fn_all.append(fn_resnet)

    plot_fp_data(fp_data_resnet, fp_holes_resnet, ax_all_fp[1, 0], sns.color_palette("RdPu", as_cmap=True), "ResNet")
    plot_cm(resnet_test_labels, resnet_test_preds, ax_all_cm[1, 0], sns.color_palette("RdPu", as_cmap=True), "ResNet")

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_cm(resnet_test_labels, resnet_test_preds, ax, sns.color_palette("RdPu", as_cmap=True), "ResNet")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrix_ResNet_small.pdf", dpi=300)

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_fp_data(fp_data_resnet, fp_holes_resnet, ax, sns.color_palette("RdPu", as_cmap=True), "ResNet")
    fig.tight_layout()
    fig.savefig("figures/fp_ResNet_small.pdf", dpi=300)

    # Vision Transformer - Finetuned
    checkpoint = tc.load(path + "/saved_models/vis_transform16b_64_300e_finetuned.ckpt")
    vt_test_preds, vt_test_labels, vt_test_probs = evaluate_trained_vt(checkpoint, n_hidden=n_hidden)
    fp_vt, fn_vt = compute_metrics(vt_test_preds, vt_test_labels, "Vision Transformer - Finetuned")
    fp_data_vt, fp_holes_vt = get_misclassified_data(real_data, real_holes, [fp_vt], ell_test)
    fp_all.append(fp_vt)
    fn_all.append(fn_vt)

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_cm(vt_test_labels, vt_test_preds, ax, sns.color_palette("Greys", as_cmap=True), "ViT")
    fig.tight_layout()
    fig.savefig("figures/confusion_matrix_ViT_small.pdf", dpi=300)

    fig, ax = plt.subplots(1, figsize=(6, 5))
    plot_fp_data(fp_data_vt, fp_holes_vt, ax, sns.color_palette("Greys", as_cmap=True), "ViT")
    fig.tight_layout()
    fig.savefig("figures/fp_ViT_small.pdf", dpi=300)

    plot_fp_data(fp_data_vt, fp_holes_vt, ax_all_fp[1, 1], sns.color_palette("Greys", as_cmap=True), "ViT")
    fig_all_fp.tight_layout()
    fig_all_fp.savefig("figures/fp_small.pdf", dpi=300)

    plot_cm(vt_test_labels, vt_test_preds, ax_all_cm[1, 1], sns.color_palette("Greys", as_cmap=True), "ViT")
    fig_all_cm.tight_layout()
    fig_all_cm.savefig("figures/confusion_matrix_small.pdf", dpi=300)

    fp_all_intersection_data, fp_all_intersection_holes = get_misclassified_data(real_data, real_holes, fp_all,
                                                                                 ell_test)
    print(f"Number of samples FP all models: {len(fp_all_intersection_data)}")
    fn_all_intersection_data, fn_all_intersection_holes = get_misclassified_data(imag_data, imag_holes, fn_all,
                                                                                 ell_test)
    print(f"Number of samples FN all models: {len(fn_all_intersection_data)}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_fp_data(real_data, real_holes, ax[0], sns.color_palette("rocket_r", as_cmap=True),
                 model_name="True", colourbar=False)
    plot_fp_data(fp_all_intersection_data, fp_all_intersection_holes, ax[1],
                 sns.color_palette("rocket_r", as_cmap=True), "FP")
    fig.tight_layout()
    fig.savefig("figures/truefp_small.pdf", dpi=300)

    fpr_vt, tpr_vt, th_pr_vt = roc_curve(vt_test_labels, vt_test_probs)
    fpr_resnet, tpr_resnet, th_pr_resnet = roc_curve(resnet_test_labels, resnet_test_probs)
    fpr_vgg19, tpr_vgg19, th_pr_vgg19 = roc_curve(vgg19_test_labels, vgg19_test_probs)

    auc_vgg19 = 1 - auc(tpr_vgg19, fpr_vgg19)
    auc_resnet = 1 - auc(tpr_resnet, fpr_resnet)
    auc_vt = 1 - auc(tpr_vt, fpr_vt)
    print("\n----------AC----------")
    print(f"AUC VGG19: {auc_vgg19}")
    print(f"AUC ResNet: {auc_resnet}")
    print(f"AUC ViT: {auc_vt}")

    fig, ax = plt.subplots(1, figsize=(6, 5))
    lw = 2
    ax.plot(fpr_vt, tpr_vt, label=f"ViT ({auc_vt:.3f})", color="black", linewidth=lw)
    ax.plot(fpr_vgg19, tpr_vgg19, label=f"VGG19 ({auc_vgg19:.3f})", color="red", linestyle="-.", linewidth=lw)
    ax.plot(fpr_resnet, tpr_resnet, label=f"ResNet ({auc_resnet:.3f})", color="magenta", linestyle="dotted", linewidth=lw)

    ax.set_xlabel("False positive rate", fontsize=20)
    ax.set_ylabel("True positive rate", fontsize=20)
    ax.set_box_aspect(1)
    ax.grid()
    ax.legend(loc="lower right", fontsize=18)
    plt.tight_layout()
    fig.savefig("figures/roc1_small.pdf", dpi=300)

    ax.set_xlim([-0.01, 0.61])
    ax.set_ylim([0.39, 1.01])
    fig.savefig("figures/roc1_small_zoom.pdf", dpi=300)

