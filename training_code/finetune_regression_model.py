import matplotlib.pyplot as plt
import torchvision
import numpy as np
from regression_dataloaders import *
from set_data_path import *
torchvision.disable_beta_transforms_warning()
import time
from read_data import *
from torch.utils.data import DataLoader as DataLoader
from torch import nn
from torchvision.transforms import Compose, ToTensor, v2
import torchvision.transforms as T
from torchvision import models
import torch as tc

plt.rcParams.update({'font.size': 14})


def training_loop(model, optimizer, loss_fn, train_loader, val_loader,
                  num_epochs, print_every, device, model_save_path):
    print(f"Starting training on device: {device}")
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        model, train_loss = train_epoch(model,
                                        optimizer,
                                        loss_fn,
                                        train_loader,
                                        val_loader,
                                        device,
                                        print_every)
        val_loss = validate(model, loss_fn, val_loader, device)

        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss) / len(train_loss):.9f}, "
              f"Val. loss: {val_loss:.9f}")
        train_losses.extend(train_loss)
        val_losses.append(val_loss)
        # Early stopping, not really necessary for pretraining
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tc.save({'model_state_dict': model.state_dict(),
                         'optimiser_state_dict': optimizer.state_dict(),
                         'train_losses': train_losses,
                         'val_losses': val_losses},
                          model_save_path + "ES_FT_" + model_name)
    return model, train_losses, val_losses


def train_epoch(model, optimizer, loss_fn, train_loader, val_loader, device,
                print_every):
    model.train()
    train_loss_batches, train_acc_batches = [], []
    num_batches = len(train_loader)
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, target = x.to(device), y.to(device)
        inputs = inputs[:, :3, :, :]
        optimizer.zero_grad()
        z = model.forward(inputs)
        if not z.shape:
            z = z.expand(1)
        loss = loss_fn(z, target.float())
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        if print_every is not None and batch_index % print_every == 0:
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            model.train()
            print(f"\tBatch {batch_index}/{num_batches}: "
                  f"\tTrain loss: \\"
                  f"{sum(train_loss_batches[-print_every:]) / print_every:.9f}, "
                  f"\tVal. loss: {val_loss:.9f}")
    return model, train_loss_batches


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    model.eval()
    with tc.no_grad():
        for batch_index, (x, y) in enumerate(val_loader, 1):
            inputs, labels = x.to(device), y.to(device)
            inputs = inputs[:, :3, :, :]
            z = model.forward(inputs)
            if not z.shape:
                z = z.expand(1)
            batch_loss = loss_fn(z, labels.float())
            val_loss_cum += batch_loss.item()
    return val_loss_cum / len(val_loader)


def plot_metrics(train_loss, val_loss, n_epochs, figure_save_path):
    nth = len(train_loss) // n_epochs
    fig, ax = plt.subplots(1, figsize=(12, 4))
    ax.plot(np.arange(n_epochs), train_loss[0::nth], 'ko-',
            markersize=4, label="Train loss")
    ax.plot(np.arange(n_epochs), val_loss, 'rx-',
            markersize=4, label="Val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_save_path + "FT_" + model_name + ".pdf", dpi=300)


def create_transforms():
    t0 = v2.Compose([ToTensor()])
    t1 = v2.Compose([ToTensor(), T.RandomHorizontalFlip(p=1.0)])
    t2 = v2.Compose([ToTensor(), T.RandomRotation((90, 90))])
    t3 = v2.Compose([ToTensor(), T.RandomRotation((90, 90)),
                     T.RandomHorizontalFlip(p=1.0)])
    t4 = v2.Compose([ToTensor(), T.RandomRotation((180, 180))])
    t5 = v2.Compose([ToTensor(), T.RandomRotation((180, 180)),
                     T.RandomHorizontalFlip(p=1.0)])
    t6 = v2.Compose([ToTensor(), T.RandomRotation((270, 270))])
    t7 = v2.Compose([ToTensor(), T.RandomRotation((270, 270)),
                     T.RandomHorizontalFlip(p=1.0)])
    transforms = [t0, t1, t2, t3, t4, t5, t6, t7]
    return transforms


def create_regressor(n_inputs, n_hidden, n_outputs):
    new_regressor = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_outputs))
    return new_regressor


if __name__ == "__main__":
    dataset_names = ["th10", "th20", "ada", "full"]
    dataset_name = dataset_names[0]
    (image_path, data_path, train_path,
     train_data_path, val_path, val_data_path) = set_data_path(dataset_name)
    (figure_save_path, model_save_path) = set_save_paths()
    device = 'cuda' if tc.cuda.is_available() else 'cpu'
    n_hidden = 256
    n_outputs = 30
    n_qs = 15
    batch_size = 128
    num_workers = 8
    img_size = 224
    transforms = create_transforms()

    data_train = read_data(train_data_path)
    used_eigenmodes = np.arange(0, n_qs)
    column_names = []

    for i in used_eigenmodes:
        name1 = "eigenfrequency" + str(i)
        name2 = "quality_factor" + str(i)
        column_names.append(name1)
        column_names.append(name2)

    column_names.sort()

    y_train_max = data_train.max()

    ell_train = Ellipses(train_path, train_data_path, transforms, y_train_max)
    ell_val = Ellipses(val_path, val_data_path, transforms, y_train_max)

    n_train_samples = len(ell_train)
    n_val_samples = len(ell_val)

    train_dataloader = DataLoader(ell_train, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(ell_val, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

    print(f"Number of train samples: {n_train_samples}")
    print(f"Number of val samples: {n_val_samples}")

    models = [models.vgg19(),
              models.resnet152(),
              models.vit_b_16()]

    model_names = ["VGG19.ckpt", "ResNet152.ckpt", "VisionTransformer16.ckpt"]
    model = models[0]
    model_name = model_names[0]
    n_inputs = None
    if model == "VGG19.ckpt":
        n_inputs = model.classifier[0].in_features
    elif model == "ResNet152.ckpt":
        n_inputs = model.fc.in_features
    elif model == "VisionTransformer16.ckpt":
        n_inputs = 768

    loss_fn = nn.HuberLoss()
    learning_rate = 2e-5
    finetuning_epochs = 300

    regressor = create_regressor(n_inputs, n_hidden, n_outputs)
    # Not all models call the last layer 'classifier', some call it 'fc', 'head', etc.
    if model == "VGG19.ckpt":
        model.classifier = regressor
    elif model == "ResNet152.ckpt":
        model.fc = regressor
    elif model == "VisionTransformer16.ckpt":
        model.heads = regressor

    checkpoint = tc.load(model_save_path + "ES_" + model_name)
    model.load_state_dict(checkpoint['model_state_dict'])

    for param in model.parameters():
        param.requires_grad = True

    model.to(device)
    optimiser = tc.optim.Adam(model.parameters(), lr=learning_rate)
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    start_time = time.time()
    print(f"Starting training of {model_name}")
    (trained_transfer_model, transfer_train_losses, transfer_val_losses) = (
    training_loop(model=model, optimizer=optimiser, loss_fn=loss_fn,
                  train_loader=train_dataloader, val_loader=val_dataloader,
                  num_epochs=finetuning_epochs, print_every=np.inf,
                  device=device, model_save_path=model_save_path))
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    h, m = divmod(m, 60)
    plot_metrics(transfer_train_losses, transfer_val_losses,
                 n_epochs=finetuning_epochs)
    tc.save({'model_state_dict': model.state_dict(),
                 'optimiser_state_dict': optimiser.state_dict(),
                 'train_losses': transfer_train_losses,
                 'val_losses': transfer_val_losses},
                  model_save_path + "FT_" + model_name)
    print(f"Training took: {h}h, {m}m, {s}s")
