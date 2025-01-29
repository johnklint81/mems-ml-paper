from create_dataloaders import *
import matplotlib.pyplot as plt
import torchvision
torchvision.disable_beta_transforms_warning()
import torch as tc
from torch.utils.data import DataLoader as DataLoader
from torch import nn
from torchvision.transforms import Compose, ToTensor, v2
import torchvision.transforms as T
from torchvision import models
plt.rcParams.update({'font.size': 14})
import time

test_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/images_split_balanced/test/"
train_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/images_split_balanced/train/"
val_path = "/home/j/PycharmProjects/Master_ADS/create_images_ellipses/images_split_balanced/val/"


def output_to_label(z):
    c = tc.zeros_like(z, dtype=tc.long)
    c[z > 0.5] = 1
    return c


def training_loop(model, optimizer, loss_fn, train_loader, val_loader, num_epochs, print_every):
    print(f"Starting training on device: {device}")
    model.to(device)
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    best_val_acc = np.inf
    for epoch in range(1, num_epochs + 1):
        model, train_loss, train_acc = train_epoch(model,
                                                   optimizer,
                                                   loss_fn,
                                                   train_loader,
                                                   val_loader,
                                                   device,
                                                   print_every)
        val_loss, val_acc = validate(model, loss_fn, val_loader, device)
        if val_acc < best_val_acc:
            best_val_acc = val_acc
            tc.save({'model_state_dict': model.state_dict()}, "saved_models/vgg19_100e_best_finetuned.ckpt")

        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train loss: {sum(train_loss) / len(train_loss):.3f}, "
              f"Train acc.: {sum(train_acc) / len(train_acc):.3f}, "
              f"Val. loss: {val_loss:.3f}, "
              f"Val. acc.: {val_acc:.3f}")
        train_losses.extend(train_loss)
        train_accs.extend(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    return model, train_losses, train_accs, val_losses, val_accs


def train_epoch(model, optimizer, loss_fn, train_loader, val_loader, device, print_every):
    # Train:
    model.train()
    train_loss_batches, train_acc_batches = [], []
    num_batches = len(train_loader)
    for batch_index, (x, y) in enumerate(train_loader, 1):
        inputs, labels = x.to(device), y.to(device)
        inputs = inputs[:, :3, :, :]
        optimizer.zero_grad()
        z = model.forward(inputs)
        if not z.shape:
            z = z.expand(1)
        loss = loss_fn(z, labels.float())
        loss.backward()
        optimizer.step()
        train_loss_batches.append(loss.item())

        hard_preds = output_to_label(z)
        acc_batch_avg = (hard_preds == labels).float().mean().item()
        train_acc_batches.append(acc_batch_avg)
        if print_every is not None and batch_index % print_every == 0:
            val_loss, val_acc = validate(model, loss_fn, val_loader, device)
            model.train()
            print(f"\tBatch {batch_index}/{num_batches}: "
                  f"\tTrain loss: {sum(train_loss_batches[-print_every:]) / print_every:.3f}, "
                  f"\tTrain acc.: {sum(train_acc_batches[-print_every:]) / print_every:.3f}, "
                  f"\tVal. loss: {val_loss:.3f}, "
                  f"\tVal. acc.: {val_acc:.3f}")

    return model, train_loss_batches, train_acc_batches


def validate(model, loss_fn, val_loader, device):
    val_loss_cum = 0
    val_acc_cum = 0
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
            hard_preds = output_to_label(z)
            acc_batch_avg = (hard_preds == labels).float().mean().item()
            val_acc_cum += acc_batch_avg
    return val_loss_cum / len(val_loader), val_acc_cum / len(val_loader)


def plot_metrics(train_loss, train_acc, val_loss, val_acc, n_epochs):
    nth = len(train_loss) // n_epochs
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].plot(np.arange(n_epochs), train_loss[0::nth], 'ko-', markersize=4, label="Train loss")
    ax[0].plot(np.arange(n_epochs), val_loss, 'rx-', markersize=4, label="Val loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss")
    ax[0].legend()

    ax[1].plot(np.arange(n_epochs), train_acc[0::nth], 'ko-', markersize=4, label="Train acc")
    ax[1].plot(np.arange(n_epochs), val_acc, 'rx-', markersize=4, label="Val acc")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("accuracy")
    ax[1].set_title("Accuracy")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig("figures_cnn/metrics_vgg19_300_finetuned.png", dpi=300)


def create_transforms():
    t0 = v2.Compose([ToTensor()])
    t1 = v2.Compose([ToTensor(), T.RandomHorizontalFlip(p=1.0)])
    t2 = v2.Compose([ToTensor(), T.RandomRotation((90, 90))])
    t3 = v2.Compose([ToTensor(), T.RandomRotation((90, 90)), T.RandomHorizontalFlip(p=1.0)])
    t4 = v2.Compose([ToTensor(), T.RandomRotation((180, 180))])
    t5 = v2.Compose([ToTensor(), T.RandomRotation((180, 180)), T.RandomHorizontalFlip(p=1.0)])
    t6 = v2.Compose([ToTensor(), T.RandomRotation((270, 270))])
    t7 = v2.Compose([ToTensor(), T.RandomRotation((270, 270)), T.RandomHorizontalFlip(p=1.0)])
    transforms = [t0, t1, t2, t3, t4, t5, t6, t7]
    return transforms


if __name__ == "__main__":

    # DATA PROCESSING
    batch_size = 64
    num_workers = 16
    img_size = 224
    device = 'cuda' if tc.cuda.is_available() else 'cpu'
    transforms = create_transforms()

    ell_train = Ellipses(train_path, transforms)
    ell_test = Ellipses(test_path, transforms)
    ell_val = Ellipses(val_path, transforms)

    train_dataloader = DataLoader(ell_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(ell_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(ell_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    n_train_samples = len(ell_train)
    n_test_samples = len(ell_test)
    n_val_samples = len(ell_val)
    print(f"Number of train samples: {n_train_samples}")
    print(f"Number of val samples: {n_val_samples}")
    print(f"Number of test samples: {n_test_samples}")

    _, label_0 = ell_train[0]
    _, label_1 = ell_train[1]
    print(f"The label of the first item in the train dataset is {label_0}")
    print(f"The label of the second item in the train dataset is {label_1}")

    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    n_inputs = model.classifier[0].in_features
    n_hidden = 64
    new_classifier = nn.Sequential(
        nn.Linear(n_inputs, n_hidden),
        nn.ReLU(),
        nn.Linear(n_hidden, n_hidden),
        nn.Linear(n_hidden, 1),
        nn.Flatten(start_dim=0),
        nn.Sigmoid())
    # This part has be adapted for the specific model used in line with the
    # regression training files
    model.classifier = new_classifier

    checkpoint = tc.load("./saved_models/vgg19_64_100e.ckpt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    n_hidden = 64
    for param in model.features.parameters():
        param.requires_grad = True

    transfer_epochs = 100
    loss_fn = nn.BCELoss()
    learning_rate = 2e-5
    optimiser = tc.optim.Adam(model.parameters(), lr=learning_rate)
    # optimiser.load_state_dict(checkpoint['optimiser_state_dict'])

    start_time = time.time()
    trained_transferVGG, transferVGG_train_losses, transferVGG_train_accs, transferVGG_val_losses, transferVGG_val_accs = \
        training_loop(model=model, optimizer=optimiser, loss_fn=loss_fn,
                      train_loader=train_dataloader, val_loader=val_dataloader,
                      num_epochs=transfer_epochs, print_every=np.inf)
    end_time = time.time() - start_time
    m, s = divmod(end_time, 60)
    print(f"Training took {int(m)} minutes and {np.round(s)} seconds")
    plot_metrics(transferVGG_train_losses, transferVGG_train_accs,
                 transferVGG_val_losses, transferVGG_val_accs,
                 n_epochs=transfer_epochs)
    tc.save({'model_state_dict': model.state_dict(),
             'train_losses': transferVGG_train_losses,
             'train_accs': transferVGG_train_accs,
             'val_losses': transferVGG_val_losses,
             'val_accs': transferVGG_val_accs,
             }, "saved_models/vgg19_64_100e_finetuned.ckpt")
