import torch as tc
from torch import nn
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.flat1 = nn.Flatten()
        self.drop1 = nn.Dropout(p=0.3)
        conv2_output_size = 128 * (img_size - 3) ** 2
        pool3_output_size = int(conv2_output_size)
        self.fc1 = nn.Linear(pool3_output_size, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input_batch):
        x = F.relu(self.conv1(input_batch))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop1(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flat1(x)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)
        x = tc.squeeze(x)
        x = F.sigmoid(x)
        return x
