import torch.nn as nn
import torch

class Resnet(nn.Module):
    def __init__(self, input_size, n_channels):
        super().__init__()
        b, c, s = input_size

        self.relu = nn.ReLU()

        # residual block 1
        self.conv1 = nn.Conv1d(in_channels=c,
                               out_channels=n_channels,
                               kernel_size=8,
                               padding_mode='replicate')
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv2 = nn.Conv1d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=5,
                               padding_mode='replicate')
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.conv3 = nn.Conv1d(in_channels=n_channels,
                               out_channels=n_channels,
                               kernel_size=3,
                               padding_mode='replicate')
        self.bn3 = nn.BatchNorm1d(n_channels)
        self.expand1 = nn.Conv1d(in_channels=c,
                                 out_channels=n_channels,
                                 kernel_size=1,
                                 padding_mode='replicate')
        # residual block 2
        self.conv4 = nn.Conv1d(in_channels=n_channels,
                               out_channels=n_channels*2,
                               kernel_size=8,
                               padding_mode='replicate')
        self.bn4 = nn.BatchNorm1d(n_channels*2)
        self.conv5 = nn.Conv1d(in_channels=n_channels*2,
                               out_channels=n_channels*2,
                               kernel_size=5,
                               padding_mode='replicate')
        self.bn5 = nn.BatchNorm1d(n_channels*2)
        self.conv6 = nn.Conv1d(in_channels=n_channels*2,
                               out_channels=n_channels*2,
                               kernel_size=3,
                               padding_mode='replicate')
        self.bn6 = nn.BatchNorm1d(n_channels*2)
        self.expand2 = nn.Conv1d(in_channels=n_channels,
                                 out_channels=n_channels*2,
                                 kernel_size=1,
                                 padding_mode='replicate')

        # residual block 3
        self.conv7 = nn.Conv1d(in_channels=n_channels*2,
                               out_channels=n_channels*2,
                               kernel_size=8,
                               padding_mode='replicate')
        self.bn7 = nn.BatchNorm1d(n_channels*2)
        self.conv8 = nn.Conv1d(in_channels=n_channels*2,
                               out_channels=n_channels*2,
                               kernel_size=5,
                               padding_mode='replicate')
        self.bn8 = nn.BatchNorm1d(n_channels*2)
        self.conv9 = nn.Conv1d(in_channels=n_channels*2,
                               out_channels=n_channels*2,
                               kernel_size=3,
                               padding_mode='replicate')
        self.bn9 = nn.BatchNorm1d(n_channels*2)

        # global average pooling, fully connected layer
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(in_features=n_channels*2,
                               out_features=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, seq):
        shortcut_y = seq

        # residual block 1
        y = self.conv1(seq)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.bn3(y)
        shortcut_y = self.expand1(shortcut_y)   # (b, c, s) -> (b, n_channels, s))
        shortcut_y = self.bn(shortcut_y)
        y = torch.add(shortcut_y, y)
        y = self.relu(y)

        # residual block 2
        shortcut_y = y
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.relu(y)
        y = self.conv5(y)
        y = self.bn5(y)
        y = self.relu(y)
        y = self.conv6(y)
        y = self.bn6(y)
        shortcut_y = self.expand2(shortcut_y)
        shortcut_y = self.bn(shortcut_y)
        y = torch.add(shortcut_y, y)
        y = self.relu(y)

        # residual block 3
        shortcut_y = y
        y = self.conv7(y)
        y = self.bn7(y)
        y = self.relu(y)
        y = self.conv8(y)
        y = self.bn8(y)
        y = self.relu(y)
        y = self.conv9(y)
        y = self.bn9(y)
        shortcut_y = self.bn(shortcut_y)
        y = torch.add(shortcut_y, y)
        y = self.relu(y)

        # global average pooling, fully connected layer
        y = self.gap(y)
        y = self.dense(y)
        y = self.sigmoid(y)

        return y
