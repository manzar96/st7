import torch.nn as nn

class ConvNet(nn.Module):

    def __init__(self,drop):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 60, kernel_size=2, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(60, 1, kernel_size=2, stride=1),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out


class ConvNet1D(nn.Module):
    def __init__(self, input_channels, kernel_sz=None, stride=None,
                 padding=None):
        super(ConvNet1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 4, kernel_size=3, stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(4),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

    def forward(self, x, lengths=None):
        x=x.unsqueeze(1)
        out1 = self.layer1(x)
        # print(out1.shape)
        out2 = self.layer2(out1)
        # print(out2.shape)
        out3 = self.layer3(out2)
        # print(out3.shape)
        out4 = self.layer4(out3)
        # print(out4.shape)
        out_flat = out4.reshape(-1, out4.size(1) * out4.size(2))
        # print(out_flat.shape)

        return out_flat