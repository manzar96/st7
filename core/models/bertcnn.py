import torch.nn as nn
from core.models.modules.cnn import ConvNet1D


class BertCNN(nn.Module):
    def __init__(self, encoder):
        super(BertCNN, self).__init__()
        self.encoder = encoder
        self.cnn = ConvNet1D(input_channels=1)

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pulled_output = outputs[1]
        out_cnn = self.cnn(pulled_output)
        return out_cnn
