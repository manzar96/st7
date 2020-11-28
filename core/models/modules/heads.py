import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,drop=0,
                 act='none'):
        super(ClassificationHead, self).__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoded_features, num_classes)
        self.drop = nn.Dropout(drop)
        if act is 'none':
            self.act = None
        elif act is 'sigmoid':
            self.act = nn.Sigmoid()


    def forward(self, *args, **kwargs):
        x = self.encoder(*args, **kwargs)
        out = self.clf(x)
        out = self.act(out)
        out = self.drop(out)
        return out
