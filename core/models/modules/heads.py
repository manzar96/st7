import torch.nn as nn


class BertClassificationHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,drop=0,
                 act='none'):
        super(BertClassificationHead, self).__init__()
        self.encoder = encoder
        self.clf1 = nn.Linear(encoded_features, 126)
        self.clf2 = nn.Linear(encoded_features, num_classes)
        self.drop = nn.Dropout(drop)
        if act is 'none':
            self.act = None
        elif act is 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pulled_output = outputs[1]
        out = self.clf1(pulled_output)
        out = self.drop(out)
        out = self.clf2(out)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out
