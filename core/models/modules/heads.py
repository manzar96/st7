import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,drop=0,
                 act='none'):
        super(ClassificationHead, self).__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoded_features, num_classes)
        self.drop = nn.Dropout(drop)
        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        out = self.clf(outputs)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out

class BertClassificationHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,drop=0,
                 act='none'):
        super(BertClassificationHead, self).__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoded_features, num_classes)
        self.drop = nn.Dropout(drop)
        if act is 'none':
            self.act = None
        elif act is 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pulled_output = outputs[1]
        out = self.clf(pulled_output)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out
