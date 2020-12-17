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
        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
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

class T5ClassificationHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,drop=0,
                 act='none'):
        super(T5ClassificationHead, self).__init__()
        self.encoder = encoder
        self.clf = nn.Linear(encoded_features, num_classes)
        self.drop = nn.Dropout(drop)
        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pulled_output = outputs[0][:,-1,:]
        out = self.clf(pulled_output)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out


class DoubleHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes1,num_classes2,
                 drop=0,
                 act='none'):
        super(DoubleHead, self).__init__()
        self.encoder = encoder
        self.clf1 = nn.Linear(encoded_features, num_classes1)
        self.clf2 = nn.Linear(encoded_features, num_classes2)
        self.drop = nn.Dropout(drop)
        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        pulled_output = outputs[1]
        out1 = self.clf1(pulled_output)
        if self.act:
            out1 = self.act(out1)
        out1 = self.drop(out1)
        out2 = self.clf2(pulled_output)
        if self.act:
            out2 = self.act(out2)
        out2 = self.drop(out2)
        return out1, out2