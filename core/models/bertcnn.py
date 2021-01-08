import torch.nn as nn
import torch.nn.functional as F
import torch
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


class BertCNNHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,
                 kernel_sizes=[2,3,5],
                 kernels_num=3,
                 drop_cnn=0,
                 drop=0,
                 act='none',
                 method=None):
        super(BertCNNHead, self).__init__()
        self.encoder = encoder
        self.method = method
        if self.method =='concat4':
            self.clf = nn.Linear(4*encoded_features, num_classes)
        else:
            self.clf = nn.Linear(encoded_features, num_classes)
        self.drop = nn.Dropout(drop)
        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()

        self.embed_dim = encoded_features
        self.num_classes = num_classes
        Ci = 1
        Co = kernels_num
        Ks = kernel_sizes
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, self.embed_dim))
                                    for K in Ks])
        self.drop_cnn = nn.Dropout(drop_cnn)
        self.fc1 = nn.Linear(len(Ks) * Co, self.num_classes)



    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        # another way is to use/handle the hidden states of the layers to
        # extract a sentence representation
        output_embed = outputs[2][0]
        hidden_states = torch.stack(outputs[2][1:])
        if self.method == 'sum12':
            # sum all 12 layers
            output = hidden_states.sum(dim=0)
        elif self.method == 'lasthidden':
            # only take the last layer
            output = hidden_states[-1]
        elif self.method == 'mean2tolast':
            output=torch.mean(hidden_states[1:], dim=0)
        elif self.method == 'sum4':
            # sum the last 4 layers
            output = hidden_states[-4:].sum(dim=0)
        elif self.method == 'concat4':
            # concatenate the last 4 layers
            last4 = hidden_states[-4:]
            output = torch.cat((last4[0], last4[1], last4[2], last4[3]), dim=-1)
        else:
            raise NotImplementedError

        output = output.unsqueeze(1) # (N, Ci, W, D)
        x = [F.relu(conv(output)).squeeze(3) for conv in self.convs] # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.drop_cnn(x)  # (N, len(Ks)*Co)
        out=self.fc1(x)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out