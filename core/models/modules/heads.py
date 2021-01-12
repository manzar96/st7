import torch
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
                 act='none', method=None, fusion=None):
        super(BertClassificationHead, self).__init__()
        self.encoder = encoder
        self.method = method
        self.fusion = fusion
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

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        # here we retrieve the last layer hidden-state of the first token [CLS]
        # which is used for seq classification (according to docs) but for
        # the NSP task
        if self.method is None:
            pulled_output = outputs[1]
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


        if self.method is not None:
            # we need to apply pooling over time (by using mean or another
            # method e.g. take last timestep)!!
            if self.fusion is None:
                pulled_output = output[:,0,:] # extract from CLS token
            elif self.fusion == 'mean':
                pulled_output = torch.mean(output, dim=1) # mean over all
                # timesteps
            elif self.fusion == 'last':
                batch_size = kwargs['attention_mask'].shape[0]
                lengths = torch.sum(kwargs['attention_mask'], dim=1,
                                    dtype=torch.long) - 1
                pulled_output = output[range(batch_size), lengths] # take
                # last-padded timestep

        out = self.clf(pulled_output)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out


class ElectraClassificationHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,drop=0,
                 act='none', method=None, fusion=None):
        super(ElectraClassificationHead, self).__init__()
        self.encoder = encoder
        self.method = method
        self.fusion = fusion
        if self.method =='concat4':
            self.clf = nn.Linear(4*encoded_features, num_classes)
        else:
            self.clf = nn.Linear(encoded_features, num_classes)
        if self.method =='concat4':
            self.dense = nn.Linear(4*encoded_features, 4*encoded_features)
        else:
            self.dense = nn.Linear(encoded_features, encoded_features)
        self.drop = nn.Dropout(drop)
        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()

    def forward(self, *args, **kwargs):
        outputs = self.encoder(*args, **kwargs)
        # here we retrieve the last layer hidden-state of the first token [CLS]
        # which is used for seq classification (according to docs) but for
        # the NSP task
        if self.method is None:
            raise Exception
        hidden_states = torch.stack(outputs[1][1:])
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


        if self.method is not None:
            # we need to apply pooling over time (by using mean or another
            # method e.g. take last timestep)!!
            if self.fusion is None:
                pulled_output = output[:,0,:] # extract from CLS token
            elif self.fusion == 'mean':
                pulled_output = torch.mean(output, dim=1) # mean over all
                # timesteps
            elif self.fusion == 'last':
                batch_size = kwargs['attention_mask'].shape[0]
                lengths = torch.sum(kwargs['attention_mask'], dim=1,
                                    dtype=torch.long) - 1
                pulled_output = output[range(batch_size), lengths] # take
                # last-padded timestep

        out = self.drop(pulled_output)
        out = self.dense(out)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        out = self.clf(out)
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
