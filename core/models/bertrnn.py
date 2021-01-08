import torch.nn as nn
import torch.nn.functional as F
import torch


class BertRNNHead(nn.Module):
    def __init__(self, encoder, encoded_features, num_classes,rnn_hidden,
                 num_layers=2, bidi=True, batch_first=True,
                 drop_rnn=0,
                 drop=0,
                 act='none',
                 method=None):
        super(BertRNNHead, self).__init__()
        self.encoder = encoder
        self.method = method
        if self.method =='concat4':
            self.clf = nn.Linear(4*encoded_features, num_classes)
        else:
            self.clf = nn.Linear(encoded_features, num_classes)

        if act == 'none':
            self.act = None
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'relu':
            self.act = nn.ReLU()

        self.drop = nn.Dropout(drop)
        self.embed_dim = encoded_features

        self.rnn_hidden = rnn_hidden
        self.rnn = nn.RNN(input_size=self.embed_dim,
                          hidden_size=self.rnn_hidden,
                          num_layers=num_layers,
                          bidirectional=bidi,
                          dropout=drop_rnn,
                          batch_first=batch_first)

        self.num_classes = num_classes
        if bidi:
            self.fc1 = nn.Linear(2*self.rnn_hidden, self.num_classes)
        else:
            self.fc1 = nn.Linear(self.rnn_hidden, self.num_classes)



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

        batch_size = kwargs['attention_mask'].shape[0]
        lengths = torch.ne(kwargs['attention_mask'],0).sum(-1).cpu()

        rnn_input = torch.nn.utils.rnn.pack_padded_sequence(output, lengths,
                                                            batch_first=True,
                                                            enforce_sorted=False)
        out, hidden = self.rnn(rnn_input)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(out,
                                                                        batch_first=True)
        # take last timestep
        lengths = torch.ne(kwargs['attention_mask'], 0).sum(-1)-1
        out = unpacked[range(batch_size),lengths]

        out = self.fc1(out)
        if self.act:
            out = self.act(out)
        out = self.drop(out)
        return out