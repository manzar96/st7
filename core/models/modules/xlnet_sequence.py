import torch.nn as nn
from transformers.modeling_utils import SequenceSummary


class XLNetSequence(nn.Module):
    """
    This class is used for summarizing the sequence of hidden states of XLNET
    into on single vector.
    """
    def __init__(self, xlnetmodel):
        super(XLNetSequence, self).__init__()
        self.xlnetmodel = xlnetmodel
        self.config = xlnetmodel.config
        self.seq_summary = SequenceSummary(xlnetmodel.config)

    def forward(self, *args, **kwargs):
        outputs = self.xlnetmodel(*args, **kwargs)
        out = self.seq_summary(outputs[0])
        return out