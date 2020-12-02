import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from core.utils.masks import pad_mask, subsequent_mask
from core.utils.tensors import mktensor


class Task71aCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, is_humor = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        is_humor = mktensor(is_humor, dtype=torch.long)

        return padded_inputs,inputs_pad_mask, is_humor


class Task71aCollatorTest(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, is_humor = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        return myid, padded_inputs, inputs_pad_mask


class Task71aCollatorFeatures(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, is_humor = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        is_humor = mktensor(is_humor, dtype=torch.long)

        return myid, padded_inputs,inputs_pad_mask, is_humor


class ShortTextDatasetCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        text_input, is_humor = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        is_humor = mktensor(is_humor, dtype=torch.long)

        return padded_inputs,inputs_pad_mask, is_humor


class Task723Collator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, humor_rating,humor_contr = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        humor_rating = mktensor(humor_rating, dtype=torch.float)
        humor_contr = mktensor(humor_contr, dtype=torch.long)

        return padded_inputs,inputs_pad_mask, humor_rating, humor_contr


class Task723CollatorTest(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, humor_rat, humor_contr = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        return myid, padded_inputs, inputs_pad_mask


class Task723CollatorFeatures(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, humor_rating,humor_contr = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in text_input], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text_input, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        humor_rating = mktensor(humor_rating, dtype=torch.float)
        humor_contr = mktensor(humor_contr, dtype=torch.long)
        return myid, padded_inputs,inputs_pad_mask, humor_rating, humor_contr