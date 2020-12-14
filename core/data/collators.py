import torch
import random
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


class Task7NegativeSamplingCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text, is_humor, pos_samples, neg_samples = map(list, zip(*batch))

        batch_size = len(text)
        pos_samples = [item for s in pos_samples for item in s]
        neg_samples = [item for s in neg_samples for item in s]

        input_lengths = torch.tensor(
            [len(s) for s in text], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(text, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        # positive text
        pos_lengths = torch.tensor(
            [len(s) for s in pos_samples], device=self.device)

        # attention mask
        max_length = max(pos_lengths)
        pos_pad_mask = pad_mask(pos_lengths, max_length=max_length,
                                   device=self.device)
        pos_pad_mask = pos_pad_mask.reshape(batch_size, -1, max_length)
        padded_pos = (
            pad_sequence(pos_samples, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device)).reshape(batch_size, -1, max_length)

        # perform sampling
        number_of_samples = padded_pos.shape[1]
        sample_indexes = [random.randint(0,number_of_samples-1) for i in \
                range(batch_size)]
        list_pos = [padded_pos[i,sample_indexes[i],:] for i in range(
            batch_size)]
        padded_pos = torch.stack(list_pos)
        list_pos_mask = [pos_pad_mask[i,sample_indexes[i],:] for i in range(
            batch_size)]
        pos_pad_mask = torch.stack(list_pos_mask)

        # negative text
        neg_lengths = torch.tensor(
            [len(s) for s in neg_samples], device=self.device)

        # attention mask
        max_length = max(neg_lengths)
        neg_pad_mask = pad_mask(neg_lengths, max_length=max_length,
                                device=self.device)
        neg_pad_mask = neg_pad_mask.reshape(batch_size, -1, max_length)

        padded_neg = (
            pad_sequence(neg_samples, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device)).reshape(batch_size, -1, max_length)

        # perform sampling
        number_of_samples = padded_neg.shape[1]
        sample_indexes = [random.randint(0,number_of_samples-1) for i in \
                range(batch_size)]
        list_neg = [padded_neg[i,sample_indexes[i],:] for i in range(
            batch_size)]
        padded_neg = torch.stack(list_neg)
        list_neg_mask = [neg_pad_mask[i,sample_indexes[i],:] for i in range(
            batch_size)]
        neg_pad_mask = torch.stack(list_neg_mask)

        return padded_inputs, inputs_pad_mask, padded_pos, pos_pad_mask, \
               padded_neg, neg_pad_mask


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


class Task74Collator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, off = map(list, zip(*batch))

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

        off = mktensor(off, dtype=torch.float)

        return padded_inputs,inputs_pad_mask, off


class Task74CollatorTest(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, off = map(list, zip(*batch))

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


class Task74CollatorFeatures(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        myid, text_input, off = map(list, zip(*batch))

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

        off = mktensor(off, dtype=torch.float)


        return myid, padded_inputs, inputs_pad_mask,off