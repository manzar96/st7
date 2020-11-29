import os
import csv
import numpy as np
from core.utils.tensors import mktensor
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


class ShortTextDataset(Dataset):

    def __init__(self, tokenizer, maxsentlen=None):
        self.csvfile = './data/short_texts_200k_kaggle/dataset.csv'
        self.maxsentlen = maxsentlen

        self.data = self.read_data()
        self.transforms = []

        # tokenizer used
        self.tokenizer = tokenizer

    def label2idx(self, humor):
        if humor == 'False':
            val = 0
        else:
            val = 1
        return val

    def read_data(self):
        data = []
        with open(self.csvfile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for index, line in enumerate(csv_reader):
                if index == 0:
                    continue
                data.append(line)

        return data

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text, humor = self.data[index]
        is_humor = self.label2idx(humor)
        if not self.transforms:
            text = self.tokenizer(text)
            text = mktensor(text['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                text = t(text)

        is_humor = int(is_humor)

        return text, is_humor


if __name__ == "__main__":

    train_dataset = Task71Dataset('train')
    import ipdb;ipdb.set_trace()