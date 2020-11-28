import os
import csv
import numpy as np
from core.utils.tensors import mktensor
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


class Task71Dataset(Dataset):

    def __init__(self, splitname, tokenizer, maxsentlen=None):
        if splitname == 'train':

            self.csvfile = os.path.join("data/train_data",
                                        f"{splitname}.csv")
        elif splitname == 'dev':
            self.csvfile = "data/public_dat_dev/public_dev.csv"
        elif splitname == "eval":
            self.csvfile = "data/public_dat_eval/public_test.csv"
        elif splitname == "post_eval":
            self.csvfile = "data/public_dat_post_eval/public_test.csv"
        else:
            raise IOError("Dataset's splitname sould be train,eval or "
                          "post_eval!")
        self.maxsentlen = maxsentlen

        self.data = self.read_data()
        self.transforms = []

        # tokenizer used
        self.tokenizer = tokenizer


    def read_data(self):
        data = []
        with open(self.csvfile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for index,line in enumerate(csv_reader):
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
        myid, text, is_humor, humor_rating, humor_controversy, offense_rating\
            = \
            self.data[index]
        import ipdb;ipdb.set_trace()
        if not self.transforms:
            text = self.tokenizer(text)
            text = mktensor(text['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                text = t(text)


        myid = mktensor(int(myid),dtype=torch.long)
        is_humor = mktensor(int(is_humor),dtype=torch.long)
        # humor_rating = mktensor(float(humor_rating))
        # humor_controversy = mktensor(int(humor_controversy),dtype=torch.long)
        # offense_rating = mktensor(float(offense_rating))
        import ipdb;ipdb.set_trace()
        return myid, text, is_humor


if __name__ == "__main__":

    train_dataset = Task71Dataset('train')
    import ipdb;ipdb.set_trace()