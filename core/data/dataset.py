import os
import csv
import random
import numpy as np
from core.utils.tensors import mktensor
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


class Task7DatasetNegativeSampling(Dataset):
    def __init__(self, splitname, tokenizer=None, maxsentlen=None):
        self.splitname = splitname
        if splitname == 'train':
            self.csvfile = os.path.join("data/train_data",
                                        f"{splitname}.csv")
        self.maxsentlen = maxsentlen

        self.data = self.read_data()

        self.data = self.negative_sampling()

        self.transforms = []

        # tokenizer used
        self.tokenizer = tokenizer

    def read_data(self):
        data = []
        with open(self.csvfile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for index, line in enumerate(csv_reader):
                if index == 0:
                    continue
                data.append(line)

        return data

    def negative_sampling(self):
        # split data to humorous and non-humorous
        humor_data = []
        no_humor_data = []
        for sample in self.data:
            if int(sample[2])==1:
                humor_data.append(sample)
            else:
                no_humor_data.append(sample)

        # negative random sampling for humor data and augment humor data
        for index,sample in enumerate(humor_data):
            neg_samples = random.sample(no_humor_data, 5)
            humor_data[index] = [sample, neg_samples]

        for index,sample in enumerate(no_humor_data):
            neg_samples = random.sample(humor_data, 5)
            no_humor_data[index] = [sample, neg_samples]

        data = humor_data + no_humor_data
        random.shuffle(data)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.splitname == 'train':
            myid, text, is_humor, humor_rating, humor_controversy, offense_rating\
                = \
                self.data[index]
            sample,neg_samples = self.data[index]


        else:
            myid, text = self.data[index]
            is_humor = None
        if not self.transforms:
            text = self.tokenizer(text)
            text = mktensor(text['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                text = t(text)


        myid = int(myid)
        if self.splitname == 'train':
            is_humor = int(is_humor)
        # humor_rating = float(humor_rating)
        # humor_controversy = int(humor_controversy)
        # offense_rating = float(offense_rating)
        return myid, text, is_humor

class Task71Dataset(Dataset):

    def __init__(self, splitname, tokenizer, maxsentlen=None):
        self.splitname = splitname
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
        if self.splitname == 'train':
            myid, text, is_humor, humor_rating, humor_controversy, offense_rating\
                = \
                self.data[index]
        else:
            myid, text = self.data[index]
            is_humor = None
        if not self.transforms:
            text = self.tokenizer(text)
            text = mktensor(text['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                text = t(text)


        myid = int(myid)
        if self.splitname == 'train':
            is_humor = int(is_humor)
        # humor_rating = float(humor_rating)
        # humor_controversy = int(humor_controversy)
        # offense_rating = float(offense_rating)
        return myid, text, is_humor


class Task723Dataset(Dataset):

    def __init__(self, splitname, tokenizer, maxsentlen=None):
        self.splitname = splitname
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
            for index, line in enumerate(csv_reader):
                if index == 0:
                    continue
                # if text is humorous then add to data else not
                if self.splitname == 'train':
                    if line[2] == '1':
                        data.append(line)
                else:
                    data.append(line)
        return data

    def map(self, t):
        self.transforms.append(t)
        return self

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.splitname == 'train':
            myid, text, is_humor, humor_rating, humor_controversy, offense_rating\
                = \
                self.data[index]
        else:
            myid, text = self.data[index]
            humor_rating = None
            humor_controversy = None
        if not self.transforms:
            text = self.tokenizer(text)
            text = mktensor(text['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                text = t(text)


        myid = int(myid)
        if self.splitname == 'train':
            humor_rating = float(humor_rating)
            humor_controversy = int(humor_controversy)
        return myid, text, humor_rating, humor_controversy


class Task74Dataset(Dataset):

    def __init__(self, splitname, tokenizer, maxsentlen=None):
        self.splitname = splitname
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
        if self.splitname == 'train':
            myid, text, is_humor, humor_rating, humor_controversy, offense_rating\
                = \
                self.data[index]
        else:
            myid, text = self.data[index]
            offense_rating = None
        if not self.transforms:
            text = self.tokenizer(text)
            text = mktensor(text['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                text = t(text)


        myid = int(myid)
        if self.splitname == 'train':
            offense_rating = float(offense_rating)
        # humor_rating = float(humor_rating)
        # humor_controversy = int(humor_controversy)
        # offense_rating = float(offense_rating)
        return myid, text, offense_rating

if __name__ == "__main__":

    train_dataset = Task7DatasetNegativeSampling('train')
    import ipdb;ipdb.set_trace()