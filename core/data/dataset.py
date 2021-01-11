import os
import csv
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from core.utils.tensors import mktensor
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


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
        self.preprocess_data()
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

    def preprocess_data(self):
        import nltk
        nltk.download('punkt')
        length = len(self.data)
        humor_data=[]
        no_humor_data=[]
        for sample in self.data:
            if sample[2]=='1':
                humor_data.append(sample)
            else:
                no_humor_data.append(sample)

        print("Humorous samples: ",len(humor_data))
        print("No Humorous samples: ",len(no_humor_data))
        humor_lengths = [len(sample) for sample in humor_data]
        no_humor_lengths = [len(sample) for sample in no_humor_data]
        print("Max Humorous length: ",len(humor_data))
        print("No Humorous samples: ",len(no_humor_data))

        new_data = []
        for sample in self.data:
            myid, sent, is_humor, humor_rating, humor_controversy, \
            offense_rating = sample
            sent = self.to_lowercase(sent)
            sent = self.remove_hyperlinks(sent)
            sent = self.clean_contractions(sent)
            sent = self.split_punct_marks(sent)
            new_data.append([myid, sent, is_humor, humor_rating, humor_controversy, \
            offense_rating])
        return new_data


    def remove_hyperlinks(self,sentence):
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', sentence,
                      flags=re.MULTILINE)
        return text

    def remove_punctuation(self,sentence):
        text = sentence.translate(str.maketrans('', '', string.punctuation))
        return text

    def remove_numbers(self,sentence):
        text = re.sub(r'[0-9]+', '', sentence)
        return text

    def remove_stopwords(self,sentence):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sentence)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        text = ' '.join(filtered_sentence)
        return text

    def to_lowercase(self,sentence):
        text = sentence.lower()
        return text

    def clean_contractions(self,sentence):
        contractions = {
            "ain't": "am not / are not / is not / has not / have not",
            "aren't": "are not / am not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he had / he would",
            "he'd've": "he would have",
            "he'll": "he shall / he will",
            "he'll've": "he shall have / he will have",
            "he's": "he has / he is",
            "how'd": "how did",
            "how'd'y": "how do you",
            "how'll": "how will",
            "how's": "how has / how is / how does",
            "I'd": "I had / I would",
            "I'd've": "I would have",
            "I'll": "I shall / I will",
            "I'll've": "I shall have / I will have",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it'd": "it had / it would",
            "it'd've": "it would have",
            "it'll": "it shall / it will",
            "it'll've": "it shall have / it will have",
            "it's": "it has / it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "mightn't've": "might not have",
            "must've": "must have",
            "mustn't": "must not",
            "mustn't've": "must not have",
            "needn't": "need not",
            "needn't've": "need not have",
            "o'clock": "of the clock",
            "oughtn't": "ought not",
            "oughtn't've": "ought not have",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "shan't've": "shall not have",
            "she'd": "she had / she would",
            "she'd've": "she would have",
            "she'll": "she shall / she will",
            "she'll've": "she shall have / she will have",
            "she's": "she has / she is",
            "should've": "should have",
            "shouldn't": "should not",
            "shouldn't've": "should not have",
            "so've": "so have",
            "so's": "so as / so is",
            "that'd": "that would / that had",
            "that'd've": "that would have",
            "that's": "that has / that is",
            "there'd": "there had / there would",
            "there'd've": "there would have",
            "there's": "there has / there is",
            "they'd": "they had / they would",
            "they'd've": "they would have",
            "they'll": "they shall / they will",
            "they'll've": "they shall have / they will have",
            "they're": "they are",
            "they've": "they have",
            "to've": "to have",
            "wasn't": "was not",
            "we'd": "we had / we would",
            "we'd've": "we would have",
            "we'll": "we will",
            "we'll've": "we will have",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what shall / what will",
            "what'll've": "what shall have / what will have",
            "what're": "what are",
            "what's": "what has / what is",
            "what've": "what have",
            "when's": "when has / when is",
            "when've": "when have",
            "where'd": "where did",
            "where's": "where has / where is",
            "where've": "where have",
            "who'll": "who shall / who will",
            "who'll've": "who shall have / who will have",
            "who's": "who has / who is",
            "who've": "who have",
            "why's": "why has / why is",
            "why've": "why have",
            "will've": "will have",
            "won't": "will not",
            "won't've": "will not have",
            "would've": "would have",
            "wouldn't": "would not",
            "wouldn't've": "would not have",
            "y'all": "you all",
            "y'all'd": "you all would",
            "y'all'd've": "you all would have",
            "y'all're": "you all are",
            "y'all've": "you all have",
            "you'd": "you had / you would",
            "you'd've": "you would have",
            "you'll": "you shall / you will",
            "you'll've": "you shall have / you will have",
            "you're": "you are",
            "you've": "you have"
        }
        word_tokens = sentence.split(' ')
        new_tokens = []
        for w in word_tokens:
            if w in contractions.keys():
                new_tokens.append(contractions[w])
            else:
                new_tokens.append(w)
        text = ' '.join(new_tokens)
        return text

    def split_punct_marks(self,sentence):
        text = re.findall(r"[\w']+|[.,!?;:]", sentence)
        text = ' '.join(text)
        return text

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

    train_dataset = Task71Dataset('train',tokenizer=None)
    import ipdb;ipdb.set_trace()