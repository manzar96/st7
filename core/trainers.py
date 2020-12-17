import random
import torch.nn as nn
import numpy as np
import copy
import sklearn
import os
import torch
import math
import time
from tqdm import tqdm
from sklearn.model_selection import KFold
from typing import cast, List, Optional, Tuple, TypeVar
from core.utils.tensors import to_device
TrainerType = TypeVar('TrainerType', bound='Trainer')


class BertTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 checkpoint_with=None,
                 checkpoint_max=False,
                 metrics=None,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        self.metrics = metrics
        self.checkpoint_with =checkpoint_with
        self.checkpoint_max = checkpoint_max


    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            f1=[]
            acc=[]
            metrics_dict={}
            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                targets = to_device(batch[2], device=self.device)
                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att)

                loss = self.criterion(outputs, targets)
                avg_val_loss += loss.item()
                if 'f1-score' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    f1.append(sklearn.metrics.f1_score(true.cpu().numpy(),
                                                  preds.cpu().numpy()))
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    acc.append(sklearn.metrics.accuracy_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))

            avg_val_loss = avg_val_loss / len(val_loader)
            if 'f1-score' in self.metrics:
                metrics_dict['f1-score'] = np.mean(f1)
            if 'accuracy' in self.metrics:
                metrics_dict['accuracy'] = np.mean(acc)
            return avg_val_loss, metrics_dict

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    metrics_dict,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} ".format(
            avg_train_epoch_loss))
        print("Val loss: {} ".format(avg_val_epoch_loss))
        for metric in metrics_dict.keys():
            print("Metric {}: {}".format(metric, metrics_dict[metric]))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint.pth'))

        # we use the proposed method for saving EncoderDecoder model
        #self.model.save_pretrained(os.path.join(self.checkpoint_dir,
        # 'model_checkpoint'))


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        targets = to_device(batch[2], device=self.device)
        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att)

        loss = self.criterion(outputs, targets)
        # print(loss)
        return loss

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_max, cur_patience = 0, 0
        best_val_min = 10000
        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss, metrics_dict = self.calc_val_loss(val_loader)
            if self.checkpoint_with:
                avg_val = metrics_dict[self.checkpoint_with]
            else:
                avg_val = avg_val_loss
            if self.checkpoint_max:
                if avg_val > best_val_max:
                    self.save_epoch(epoch)
                    best_val_max = avg_val
                    cur_patience = 0
                else:
                    cur_patience += 1
            else:
                if avg_val < best_val_min:
                    self.save_epoch(epoch)
                    best_val_min = avg_val
                    cur_patience = 0
                else:
                    cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss, metrics_dict,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)


class BertTrainerKfold:
    # https: // github.com / PyTorchLightning / pytorch - lightning / issues / 839
    def __init__(self, folds, trainer):
        self.folds = folds
        self.trainer = trainer
        self.kf = KFold(n_splits=folds)

    def fit(self, dataset):
        raise NotImplementedError("Not implemented yet!")
        # for forld, (train_idx,valid_idx) in enumerate (self.kf.split(
        #         train_df)):
        #     train_loader = ...
        #     valid_loader = ...


class T5Trainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
                 checkpoint_with=None,
                 checkpoint_max=False,
                 metrics=None,
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion = criterion
        self.metrics = metrics
        self.checkpoint_with =checkpoint_with
        self.checkpoint_max = checkpoint_max


    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            f1=[]
            acc=[]
            metrics_dict={}
            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                targets = to_device(batch[2], device=self.device)
                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att)

                loss = self.criterion(outputs, targets)
                avg_val_loss += loss.item()
                if 'f1-score' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    f1.append(sklearn.metrics.f1_score(true.cpu().numpy(),
                                                  preds.cpu().numpy()))
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    acc.append(sklearn.metrics.accuracy_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))

            avg_val_loss = avg_val_loss / len(val_loader)
            if 'f1-score' in self.metrics:
                metrics_dict['f1-score'] = np.mean(f1)
            if 'accuracy' in self.metrics:
                metrics_dict['accuracy'] = np.mean(acc)
            return avg_val_loss, metrics_dict

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    metrics_dict,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} ".format(
            avg_train_epoch_loss))
        print("Val loss: {} ".format(avg_val_epoch_loss))
        for metric in metrics_dict.keys():
            print("Metric {}: {}".format(metric, metrics_dict[metric]))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint.pth'))

        # we use the proposed method for saving EncoderDecoder model
        #self.model.save_pretrained(os.path.join(self.checkpoint_dir,
        # 'model_checkpoint'))


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        targets = to_device(batch[2], device=self.device)
        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att)

        loss = self.criterion(outputs, targets)
        # print(loss)
        return loss

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_max, cur_patience = 0, 0
        best_val_min = 10000
        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss = self.train_step(sample_batch)
                avg_train_loss += loss.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_val_loss, metrics_dict = self.calc_val_loss(val_loader)
            if self.checkpoint_with:
                avg_val = metrics_dict[self.checkpoint_with]
            else:
                avg_val = avg_val_loss
            if self.checkpoint_max:
                if avg_val > best_val_max:
                    self.save_epoch(epoch)
                    best_val_max = avg_val
                    cur_patience = 0
                else:
                    cur_patience += 1
            else:
                if avg_val < best_val_min:
                    self.save_epoch(epoch)
                    best_val_min = avg_val
                    cur_patience = 0
                else:
                    cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss, metrics_dict,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)

        
class BertTrainerTask72(BertTrainer):

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        humor_rating = to_device(batch[2], device=self.device)
        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att)
        outputs = outputs.squeeze(1)
        loss = self.criterion(outputs, humor_rating)

        return loss

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0

            metrics_dict={}
            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                humor_rating = to_device(batch[2], device=self.device)
                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att)
                outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, humor_rating)
                avg_val_loss += loss.item()

            avg_val_loss = avg_val_loss / len(val_loader)

            return avg_val_loss, metrics_dict


class BertTrainerTask73(BertTrainer):
    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        targets = to_device(batch[3], device=self.device)
        outputs = self.model(input_ids=inputs,
                             attention_mask=inputs_att)
        loss = self.criterion(outputs, targets)
        # print(loss)
        return loss

    def calc_val_loss(self, val_loader):

        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            f1=[]
            acc=[]
            prec = []
            rec = []
            metrics_dict={}
            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                targets = to_device(batch[3], device=self.device)
                outputs = self.model(input_ids=inputs,
                                     attention_mask=inputs_att)
                loss = self.criterion(outputs, targets)
                avg_val_loss += loss.item()
                if 'f1-score' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    f1.append(sklearn.metrics.f1_score(true.cpu().numpy(),
                                                  preds.cpu().numpy()))
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    acc.append(sklearn.metrics.accuracy_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))
                if 'precision' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    prec.append(sklearn.metrics.precision_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))
                if 'recall' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    rec.append(sklearn.metrics.recall_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))
            avg_val_loss = avg_val_loss / len(val_loader)
            if 'f1-score' in self.metrics:
                metrics_dict['f1-score'] = np.mean(f1)
            if 'accuracy' in self.metrics:
                metrics_dict['accuracy'] = np.mean(acc)
            if 'recall' in self.metrics:
                metrics_dict['recall'] = np.mean(rec)
            if 'precision' in self.metrics:
                metrics_dict['precision'] = np.mean(prec)
            return avg_val_loss, metrics_dict


class BertTrainerTask73Multitask:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion1,
                 criterion2,
                 multitask1=1.0,
                 multitask2=1.0,
                 checkpoint_with=None,
                 checkpoint_max=False,
                 metrics=[],
                 scheduler=None,
                 checkpoint_dir=None,
                 clip=None,
                 device='cpu'):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.clip = clip
        self.device = device
        self.patience = patience
        self.criterion1 = criterion1
        self.criterion2 = criterion2
        self.multitask1 = multitask1
        self.multitask2 = multitask2
        self.metrics = metrics
        self.checkpoint_with =checkpoint_with
        self.checkpoint_max = checkpoint_max


    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        inputs = to_device(batch[0], device=self.device)
        inputs_att = to_device(batch[1], device=self.device)
        humor_rating = to_device(batch[2], device=self.device)
        humor_contr = to_device(batch[3], device=self.device)
        outputs1,outputs2 = self.model(input_ids=inputs,
                             attention_mask=inputs_att)
        outputs1 = outputs1.squeeze(1)
        loss1 = self.criterion1(outputs1, humor_rating)
        loss2 = self.criterion2(outputs2,humor_contr)
        return loss1,loss2


    def calc_val_loss(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            avg_val_loss = 0
            avg_val_loss1 = 0
            avg_val_loss2 = 0
            f1 = []
            acc = []
            prec = []
            rec = []
            metrics_dict = {}
            for index, batch in enumerate(tqdm(val_loader)):
                inputs = to_device(batch[0], device=self.device)
                inputs_att = to_device(batch[1], device=self.device)
                humor_rating = to_device(batch[2], device=self.device)
                humor_contr = to_device(batch[3], device=self.device)
                outputs1,outputs2 = self.model(input_ids=inputs,
                                     attention_mask=inputs_att)
                outputs1 = outputs1.squeeze(1)
                loss1 = self.criterion1(outputs1, humor_rating)
                loss2 = self.criterion2(outputs2, humor_contr)
                loss = self.multitask1*loss1+self.multitask2*loss2
                avg_val_loss += loss.item()
                avg_val_loss1 += loss1.item()
                avg_val_loss2 += loss2.item()
                if 'f1-score' in self.metrics:
                    preds = torch.argmax(outputs2, dim=1)
                    true = copy.deepcopy(humor_contr)
                    f1.append(sklearn.metrics.f1_score(true.cpu().numpy(),
                                                       preds.cpu().numpy()))
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs2, dim=1)
                    true = copy.deepcopy(humor_contr)
                    acc.append(sklearn.metrics.accuracy_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))
                if 'precision' in self.metrics:
                    preds = torch.argmax(outputs2, dim=1)
                    true = copy.deepcopy(humor_contr)
                    prec.append(sklearn.metrics.precision_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))
                if 'recall' in self.metrics:
                    preds = torch.argmax(outputs2, dim=1)
                    true = copy.deepcopy(humor_contr)
                    rec.append(sklearn.metrics.recall_score(true.cpu(
                    ).numpy(), preds.cpu().numpy()))
            avg_val_loss = avg_val_loss / len(val_loader)
            avg_val_loss1 = avg_val_loss1 / len(val_loader)
            avg_val_loss2 = avg_val_loss2 / len(val_loader)
            if 'f1-score' in self.metrics:
                metrics_dict['f1-score'] = np.mean(f1)
            if 'accuracy' in self.metrics:
                metrics_dict['accuracy'] = np.mean(acc)
            if 'recall' in self.metrics:
                metrics_dict['recall'] = np.mean(rec)
            if 'precision' in self.metrics:
                metrics_dict['precision'] = np.mean(prec)
            print("avg val loss1 {} | avg val loss 2 {}".format(
                avg_val_loss1,avg_val_loss2))
            return avg_val_loss, metrics_dict

    def print_epoch(self, epoch, avg_train_epoch_loss, avg_val_epoch_loss,
                    metrics_dict,
                    cur_patience, strt):

        print("Epoch {}:".format(epoch+1))
        print("Train loss: {} ".format(
            avg_train_epoch_loss))
        print("Val loss: {} ".format(avg_val_epoch_loss))
        for metric in metrics_dict.keys():
            print("Metric {}: {}".format(metric, metrics_dict[metric]))
        print("Patience left: {}".format(self.patience-cur_patience))
        print("Time: {} mins".format((time.time() - strt) / 60.0))
        print("++++++++++++++++++")

    def save_epoch(self, epoch, loss=None):

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        torch.save(self.model.state_dict(), os.path.join(
            self.checkpoint_dir, 'model_checkpoint.pth'))

        # we use the proposed method for saving EncoderDecoder model
        #self.model.save_pretrained(os.path.join(self.checkpoint_dir,
        # 'model_checkpoint'))

    def train_epochs(self, n_epochs, train_loader, val_loader):

        best_val_max, cur_patience = 0, 0
        best_val_min = 10000
        print("Training model....")
        self.model.train()

        for epoch in range(n_epochs):
            if cur_patience == self.patience:
                break

            avg_train_loss = 0
            avg_train_loss1 = 0
            avg_train_loss2 = 0
            strt = time.time()

            for index, sample_batch in enumerate(tqdm(train_loader)):

                loss1,loss2 = self.train_step(sample_batch)
                loss = self.multitask1*loss1+self.multitask2*loss2
                avg_train_loss += loss.item()
                avg_train_loss1 += loss1.item()
                avg_train_loss2 += loss2.item()
                loss.backward(retain_graph=False)
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clip)
                self.optimizer.step()
            avg_train_loss = avg_train_loss / len(train_loader)
            avg_train_loss1 = avg_train_loss1 / len(train_loader)
            avg_train_loss2 = avg_train_loss2 / len(train_loader)
            avg_val_loss, metrics_dict = self.calc_val_loss(val_loader)
            if self.checkpoint_with:
                avg_val = metrics_dict[self.checkpoint_with]
            else:
                avg_val = avg_val_loss
            if self.checkpoint_max:
                if avg_val > best_val_max:
                    self.save_epoch(epoch)
                    best_val_max = avg_val
                    cur_patience = 0
                else:
                    cur_patience += 1
            else:
                if avg_val < best_val_min:
                    self.save_epoch(epoch)
                    best_val_min = avg_val
                    cur_patience = 0
                else:
                    cur_patience += 1
            print("Avg Train loss1 {} | Avg Train loss2 {}".format(
                avg_train_loss1, avg_train_loss2))
            self.print_epoch(epoch, avg_train_loss, avg_val_loss, metrics_dict,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)