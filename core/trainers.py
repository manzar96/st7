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
from typing import cast, List, Optional, Tuple, TypeVar
from core.utils.tensors import to_device
TrainerType = TypeVar('TrainerType', bound='Trainer')


class BertTrainer:

    def __init__(self, model,
                 optimizer,
                 patience,
                 criterion,
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
                    import pdb;pdb.set_trace()
                    f1.append(sklearn.metrics.f1_score(true.item().numpy(),
                                                  preds.item().numpy()))
                if 'accuracy' in self.metrics:
                    preds = torch.argmax(outputs, dim=1)
                    true = copy.deepcopy(targets)
                    acc.append(sklearn.metrics.accuracy_score(true.item(
                    ).numpy(), preds.item().numpy()))

            avg_val_loss = avg_val_loss / len(val_loader)
            if 'f1-score' in self.metrics:
                metrics_dict['fi-score'] = np.mean(f1)
            if 'accuracy' in self.metrics:
                metrics_dict['accuracy'] = np.mean(f1)
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
        # torch.save(self.model.state_dict(), os.path.join(
        #     self.checkpoint_dir, '{}_{}.pth'.format(epoch, 'model_checkpoint')))

        # we use the proposed method for saving EncoderDecoder model
        self.model.save_pretrained(os.path.join(self.checkpoint_dir,'model_checkpoint'))


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

        best_val_loss, cur_patience  = 10000, 0

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

            if avg_val_loss < best_val_loss:
                self.save_epoch(epoch)
                best_val_loss = avg_val_loss
                cur_patience = 0
            else:
                cur_patience += 1
            self.print_epoch(epoch, avg_train_loss, avg_val_loss,
                             cur_patience, strt)

    def fit(self, train_loader, val_loader, epochs):
        self.train_epochs(epochs, train_loader, val_loader)
