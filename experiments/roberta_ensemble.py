import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel

from core.data.dataset import Task71Dataset
from core.data.collators import Task71aCollator
from core.models.modules.heads import BertClassificationHead
from core.trainers import BertTrainer
from core.utils.parser import get_train_parser

from core.torchensemble import VotingClassifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# make transforms using only bert tokenizer!
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

dataset = Task71Dataset("train", tokenizer=tokenizer)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [7200,
                                                                     800],
                              generator=torch.Generator().manual_seed(42))

collator_fn = Task71aCollator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                        drop_last=False, shuffle=True,
                        collate_fn=collator_fn)

# create model
encoder = RobertaModel.from_pretrained('roberta-base')

# change config if you want
# encoder.config.output_hidden_states = True
base_estimator = BertClassificationHead(encoder, encoder.config.hidden_size,
                                        num_classes=2, drop=0.2)

model = VotingClassifier(estimator=base_estimator, n_estimators=2,
                         output_dim=2,lr=1.5e-5,epochs=1,weight_decay=1e-6)
# import ipdb;ipdb.set_trace()
# model.fit(train_loader)
import ipdb;ipdb.set_trace()
model.predict(val_loader)
