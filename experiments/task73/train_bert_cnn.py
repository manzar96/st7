import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

from core.data.dataset import Task723Dataset
from core.data.collators import Task723Collator
from core.models.modules.heads import ClassificationHead
from core.models.bertcnn import BertCNN
from core.trainers import BertTrainerTask73
from core.utils.parser import get_train_parser



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# make transforms using only bert tokenizer!
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# CLS token will work as BOS token
# tokenizer.bos_token = tokenizer.cls_token
# SEP token will work as EOS token
# tokenizer.eos_token = tokenizer.sep_token

# load dataset
dataset = Task723Dataset("train", tokenizer=tokenizer)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [4442,
                                                                     490],
                              generator=torch.Generator().manual_seed(42))

collator_fn = Task723Collator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                        drop_last=False, shuffle=True,
                        collate_fn=collator_fn)

# create model
if options.modelckpt is not None:
    model = BertModel.from_pretrained(options.modelckpt)
else:
    model = BertModel.from_pretrained('bert-base-uncased')

# change config if you want
# model.config.output_hidden_states = True

model = BertCNN(model)
model = ClassificationHead(encoder=ClassificationHead(model,
                                                      encoded_features=1344,
                                                      num_classes=256,
                                                      act='relu',
                                                      drop=0.2),
                           encoded_features=256, num_classes=2, drop=0.2)

model.to(DEVICE)

# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
metrics = ['f1-score','accuracy']
import ipdb;ipdb.set_trace()

# create trainer

trainer = BertTrainerTask73(model=model, optimizer=optimizer,
                           criterion=criterion,
                      metrics=metrics,
                      checkpoint_max=True,
                      checkpoint_with='f1-score',
                      patience=5, scheduler=None,
                      checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)
