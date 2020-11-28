import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from core.models.modules.heads import ClassificationHead
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


# create model
if options.modelckpt is not None:
    model = BertModel.from_pretrained(options.modelckpt)
else:
    model = BertModel.from_pretrained('bert-base-uncased')

# change config if you want
# model.config.output_hidden_states = True
model = ClassificationHead(model, model.config.hidden_size, num_classes=2,
                           drop=0.2,
                           act='sigmoid')

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


import ipdb;ipdb.set_trace()

# # create trainer
# trainer = BertTrainer(model=model, optimizer=optimizer,
#                                  patience=5, scheduler=None,
#                                  checkpoint_dir=options.ckpt, device=DEVICE)
# # train model
# trainer.fit(train_loader, val_loader, epochs=options.epochs)