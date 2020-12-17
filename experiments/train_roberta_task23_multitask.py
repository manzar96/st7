import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel

from core.data.dataset import Task723Dataset
from core.data.collators import Task723Collator
from core.models.modules.heads import DoubleHead
from core.trainers import BertTrainerTask73Multitask
from core.utils.parser import get_train_parser



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# make transforms using only bert tokenizer!
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
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
encoder = RobertaModel.from_pretrained('roberta-base')

# change config if you want
# encoder.config.output_hidden_states = True
model = DoubleHead(encoder, encoder.config.hidden_size,act='relu',
                               num_classes1=1,num_classes2=2, drop=0.2)
if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt,map_location='cpu')
    model.load_state_dict(state_dict)

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
criterion1 = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
import ipdb;ipdb.set_trace()

# create trainer

trainer = BertTrainerTask73Multitask(model=model, optimizer=optimizer,
                          criterion1=criterion1,
                          criterion2=criterion2,
                                     multitask1=options.multitask1,
                                     multitask2=options.multitask2,

                      checkpoint_max=False,
                      checkpoint_with=None,
                      patience=5, scheduler=None,
                      checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)
