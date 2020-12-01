import torch
import os
import csv
from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel

from core.data.dataset import Task71Dataset
from core.data.collators import Task71aCollatorTest
from core.models.modules.heads import BertClassificationHead
from core.trainers import BertTrainer
from core.utils.parser import get_test_parser

from core.torchensemble import VotingClassifier


def create_submition_file(outfolder, mymodel, loader, device):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outfile = os.path.join(outfolder, 'output.csv')

    all_ids,all_outputs = mymodel.predict2(loader)
    import ipdb;ipdb.set_trace()
    ids_list = [item for sublist in all_ids for item in sublist]
    outs_list = [item for sublist in all_outputs for item in sublist]

    with open(outfile, 'w') as output:
        csv_writer = csv.writer(output, delimiter=',',
                                     quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for id,out in zip(ids_list, outs_list):
            csv_writer.writerow([id, int(out)])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_test_parser()
options = parser.parse_args()

# make transforms using only bert tokenizer!
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

dataset = Task71Dataset("dev", tokenizer=tokenizer)


collator_fn = Task71aCollatorTest(device='cpu')
loader = DataLoader(dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
encoder = RobertaModel.from_pretrained('roberta-base')

# change config if you want
# encoder.config.output_hidden_states = True
base_estimator = BertClassificationHead(encoder, encoder.config.hidden_size,
                                        num_classes=2, drop=0.2,act='sigmoid')

model = VotingClassifier(estimator=base_estimator, n_estimators=3,
                         output_dim=2,lr=1.5e-5,epochs=3,weight_decay=1e-6)

if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt,map_location='cpu')
    model.load_state_dict(state_dict)

model.to(DEVICE)

create_submition_file(options.outfolder, model, loader, DEVICE)
