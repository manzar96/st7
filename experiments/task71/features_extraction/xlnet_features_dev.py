import math
import os
import pickle
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import XLNetTokenizer, XLNetModel
from core.models.modules.xlnet_sequence import XLNetSequence

from core.data.dataset import Task71Dataset
from core.data.collators import Task71aCollatorTest
from core.models.modules.heads import ClassificationHead
from core.utils.parser import get_train_parser
from core.utils.tensors import to_device

def get_features(data_loader,mymodel,device):
    features = []
    ids = []
    mymodel.eval()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader)):
            myid = batch[0]
            inputs = to_device(batch[1], device=device)
            inputs_att = to_device(batch[2], device=device)
            outputs = mymodel.encoder(input_ids=inputs, attention_mask=inputs_att)
            features.append(outputs.cpu().numpy())
            ids.append(myid)
    ids = [item for sublist in ids for item in sublist]
    features = [item for sublist in features for item in sublist]
    res = {ids[i]: features[i] for i in range(len(ids))}
    return res


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# make transforms using only bert tokenizer!
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')


# load dataset
dataset = Task71Dataset("dev", tokenizer=tokenizer)

collator_fn = Task71aCollatorTest(device='cpu')
loader = DataLoader(dataset, batch_size=options.batch_size,
                    drop_last=False, shuffle=True,
                    collate_fn=collator_fn)


# create model
encoder = XLNetModel.from_pretrained('xlnet-base-cased')
encoder = XLNetSequence(encoder)
# change config if you want
# encoder.config.output_hidden_states = True
model = ClassificationHead(encoder, encoder.config.hidden_size,act='sigmoid',
                               num_classes=2, drop=0.2)
if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt,map_location='cpu')
    model.load_state_dict(state_dict)

model.to(DEVICE)

res_dict = get_features(loader, model, DEVICE)
if not os.path.exists('./features_dev/'):
    os.makedirs('./features_dev')
pickle.dump(res_dict, open("./features_dev/xlnet_features.pkl", "wb"))
