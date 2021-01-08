import torch
import csv
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer,RobertaModel

from core.data.dataset import Task71Dataset
from core.data.collators import Task71aCollatorTest
from core.models.bertcnn import BertCNNHead
from core.utils.parser import get_test_parser
from core.utils.tensors import to_device



def create_submition_file(outfolder, mymodel, loader, device):
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    outfile = os.path.join(outfolder, 'output.csv')
    mymodel.eval()
    all_ids = []
    all_outputs = []
    for index, batch in enumerate(tqdm(loader)):
        myid = batch[0]
        inputs = to_device(batch[1], device=device)
        inputs_att = to_device(batch[2], device=device)

        outputs = mymodel(input_ids=inputs,
                             attention_mask=inputs_att)
        if not mymodel.act:
            outputs = torch.softmax(outputs,dim=1)
        outputs = torch.argmax(outputs,dim=1)
        all_ids.append(myid)
        all_outputs.append(outputs)

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


# load dataset
test_dataset = Task71Dataset("dev", tokenizer=tokenizer)


collator_fn = Task71aCollatorTest(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)


# create model
encoder = RobertaModel.from_pretrained('roberta-base')
encoder.config.output_hidden_states = True
model = BertCNNHead(encoder, encoder.config.hidden_size,
                               num_classes=2, drop=0.2,
                    drop_cnn=0.3, method=options.method)


if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt,map_location='cpu')
    model.load_state_dict(state_dict)

model.to(DEVICE)

create_submition_file(options.outfolder, model, test_loader, DEVICE)
