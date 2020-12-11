import pickle
import os
import numpy as np
import csv
import pandas as pd
from core.utils.parser import get_feat_test_parser

parser = get_feat_test_parser()
options = parser.parse_args()


if options.features is None:
    raise IOError("Enter features!")

dict = pickle.load(open(options.features, "rb"))

ids = []
feats=[]
for key in dict.keys():
    value = dict[key]
    ids.append(int(key))
    feats.append(value.tolist())
feats = np.array(feats)

pca = pickle.load(open(options.pcackpt,'rb'))
feats = pca.transform(feats)

clf = pickle.load(open(options.modelckpt,'rb'))
preds = clf.predict(feats)


if not os.path.exists(options.outfolder):
    os.makedirs(options.outfolder)
outfile = os.path.join(options.outfolder, 'output.csv')

with open(outfile, 'w') as output:
    csv_writer = csv.writer(output, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for id, out in zip(ids, preds):
        csv_writer.writerow([id, '{}'.format(int(out))])
