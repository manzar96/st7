import pickle
import os
import numpy as np
import csv
import pandas as pd
from core.utils.parser import get_feat_test_parser

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor

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
        csv_writer.writerow([id, '{:.3f}'.format(float(out))])
