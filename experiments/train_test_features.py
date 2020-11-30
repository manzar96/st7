import torch
import pickle
import csv
import numpy as np
from core.utils.parser import get_feat_parser

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,LinearSVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

parser = get_feat_parser()
options = parser.parse_args()

if options.features is None:
    raise IOError("Enter features!")

dict  = pickle.load(open(options.features, "rb"))


feats=[]
humor = []
for key in dict.keys():
    value = dict[key]
    feats.append(value[0].tolist())
    humor.append(value[1].tolist())

# X_train, X_test, y_train, y_test = train_test_split(feats,humor,test_size=0.2,
#                                                     random_state=42)
if options.clf is 'GaussianProcess':
    clf = GaussianProcessClassifier()
elif options.clf is "SVC":
    clf = SVC()
elif options.clf is "LinearSVC":
    clf = LinearSVC(max_iter=10000,dual=False)
elif options.clf is "LinearSVR":
    clf = LinearSVR(dual=False)
elif options.clf is "DecisionTree":
    clf = DecisionTreeClassifier()
elif options.clf is "RandomForest":
    clf = RandomForestClassifier()
elif options.clf is "AdaBoost":
    clf = AdaBoostClassifier()
elif options.clf is "KNN":
    clf = KNeighborsClassifier(n_neighbors=5)
elif options.clf == "GaussianNB":
    clf = GaussianNB()
elif options.clf is "RBF":
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
else:
    raise IOError("Please a valid select clf!")

# perform kfold cross-validation with k=5
kf = KFold(n_splits=5)

f1 = []
acc = []
for train_index, test_index in kf.split(humor):
    X_train,X_test = feats[train_index], feats[test_index]
    y_train,y_test = humor[train_index],humor[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    f1.append(f1_score(y_test, pred))
    acc.append(accuracy_score(y_test, pred))

print(np.mean(f1))
print(np.mean(acc))