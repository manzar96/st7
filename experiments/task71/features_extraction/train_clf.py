import pickle
import os
import numpy as np
from tqdm import tqdm
from core.utils.parser import get_feat_parser

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
from xgboost import XGBClassifier
from EvoDAG.model import EvoDAG,EvoDAGE
# from EvoMSA.base import EvoMSA
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
feats = np.array(feats)
humor = np.array(humor)


if options.clf == 'GaussianProc':
    clf = GaussianProcessClassifier()
elif options.clf == "SVC":
    clf = SVC()
elif options.clf == "LinearSVC":
    clf = LinearSVC(max_iter=10000,dual=False)
elif options.clf == "DecisionTree":
    clf = DecisionTreeClassifier()
elif options.clf == "RandomForest":
    clf = RandomForestClassifier()
elif options.clf == "AdaBoost":
    clf = AdaBoostClassifier(n_estimators=100)
elif options.clf == "XGBoost":
    clf = XGBClassifier()
elif options.clf == "KNN":
    clf = KNeighborsClassifier(n_neighbors=5)
elif options.clf == "GaussianNB":
    clf = GaussianNB()
elif options.clf == "RBF":
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
elif options.clf == "EvoDAGE":
    clf = EvoDAGE(n_estimators=30, n_jobs=4)
elif options.clf == "EvoDAG":
    clf = EvoDAG()
# elif options.clf == "EvoMSA":
#     clf = EvoMSa(Emo=True, lang='es')
else:
    raise IOError("Please select a valid clf!")

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

print("F1-score: ",np.mean(f1))
print("Accuracy score: ",np.mean(acc))

if options.clf == 'GaussianProc':
    clf = GaussianProcessClassifier()
elif options.clf == "SVC":
    clf = SVC()
elif options.clf == "LinearSVC":
    clf = LinearSVC(max_iter=10000,dual=False)
elif options.clf == "DecisionTree":
    clf = DecisionTreeClassifier()
elif options.clf == "RandomForest":
    clf = RandomForestClassifier()
elif options.clf == "AdaBoost":
    clf = AdaBoostClassifier()
elif options.clf == "XGBoost":
    clf = XGBClassifier()
elif options.clf == "KNN":
    clf = KNeighborsClassifier(n_neighbors=5)
elif options.clf == "GaussianNB":
    clf = GaussianNB()
elif options.clf == "RBF":
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessClassifier(kernel=kernel, random_state=0)
elif options.clf == "EvoDAGE":
    clf = EvoDAGE(n_estimators=30, n_jobs=4)
elif options.clf == "EvoDAG":
    clf = EvoDAG()
# elif options.clf == "EvoMSA":
#     clf = EvoMSa(Emo=True, lang='es')
else:
    raise IOError("Please select a valid clf!")

clf.fit(feats,humor)
if not os.path.exists(options.ckpt):
    os.makedirs(options.ckpt)
pickle.dump(clf, open(os.path.join(options.ckpt,"{}.pth".format(options.clf)),
                   "wb"))
