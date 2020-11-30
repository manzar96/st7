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

from sklearn.ensemble import StackingClassifier


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


# classifiers for voting
clf1 = SVC(probability=True)
clf2 = RandomForestClassifier()
clf3= DecisionTreeClassifier()
clf4 = AdaBoostClassifier()
clf5 = KNeighborsClassifier()

XGBClassifier()

ensemble = StackingClassifier(estimators=[('SVC',clf1),('rf',clf2),('dectree',
                                                                  clf3),
                                        ('ada',clf4),('knn',clf5)],
                            final_estimator=XGBClassifier(),n_jobs=-1)

# perform kfold cross-validation with k=5
kf = KFold(n_splits=5)

f1 = []
acc = []
for train_index, test_index in kf.split(humor):
    X_train,X_test = feats[train_index], feats[test_index]
    y_train,y_test = humor[train_index], humor[test_index]
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict(X_test)
    f1.append(f1_score(y_test, pred))
    acc.append(accuracy_score(y_test, pred))

print("F1-score: ", np.mean(f1))
print("Accuracy score: ", np.mean(acc))

ensemble.fit(feats, humor)

if not os.path.exists(options.ckpt):
    os.makedirs(options.ckpt)
pickle.dump(ensemble, open(os.path.join(options.ckpt,"modelcheckpoint.pth"),
                           "wb"))
