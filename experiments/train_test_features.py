import torch
import pickle
import csv
from core.utils.parser import get_feat_parser

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

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

dict = favorite_color = pickle.load(open(options.features, "rb"))


feats=[]
humor = []
for key in dict.keys():
    value = dict[key]
    feats.append(value[0])
    humor.append(value[1])

X_train, X_test, y_train, y_test = train_test_split(feats,humor,test_size=0.2,
                                                    random_state=42)
# clf = GaussianProcessClassifier()
# clf = SVC()
# clf = LinearSVC(max_iter=10000,dual=False)
clf = AdaBoostClassifier()


clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(f1_score(y_test,pred))
print(accuracy_score(y_test,pred))