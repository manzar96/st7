import pickle
import os
import numpy as np
from tqdm import tqdm
from core.utils.parser import get_feat_parser

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
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
    clf = GaussianProcessRegressor()
elif options.clf == "SVR":
    clf = SVR()
elif options.clf == "LinearSVR":
    clf = LinearSVR(max_iter=10000)
elif options.clf == "DecisionTree":
    clf = DecisionTreeRegressor()
elif options.clf == "RandomForest":
    clf = RandomForestRegressor()
elif options.clf == "AdaBoost":
    clf = AdaBoostRegressor(n_estimators=100)
elif options.clf == "XGBoost":
    clf = XGBRegressor()
elif options.clf == "KNN":
    clf = KNeighborsRegressor(n_neighbors=5)
elif options.clf == "RBF":
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessRegressor(kernel=kernel, random_state=0)
elif options.clf == "Ridge":
    clf = KernelRidge(alpha=1.0)
else:
    raise IOError("Please select a valid clf!")

# perform kfold cross-validation with k=5
kf = KFold(n_splits=5)

mse = []

for train_index, test_index in kf.split(humor):
    X_train,X_test = feats[train_index], feats[test_index]
    y_train,y_test = humor[train_index],humor[test_index]
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    mse.append(mean_squared_error(y_test, pred))

print("MSE: ",np.mean(mse))
print("RMSE: ",np.sqrt(np.mean(mse)))

if options.clf == 'GaussianProc':
    clf = GaussianProcessRegressor()
elif options.clf == "SVR":
    clf = SVR()
elif options.clf == "LinearSVR":
    clf = LinearSVR(max_iter=10000)
elif options.clf == "DecisionTree":
    clf = DecisionTreeRegressor()
elif options.clf == "RandomForest":
    clf = RandomForestRegressor()
elif options.clf == "AdaBoost":
    clf = AdaBoostRegressor(n_estimators=100)
elif options.clf == "XGBoost":
    clf = XGBRegressor()
elif options.clf == "KNN":
    clf = KNeighborsRegressor(n_neighbors=5)
elif options.clf == "RBF":
    kernel = 1.0 * RBF(1.0)
    clf = GaussianProcessRegressor(kernel=kernel, random_state=0)
elif options.clf == "Ridge":
    clf = KernelRidge(alpha=1.0)
else:
    raise IOError("Please select a valid clf!")

clf.fit(feats,humor)
if not os.path.exists(options.ckpt):
    os.makedirs(options.ckpt)
pickle.dump(clf, open(os.path.join(options.ckpt,"{}.pth".format(options.clf)),
                   "wb"))
