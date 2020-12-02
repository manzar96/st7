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
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,  \
    GradientBoostingRegressor,BaggingRegressor,VotingRegressor
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
    clf1 = GaussianProcessRegressor()
elif options.clf == "SVR":
    clf1 = SVR()
elif options.clf == "LinearSVR":
    clf1 = LinearSVR(max_iter=10000)
elif options.clf == "DecisionTree":
    clf1 = DecisionTreeRegressor()
elif options.clf == "RandomForest":
    clf1 = RandomForestRegressor()
elif options.clf == "AdaBoost":
    clf1 = AdaBoostRegressor(n_estimators=100)
elif options.clf == "XGBoost":
    clf1 = XGBRegressor()
elif options.clf == "KNN":
    clf1 = KNeighborsRegressor(n_neighbors=5)
elif options.clf == "RBF":
    kernel = 1.0 * RBF(1.0)
    clf1 = GaussianProcessRegressor(kernel=kernel, random_state=0)
elif options.clf == "Ridge":
    clf1 = KernelRidge(alpha=2.3)
elif options.clf == 'gradboost':
    clf1 = GradientBoostingRegressor()
else:
    raise IOError("Please select a valid clf!")

# perform kfold cross-validation with k=5
kf = KFold(n_splits=5)

# clf2 = BaggingRegressor(base_estimator=clf1,n_estimators=30,n_jobs=-1)
clf2 = VotingRegressor(estimators=[('svr',SVR()),('ridge',KernelRidge(
    alpha=2.3))])
mse = []

for train_index, test_index in kf.split(humor):
    X_train,X_test = feats[train_index], feats[test_index]
    y_train,y_test = humor[train_index],humor[test_index]
    clf2.fit(X_train, y_train)
    pred = clf2.predict(X_test)
    mse.append(mean_squared_error(y_test, pred))

print("MSE: ",np.mean(mse))
print("RMSE: ",np.sqrt(np.mean(mse)))

if options.clf == 'GaussianProc':
    clf1 = GaussianProcessRegressor()
elif options.clf == "SVR":
    clf1 = SVR()
elif options.clf == "LinearSVR":
    clf1 = LinearSVR(max_iter=10000)
elif options.clf == "DecisionTree":
    clf1 = DecisionTreeRegressor()
elif options.clf == "RandomForest":
    clf1 = RandomForestRegressor()
elif options.clf == "AdaBoost":
    clf1 = AdaBoostRegressor(n_estimators=100)
elif options.clf == "XGBoost":
    clf1 = XGBRegressor()
elif options.clf == "KNN":
    clf1 = KNeighborsRegressor(n_neighbors=5)
elif options.clf == "RBF":
    kernel = 1.0 * RBF(1.0)
    clf1 = GaussianProcessRegressor(kernel=kernel, random_state=0)
elif options.clf == "Ridge":
    clf1 = KernelRidge(alpha=2.3)
elif options.clf == 'gradboost':
    clf1 = GradientBoostingRegressor()
else:
    raise IOError("Please select a valid clf!")

clf2.fit(feats,humor)
if not os.path.exists(options.ckpt):
    os.makedirs(options.ckpt)
pickle.dump(clf2, open(os.path.join(options.ckpt,"bagging_{}.pth".format(
    options.clf)),
                   "wb"))
