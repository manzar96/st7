import torch
import csv
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.metrics import f1_score
from EvoDAG.model import EvoDAGE

from sklearn.datasets import load_iris

# Reading data
data = load_iris()
X = data.data
y = data.target

clf = EvoDAGE(n_estimators=30, n_jobs=4).fit(X, y)

clf.fit(X,y)


# def read_dataset(csvfile):
#     ids = []
#     texts = []
#     humor = []
#     with open(csvfile) as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         for index, line in enumerate(csv_reader):
#             if index == 0:
#                 continue
#             ids.append(int(line[0]))
#             texts.append(line[1])
#             humor.append(int(line[2]))
#
#     return ids,texts,humor
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(DEVICE)
#
# # load dataset
# ids,texts,humor = read_dataset('./data/train_data/train.csv')
# X_train, X_test, y_train, y_test = train_test_split(texts,humor,
#                                                     test_size=0.2,
#                                                     random_state=42)
#
#
# evo = EvoMSA(lang='en',n_jobs=4,HA=True,
#              # stacked_method='sklearn.svm.LinearSVC')
#              stacked_method='sklearn.naive_bayes.GaussianNB')
#              # stacked_method = 'sklearn.tree.DecisionTreeClassifier')
#              # stacked_method= 'sklearn.ensemble.RandomForestClassifier')
#              # stacked_method= 'sklearn.ensemble.AdaBoostClassifier')
#              # stacked_method='sklearn.gaussian_process.GaussianProcessClassifier')
#              # stacked_method='sklearn.gaussian_process.GaussianProcessClassifier')
#              # stacked_method='sklearn.neural_network.MLPClassifier')

# evo.fit(X_train,y_train)
# pred = evo.predict(X_test)
# print(f1_score(y_test,pred))