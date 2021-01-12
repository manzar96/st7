import argparse
import csv
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score

def calc_metrics(inputfile):
    devfile = open('./data/dev_labeled.csv', "r")
    csv_reader = csv.reader(devfile, delimiter=',')

    dict_res = {}
    for index,row in enumerate(csv_reader):
        if index==0:
            continue
        else:
            id,text,is_humor,rating,humor_contr,off = row
            dict_res[int(id)] = [text,is_humor,rating,humor_contr,off]

    infile = open(inputfile, "r")
    csv_reader = csv.reader(infile, delimiter=',')
    for index, row in enumerate(csv_reader):
        if index==0:
            continue
        else:
            id, is_humor = row
            dict_res[int(id)].append(is_humor)

    y_true = []
    y_pred = []
    for id in dict_res.keys():
        y_true.append(int(dict_res[int(id)][1]))
        y_pred.append(int(dict_res[int(id)][5]))

    print("F1-score ",f1_score(y_true,y_pred,pos_label=1))
    print("Accuracy ",accuracy_score(y_true,y_pred))
    print("Recall ",recall_score(y_true,y_pred))
    print("Precision ",precision_score(y_true,y_pred))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="File where of the generated outputs",
    )

    options = parser.parse_args()
    calc_metrics(options.inputfile)
