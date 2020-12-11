import pickle
from core.utils.parser import get_feat_parser
import os
import argparse
import numpy as np


def merge_features(features1,features2):
    dict1 = pickle.load(open(features1, "rb"))
    dict2 = pickle.load(open(features2, "rb"))

    new_dict={}

    for key in dict1.keys():
        value1 = dict1[key]
        value2 = dict2[key]
        new_dict[key] = np.array(value1[0].tolist()+value2[0].tolist())

    return new_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features1",
        type=str,
        required=True,
        help="Features pickle 1 to be loaded.",
    )

    parser.add_argument(
        "--features2",
        type=str,
        required=True,
        help="Features pickle 2 to be loaded.",
    )

    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="output file for merged features",
    )

    options = parser.parse_args()

    merged_dict = merge_features(options.features1,options.features2)

    pickle.dump(merged_dict, open(options.outfile, "wb"))
