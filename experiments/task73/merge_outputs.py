import csv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--csv1",
    type=str,
    required=True,
    help="csv1",
)

parser.add_argument(
    "--csv2",
    type=str,
    required=True,
    help="csv1",
)

parser.add_argument(
    "--out",
    type=str,
    required=True,
    help="outputcsv",
)

options = parser.parse_args()

# read csv 1
dict1 = {}
with open(options.csv1) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for index, line in enumerate(csv_reader):
        if index == 0:
            continue
        dict1[int(line[0])] = [int(line[1]),float(line[2])]

# read csv 2
dict2 = {}
with open(options.csv2) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for index, line in enumerate(csv_reader):
        if index == 0:
            continue
        dict2[int(line[0])] = int(line[1])

    with open(options.out, 'w') as output:
        csv_writer = csv.writer(output, delimiter=',',
                                     quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id','is_humor','humor_rating',
                             'humor_controversy'])

        for key in dict1.keys():
            csv_writer.writerow([key,dict1[key][0],dict1[key][1],dict2[key]])


