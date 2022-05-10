from pathlib import Path

import argparse
from tokenize import String
from xmlrpc.client import Boolean

parser = argparse.ArgumentParser(description='Program parasms')

#file path
parser.add_argument('-data_path', type=Path,default="data/train.csv")
parser.add_argument('-finalfile_path', type=Path,default="data/model.joblib")

#Model select
parser.add_argument('model_choise',type=int)

#First model
parser.add_argument('-max_iter', type=int,default=100)
parser.add_argument('-logreg_C', type=float,default= 1)
parser.add_argument('-random_state', type=int,default= 42)


#Second model

parser.add_argument('-n_estimators', type=int,default=100)
parser.add_argument('-criterion', type=ascii,default="gini")
parser.add_argument('-max_depth', type=int,default=None)

args = parser.parse_args()
print(args)