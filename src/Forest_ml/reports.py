import pandas as pd
import pandas_profiling
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description='Program parasms')

# file path
parser.add_argument(
    '-dataset_path',
     type=Path, default="data/train.csv",
     help='Path to dataset')
args = parser.parse_args()

def create_profile_report(dataset_path=args.dataset_path):
    """
    Crating report with pandas profile
    File name is train.csv it must be in data folder
    """
    df = pd.read_csv(dataset_path)
    profile = df.profile_report()
    profile.to_file("report/pandas_profiling.html")



