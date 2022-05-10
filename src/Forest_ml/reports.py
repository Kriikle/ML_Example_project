import pandas as pd
import pandas_profiling


def create_profile_report():
    """
    Crating report with pandas profile
    File name is train.csv it must be in data folder
    """
    df = pd.read_csv('data/train.csv')
    profile = df.profile_report()
    profile.to_file("report/pandas_profiling.html")



