from pathlib import Path
from joblib import dump

import argparse

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score


from .data import get_dataset
from .pipeline import create_pipeline

parser = argparse.ArgumentParser(description='Program parasms')

#file path
parser.add_argument('-data_path', type=Path,default="data/train.csv")
parser.add_argument('-finalfile_path', type=Path,default="data/model.joblib")

#Model select
parser.add_argument('-model_choise',type=int,default=0)

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


def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        #click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        #click.echo(f"Model is saved to {save_model_path}.")