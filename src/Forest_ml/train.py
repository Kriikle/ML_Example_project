from ast import Str
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
parser.add_argument('-dataset_path', type=Path,default="data/train.csv")
parser.add_argument('-save_model_path', type=Path,default="data/model.joblib")

#Model select
parser.add_argument('-model_choise',type=int,default=0)


parser.add_argument('-use_scaler',type=bool,default=True)

#First model
parser.add_argument('-max_iter', type=int,default=100)
parser.add_argument('-logreg_C', type=float,default= 1)
parser.add_argument('-random_state', type=int,default= 42)


#Second model

parser.add_argument('-n_estimators', type=int,default=100)
parser.add_argument('-criterion', type=str,default='gini',choices=['gini', 'entropy'])
parser.add_argument('-max_depth', type=int,default=None)

args = parser.parse_args()
print(args)


def train(
    dataset_path = args.dataset_path ,
    save_model_path= args.save_model_path,
    model_num = args.model_choise,
    random_state = args.random_state,
    test_split_ratio = 0.2,
    use_scaler = args.use_scaler,
    max_iter = args.max_iter,
    logreg_c = args.logreg_C,
    n_estimators = args.n_estimators,
    criterion =args.criterion,
    max_depth = args.max_depth,
    target = "Cover_Type"
):
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
        target,
    )
    with mlflow.start_run():
        pipeline = create_pipeline(
            model_num = model_num,
            use_scaler = use_scaler,
            max_iter = max_iter,
            logreg_C = logreg_c,
            random_state = random_state,
            n_estimators = n_estimators,
            criterion = criterion,
            max_depth = max_depth
        )
        pipeline.fit(features_train, target_train)
        accuracy = accuracy_score(target_val, pipeline.predict(features_val))
        if model_num == 0:
            mlflow.log_param("Models", "LogisticRegression")
            mlflow.log_param("use_scaler", use_scaler)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
        else:
            mlflow.log_param("Models", "RandomForestClassifier")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)
        #click.echo(f"Accuracy: {accuracy}.")
        dump(pipeline, save_model_path)
        #click.echo(f"Model is saved to {save_model_path}.")