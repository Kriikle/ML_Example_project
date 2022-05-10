
from errno import ESTALE
from pathlib import Path
from xmlrpc.client import Boolean
from joblib import dump

import argparse

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score

from sklearn.model_selection import KFold

import numpy as np

from .data import get_dataset
from .pipeline import create_pipeline
from .hyperparam_finder import finder

parser = argparse.ArgumentParser(description='Program parasms')

# file path
parser.add_argument(
    '-dataset_path',
    type=Path, default="data/train.csv",
    help='Path to dataset')
parser.add_argument(
    '-save_model_path',
    type=Path, default="data/model.joblib",
    help='Path to output file')

# Model select
parser.add_argument(
    '-model_choise',
    type=int,
    default=0,
    help="0:LogisticRegression;else:LogisticRegression"
)

# Param finder
parser.add_argument(
    '-finder',
    type=bool,
    default=False,
    help="If you have a lot of time use True"
)


parser.add_argument(
    '-use_scaler',
    type=bool,
    default=True,
    help="Scalar using or not"
)

# First model LogisticRegression
parser.add_argument(
    '-max_iter', 
    type=int, 
    default=100,
    help = "LogisticRegression parametr"
)
parser.add_argument(
    '-logreg_C',
    type=float,
    default=1,
    help = "LogisticRegression parametr"
)
parser.add_argument(
    '-random_state',
    type=int,
    default=42,
    help = "LogisticRegression parametr"
)


# Second model RandomForestClassifier
parser.add_argument(
    '-n_estimators',
    type=int, 
    default=1000,
    help = "RandomForestClassifier parametr"
)

parser.add_argument(
    '-criterion',
    type=str,
    default='gini',
    choices=['gini', 'entropy'],
    help = "RandomForestClassifier parametr"
)
parser.add_argument(
    '-max_depth',
    type=int, 
    default=None,
    help = "RandomForestClassifier parametr"
)

args = parser.parse_args()

def train(
    dataset_path=args.dataset_path,
    save_model_path=args.save_model_path,
    model_num=args.model_choise,
    random_state=args.random_state,
    test_split_ratio=0.2,
    use_scaler=args.use_scaler,
    max_iter=args.max_iter,
    logreg_c=args.logreg_C,
    n_estimators=args.n_estimators,
    criterion=args.criterion,
    max_depth=args.max_depth,
    auto_finder = args.finder,
    target="Cover_Type"
):
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
        target,
    )
    with mlflow.start_run():
        if auto_finder == True:
            best_params,score = finder(features_train,target_train,model_num)
            if model_num==0:
                max_iter=best_params['max_iter']
                logreg_C=best_params['C']
                random_state=best_params['random_state']
            else:
                n_estimators=n_estimators=best_params['n_estimators']
                criterion=criterion=best_params['criterion']
                max_depth=best_params['max_depth']
            # mlflow.log_metric("Nested_score", score)
            # print(best_params,score)

        pipeline = create_pipeline(
            model_num=model_num,
            use_scaler=use_scaler,
            max_iter=max_iter,
            logreg_C=logreg_c,
            random_state=random_state,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth
        )

        
        pipeline.fit(features_train, target_train)
        predited = pipeline.predict(features_val)
        accuracy = accuracy_score(target_val, predited)
        f1 = f1_score(target_val, predited,average='micro')
        recall = recall_score(target_val, predited, average='micro')
        # roc_auc = roc_auc_score(target_val, predited,multi_class="ovo")
        mlflow.log_param("use_scaler", use_scaler)
        if model_num == 0:
            mlflow.log_param("Model_type", "LogisticRegression")
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
        else:
            mlflow.log_param("Model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        # mlflow.log_metric("roc_auc", roc_auc)

        dump(pipeline, save_model_path)