import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    output_path = sys.argv[1]

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    wine_path = 'https://raw.githubusercontent.com/mlflow/mlflow-example/master/wine-quality.csv'
    data = pd.read_csv(wine_path)

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # save features and targets
    train_x.to_csv(f'{output_path}/feature_train.csv')
    test_x.to_csv(f'{output_path}/feature_test.csv')

    train_y.to_csv(f'{output_path}/target_train.csv')
    test_y.to_csv(f'{output_path}/target_test.csv')