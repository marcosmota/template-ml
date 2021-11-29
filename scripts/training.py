
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


EXPERIMENT_ID = 'wine_regression'


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print('Iniciando o treinamento')

    train_x = pd.read_csv('./data/feature_train.csv')
    test_x = pd.read_csv('./data/feature_test.csv')
    train_y = pd.read_csv('./data/target_train.csv')
    test_y = pd.read_csv('./data/target_test.csv')
    print('Obtendo os dados pre-processado')

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    print(f'Obtendo parâmetros de treinamento | alpha: {alpha} - l1 ratio: {l1_ratio}')

    with mlflow.start_run(mlflow.set_experiment(EXPERIMENT_ID)):
        print('Iniciando o experimento')
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        
        print('Aplicando o fit nos dados')
        lr.fit(train_x, train_y)

        print('Gerando previsões c/ o modelo')
        predicted_qualities = lr.predict(test_x)

        print('Avaliando modelo')
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        print('Salvando modelo, parâmentros e métricas')
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")