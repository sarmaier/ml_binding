import os
import sys
from datetime import date
import glob
import json_numpy
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error

import logging


# Function to load data from a json
def load_json(name):
    with open(name, 'r') as file:
        return json_numpy.load(file)


def train_rf(train_x, train_y):
    """
    train random forest model
    """
    rf = RandomForestRegressor(n_estimators=2000, max_features=0.5, max_depth=40, min_samples_split=2,
                               min_samples_leaf=2, random_state=42)
    trained_model = rf.fit(train_x, train_y)

    return trained_model


def evaluate_rf(trained_model, test_x, test_y):
    """
    get mean absolute error (MAE),
    get root mean squared error (RMSE),
    get coefficient of determination (R^2)
    """
    pred = trained_model.predict(test_x)
    mae = mean_absolute_error(test_y, pred)
    rmse = mean_squared_error(test_y, pred, squared=False)
    r2 = r2_score(test_y, pred)

    return mae, rmse, r2


def run_model():
    """
    loads the datasets, trains the RF model, and predicts the pKas for
    test split, sampl6, and novartis
    """

    # load the datasets

    # get features and exp_pKa of datasets

    # train random forest model
    # trained_model = train_rf(train_x, train_y)

    # evaluate trained model on test sets
    # test_mae, test_rmse, test_r2 = evaluate_rf(trained_model, test_x, test_y)

    return


if __name__ == "__main__":
    # get src directory
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    my_dir = os.getcwd()

    # execute script to build necessary data files
    pairwise_features = load_json(my_dir + "/all_features.json")
    for i in pairwise_features:
        print(pairwise_features[i][0].shape)
    non_pairwise_features = load_json("no_pairwise_features.json")


    exit()


