import os
import sys
import re
import numpy as np
import math
from datetime import date
import json_numpy
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error

import logging


# Function to load data from a json
def load_json(name):
    with open(name, 'r') as file:
        return json_numpy.load(file)


def get_regression_label(filename):
    input_lines = open(filename, "r").read().split("\n")
    lines = [[x for x in y.split(" ") if x] for y in input_lines if "//" in y]
    ki = {x[0]: re.split("[a-z][A-Z]", x[4].split("=")[1])[0] for x in lines}
    ki_scale = {x[0]: re.split("([a-z][A-Z])", x[4].split("=")[1])[1].split("M")[0] for x in lines}
    exp_ki = {x: np.format_float_positional(float(ki[x]) * scales[ki_scale[x]], precision=4, fractional=False)
              for x in ki}
    exp_pki = {x: np.format_float_positional(math.log10(float(exp_ki[x])), precision=4, fractional=False)
               for x in exp_ki}
    return exp_pki


def generate_rf_from_search(grid_or_random, rf, parameters, scorer, X, y):
    if grid_or_random == "Grid":
        search_obj = GridSearchCV(rf, parameters, scoring=scorer)
    elif grid_or_random == "Random":
        search_obj = RandomizedSearchCV(rf, parameters, scoring=scorer)
    fit_obj = search_obj.fit(X, y)
    best_rf = fit_obj.best_estimator_
    return best_rf


def train_rf(train_x, train_y):
    """
    train random forest model
    """
    rf = RandomForestRegressor(n_estimators=2000, max_features=0.3, max_depth=50, min_samples_split=2,
                               min_samples_leaf=2, random_state=42)
    trained_model = rf.fit(train_x, train_y)

    return trained_model


def evaluate_rf(trained_model, test_x, test_y):
    """
    get mean absolute error (MAE),
    get root mean squared error (RMSE),
    get coefficient of determination (R^2)
    """
    predicted = trained_model.predict(test_x)
    mae = mean_absolute_error(test_y, predicted)
    rmse = mean_squared_error(test_y, predicted, squared=False)
    r2 = r2_score(test_y, predicted)

    return mae, rmse, r2


if __name__ == "__main__":
    # get src directory
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    my_dir = os.getcwd()
    # execute script to get necessary features
    # pairwise_features = load_json(my_dir + "/all_features_01_10_24.json")
    pairwise_features = load_json(my_dir + "/all_features_01_10_24.json")
    pdb_ids = [x for x in pairwise_features]

    # get experimental values for training
    scales = {'f': 10 ** -15, 'p': 10 ** -12, 'n': 10 ** -9, 'u': 10 ** -6, 'm': 10 ** -3}
    try:
        experiment_f = "INDEX_refined_data.2020"
        assert os.path.isfile(experiment_f)
        experiment = get_regression_label(experiment_f)
        regression_labels = {x: experiment[x] for x in experiment if x in pdb_ids}

    except AssertionError:
        print("INDEX_refined_data.2020 file needed to run.")
        exit()
    # make dataframes of features and regression labels and split data
    features = {x: pairwise_features[x].flatten() for x in pairwise_features}
    X_df = pd.DataFrame.from_dict(features, orient='index').sort_index()
    y_df = pd.DataFrame.from_dict(regression_labels, orient='index').sort_index()
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.1, random_state=42)
    model = train_rf(X_train, y_train)
    test_mae, test_rmse, test_r2 = evaluate_rf(model, X_test, y_test)
    print(test_mae, test_rmse, test_r2)

