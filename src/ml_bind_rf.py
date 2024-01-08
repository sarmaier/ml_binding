import os
import sys
from datetime import date
import glob
import json_numpy
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


if __name__ == "__main__":
    # get src directory
    py_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    my_dir = os.getcwd()
    # execute script to build necessary data files
    pairwise_features = load_json("all_features.json")
    non_pairwise_features = load_json("no_pairwise_features.json")


    exit()

