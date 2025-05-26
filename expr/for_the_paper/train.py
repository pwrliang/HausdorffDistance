import json
import glob
import os
from collections import OrderedDict
import numpy as np
import pandas as pd

# from sklearn.tree import DecisionTreeRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error
from xgboost import XGBRegressor
from math import log
import matplotlib.pyplot as plt
import re
import duckdb
import m2cgen as m2c
from pandas import json_normalize
import hashlib


def load_df(dir, param_name):
    cache_file = hashlib.md5((dir + param_name).encode()).hexdigest()
    # If cache exists, load from pickle
    if os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
        print("Loaded DataFrame from cache.", cache_file)
    else:
        json_records = []
        i = 0
        for root, _, files in os.walk(dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                if os.path.basename(root) == param_name and filename.lower().endswith(".json") and os.path.getsize(
                        file_path) > 0:
                    with open(file_path) as f:
                        json_records.append(json.load(f))
                    i += 1
        df = pd.json_normalize(json_records)

        # Serialize to pickle
        df.to_pickle(cache_file)
        print("Loaded DataFrame from JSON and saved to cache.")
    return df


def export_regressor(regressor, key, vars, isPoisson=False, base_score=1):
    c_code = m2c.export_to_c(regressor, function_name="Predict" + key)

    variables = []
    members = []

    # Helper to strip only the last numeric suffix
    def collapse_key(var):
        return re.sub(r'_\d+$', '', var)

    # Ordered dictionary to maintain insertion order of collapsed prefixes
    collapsed = OrderedDict()

    # O(1) time per variable processing
    for var in vars:
        strippd_key = collapse_key(var)
        collapsed[strippd_key] = collapsed.get(strippd_key, 0) + 1

    # Output results
    for var, count in collapsed.items():
        if count == 1:
            members.append(f"    double {var};")
        else:
            members.append(f"    double {var}[{count}];")

    for i, var in enumerate(vars):
        variables.append(f"{i} {var}")

    header = """
#include <math.h>
#ifndef DECISION_TREE_{name}
#define DECISION_TREE_{name}
/*
{comments}

struct Input {{
{members}
}};

*/
inline {code}
#endif // DECISION_TREE_{name}
""".format(name=key.upper(), code=c_code, comments='\n'.join(variables), members='\n'.join(members))
    if base_score != 0:
        base_score = float(regressor.get_params()["base_score"])
    print("Base Score: " + str(base_score))
    if isPoisson:
        header = header.replace("nan", f"{base_score:.8f}")
        header = header.replace("return", f"double ret=")
        pattern = r"(ret\s*=\s*[^;]+;)"
        header = re.sub(pattern, r"\1\nreturn exp(ret);", header)
    else:
        header = header.replace("nan", f"{base_score:.8f}")

    with open("tree_{name}.h".format(name=key.lower()), "w") as f:
        f.write(header)


def train_param(name, X, Y, n_dims, regressor, return_model=False, isPoisson=False, base_score=1):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse}")
    mabs = median_absolute_error(y_test, y_pred)
    print(f"Median Absolute Error on test set: {mabs}")

    header = X_train.columns.tolist()
    export_regressor(regressor, name + "_" + str(n_dims) + "D", header, isPoisson=isPoisson, base_score=base_score)

    if return_model:
        return regressor, header


def train_eb_only_threshold(best_params_df, n_dims):
    df_features = extract_stats(best_params_df, 3)
    df_labels = best_params_df['Running.EBOnlyThreshold']
    base_score = df["Running.EBOnlyThreshold"].mean()
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.4,
                             n_estimators=5, min_child_weight=10, gamma=0.0, base_score=base_score)
    model, features = train_param(
        "EBOnlyThreshold", df_features, df_labels, n_dims, regressor,
        return_model=True, base_score=base_score  # , isPoisson=True
    )


def train_num_points_per_cell(best_params_df, n_dims):
    df_features = extract_stats(best_params_df, 3)
    df_labels = best_params_df['Running.NumPointsPerCell']
    base_score = df["Running.NumPointsPerCell"].mean()
    # base_score = 1
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.2,
                             n_estimators=20, min_child_weight=10, gamma=1.0, max_leaves=4, reg_lambda=1.0
                             , base_score=base_score)
    model, features = train_param(
        "NumPointsPerCell", df_features, df_labels, n_dims, regressor,
        return_model=True, base_score=base_score  # , isPoisson=True
    )


def train_max_hit(best_params_df, n_dims):
    df_features = extract_stats(best_params_df, 3)
    df_labels = best_params_df['Running.MaxHit']
    base_score = df["Running.MaxHit"].mean()
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.2,
                             n_estimators=5, min_child_weight=10, gamma=0.0, base_score=base_score)
    model, features = train_param(
        "MaxHit", df_features, df_labels, n_dims, regressor,
        return_model=True, base_score=base_score  # , isPoisson=True
    )


def extract_stats(df, n_dims):
    ft_df = pd.DataFrame()
    def extract_percentile(histo):
        for bucket in reversed(histo):
            if bucket['percentile'] < percentile:
                return bucket[key]
    for i in range(2):
        ft_df[f'File_{i}_Density'] = df['Input.Files'].apply(lambda x: x[i]['Density'])
        ft_df[f'File_{i}_NumPoints'] = df['Input.Files'].apply(lambda x: x[i]['NumPoints'])
        for dim in range(n_dims):
            ft_df[f'File_{i}_MBR_Dim_{dim}_Lower'] = df['Input.Files'].apply(
                lambda x: x[i]['MBR'][i]['Lower'] if i < len(x[i]['MBR']) else 0)
        for dim in range(n_dims):
            ft_df[f'File_{i}_MBR_Dim_{dim}_Upper'] = df['Input.Files'].apply(
                lambda x: x[i]['MBR'][i]['Upper'] if i < len(x[i]['MBR']) else 0)
        ft_df[f'File_{i}_GINI'] = df['Input.Files'].apply(lambda x: x[i]['Grid']['GiniIndex'])

        percentiles = (0.99, 0.95, 0.5, 0.1)
        key = 'value'
        for percentile in percentiles:
            ft_df[f'File_{i}_Cell_P{percentile}_Value'] = df['Input.Files'].apply(
                lambda x: extract_percentile(x[i]['Grid']['Histogram']))

        key = 'count'
        for percentile in percentiles:
            ft_df[f'File_{i}_Cell_P{percentile}_Count'] = df['Input.Files'].apply(
                lambda x: extract_percentile(x[i]['Grid']['Histogram']))

        for dim in range(n_dims):
            ft_df[f'File_{i}_Dim{dim}_GridSize'] = df['Input.Files'].apply(
                lambda x: x[i]['Grid']['GridSize'][dim] if dim < len(x[i]['Grid']['GridSize']) else 0)

        ft_df[f'File_{i}_NonEmptyCells'] = df['Input.Files'].apply(lambda x: x[i]['Grid']['NonEmptyCells'])
        ft_df[f'File_{i}_TotalCells'] = df['Input.Files'].apply(lambda x: x[i]['Grid']['TotalCells'])

    percentiles = (0.99, 0.95, 0.5, 0.1)
    key = 'value'
    for percentile in percentiles:
        ft_df[f'Cell_P{percentile}_Value'] = df['Input.Grid.Histogram'].apply(
            lambda x: extract_percentile(x))

    key = 'count'
    for percentile in percentiles:
        ft_df[f'Cell_P{percentile}_Count'] = df['Input.Grid.Histogram'].apply(
            lambda x: extract_percentile(x))

    for dim in range(n_dims):
        ft_df[f'Dim{dim}_GridSize'] = df['Input.Grid.GridSize'].apply(
            lambda x: x[dim] if dim < len(x) else 0)

    ft_df["HDLB"] = df['Running.Repeats'].apply(lambda x: x[0]['HDLowerBound'])
    ft_df["HDUP"] = df['Running.Repeats'].apply(lambda x: x[0]['HDUpperBound'])
    return ft_df


def get_best_params(df):
    df['RunID'] = df['Input.Files'].apply(
        lambda x: x[0]['Path'] + '-' + x[1]['Path']) + '-' + df['Input.Translate'].astype(str)
    min_idx = df.groupby('RunID')['Running.AvgTime'].idxmin()
    return df.loc[min_idx].reset_index(drop=True)


if __name__ == '__main__':
    # Set the path to the directory containing your JSON files
    directory_path = 'logs/train'  # <-- change this to your JSON directory

    df = load_df(directory_path, "eb_only_threshold")
    best_params_df = get_best_params(df)
    train_eb_only_threshold(best_params_df, 3)

    df = load_df(directory_path, "n_points_cell")
    best_params_df = get_best_params(df)
    train_num_points_per_cell(best_params_df, 3)

    df = load_df(directory_path, "max_hit")
    best_params_df = get_best_params(df)
    train_max_hit(best_params_df, 3)
