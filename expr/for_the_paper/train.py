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
from sql import *


def load_df(dir):
    cache_file = hashlib.md5((dir).encode()).hexdigest()
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
                if filename.lower().endswith(".json"):
                    with open(file_path) as f:
                        json_records.append(json.load(f))
                    i += 1
        df = pd.json_normalize(json_records)

        # Serialize to pickle
        df.to_pickle(cache_file)
        print("Loaded DataFrame from JSON and saved to cache.")
    return df


def expand_column(df, key):
    if key not in df.columns:
        return df
    max_len = df[key].apply(lambda x: len(x)).max()

    expanded = pd.DataFrame(df[key].tolist(), columns=[key + "_" + str(i) for i in range(max_len)])
    return pd.concat([df.drop(columns=[key]), expanded], axis=1)


def expand_dict(df, dict_key, limit=8):
    def flatten_histogram(hist_list):
        result = {}
        for i, item in enumerate(hist_list):
            if i >= limit:
                break
            for key, value in item.items():
                result[f"{dict_key}_{key}_{i}"] = value

        return pd.Series(result)

    if dict_key not in df.columns:
        return df
    hist_expanded = df[dict_key].apply(flatten_histogram)
    # Concatenate with original DataFrame
    return pd.concat([df.drop(columns=[dict_key]), hist_expanded], axis=1)


def def_filter_best_parameter(raw_df, sql):
    duckdb.register("raw_df", raw_df)
    df = duckdb.query(sql).to_df()

    df = expand_column(df, "A_GridSize")
    df = expand_column(df, "B_GridSize")
    df = expand_dict(df, "A_MBR")
    df = expand_dict(df, "B_MBR")
    df = expand_dict(df, "A_Histogram")
    df = expand_dict(df, "B_Histogram")
    sorted_cols = sorted(df.columns)
    return df[sorted_cols]


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


def train_param(df, key, n_dims, regressor, extra_columns=None, return_model=False, isPoisson=False, base_score=1):
    df = df.dropna(subset=[key])
    exclude_cols = [key]
    if extra_columns:
        exclude_cols += extra_columns

    X = df.drop(columns=exclude_cols)

    #    if extra_columns:
    #        for col in extra_columns:
    #            if col not in X.columns:
    #                raise ValueError(f"Extra column '{col}' not found in DataFrame.")

    y = df[key]
    X = X.select_dtypes(include=[np.number]).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse}")
    mabs = median_absolute_error(y_test, y_pred)
    print(f"Median Absolute Error on test set: {mabs}")

    header = X_train.columns.tolist()
    export_regressor(regressor, key + "_" + str(n_dims) + "D", header, isPoisson=isPoisson, base_score=base_score)

    if return_model:
        return regressor, header


def train_sample_rate(df, n_dims):
    # Load data into a DataFrame
    df_filtered = def_filter_best_parameter(df, sql_vary_sample_rate)
    base_score = df_filtered["SampleRate"].mean()
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.4,
                             n_estimators=5, min_child_weight=10, gamma=0.0, base_score=base_score)
    model, features = train_param(df_filtered, "SampleRate", n_dims, regressor, return_model=True,
                                  base_score=base_score)


def train_num_points_per_cell(df, n_dims):
    df_filtered = def_filter_best_parameter(df, sql_vary_n_points_per_cell)

    base_score = df_filtered["NumPointsPerCell"].mean()
    # print(base_score)
    base_score = 1.0
    regressor = XGBRegressor(objective='reg:pseudohubererror', random_state=42, max_depth=10, learning_rate=0.4,
                             n_estimators=20, min_child_weight=10, gamma=1.0, max_leaves=10, reg_lambda=1.0
                             , base_score=base_score)
    model, features = train_param(
        df_filtered, "NumPointsPerCell", n_dims, regressor,
        return_model=True, base_score=base_score  # , isPoisson=True
    )


if __name__ == '__main__':
    # Set the path to the directory containing your JSON files
    directory_path = 'logs/train'  # <-- change this to your JSON directory
    df = load_df(directory_path)
    # train_sample_rate(df, 3)
    train_num_points_per_cell(df, 3)
