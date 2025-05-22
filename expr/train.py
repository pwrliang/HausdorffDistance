import json
import glob
import os
from collections import OrderedDict
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error
from xgboost import XGBRegressor
from math import log
import matplotlib.pyplot as plt
import re
import duckdb
import m2cgen as m2c
from pandas import json_normalize
from sql2 import *

def load_df(files, cache_file):
    # If cache exists, load from pickle
    if os.path.exists(cache_file):
        df = pd.read_pickle(cache_file)
        print("Loaded DataFrame from cache.")
    else:
        # Otherwise, load from JSON files and serialize
        json_records = []
        for file in files:
            with open(file) as f:
                json_records.append(json.load(f))

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
    df_filtered = def_filter_best_parameter(df, sql_vary_sample_rate)

    base_score = df_filtered["SampleRate"].mean()
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.4, n_estimators=5, min_child_weight=10, gamma=0.0, base_score=base_score)
    #regressor = DecisionTreeRegressor(random_state=42,
    #                                  max_depth=10,
    #                                  min_samples_split=10,
    #                                  min_samples_leaf=5,
    #                                  max_leaf_nodes=10)
    model, features = train_param(df_filtered, "SampleRate", n_dims, regressor, return_model=True, base_score=base_score)

    #df_filtered["Predicted.SampleRate"] = model.predict(df_filtered[features].fillna(0))

    #df = df.copy()
    #df["Dataset"] = df["Input.FileA.Path"] + df["Input.FileB.Path"]
    #df = pd.merge(df, df_filtered[["Dataset", "Predicted.SampleRate"]], on="Dataset", how="left")
    return df


def train_num_points_per_cell(df, n_dims):
    df_filtered = def_filter_best_parameter(df, sql_vary_n_points_per_cell)

    # merge Predicted.SampleRate from full df using Dataset
    #df["Dataset"] = df["Input.FileA.Path"] + df["Input.FileB.Path"]
    #df_filtered = df_filtered.copy()
    #df_filtered = pd.merge(
    #    df_filtered,
    #    df[["Dataset", "Predicted.SampleRate"]],
    #    on="Dataset",
    #    how="left"
    #)

    base_score = df_filtered["NumPointsPerCell"].mean()
    #base_score = 1.0
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.4, n_estimators=20, min_child_weight=10, gamma=1.0, max_leaves=10, reg_lambda=1.0
                            , base_score=base_score)
    #regressor = DecisionTreeRegressor(criterion="poisson", random_state=42,
    #                                  max_depth=10,
    #                                  min_samples_split=10,
    #                                  min_samples_leaf=5,
    #                                  max_leaf_nodes=10)
    model, features = train_param(
        df_filtered, "NumPointsPerCell", n_dims, regressor,
        return_model=True, base_score=base_score#, isPoisson=True
    )
    

    #df_filtered["Predicted.NumPointsPerCell"] = model.predict(df_filtered[features].fillna(0))

    #df = df.copy()
    #df = pd.merge(
    #    df,
    #    df_filtered[["Dataset", "Predicted.NumPointsPerCell"]],
    #    on="Dataset",
    #    how="left"
    #)
    return df


def train_max_hit(df, n_dims):
    df = df[df['Running.MaxHitList'].str.contains(',', na=False)]
    columns = df.columns
    repeat_nums = [
        int(re.search(r'Running\.Repeat(\d+)\.TotalTime', col).group(1))
        for col in columns if re.search(r'Running\.Repeat(\d+)\.TotalTime', col)
    ]
    max_repeat = max(repeat_nums) if repeat_nums else None
    print("Max repeat:", max_repeat)
    target_cols = [f'Running.Repeat{i}.TotalTime' for i in range(max_repeat + 1)]
    existing_cols = [col for col in target_cols if col in df.columns]

    df.loc[:, 'TrainRepeat'] = df[existing_cols].idxmax(axis=1)
    df['TrainRepeat'] = df['TrainRepeat'].str.replace('.TotalTime', '', regex=False)

    def extract_matching_values(row):
        substring = row['TrainRepeat']
        matching_cols = [col for col in df.columns if substring in col]
        s = row[matching_cols]
        s.index = s.index.str.replace(substring, 'Running.Train', regex=False)
        def is_iter_gt_i(index_str):
            match = re.search(r'Iter(\d+)', index_str)
            return int(match.group(1)) > 2 if match else False
        return s[~s.index.map(is_iter_gt_i)]

    extracted_df = df.apply(extract_matching_values, axis=1, result_type='expand')
    df = df.loc[:, ~df.columns.str.contains("Repeat")]
    df = pd.concat([df, extracted_df], axis=1)

    df_train_init = def_filter_best_parameter(df, sql_vary_max_hit_init)
    #df_train_init["Predicted.SampleRate"] = df["Predicted.SampleRate"]
    #df_train_init["Predicted.NumPointsPerCell"] = df["Predicted.NumPointsPerCell"]

    base_score = df_train_init["MaxHitInit"].mean()
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.2, n_estimators=20, min_child_weight=10, gamma=1.0, base_score=base_score, max_leaves=10)
    #regressor = DecisionTreeRegressor(criterion="poisson", random_state=42,
    #                                  max_depth=10,
    #                                  min_samples_split=10,
    #                                  min_samples_leaf=5,
    #                                  max_leaf_nodes=10)
    train_param(df_train_init, "MaxHitInit", n_dims, regressor
                , base_score=base_score)
    
    df_train_next = def_filter_best_parameter(df, sql_vary_max_hit_next)
    #df_train_next["Predicted.SampleRate"] = df["Predicted.SampleRate"]
    #df_train_next["Predicted.NumPointsPerCell"] = df["Predicted.NumPointsPerCell"]
    base_score = df_train_next["MaxHitNext"].mean()
    regressor = XGBRegressor(objective='reg:squarederror', random_state=42, max_depth=10, learning_rate=0.2, n_estimators=20, min_child_weight=10, gamma=1.0, base_score=base_score, max_leaves=10)
    #regressor = DecisionTreeRegressor(criterion="poisson", random_state=42,
    #                                  max_depth=10,
    #                                  min_samples_split=10,
    #                                  min_samples_leaf=5,
    #                                  max_leaf_nodes=10)
    train_param(df_train_next, "MaxHitNext", n_dims, regressor
                , base_score=base_score)


if __name__ == '__main__':
    directory_path = '../../data/BraTS2020_TrainingData'
    cache_file = "BraTS2020_TrainingData.pkl"
    all_files = glob.glob(os.path.join(directory_path, '*.json'))
    df_raw = load_df(all_files, cache_file)

    df_sample = train_sample_rate(df_raw.copy(), 3)
    df_points = train_num_points_per_cell(df_raw.copy(), 3)
    train_max_hit(df_raw.copy(), 3)

