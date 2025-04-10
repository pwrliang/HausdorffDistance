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
import matplotlib.pyplot as plt
import re
import duckdb
import m2cgen as m2c
from pandas import json_normalize
from sql import *


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
    df = duckdb.query(sql).to_df()

    df = expand_column(df, "A_GridSize")
    df = expand_column(df, "B_GridSize")
    df = expand_dict(df, "A_MBR")
    df = expand_dict(df, "B_MBR")
    df = expand_dict(df, "A_Histogram")
    df = expand_dict(df, "B_Histogram")
    sorted_cols = sorted(df.columns)
    return df[sorted_cols]


def export_regressor(regressor, key, vars):
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

    with open("tree_{name}.h".format(name=key.lower()), "w") as f:
        f.write(header)


def train_param(df, key, n_dims, regressor):
    # Drop rows where the target value is missing
    df = df.dropna(subset=[key])

    # Separate features (X) and target (y)
    X = df.drop(key, axis=1)
    y = df[key]

    # Fill missing values (if any) with 0. You may wish to use a different strategy.
    X = X.fillna(0)

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor.fit(X_train, y_train)

    # Predict on the test set and evaluate performance
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse}")
    mabs = median_absolute_error(y_test, y_pred)
    print(f"Median Absolute Error on test set: {mabs}")

    header = X_train.columns.tolist()
    export_regressor(regressor, key + "_" + str(n_dims) + "D", header)
    # # Optionally, export the decision tree structure (requires graphviz)
    # from sklearn.tree import export_graphviz
    # export_graphviz(regressor, out_file='tree.dot', feature_names=X.columns)


def train_sample_rate(df, n_dims):
    # Load data into a DataFrame
    df = def_filter_best_parameter(df, sql_vary_sample_rate)
    train_param(df, "SampleRate", n_dims)


def train_num_points_per_cell(df, n_dims):
    df = def_filter_best_parameter(df, sql_vary_n_points_per_cell)

    # import scipy.stats as stats
    #
    # groups = [group["AvgTime"].values for name, group in df.groupby("NumPointsPerCell")]
    # f_stat, p_value = stats.f_oneway(*groups)
    #
    # print(f"F-statistic: {f_stat}, p-value: {p_value}")

    # from sklearn.feature_selection import mutual_info_regression
    #
    # X = df[['NumPointsPerCell']]
    # y = df['AvgTime']
    # mi = mutual_info_regression(X, y)
    # print(f"Mutual information: {mi[0]}")
    #
    # plt.scatter(df['NumPointsPerCell'], df['AvgTime'])
    # plt.xlabel("NumPointsPerCell")
    # plt.ylabel("AvgTime")
    # plt.title("Scatter plot of NumPointsPerCell vs AvgTime")
    # plt.show()
    regressor = DecisionTreeRegressor(random_state=42,
                                      max_depth=5,
                                      min_samples_split=10,
                                      min_samples_leaf=5,
                                      max_leaf_nodes=10)
    train_param(df, "NumPointsPerCell", n_dims, regressor)


def train_max_hit(df, n_dims):
    df = df[df['Running.MaxHitList'].str.contains(',', na=False)]
    # Extract the column names from the DataFrame
    columns = df.columns

    # Extract repeat numbers using regex
    repeat_nums = [
        int(re.search(r'Running\.Repeat(\d+)\.TotalTime', col).group(1))
        for col in columns if re.search(r'Running\.Repeat(\d+)\.TotalTime', col)
    ]

    # Get the maximum repeat number
    max_repeat = max(repeat_nums) if repeat_nums else None
    print("Max repeat:", max_repeat)
    target_cols = [f'Running.Repeat{i}.TotalTime' for i in range(max_repeat + 1)]

    # Make sure all columns exist in the DataFrame
    existing_cols = [col for col in target_cols if col in df.columns]

    # Find the column with the minimum value per row
    df.loc[:, 'TrainRepeat'] = df[existing_cols].idxmax(axis=1)
    df['TrainRepeat'] = df['TrainRepeat'].str.replace('.TotalTime', '', regex=False)

    def extract_matching_values(row):
        substring = row['TrainRepeat']
        # Get all columns that contain the substring
        matching_cols = [col for col in df.columns if substring in col]
        # Return just the values for this row
        s = row[matching_cols]
        s.index = s.index.str.replace(substring, 'Running.Train', regex=False)

        def is_iter_gt_i(index_str):
            match = re.search(r'Iter(\d+)', index_str)
            if match:
                return int(match.group(1)) > 2
            return False  # No 'Iter' in the string, so keep it

        return s[~s.index.map(is_iter_gt_i)]

    # Apply row-wise and build a new DataFrame
    extracted_df = df.apply(extract_matching_values, axis=1, result_type='expand')
    df = df.loc[:, ~df.columns.str.contains("Repeat")]
    df = pd.concat([df, extracted_df], axis=1)

    df_train_init = def_filter_best_parameter(df, sql_vary_max_hit_init)


    # Create and train the decision tree regressor
    regressor = DecisionTreeRegressor(random_state=42,
                                      max_depth=5,
                                      min_samples_split=10,
                                      min_samples_leaf=5,
                                      max_leaf_nodes=10)
    train_param(df_train_init, "MaxHitInit", n_dims, regressor)

    df_train_next = def_filter_best_parameter(df, sql_vary_max_hit_next)
    train_param(df_train_next, "MaxHitNext", n_dims, regressor)

    # feature_key = 'MaxPoints'
    # predict_value = 'MaxHit'
    # df[feature_key] = df[feature_key].round(2)
    #
    # percentages = (
    #     df.groupby(feature_key)[predict_value]
    #     .value_counts(normalize=True)
    #     .rename('Percentage')
    #     .reset_index()
    # )
    # num_b = percentages[predict_value].nunique()
    # reds = plt.cm.Reds(np.linspace(0.1, 1, num_b))  # use lighter to darker reds
    #
    # # Step 2: Pivot for plotting
    # pivot_df = percentages.pivot(index=feature_key, columns=predict_value, values='Percentage')
    #
    # # Step 3: Plot
    # pivot_df.plot(kind='bar', color=reds)
    # plt.xlabel(feature_key)
    # plt.ylabel("Percentage")
    # plt.title("Scatter plot of NumPointsPerCell vs AvgTime")
    # plt.show()


if __name__ == '__main__':
    # Set the path to the directory containing your JSON files
    directory_path = '/Users/liang/BraTS2020_TrainingData'  # <-- change this to your JSON directory
    cache_file = "BraTS2020_TrainingData.pkl"
    all_files = glob.glob(os.path.join(directory_path, '*.json'))
    df = load_df(all_files, cache_file)
    # train_sample_rate(df, 3)
    train_num_points_per_cell(df, 3)
    # train_max_hit(df, 3)
