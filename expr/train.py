import json
import glob
import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import duckdb
import m2cgen as m2c
from pandas import json_normalize

sql_renamed_df = """
WITH renamed_df AS (
    SELECT concat("Input.FileA.Path", "Input.FileB.Path") as Dataset,
           "Running.NumPointsPerCell" as NumPointsPerCell, 
           "Running.MaxHit" as MaxHit,
           "Running.MaxHitReduceFactor" as MaxHitReduceFactor,
           "Running.RadiusStep" as RadiusStep,
           "Running.SampleRate" as SampleRate,
           "Running.AvgTime" as AvgTime
    FROM df
)
"""

sql_vary_sample_rate = """
{renamed_df}, fixed_params AS (
    SELECT Dataset, 
           NumPointsPerCell, 
           MaxHit,
           MaxHitReduceFactor,
           RadiusStep,
    FROM renamed_df
    GROUP BY Dataset, NumPointsPerCell, MaxHit, MaxHitReduceFactor, RadiusStep
    HAVING COUNT(DISTINCT SampleRate) > 1
)

SELECT "Input.FileA.Grid.GiniIndex" AS A_GiniIndex,
       "Input.FileA.Grid.GridSize" AS A_GridSize,
       "Input.FileA.Grid.Histogram" AS A_Histogram,
       "Input.FileA.Grid.MaxPoints" AS A_MaxPoints,
       "Input.FileA.Grid.NonEmptyCells" AS A_NonEmptyCells,
       "Input.FileA.Grid.TotalCells" AS A_TotalCells,
       "Input.FileA.MBR" AS A_MBR,
       "Input.FileA.NumPoints" AS A_NumPoints,
       "Input.FileB.Grid.GiniIndex" AS B_GiniIndex,
       "Input.FileB.Grid.GridSize" AS B_GridSize,
       "Input.FileB.Grid.Histogram" AS B_Histogram,
       "Input.FileB.Grid.MaxPoints" AS B_MaxPoints,
       "Input.FileB.Grid.NonEmptyCells" AS B_NonEmptyCells,
       "Input.FileB.Grid.TotalCells" AS B_TotalCells,
       "Input.FileB.MBR" AS B_MBR,
       "Input.FileB.NumPoints" AS B_NumPoints,
       "Running.SampleRate" as SampleRate
FROM df
JOIN (
    SELECT renamed_df.Dataset, 
           renamed_df.NumPointsPerCell, 
           renamed_df.MaxHit, 
           renamed_df.MaxHitReduceFactor, 
           renamed_df.RadiusStep,
           MIN(renamed_df.AvgTime) AS MinTime
    FROM fixed_params, renamed_df
    WHERE fixed_params.Dataset = renamed_df.Dataset AND
          fixed_params.NumPointsPerCell = renamed_df.NumPointsPerCell AND
          fixed_params.MaxHit = renamed_df.MaxHit AND
          fixed_params.MaxHitReduceFactor = renamed_df.MaxHitReduceFactor AND
          fixed_params.RadiusStep = renamed_df.RadiusStep
    GROUP BY renamed_df.Dataset, 
             renamed_df.NumPointsPerCell, 
             renamed_df.MaxHit, 
             renamed_df.MaxHitReduceFactor, 
             renamed_df.RadiusStep
) t_min ON concat("Input.FileA.Path", "Input.FileB.Path") = t_min.Dataset AND
           "Running.NumPointsPerCell" = t_min.NumPointsPerCell AND
           "Running.MaxHit" = t_min.MaxHit AND
           "Running.MaxHitReduceFactor" = t_min.MaxHitReduceFactor AND
           "Running.RadiusStep" = t_min.RadiusStep AND
           "Running.AvgTime" = t_min.MinTime
""".format(renamed_df=sql_renamed_df)


def def_filter_best_parameter(files, sql):
    # Path to cache file
    cache_file = "dataframe_cache.pkl"

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
    df = duckdb.query(sql).to_df()

    def expand_column(df, key):
        max_len = df[key].apply(lambda x: len(x)).max()

        expanded = pd.DataFrame(df[key].tolist(), columns=[key + "_" + str(i) for i in range(max_len)])
        return pd.concat([df.drop(columns=[key]), expanded], axis=1)

    def expand_dict(df, dict_key, limit=10):
        def flatten_histogram(hist_list):
            result = {}
            for i, item in enumerate(hist_list):
                if i >= 10:
                    break
                for key, value in item.items():
                    result[f"{dict_key}_{key}_{i}"] = value

            return pd.Series(result)

        hist_expanded = df[dict_key].apply(flatten_histogram)
        # Concatenate with original DataFrame
        return pd.concat([df.drop(columns=[dict_key]), hist_expanded], axis=1)

    df = expand_column(df, "A_GridSize")
    df = expand_column(df, "B_GridSize")
    df = expand_dict(df, "A_MBR")
    df = expand_dict(df, "B_MBR")
    df = expand_dict(df, "A_Histogram")
    df = expand_dict(df, "B_Histogram")
    # Sort column names by their first 2 characters
    sorted_cols = sorted(df.columns)
    return df[sorted_cols]


def export_regressor(regressor, name, comments):
    c_code = m2c.export_to_c(regressor, function_name="Predicate" + name)
    vars = []
    for i, name in enumerate(comments):
        vars.append("%i %s" % (i, name))

    header = """
#ifndef DECISION_TREE_{name}
#define DECISION_TREE_{name}
/*
{comments}
*/
{code}
#endif // DECISION_TREE_{name}
""".format(name=name.upper(), code=c_code, comments='\n'.join(vars))

    with open("tree_sample_rate.h", "w") as f:
        f.write(header)


def main():
    # Set the path to the directory containing your JSON files
    directory_path = '/Users/liang/BraTS2020_TrainingData'  # <-- change this to your JSON directory

    # Load data into a DataFrame
    all_files = glob.glob(os.path.join(directory_path, '*.json'))
    df = def_filter_best_parameter(all_files, sql_vary_sample_rate)
    print(df)
    key = "SampleRate"

    # Drop rows where the target value is missing
    df = df.dropna(subset=[key])

    # Separate features (X) and target (y)
    X = df.drop(key, axis=1)
    y = df[key]

    # Fill missing values (if any) with 0. You may wish to use a different strategy.
    X = X.fillna(0)

    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the decision tree regressor
    regressor = DecisionTreeRegressor(random_state=42)
    regressor.fit(X_train, y_train)

    # Predict on the test set and evaluate performance
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on test set: {mse}")

    header = X_train.columns.tolist()
    export_regressor(regressor, "SampleRate", header)
    # Optionally, export the decision tree structure (requires graphviz)
    from sklearn.tree import export_graphviz
    export_graphviz(regressor, out_file='tree.dot', feature_names=X.columns)


if __name__ == '__main__':
    main()
