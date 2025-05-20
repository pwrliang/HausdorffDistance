sql_vary_sample_rate = """
WITH renamed_df AS (
    SELECT concat("Input.FileA.Path", "Input.FileB.Path") as Dataset,
           "Running.NumPointsPerCell" as NumPointsPerCell, 
           "Running.SampleRate" as SampleRate,
           "Running.AvgTime" as AvgTime
    FROM raw_df
), fixed_params AS (
    SELECT Dataset, 
           NumPointsPerCell
    FROM renamed_df
    GROUP BY Dataset, NumPointsPerCell
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
FROM raw_df
JOIN (
    SELECT renamed_df.Dataset, 
           renamed_df.NumPointsPerCell, 
           MIN(renamed_df.AvgTime) AS MinTime
    FROM fixed_params, renamed_df
    WHERE fixed_params.Dataset = renamed_df.Dataset AND
          fixed_params.NumPointsPerCell = renamed_df.NumPointsPerCell
    GROUP BY renamed_df.Dataset, 
             renamed_df.NumPointsPerCell
) t_min ON concat("Input.FileA.Path", "Input.FileB.Path") = t_min.Dataset AND
           "Running.NumPointsPerCell" = t_min.NumPointsPerCell AND
           "Running.AvgTime" = t_min.MinTime
"""

sql_vary_n_points_per_cell = """
WITH renamed_df AS (
    SELECT concat("Input.FileA.Path", "Input.FileB.Path") as Dataset,
           "Running.NumPointsPerCell" as NumPointsPerCell, 
           "Running.SampleRate" as SampleRate,
           "Running.AvgTime" as AvgTime
    FROM raw_df
), fixed_params AS (
    SELECT Dataset, 
           SampleRate
    FROM renamed_df
    GROUP BY Dataset, SampleRate
    HAVING COUNT(DISTINCT NumPointsPerCell) > 1
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
       "Running.SampleRate" as SampleRate,
       "Running.NumPointsPerCell" as NumPointsPerCell
FROM raw_df
JOIN (
    SELECT renamed_df.Dataset, 
           renamed_df.SampleRate, 
           MIN(renamed_df.AvgTime) AS MinTime
    FROM fixed_params, renamed_df
    WHERE fixed_params.Dataset = renamed_df.Dataset AND
          fixed_params.SampleRate = renamed_df.SampleRate
    GROUP BY renamed_df.Dataset, 
             renamed_df.SampleRate
) t_min ON concat("Input.FileA.Path", "Input.FileB.Path") = t_min.Dataset AND
           "Running.SampleRate" = t_min.SampleRate AND
           "Running.AvgTime" = t_min.MinTime
"""
