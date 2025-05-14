sql_vary_sample_rate = """
WITH renamed_df AS (
    SELECT concat("Input.FileA.Path", "Input.FileB.Path") as Dataset,
           "GPU.multiProcessorCount" as MultiProcessorCount,
           "GPU.l2CacheSize" / 1024 as L2CacheSize,
           "Running.NumPointsPerCell" as NumPointsPerCell, 
           "Running.RadiusStep" as RadiusStep,
           "Running.MaxHitList" as MaxHitList,
           "Running.SampleRate" as SampleRate,
           "Running.AvgTime" as AvgTime
    FROM raw_df
), fixed_params AS (
    SELECT Dataset, 
           NumPointsPerCell, 
           MaxHitList,
           RadiusStep,
    FROM renamed_df
    GROUP BY Dataset, NumPointsPerCell, MaxHitList, RadiusStep
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
       "Running.SampleRate" as SampleRate,
	   concat("Input.FileA.Path", "Input.FileB.Path") as Dataset
FROM raw_df
JOIN (
    SELECT renamed_df.Dataset, 
           renamed_df.NumPointsPerCell, 
           renamed_df.MaxHitList, 
           renamed_df.RadiusStep,
           MIN(renamed_df.AvgTime) AS MinTime
    FROM fixed_params, renamed_df
    WHERE fixed_params.Dataset = renamed_df.Dataset AND
          fixed_params.NumPointsPerCell = renamed_df.NumPointsPerCell AND
          fixed_params.MaxHitList = renamed_df.MaxHitList AND
          fixed_params.RadiusStep = renamed_df.RadiusStep
    GROUP BY renamed_df.Dataset, 
             renamed_df.NumPointsPerCell, 
             renamed_df.MaxHitList, 
             renamed_df.RadiusStep
) t_min ON concat("Input.FileA.Path", "Input.FileB.Path") = t_min.Dataset AND
           "Running.NumPointsPerCell" = t_min.NumPointsPerCell AND
           "Running.MaxHitList" = t_min.MaxHitList AND
           "Running.RadiusStep" = t_min.RadiusStep AND
           "Running.AvgTime" = t_min.MinTime
"""

sql_vary_n_points_per_cell = """
WITH renamed_df AS (
    SELECT concat("Input.FileA.Path", "Input.FileB.Path") as Dataset,
           "GPU.multiProcessorCount" as MultiProcessorCount,
           "GPU.l2CacheSize" / 1024 as L2CacheSize,
           "Running.NumPointsPerCell" as NumPointsPerCell, 
           "Running.RadiusStep" as RadiusStep,
           "Running.MaxHitList" as MaxHitList,
           "Running.SampleRate" as SampleRate,
           "Running.AvgTime" as AvgTime
    FROM raw_df
), fixed_params AS (
    SELECT Dataset, 
           SampleRate,
           MaxHitList,
           RadiusStep,
    FROM renamed_df
    GROUP BY Dataset, SampleRate, MaxHitList, RadiusStep
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
       "Running.NumPointsPerCell" as NumPointsPerCell,
       concat("Input.FileA.Path", "Input.FileB.Path") as Dataset
FROM raw_df
JOIN (
    SELECT renamed_df.Dataset, 
           renamed_df.MaxHitList, 
           renamed_df.SampleRate, 
           renamed_df.RadiusStep,
           MIN(renamed_df.AvgTime) AS MinTime
    FROM fixed_params, renamed_df
    WHERE fixed_params.Dataset = renamed_df.Dataset AND
          fixed_params.MaxHitList = renamed_df.MaxHitList AND
          fixed_params.SampleRate = renamed_df.SampleRate AND
          fixed_params.RadiusStep = renamed_df.RadiusStep
    GROUP BY renamed_df.Dataset, 
             renamed_df.MaxHitList, 
             renamed_df.SampleRate, 
             renamed_df.RadiusStep
) t_min ON concat("Input.FileA.Path", "Input.FileB.Path") = t_min.Dataset AND
           "Running.MaxHitList" = t_min.MaxHitList AND
           "Running.SampleRate" = t_min.SampleRate AND
           "Running.RadiusStep" = t_min.RadiusStep AND
           "Running.AvgTime" = t_min.MinTime


"""

sql_vary_max_hit_init = """
SELECT "Input.FileA.Density" AS A_Density,
       "Input.FileA.Grid.GiniIndex" AS A_GiniIndex,
       "Input.FileA.Grid.GridSize" AS A_GridSize,
       "Input.FileA.Grid.MaxPoints" AS A_MaxPoints,
       "Input.FileA.Grid.NonEmptyCells" AS A_NonEmptyCells,
       "Input.FileA.NumPoints" AS A_NumPoints,
       "Input.FileB.Density" AS B_Density,
       "Input.FileB.Grid.GiniIndex" AS B_GiniIndex,
       "Input.FileB.Grid.GridSize" AS B_GridSize,
       "Input.FileB.Grid.MaxPoints" AS B_MaxPoints,
       "Input.FileB.Grid.NonEmptyCells" AS B_NonEmptyCells,
       "Input.FileB.NumPoints" AS B_NumPoints,
       "Input.Density" AS Density,
       "Running.Train.Grid.GiniIndex" as GiniIndex,
       "Running.Train.Grid.MaxPoints" as MaxPoints,
       "Running.Train.Grid.NonEmptyCells" as NonEmptyCells,
       "Running.Train.HDLowerBound" as HDLowerBound,
       "Running.Train.HDUpperBound" as HDUpperBound,
       "Running.Train.Iter1.MaxHit" as MaxHitInit,
       "Running.SampleRate" as SampleRate,
       "Running.NumPointsPerCell" as NumPointsPerCell
FROM raw_df
"""

sql_vary_max_hit_next = """
SELECT "Input.FileA.Density" AS A_Density,
       "Input.FileA.Grid.GiniIndex" AS A_GiniIndex,
       "Input.FileA.Grid.GridSize" AS A_GridSize,
       "Input.FileA.Grid.MaxPoints" AS A_MaxPoints,
       "Input.FileA.Grid.NonEmptyCells" AS A_NonEmptyCells,
       "Input.FileA.NumPoints" AS A_NumPoints,
       "Input.FileB.Density" AS B_Density,
       "Input.FileB.Grid.GiniIndex" AS B_GiniIndex,
       "Input.FileB.Grid.GridSize" AS B_GridSize,
       "Input.FileB.Grid.MaxPoints" AS B_MaxPoints,
       "Input.FileB.Grid.NonEmptyCells" AS B_NonEmptyCells,
       "Input.FileB.NumPoints" AS B_NumPoints,
       "Input.Density" AS Density,
       "Running.Train.Iter1.Hits" as Hits1,
       "Running.Train.Iter1.CMax2" as CMax2,
       "Running.Train.Iter1.NumInputPoints" as NumInputPoints,
       "Running.Train.Iter1.NumOutputPoints" as NumOutputPoints,
       "Running.Train.Iter1.NumTermPoints" as NumTermPoints,
       "Running.Train.Iter1.ComparedPairs" as ComparedPairs,
       "Running.Train.Iter1.EBTime" as EBTime,
       "Running.Train.Iter1.RTTime" as RTTime,
       "Running.Train.Iter2.MaxHit" as MaxHitNext,
       "Running.SampleRate" as SampleRate,
       "Running.NumPointsPerCell" as NumPointsPerCell
FROM raw_df
"""
