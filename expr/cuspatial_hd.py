# Modified from https://gist.github.com/trxcllnt/38e44bc86cabff23352f3541add6228f
import cudf
import cupy
import cuspatial
import geopandas as gpd
import pandas as pd
import os.path
import pyarrow
import time
import rmm
import shapely
import sys

from shapely.geometry import MultiPoint


def read_wkt(path):
    points = []

    def extract_points(polygon):
        # Extract points from the exterior
        exterior_points = list(polygon.exterior.coords)

        # Extract points from the interior (holes)
        interior_points = [list(interior.coords) for interior in polygon.interiors]

        # Combine all points into a single list
        return exterior_points + [pt for hole in interior_points for pt in hole]

    with open(path, 'r') as fi:
        for line in fi:
            line = line.strip()
            if len(line) > 0:
                geom = shapely.from_wkt(line)
                if geom.geom_type == "Polygon":
                    points += extract_points(geom)
                elif geom.geom_type == "MultiPolygon":
                    for polygon in geom.geoms:
                        points += extract_points(polygon)
    points = MultiPoint(points)
    return points


def create_df(points1, points2):
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1]},  # Example column with IDs
        geometry=[points1, points2],  # Geometry column with individual points
    )
    return cuspatial.from_geopandas(gdf)


rmm.mr.set_current_device_resource(
    rmm.mr.PoolMemoryResource(rmm.mr.get_current_device_resource())
)

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")
    geom_path1 = sys.argv[1]
    geom_path2 = sys.argv[2]
    points1 = read_wkt(geom_path1)
    points2 = read_wkt(geom_path2)
    df = create_df(points1, points2)
    begin = time.time()
    for _ in range(5):
        res = cuspatial.directed_hausdorff_distance(df.geometry)
    time_ms = (time.time() - begin) * 1000
    print("Running time", time_ms / 5, " ms")
