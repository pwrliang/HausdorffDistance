
#include <math.h>
#ifndef DECISION_TREE_EBONLYTHRESHOLD_3D
#define DECISION_TREE_EBONLYTHRESHOLD_3D
/*
0 Density_0
1 NumPoints_0
2 MBR_0Dim_0_Lower
3 MBR_0Dim_0_Upper
4 MBR_0Dim_1_Lower
5 MBR_0Dim_1_Upper
6 MBR_0Dim_2_Lower
7 MBR_0Dim_2_Upper
8 GINI_0
9 CellP99_0
10 CellP95_0
11 CellP50_0
12 NonEmptyCells_0
13 TotalCells_0
14 Density_1
15 NumPoints_1
16 MBR_1Dim_0_Lower
17 MBR_1Dim_0_Upper
18 MBR_1Dim_1_Lower
19 MBR_1Dim_1_Upper
20 MBR_1Dim_2_Lower
21 MBR_1Dim_2_Upper
22 GINI_1
23 CellP99_1
24 CellP95_1
25 CellP50_1
26 NonEmptyCells_1
27 TotalCells_1
28 HDLB
29 HDUP

struct Input {
    double Density[2];
    double NumPoints[2];
    double MBR_0Dim_0_Lower;
    double MBR_0Dim_0_Upper;
    double MBR_0Dim_1_Lower;
    double MBR_0Dim_1_Upper;
    double MBR_0Dim_2_Lower;
    double MBR_0Dim_2_Upper;
    double GINI[2];
    double CellP99[2];
    double CellP95[2];
    double CellP50[2];
    double NonEmptyCells[2];
    double TotalCells[2];
    double MBR_1Dim_0_Lower;
    double MBR_1Dim_0_Upper;
    double MBR_1Dim_1_Lower;
    double MBR_1Dim_1_Upper;
    double MBR_1Dim_2_Lower;
    double MBR_1Dim_2_Upper;
    double HDLB;
    double HDUP;
};

*/
inline double PredictEBOnlyThreshold_3D(double * input) {
    return 500.1666666666667 + (0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0);
}

#endif // DECISION_TREE_EBONLYTHRESHOLD_3D
