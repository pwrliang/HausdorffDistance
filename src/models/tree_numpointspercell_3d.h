
#include <math.h>
#ifndef DECISION_TREE_NUMPOINTSPERCELL_3D
#define DECISION_TREE_NUMPOINTSPERCELL_3D
/*
0 File_0_Density
1 File_0_NumPoints
2 File_0_MBR_Dim_0_Lower
3 File_0_MBR_Dim_1_Lower
4 File_0_MBR_Dim_2_Lower
5 File_0_MBR_Dim_0_Upper
6 File_0_MBR_Dim_1_Upper
7 File_0_MBR_Dim_2_Upper
8 File_0_GINI
9 File_0_Cell_P0.99_Value
10 File_0_Cell_P0.95_Value
11 File_0_Cell_P0.5_Value
12 File_0_Cell_P0.1_Value
13 File_0_Cell_P0.99_Count
14 File_0_Cell_P0.95_Count
15 File_0_Cell_P0.5_Count
16 File_0_Cell_P0.1_Count
17 File_0_Dim0_GridSize
18 File_0_Dim1_GridSize
19 File_0_Dim2_GridSize
20 File_0_NonEmptyCells
21 File_0_TotalCells
22 File_1_Density
23 File_1_NumPoints
24 File_1_MBR_Dim_0_Lower
25 File_1_MBR_Dim_1_Lower
26 File_1_MBR_Dim_2_Lower
27 File_1_MBR_Dim_0_Upper
28 File_1_MBR_Dim_1_Upper
29 File_1_MBR_Dim_2_Upper
30 File_1_GINI
31 File_1_Cell_P0.99_Value
32 File_1_Cell_P0.95_Value
33 File_1_Cell_P0.5_Value
34 File_1_Cell_P0.1_Value
35 File_1_Cell_P0.99_Count
36 File_1_Cell_P0.95_Count
37 File_1_Cell_P0.5_Count
38 File_1_Cell_P0.1_Count
39 File_1_Dim0_GridSize
40 File_1_Dim1_GridSize
41 File_1_Dim2_GridSize
42 File_1_NonEmptyCells
43 File_1_TotalCells
44 Cell_P0.99_Value
45 Cell_P0.95_Value
46 Cell_P0.5_Value
47 Cell_P0.1_Value
48 Cell_P0.99_Count
49 Cell_P0.95_Count
50 Cell_P0.5_Count
51 Cell_P0.1_Count
52 Dim0_GridSize
53 Dim1_GridSize
54 Dim2_GridSize
55 HDLB
56 HDUP

struct Input {
    double File_0_Density;
    double File_0_NumPoints;
    double File_0_MBR_Dim_0_Lower;
    double File_0_MBR_Dim_1_Lower;
    double File_0_MBR_Dim_2_Lower;
    double File_0_MBR_Dim_0_Upper;
    double File_0_MBR_Dim_1_Upper;
    double File_0_MBR_Dim_2_Upper;
    double File_0_GINI;
    double File_0_Cell_P0.99_Value;
    double File_0_Cell_P0.95_Value;
    double File_0_Cell_P0.5_Value;
    double File_0_Cell_P0.1_Value;
    double File_0_Cell_P0.99_Count;
    double File_0_Cell_P0.95_Count;
    double File_0_Cell_P0.5_Count;
    double File_0_Cell_P0.1_Count;
    double File_0_Dim0_GridSize;
    double File_0_Dim1_GridSize;
    double File_0_Dim2_GridSize;
    double File_0_NonEmptyCells;
    double File_0_TotalCells;
    double File_1_Density;
    double File_1_NumPoints;
    double File_1_MBR_Dim_0_Lower;
    double File_1_MBR_Dim_1_Lower;
    double File_1_MBR_Dim_2_Lower;
    double File_1_MBR_Dim_0_Upper;
    double File_1_MBR_Dim_1_Upper;
    double File_1_MBR_Dim_2_Upper;
    double File_1_GINI;
    double File_1_Cell_P0.99_Value;
    double File_1_Cell_P0.95_Value;
    double File_1_Cell_P0.5_Value;
    double File_1_Cell_P0.1_Value;
    double File_1_Cell_P0.99_Count;
    double File_1_Cell_P0.95_Count;
    double File_1_Cell_P0.5_Count;
    double File_1_Cell_P0.1_Count;
    double File_1_Dim0_GridSize;
    double File_1_Dim1_GridSize;
    double File_1_Dim2_GridSize;
    double File_1_NonEmptyCells;
    double File_1_TotalCells;
    double Cell_P0.99_Value;
    double Cell_P0.95_Value;
    double Cell_P0.5_Value;
    double Cell_P0.1_Value;
    double Cell_P0.99_Count;
    double Cell_P0.95_Count;
    double Cell_P0.5_Count;
    double Cell_P0.1_Count;
    double Dim0_GridSize;
    double Dim1_GridSize;
    double Dim2_GridSize;
    double HDLB;
    double HDUP;
};

*/
inline double PredictNumPointsPerCell_3D(double * input) {
    double var0;
    if (input[55] < 2.4341278) {
        var0 = -1.7636365;
    } else {
        var0 = 0.49333334;
    }
    double var1;
    if (input[55] < 2.4341278) {
        var1 = -1.4429752;
    } else {
        var1 = 0.40124437;
    }
    double var2;
    if (input[55] < 2.4341278) {
        var2 = -1.1806161;
    } else {
        var2 = 0.3263453;
    }
    double var3;
    if (input[22] < 616.4713) {
        var3 = -0.8730707;
    } else {
        var3 = 0.46490484;
    }
    double var4;
    if (input[33] < 20.0) {
        var4 = 0.3769337;
    } else {
        var4 = -0.79208285;
    }
    double var5;
    if (input[55] < 3.3200378) {
        var5 = -0.7711192;
    } else {
        var5 = 0.43261296;
    }
    double var6;
    if (input[33] < 20.0) {
        var6 = 0.35711157;
    } else {
        var6 = -0.6331244;
    }
    double var7;
    if (input[55] < 3.3200378) {
        var7 = -0.618515;
    } else {
        var7 = 0.3934585;
    }
    double var8;
    if (input[22] < 616.4713) {
        var8 = -0.50066984;
    } else {
        var8 = 0.38531473;
    }
    double var9;
    if (input[33] < 20.0) {
        var9 = 0.3245441;
    } else {
        var9 = -0.47312504;
    }
    double var10;
    if (input[55] < 3.3200378) {
        var10 = -0.4649751;
    } else {
        var10 = 0.34382445;
    }
    double var11;
    if (input[33] < 20.0) {
        var11 = 0.28825444;
    } else {
        var11 = -0.3870388;
    }
    double var12;
    if (input[55] < 3.3200378) {
        var12 = -0.3804041;
    } else {
        var12 = 0.29985696;
    }
    double var13;
    if (input[22] < 616.4713) {
        var13 = -0.30251762;
    } else {
        var13 = 0.28178737;
    }
    double var14;
    if (input[33] < 20.0) {
        var14 = 0.24586324;
    } else {
        var14 = -0.2987237;
    }
    double var15;
    if (input[55] < 3.3200378) {
        var15 = -0.2937823;
    } else {
        var15 = 0.25068069;
    }
    double var16;
    if (input[0] < 616.4713) {
        var16 = 0.23081653;
    } else {
        var16 = -0.26596093;
    }
    double var17;
    if (input[51] < 4838.0) {
        var17 = -0.25734487;
    } else {
        var17 = 0.22868867;
    }
    double var18;
    if (input[2] < -176.7131) {
        var18 = -0.21855891;
    } else {
        var18 = 0.27042076;
    }
    double var19;
    if (input[0] < 616.4713) {
        var19 = 0.22358549;
    } else {
        var19 = -0.24148016;
    }
    return 15.5 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_NUMPOINTSPERCELL_3D
