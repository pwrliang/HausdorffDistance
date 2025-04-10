
#ifndef DECISION_TREE_NUMPOINTSPERCELL_3D
#define DECISION_TREE_NUMPOINTSPERCELL_3D
/*
0 A_GiniIndex
1 A_GridSize_0
2 A_GridSize_1
3 A_GridSize_2
4 A_Histogram_count_0
5 A_Histogram_count_1
6 A_Histogram_count_2
7 A_Histogram_count_3
8 A_Histogram_count_4
9 A_Histogram_count_5
10 A_Histogram_count_6
11 A_Histogram_count_7
12 A_Histogram_percentile_0
13 A_Histogram_percentile_1
14 A_Histogram_percentile_2
15 A_Histogram_percentile_3
16 A_Histogram_percentile_4
17 A_Histogram_percentile_5
18 A_Histogram_percentile_6
19 A_Histogram_percentile_7
20 A_Histogram_value_0
21 A_Histogram_value_1
22 A_Histogram_value_2
23 A_Histogram_value_3
24 A_Histogram_value_4
25 A_Histogram_value_5
26 A_Histogram_value_6
27 A_Histogram_value_7
28 A_MBR_Lower_0
29 A_MBR_Lower_1
30 A_MBR_Lower_2
31 A_MBR_Upper_0
32 A_MBR_Upper_1
33 A_MBR_Upper_2
34 A_MaxPoints
35 A_NonEmptyCells
36 A_NumPoints
37 A_TotalCells
38 B_GiniIndex
39 B_GridSize_0
40 B_GridSize_1
41 B_GridSize_2
42 B_Histogram_count_0
43 B_Histogram_count_1
44 B_Histogram_count_2
45 B_Histogram_count_3
46 B_Histogram_count_4
47 B_Histogram_count_5
48 B_Histogram_count_6
49 B_Histogram_count_7
50 B_Histogram_percentile_0
51 B_Histogram_percentile_1
52 B_Histogram_percentile_2
53 B_Histogram_percentile_3
54 B_Histogram_percentile_4
55 B_Histogram_percentile_5
56 B_Histogram_percentile_6
57 B_Histogram_percentile_7
58 B_Histogram_value_0
59 B_Histogram_value_1
60 B_Histogram_value_2
61 B_Histogram_value_3
62 B_Histogram_value_4
63 B_Histogram_value_5
64 B_Histogram_value_6
65 B_Histogram_value_7
66 B_MBR_Lower_0
67 B_MBR_Lower_1
68 B_MBR_Lower_2
69 B_MBR_Upper_0
70 B_MBR_Upper_1
71 B_MBR_Upper_2
72 B_MaxPoints
73 B_NonEmptyCells
74 B_NumPoints
75 B_TotalCells
76 SampleRate

struct Input {
    double A_GiniIndex;
    double A_GridSize[3];
    double A_Histogram_count[8];
    double A_Histogram_percentile[8];
    double A_Histogram_value[8];
    double A_MBR_Lower[3];
    double A_MBR_Upper[3];
    double A_MaxPoints;
    double A_NonEmptyCells;
    double A_NumPoints;
    double A_TotalCells;
    double B_GiniIndex;
    double B_GridSize[3];
    double B_Histogram_count[8];
    double B_Histogram_percentile[8];
    double B_Histogram_value[8];
    double B_MBR_Lower[3];
    double B_MBR_Upper[3];
    double B_MaxPoints;
    double B_NonEmptyCells;
    double B_NumPoints;
    double B_TotalCells;
    double SampleRate;
};

*/
inline double PredictNumPointsPerCell_3D(double * input) {
    double var0;
    if (input[7] <= 27826.0) {
        if (input[73] <= 65466.0) {
            if (input[63] <= 34.5) {
                if (input[31] <= 181.0) {
                    var0 = 12.461538461538462;
                } else {
                    var0 = 18.90909090909091;
                }
            } else {
                var0 = 7.523809523809524;
            }
        } else {
            var0 = 6.689655172413793;
        }
    } else {
        if (input[41] <= 43.5) {
            if (input[43] <= 328.0) {
                var0 = 15.15;
            } else {
                var0 = 19.27659574468085;
            }
        } else {
            if (input[32] <= 219.5) {
                if (input[68] <= 10.5) {
                    if (input[39] <= 47.5) {
                        var0 = 13.36;
                    } else {
                        var0 = 15.189565217391305;
                    }
                } else {
                    var0 = 9.466666666666667;
                }
            } else {
                var0 = 9.80952380952381;
            }
        }
    }
    return var0;
}

#endif // DECISION_TREE_NUMPOINTSPERCELL_3D
