
#ifndef DECISION_TREE_SAMPLERATE_3D
#define DECISION_TREE_SAMPLERATE_3D
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
};

*/
inline double PredictSampleRate_3D(double * input) {
    return -0.54035780668492 + input[0] * -0.01231300928974244 + input[1] * -0.00022241749158827522 + input[2] * -0.00014875084291075233 + input[3] * -0.0004180150240400014 + input[4] * -0.0000004460253167714001 + input[5] * 0.000000006183843792546584 + input[6] * 0.0000000030242655901102218 + input[7] * 0.000000034984953492089436 + input[8] * 0.00000006541462745955725 + input[9] * 0.000000025335468836446 + input[10] * 0.00000001624147585449038 + input[11] * 0.000000017613304310769537 + input[12] * 0.00000000026924862450390593 + input[13] * -0.00000000011553854626556755 + input[14] * 0.0000000006720897085495451 + input[15] * 0.11036954598933449 + input[16] * 0.054422167483515804 + input[17] * 0.023043017389526715 + input[18] * 0.009463908149734355 + input[19] * 0.0013798534730791602 + input[20] * 0.00000000011365596245546951 + input[21] * 0.0000124521145703549 + input[22] * -0.000021590849032407725 + input[23] * -0.00003993439768741486 + input[24] * 0.00000830850275910742 + input[25] * -0.000016443011874807322 + input[26] * -0.00004954501714335171 + input[27] * -0.000004231613545813324 + input[28] * -0.00004827950746167769 + input[29] * -0.00005064373132469512 + input[30] * -0.00010650000748631106 + input[31] * 0.00005034194825286188 + input[32] * 0.000032797920521473965 + input[33] * 0.00011755368831449075 + input[34] * 0.000021192895722978167 + input[35] * 0.00000010352180250007559 + input[36] * -0.000000013678516206858116 + input[37] * 0.0000000800717237930959 + input[38] * 0.024225244260039972 + input[39] * 0.00006733177282829632 + input[40] * 0.0004822325878502368 + input[41] * -0.00015194257872607278 + input[42] * 0.00000009907733534639251 + input[43] * -0.0000000498859281730657 + input[44] * -0.0000000730752775825183 + input[45] * 0.00000017804717797331726 + input[46] * -0.00000016142730075927036 + input[47] * 0.00000007881004346343978 + input[48] * -0.0000000579688631402453 + input[49] * 0.000000007756986983298297 + input[50] * -0.00000000000015033417219423306 + input[51] * -0.00000000000004332298408904478 + input[52] * -0.00000000000006746187809769655 + input[53] * -0.00000000000007565454339386601 + input[54] * 0.4283079315218469 + input[55] * 0.18022632368365657 + input[56] * 0.06011535487536178 + input[57] * 0.017721753115407007 + input[58] * 0.000000000000019723805921856297 + input[59] * 0.00012283413908262004 + input[60] * 0.00004614387202317666 + input[61] * 0.00010288046773765044 + input[62] * 0.000056549457620992455 + input[63] * -0.000038293908162888865 + input[64] * -0.00001970764907415195 + input[65] * -0.00002718058430161395 + input[66] * -0.00001317780244426952 + input[67] * 0.00015332999661058291 + input[68] * -0.00003236812683697805 + input[69] * 0.000011815233317251025 + input[70] * -0.0001464166917778363 + input[71] * 0.00006278065766116003 + input[72] * 0.0000020294583501050123 + input[73] * -0.000000022624350123380543 + input[74] * -0.000000009582643304729965 + input[75] * 0.000000052732841364366845;
}

#endif // DECISION_TREE_SAMPLERATE_3D
