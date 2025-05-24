
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
9 File_0_CellP99Value
10 File_0_CellP95Value
11 File_0_CellP50Value
12 File_0_CellP99Count
13 File_0_CellP95Count
14 File_0_CellP50Count
15 File_0_Dim0_GridSize
16 File_0_Dim1_GridSize
17 File_0_Dim2_GridSize
18 File_0_MedianPointsPerCell
19 File_0_NonEmptyCells
20 File_0_TotalCells
21 File_1_Density
22 File_1_NumPoints
23 File_1_MBR_Dim_0_Lower
24 File_1_MBR_Dim_1_Lower
25 File_1_MBR_Dim_2_Lower
26 File_1_MBR_Dim_0_Upper
27 File_1_MBR_Dim_1_Upper
28 File_1_MBR_Dim_2_Upper
29 File_1_GINI
30 File_1_CellP99Value
31 File_1_CellP95Value
32 File_1_CellP50Value
33 File_1_CellP99Count
34 File_1_CellP95Count
35 File_1_CellP50Count
36 File_1_Dim0_GridSize
37 File_1_Dim1_GridSize
38 File_1_Dim2_GridSize
39 File_1_MedianPointsPerCell
40 File_1_NonEmptyCells
41 File_1_TotalCells
42 HDLB
43 HDUP

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
    double File_0_CellP99Value;
    double File_0_CellP95Value;
    double File_0_CellP50Value;
    double File_0_CellP99Count;
    double File_0_CellP95Count;
    double File_0_CellP50Count;
    double File_0_Dim0_GridSize;
    double File_0_Dim1_GridSize;
    double File_0_Dim2_GridSize;
    double File_0_MedianPointsPerCell;
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
    double File_1_CellP99Value;
    double File_1_CellP95Value;
    double File_1_CellP50Value;
    double File_1_CellP99Count;
    double File_1_CellP95Count;
    double File_1_CellP50Count;
    double File_1_Dim0_GridSize;
    double File_1_Dim1_GridSize;
    double File_1_Dim2_GridSize;
    double File_1_MedianPointsPerCell;
    double File_1_NonEmptyCells;
    double File_1_TotalCells;
    double HDLB;
    double HDUP;
};

*/
inline double PredictNumPointsPerCell_3D(double * input) {
    double var0;
    if (input[22] < 1108239.0) {
        if (input[21] < 5121.197) {
            var0 = 3.1070735;
        } else {
            var0 = -1.5372366;
        }
    } else {
        if (input[42] < 18.0) {
            var0 = -3.527924;
        } else {
            var0 = 1.5296704;
        }
    }
    double var1;
    if (input[22] < 1108239.0) {
        if (input[30] >= 66.0) {
            var1 = -1.7044319;
        } else {
            var1 = 2.3029287;
        }
    } else {
        if (input[42] < 14.0) {
            var1 = -2.8781505;
        } else {
            var1 = 0.3672297;
        }
    }
    double var2;
    if (input[42] < 18.0) {
        if (input[41] < 350.0) {
            var2 = 1.8771461;
        } else {
            var2 = -2.1908662;
        }
    } else {
        if (input[41] < 4896.0) {
            var2 = 4.25825;
        } else {
            var2 = 1.3658618;
        }
    }
    double var3;
    if (input[36] < 26.0) {
        if (input[29] < 0.4855053) {
            var3 = 1.8495814;
        } else {
            var3 = -0.6457538;
        }
    } else {
        if (input[42] < 14.0) {
            var3 = -1.8866377;
        } else {
            var3 = 0.5831558;
        }
    }
    double var4;
    if (input[36] < 26.0) {
        if (input[5] < 0.50143903) {
            var4 = -2.0555205;
        } else {
            var4 = 1.1737328;
        }
    } else {
        if (input[42] < 18.0) {
            var4 = -1.4878913;
        } else {
            var4 = 0.8752422;
        }
    }
    double var5;
    if (input[42] < 32.0) {
        if (input[30] >= 25.0) {
            var5 = -1.0805357;
        } else {
            var5 = 2.6603734;
        }
    } else {
        if (input[33] < 2119.0) {
            var5 = 3.0553038;
        } else {
            var5 = 0.4761544;
        }
    }
    double var6;
    if (input[41] < 130152.0) {
        if (input[21] < 53908.19) {
            var6 = 0.589181;
        } else {
            var6 = -1.427296;
        }
    } else {
        if (input[35] < 29870.0) {
            var6 = -2.8332164;
        } else {
            var6 = -0.79730105;
        }
    }
    double var7;
    if (input[42] < 0.09082097) {
        if (input[31] < 64.0) {
            var7 = -1.2547548;
        } else {
            var7 = -3.3245373;
        }
    } else {
        if (input[22] < 157275.0) {
            var7 = 0.63723856;
        } else {
            var7 = -0.4566326;
        }
    }
    double var8;
    if (input[42] < 18.0) {
        if (input[30] >= 25.0) {
            var8 = -0.65068036;
        } else {
            var8 = 1.9906514;
        }
    } else {
        if (input[38] < 25.0) {
            var8 = 0.23140013;
        } else {
            var8 = 2.637587;
        }
    }
    double var9;
    if (input[42] < 0.09082097) {
        if (input[31] < 64.0) {
            var9 = -0.8907822;
        } else {
            var9 = -2.5824966;
        }
    } else {
        if (input[36] < 16.0) {
            var9 = 0.8231644;
        } else {
            var9 = -0.22099772;
        }
    }
    double var10;
    if (input[42] < 34.0) {
        if (input[35] >= 7.0) {
            var10 = -0.43549964;
        } else {
            var10 = 1.6080639;
        }
    } else {
        if (input[8] < 0.17611119) {
            var10 = -1.8686408;
        } else {
            var10 = 1.1466987;
        }
    }
    double var11;
    if (input[9] >= 584.0) {
        if (input[42] < 3.0) {
            var11 = -2.3285851;
        } else {
            var11 = -0.13746358;
        }
    } else {
        if (input[0] < 118938.47) {
            var11 = -0.15447004;
        } else {
            var11 = 2.7133944;
        }
    }
    double var12;
    if (input[42] < 6.0) {
        if (input[30] >= 25.0) {
            var12 = -0.5016114;
        } else {
            var12 = 1.1976389;
        }
    } else {
        if (input[0] < 0.45603558) {
            var12 = -0.2730125;
        } else {
            var12 = 0.813837;
        }
    }
    double var13;
    if (input[0] < 0.4532164) {
        if (input[37] < 24.0) {
            var13 = 1.2037052;
        } else {
            var13 = -0.6327887;
        }
    } else {
        if (input[21] < 0.3467023) {
            var13 = 1.8903885;
        } else {
            var13 = -0.002154394;
        }
    }
    double var14;
    if (input[29] < 0.8366958) {
        if (input[33] < 99111.0) {
            if (input[9] >= 584.0) {
                var14 = -2.317414;
            } else {
                var14 = -0.0078668995;
            }
        } else {
            var14 = 2.0663688;
        }
    } else {
        var14 = -2.0203245;
    }
    double var15;
    if (input[8] < 0.17611119) {
        if (input[0] < 0.44486347) {
            var15 = -0.37618196;
        } else {
            var15 = -2.6328592;
        }
    } else {
        if (input[30] >= 66.0) {
            var15 = -0.6910357;
        } else {
            var15 = 0.18317863;
        }
    }
    double var16;
    if (input[42] < 0.09082097) {
        if (input[33] < 82691.0) {
            var16 = -0.7780835;
        } else {
            var16 = 0.57679147;
        }
    } else {
        if (input[15] < 54.0) {
            var16 = 0.16102272;
        } else {
            var16 = -0.8065858;
        }
    }
    double var17;
    if (input[39] < 70.0) {
        if (input[21] < 53908.19) {
            if (input[21] < 15201.745) {
                var17 = -0.07418035;
            } else {
                var17 = 2.4530942;
            }
        } else {
            var17 = -1.5907415;
        }
    } else {
        var17 = 1.410098;
    }
    double var18;
    if (input[0] < 0.45603558) {
        if (input[8] < 0.28502557) {
            var18 = -0.4353555;
        } else {
            var18 = 0.5381293;
        }
    } else {
        if (input[21] < 0.2945735) {
            var18 = 2.473594;
        } else {
            var18 = 0.07078718;
        }
    }
    double var19;
    if (input[8] < 0.707439) {
        if (input[36] < 16.0) {
            var19 = 0.7933169;
        } else {
            var19 = -0.10573377;
        }
    } else {
        if (input[36] < 15.0) {
            var19 = -2.0404136;
        } else {
            var19 = -0.26630646;
        }
    }
    return 40.04761904761905 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_NUMPOINTSPERCELL_3D
