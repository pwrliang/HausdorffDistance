
#include <math.h>
#ifndef DECISION_TREE_EBONLYTHRESHOLD_3D
#define DECISION_TREE_EBONLYTHRESHOLD_3D
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
inline double PredictEBOnlyThreshold_3D(double * input) {
    double var0;
    if (input[39] < 43.0) {
        if (input[46] < 27.0) {
            if (input[27] < 200.0) {
                if (input[38] < 2.0) {
                    var0 = 103.7973;
                } else {
                    if (input[17] < 52.0) {
                        var0 = 180.0755;
                    } else {
                        var0 = 103.357155;
                    }
                }
            } else {
                var0 = 88.78581;
            }
        } else {
            var0 = 79.48559;
        }
    } else {
        var0 = -146.96712;
    }
    double var1;
    if (input[39] < 43.0) {
        if (input[8] < 0.6733632) {
            if (input[22] < 0.22304402) {
                if (input[39] < 18.0) {
                    var1 = 104.37416;
                } else {
                    var1 = -5.14012;
                }
            } else {
                if (input[20] < 28.0) {
                    var1 = 51.70161;
                } else {
                    if (input[18] < 62.0) {
                        if (input[56] < 1.339206) {
                            var1 = 76.13973;
                        } else {
                            if (input[0] < 0.43537536) {
                                var1 = 78.92199;
                            } else {
                                if (input[44] < 240.0) {
                                    var1 = 137.52956;
                                } else {
                                    var1 = 96.98639;
                                }
                            }
                        }
                    } else {
                        var1 = 56.08604;
                    }
                }
            }
        } else {
            var1 = 35.546356;
        }
    } else {
        var1 = -88.2794;
    }
    double var2;
    if (input[39] < 43.0) {
        if (input[30] < 0.32161808) {
            if (input[35] < 2243.0) {
                var2 = 109.8979;
            } else {
                if (input[36] < 3914.0) {
                    var2 = 24.86481;
                } else {
                    if (input[40] < 28.0) {
                        var2 = 97.67054;
                    } else {
                        var2 = 61.30899;
                    }
                }
            }
        } else {
            if (input[1] < 1377817.0) {
                if (input[8] < 0.6733632) {
                    if (input[38] < 3.0) {
                        if (input[18] < 8.0) {
                            var2 = 5.9048386;
                        } else {
                            var2 = 72.79105;
                        }
                    } else {
                        if (input[45] < 112.0) {
                            if (input[38] >= 45.0) {
                                var2 = 58.36027;
                            } else {
                                var2 = 98.241936;
                            }
                        } else {
                            var2 = 35.127872;
                        }
                    }
                } else {
                    var2 = 12.389528;
                }
            } else {
                var2 = -18.794567;
            }
        }
    } else {
        var2 = -53.027187;
    }
    double var3;
    if (input[39] < 43.0) {
        if (input[41] < 24.0) {
            if (input[33] >= 25.0) {
                var3 = -18.644909;
            } else {
                if (input[30] < 0.58482856) {
                    if (input[55] < 0.57107115) {
                        if (input[22] < 8965.423) {
                            var3 = 81.69094;
                        } else {
                            var3 = 38.874783;
                        }
                    } else {
                        if (input[33] < 10.0) {
                            var3 = -13.2566595;
                        } else {
                            if (input[44] < 53.0) {
                                var3 = 72.96572;
                            } else {
                                if (input[40] < 23.0) {
                                    var3 = 2.2153862;
                                } else {
                                    if (input[1] < 1294182.0) {
                                        var3 = 20.170395;
                                    } else {
                                        var3 = 66.17673;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[41] < 9.0) {
                        var3 = 50.53018;
                    } else {
                        var3 = -15.118083;
                    }
                }
            }
        } else {
            if (input[18] < 59.0) {
                var3 = 84.37261;
            } else {
                var3 = 32.271606;
            }
        }
    } else {
        var3 = -31.85208;
    }
    double var4;
    if (input[39] < 43.0) {
        if (input[24] < 126.0) {
            if (input[15] < 47832.0) {
                if (input[30] < 0.58482856) {
                    if (input[46] < 27.0) {
                        if (input[46] < 12.0) {
                            if (input[40] < 8.0) {
                                var4 = 2.6868036;
                            } else {
                                var4 = 39.28924;
                            }
                        } else {
                            if (input[0] < 0.46365568) {
                                if (input[30] < 0.2988276) {
                                    if (input[30] < 0.23783533) {
                                        var4 = 36.823067;
                                    } else {
                                        var4 = -3.2998369;
                                    }
                                } else {
                                    var4 = 47.61289;
                                }
                            } else {
                                var4 = 58.941803;
                            }
                        }
                    } else {
                        var4 = 4.46165;
                    }
                } else {
                    if (input[42] < 131.0) {
                        var4 = -14.278162;
                    } else {
                        var4 = 28.986801;
                    }
                }
            } else {
                var4 = -6.40086;
            }
        } else {
            var4 = -13.2576475;
        }
    } else {
        var4 = -19.132734;
    }
    return 368.0384341637011 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_EBONLYTHRESHOLD_3D
