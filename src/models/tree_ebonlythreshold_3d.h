
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
11 File_0_Cell_P0.75_Value
12 File_0_Cell_P0.5_Value
13 File_0_Cell_P0.25_Value
14 File_0_Cell_P0.1_Value
15 File_0_Cell_P0.99_Count
16 File_0_Cell_P0.95_Count
17 File_0_Cell_P0.75_Count
18 File_0_Cell_P0.5_Count
19 File_0_Cell_P0.25_Count
20 File_0_Cell_P0.1_Count
21 File_0_Dim0_GridSize
22 File_0_Dim1_GridSize
23 File_0_Dim2_GridSize
24 File_0_NonEmptyCells
25 File_0_TotalCells
26 File_1_Density
27 File_1_NumPoints
28 File_1_MBR_Dim_0_Lower
29 File_1_MBR_Dim_1_Lower
30 File_1_MBR_Dim_2_Lower
31 File_1_MBR_Dim_0_Upper
32 File_1_MBR_Dim_1_Upper
33 File_1_MBR_Dim_2_Upper
34 File_1_GINI
35 File_1_Cell_P0.99_Value
36 File_1_Cell_P0.95_Value
37 File_1_Cell_P0.75_Value
38 File_1_Cell_P0.5_Value
39 File_1_Cell_P0.25_Value
40 File_1_Cell_P0.1_Value
41 File_1_Cell_P0.99_Count
42 File_1_Cell_P0.95_Count
43 File_1_Cell_P0.75_Count
44 File_1_Cell_P0.5_Count
45 File_1_Cell_P0.25_Count
46 File_1_Cell_P0.1_Count
47 File_1_Dim0_GridSize
48 File_1_Dim1_GridSize
49 File_1_Dim2_GridSize
50 File_1_NonEmptyCells
51 File_1_TotalCells
52 Cell_P0.99_Value
53 Cell_P0.95_Value
54 Cell_P0.75_Value
55 Cell_P0.5_Value
56 Cell_P0.25_Value
57 Cell_P0.1_Value
58 Cell_P0.99_Count
59 Cell_P0.95_Count
60 Cell_P0.75_Count
61 Cell_P0.5_Count
62 Cell_P0.25_Count
63 Cell_P0.1_Count
64 Dim0_GridSize
65 Dim1_GridSize
66 Dim2_GridSize
67 HDLB
68 HDUP

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
    double File_0_Cell_P0.75_Value;
    double File_0_Cell_P0.5_Value;
    double File_0_Cell_P0.25_Value;
    double File_0_Cell_P0.1_Value;
    double File_0_Cell_P0.99_Count;
    double File_0_Cell_P0.95_Count;
    double File_0_Cell_P0.75_Count;
    double File_0_Cell_P0.5_Count;
    double File_0_Cell_P0.25_Count;
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
    double File_1_Cell_P0.75_Value;
    double File_1_Cell_P0.5_Value;
    double File_1_Cell_P0.25_Value;
    double File_1_Cell_P0.1_Value;
    double File_1_Cell_P0.99_Count;
    double File_1_Cell_P0.95_Count;
    double File_1_Cell_P0.75_Count;
    double File_1_Cell_P0.5_Count;
    double File_1_Cell_P0.25_Count;
    double File_1_Cell_P0.1_Count;
    double File_1_Dim0_GridSize;
    double File_1_Dim1_GridSize;
    double File_1_Dim2_GridSize;
    double File_1_NonEmptyCells;
    double File_1_TotalCells;
    double Cell_P0.99_Value;
    double Cell_P0.95_Value;
    double Cell_P0.75_Value;
    double Cell_P0.5_Value;
    double Cell_P0.25_Value;
    double Cell_P0.1_Value;
    double Cell_P0.99_Count;
    double Cell_P0.95_Count;
    double Cell_P0.75_Count;
    double Cell_P0.5_Count;
    double Cell_P0.25_Count;
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
    if (input[47] < 38.0) {
        if (input[27] < 13212.0) {
            if (input[15] >= 22.0) {
                if (input[35] < 65.0) {
                    if (input[65] < 11.0) {
                        var0 = -60.067677;
                    } else {
                        var0 = -88.04284;
                    }
                } else {
                    var0 = 11.985591;
                }
            } else {
                var0 = 58.741035;
            }
        } else {
            if (input[58] < 937.0) {
                var0 = 38.59899;
            } else {
                if (input[47] < 18.0) {
                    if (input[63] < 683.0) {
                        if (input[2] < 51.0) {
                            var0 = 199.4935;
                        } else {
                            var0 = 108.38592;
                        }
                    } else {
                        var0 = 86.15716;
                    }
                } else {
                    if (input[34] < 0.2988276) {
                        if (input[67] < 71.7491) {
                            var0 = 124.1454;
                        } else {
                            var0 = 48.022392;
                        }
                    } else {
                        var0 = 20.500013;
                    }
                }
            }
        }
    } else {
        var0 = -146.83286;
    }
    double var1;
    if (input[47] < 43.0) {
        if (input[27] < 26672.0) {
            if (input[19] < 71.0) {
                if (input[19] < 5.0) {
                    var1 = 29.053516;
                } else {
                    if (input[45] >= 35.0) {
                        var1 = -9.340312;
                    } else {
                        if (input[68] < 1.4768549) {
                            var1 = -42.46235;
                        } else {
                            var1 = -66.21763;
                        }
                    }
                }
            } else {
                if (input[34] < 0.3723396) {
                    var1 = -23.969307;
                } else {
                    var1 = 83.916534;
                }
            }
        } else {
            if (input[8] < 0.18825471) {
                if (input[5] < 186.0) {
                    var1 = -26.418768;
                } else {
                    if (input[0] < 0.45568952) {
                        var1 = 95.52427;
                    } else {
                        var1 = 41.461105;
                    }
                }
            } else {
                if (input[8] < 0.29900962) {
                    if (input[31] < 179.0) {
                        var1 = 144.09291;
                    } else {
                        var1 = 75.377846;
                    }
                } else {
                    var1 = 61.101723;
                }
            }
        }
    } else {
        var1 = -88.333015;
    }
    double var2;
    if (input[47] < 27.0) {
        if (input[41] >= 230.0) {
            if (input[28] < 126.0) {
                if (input[56] < 10.0) {
                    var2 = 91.76711;
                } else {
                    if (input[67] < 51.0) {
                        var2 = -16.643118;
                    } else {
                        if (input[15] < 69932.0) {
                            if (input[19] < 11139.0) {
                                var2 = 72.67171;
                            } else {
                                var2 = -8.333812;
                            }
                        } else {
                            if (input[62] < 19251.0) {
                                var2 = 37.478893;
                            } else {
                                var2 = 97.899055;
                            }
                        }
                    }
                }
            } else {
                var2 = -16.299658;
            }
        } else {
            if (input[20] >= 4.0) {
                if (input[63] < 4.0) {
                    var2 = -14.224726;
                } else {
                    var2 = -53.088905;
                }
            } else {
                if (input[12] >= 7.0) {
                    var2 = -18.111364;
                } else {
                    var2 = 85.1145;
                }
            }
        }
    } else {
        if (input[27] < 1108239.0) {
            var2 = -33.543175;
        } else {
            var2 = -53.05879;
        }
    }
    double var3;
    if (input[47] < 27.0) {
        if (input[27] < 26672.0) {
            if (input[53] < 112.0) {
                if (input[35] >= 98.0) {
                    var3 = 68.73663;
                } else {
                    if (input[38] >= 8.0) {
                        if (input[15] < 73.0) {
                            var3 = -47.57995;
                        } else {
                            var3 = -13.741836;
                        }
                    } else {
                        if (input[12] >= 8.0) {
                            var3 = -15.34872;
                        } else {
                            var3 = 59.28909;
                        }
                    }
                }
            } else {
                if (input[0] < 52142.832) {
                    var3 = -52.98451;
                } else {
                    var3 = -14.39198;
                }
            }
        } else {
            if (input[51] < 4968.0) {
                var3 = 70.38247;
            } else {
                if (input[46] < 80.0) {
                    var3 = -25.972385;
                } else {
                    if (input[63] < 743.0) {
                        if (input[43] < 3793.0) {
                            var3 = 29.313807;
                        } else {
                            var3 = 78.25291;
                        }
                    } else {
                        var3 = 12.736502;
                    }
                }
            }
        }
    } else {
        if (input[47] < 45.0) {
            var3 = -18.641928;
        } else {
            var3 = -31.924316;
        }
    }
    double var4;
    if (input[48] < 35.0) {
        if (input[21] < 53.0) {
            if (input[55] < 10.0) {
                if (input[47] < 13.0) {
                    if (input[41] < 66.0) {
                        var4 = -2.5843809;
                    } else {
                        var4 = -57.840607;
                    }
                } else {
                    var4 = 19.962227;
                }
            } else {
                if (input[56] < 10.0) {
                    if (input[5] < 0.618559) {
                        var4 = 72.31167;
                    } else {
                        if (input[26] < 6159.1934) {
                            var4 = 40.001537;
                        } else {
                            var4 = -9.457642;
                        }
                    }
                } else {
                    if (input[26] < 0.37985972) {
                        if (input[46] < 80.0) {
                            var4 = -11.618526;
                        } else {
                            if (input[64] < 50.0) {
                                var4 = 3.5183914;
                            } else {
                                if (input[46] < 137.0) {
                                    var4 = 55.11268;
                                } else {
                                    var4 = 27.461176;
                                }
                            }
                        }
                    } else {
                        var4 = -33.76881;
                    }
                }
            }
        } else {
            var4 = 62.07685;
        }
    } else {
        var4 = -19.234512;
    }
    return 368.0384341637011 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_EBONLYTHRESHOLD_3D
