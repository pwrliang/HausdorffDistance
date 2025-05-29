
#include <math.h>
#ifndef DECISION_TREE_EBONLYTHRESHOLD_3D
#define DECISION_TREE_EBONLYTHRESHOLD_3D
/*
0 File_0_Density
1 File_0_NumPoints
2 File_0_MBR_Dim_0_Range
3 File_0_MBR_Dim_1_Range
4 File_0_MBR_Dim_2_Range
5 File_0_GINI
6 File_0_Cell_P0.99_Value
7 File_0_Cell_P0.95_Value
8 File_0_Cell_P0.75_Value
9 File_0_Cell_P0.5_Value
10 File_0_Cell_P0.25_Value
11 File_0_Cell_P0.1_Value
12 File_0_Cell_P0.99_Count
13 File_0_Cell_P0.95_Count
14 File_0_Cell_P0.75_Count
15 File_0_Cell_P0.5_Count
16 File_0_Cell_P0.25_Count
17 File_0_Cell_P0.1_Count
18 File_0_Dim0_GridSize
19 File_0_Dim1_GridSize
20 File_0_Dim2_GridSize
21 File_0_NonEmptyCells
22 File_0_TotalCells
23 File_1_Density
24 File_1_NumPoints
25 File_1_MBR_Dim_0_Range
26 File_1_MBR_Dim_1_Range
27 File_1_MBR_Dim_2_Range
28 File_1_GINI
29 File_1_Cell_P0.99_Value
30 File_1_Cell_P0.95_Value
31 File_1_Cell_P0.75_Value
32 File_1_Cell_P0.5_Value
33 File_1_Cell_P0.25_Value
34 File_1_Cell_P0.1_Value
35 File_1_Cell_P0.99_Count
36 File_1_Cell_P0.95_Count
37 File_1_Cell_P0.75_Count
38 File_1_Cell_P0.5_Count
39 File_1_Cell_P0.25_Count
40 File_1_Cell_P0.1_Count
41 File_1_Dim0_GridSize
42 File_1_Dim1_GridSize
43 File_1_Dim2_GridSize
44 File_1_NonEmptyCells
45 File_1_TotalCells
46 Cell_P0.99_Value
47 Cell_P0.95_Value
48 Cell_P0.75_Value
49 Cell_P0.5_Value
50 Cell_P0.25_Value
51 Cell_P0.1_Value
52 Cell_P0.99_Count
53 Cell_P0.95_Count
54 Cell_P0.75_Count
55 Cell_P0.5_Count
56 Cell_P0.25_Count
57 Cell_P0.1_Count
58 Dim0_GridSize
59 Dim1_GridSize
60 Dim2_GridSize
61 HDLBRatio
62 HDUBRatio

struct Input {
    double File_0_Density;
    double File_0_NumPoints;
    double File_0_MBR_Dim_0_Range;
    double File_0_MBR_Dim_1_Range;
    double File_0_MBR_Dim_2_Range;
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
    double File_1_MBR_Dim_0_Range;
    double File_1_MBR_Dim_1_Range;
    double File_1_MBR_Dim_2_Range;
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
    double HDLBRatio;
    double HDUBRatio;
};

*/
inline double PredictEBOnlyThreshold_3D(double * input) {
    double var0;
    if (input[41] < 38.0) {
        if (input[24] < 13212.0) {
            if (input[12] >= 22.0) {
                if (input[29] < 65.0) {
                    if (input[59] < 11.0) {
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
            if (input[25] < 84.0) {
                if (input[52] < 937.0) {
                    var0 = 38.59899;
                } else {
                    if (input[5] < 0.18508628) {
                        if (input[5] < 0.17880093) {
                            var0 = 126.59899;
                        } else {
                            var0 = 11.785911;
                        }
                    } else {
                        if (input[6] < 49.0) {
                            if (input[17] < 680.0) {
                                var0 = 170.1454;
                            } else {
                                var0 = 111.71924;
                            }
                        } else {
                            var0 = 102.53149;
                        }
                    }
                }
            } else {
                var0 = 34.757156;
            }
        }
    } else {
        var0 = -146.83286;
    }
    double var1;
    if (input[41] < 43.0) {
        if (input[59] < 17.0) {
            if (input[8] >= 16.0) {
                if (input[9] < 10.0) {
                    var1 = -5.7463565;
                } else {
                    if (input[25] < 0.45042732) {
                        var1 = -32.14217;
                    } else {
                        var1 = -59.8061;
                    }
                }
            } else {
                if (input[34] < 3.0) {
                    var1 = 97.302635;
                } else {
                    var1 = -33.708927;
                }
            }
        } else {
            if (input[61] < 0.5411409) {
                if (input[12] < 69932.0) {
                    if (input[20] < 49.0) {
                        if (input[30] < 55.0) {
                            var1 = 35.844784;
                        } else {
                            if (input[31] < 55.0) {
                                var1 = 152.7424;
                            } else {
                                var1 = 72.09062;
                            }
                        }
                    } else {
                        var1 = -22.145359;
                    }
                } else {
                    if (input[28] < 0.28919795) {
                        var1 = 129.05742;
                    } else {
                        var1 = 67.288216;
                    }
                }
            } else {
                var1 = -0.027120268;
            }
        }
    } else {
        var1 = -88.333015;
    }
    double var2;
    if (input[41] < 27.0) {
        if (input[35] >= 230.0) {
            if (input[3] < 168.62482) {
                if (input[35] < 389.0) {
                    var2 = 97.6616;
                } else {
                    if (input[57] < 150.0) {
                        var2 = 10.506769;
                    } else {
                        if (input[57] < 699.0) {
                            var2 = 83.8223;
                        } else {
                            if (input[14] < 69527.0) {
                                var2 = 13.447975;
                            } else {
                                var2 = 72.94889;
                            }
                        }
                    }
                }
            } else {
                if (input[59] < 64.0) {
                    var2 = -54.05954;
                } else {
                    var2 = 65.37541;
                }
            }
        } else {
            if (input[16] < 71.0) {
                if (input[47] < 49.0) {
                    var2 = 19.762602;
                } else {
                    if (input[3] < 0.5833333) {
                        var2 = -13.415964;
                    } else {
                        if (input[30] < 59.0) {
                            var2 = -33.37046;
                        } else {
                            var2 = -68.85897;
                        }
                    }
                }
            } else {
                var2 = 39.39915;
            }
        }
    } else {
        if (input[24] < 1108239.0) {
            var2 = -29.48056;
        } else {
            var2 = -53.05879;
        }
    }
    double var3;
    if (input[25] < 92.0) {
        if (input[45] < 1287.0) {
            if (input[17] >= 4.0) {
                if (input[57] < 4.0) {
                    var3 = -8.671168;
                } else {
                    var3 = -54.52247;
                }
            } else {
                if (input[48] < 27.0) {
                    var3 = 68.8396;
                } else {
                    var3 = -30.547468;
                }
            }
        } else {
            if (input[18] < 52.0) {
                if (input[16] < 11279.0) {
                    if (input[48] < 28.0) {
                        var3 = 70.14331;
                    } else {
                        if (input[30] < 62.0) {
                            var3 = -10.323909;
                        } else {
                            var3 = 37.90133;
                        }
                    }
                } else {
                    if (input[56] < 20096.0) {
                        if (input[55] < 42832.0) {
                            var3 = -42.45442;
                        } else {
                            var3 = 2.4814742;
                        }
                    } else {
                        var3 = 44.422283;
                    }
                }
            } else {
                var3 = 74.88711;
            }
        }
    } else {
        if (input[58] < 52.0) {
            if (input[57] < 765.0) {
                var3 = -31.991882;
            } else {
                var3 = -6.587064;
            }
        } else {
            if (input[24] < 1169456.0) {
                var3 = -44.642223;
            } else {
                var3 = -31.867035;
            }
        }
    }
    double var4;
    if (input[41] < 27.0) {
        if (input[37] >= 3914.0) {
            var4 = 49.56683;
        } else {
            if (input[34] >= 3.0) {
                if (input[43] < 8.0) {
                    var4 = -46.49889;
                } else {
                    var4 = 15.127875;
                }
            } else {
                if (input[61] < 0.5411409) {
                    if (input[31] < 27.0) {
                        if (input[19] < 9.0) {
                            var4 = 93.816086;
                        } else {
                            var4 = 10.20737;
                        }
                    } else {
                        if (input[56] < 182.0) {
                            var4 = -29.898407;
                        } else {
                            if (input[45] < 8800.0) {
                                if (input[61] < 0.38381433) {
                                    var4 = 61.019836;
                                } else {
                                    var4 = 15.8533125;
                                }
                            } else {
                                var4 = -13.282119;
                            }
                        }
                    }
                } else {
                    var4 = -26.190292;
                }
            }
        }
    } else {
        if (input[25] < 92.45553) {
            if (input[55] < 29431.0) {
                var4 = -8.213611;
            } else {
                var4 = -22.404097;
            }
        } else {
            var4 = -19.318113;
        }
    }
    return 368.0384341637011 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_EBONLYTHRESHOLD_3D
