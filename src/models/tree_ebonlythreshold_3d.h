
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
inline double PredictEBOnlyThreshold_3D(double * input) {
    double var0;
    if (input[22] < 210327.0) {
        if (input[0] < 0.44433957) {
            if (input[2] < 55.0) {
                var0 = 53.844692;
            } else {
                var0 = 127.4297;
            }
        } else {
            if (input[5] < 0.461435) {
                var0 = 88.84322;
            } else {
                if (input[33] < 17.0) {
                    var0 = 101.24708;
                } else {
                    if (input[35] >= 105.0) {
                        if (input[35] < 185.0) {
                            var0 = 68.95673;
                        } else {
                            if (input[26] < 179.0) {
                                var0 = 176.1816;
                            } else {
                                var0 = 117.4697;
                            }
                        }
                    } else {
                        var0 = 185.92982;
                    }
                }
            }
        }
    } else {
        var0 = -163.65092;
    }
    double var1;
    if (input[36] < 43.0) {
        if (input[34] < 5187.0) {
            if (input[14] >= 71.0) {
                if (input[43] < 204.71687) {
                    if (input[36] < 18.0) {
                        if (input[26] < 166.0) {
                            if (input[0] < 118938.47) {
                                var1 = 13.253548;
                            } else {
                                var1 = 71.44621;
                            }
                        } else {
                            var1 = 109.13924;
                        }
                    } else {
                        var1 = 7.8064537;
                    }
                } else {
                    var1 = 104.40905;
                }
            } else {
                if (input[0] < 2693.0308) {
                    var1 = 65.13973;
                } else {
                    if (input[18] < 31.0) {
                        var1 = 129.85661;
                    } else {
                        var1 = 85.30836;
                    }
                }
            }
        } else {
            var1 = 131.0088;
        }
    } else {
        var1 = -98.5529;
    }
    double var2;
    if (input[22] < 966204.0) {
        if (input[16] < 60.0) {
            if (input[14] < 1853.0) {
                if (input[33] >= 862.0) {
                    var2 = -9.765835;
                } else {
                    if (input[9] < 200.0) {
                        if (input[5] < 0.5904136) {
                            if (input[42] < 0.40750378) {
                                var2 = 64.822624;
                            } else {
                                var2 = 103.573;
                            }
                        } else {
                            if (input[0] < 14018.713) {
                                if (input[39] < 19.0) {
                                    var2 = -6.3509645;
                                } else {
                                    var2 = 52.673553;
                                }
                            } else {
                                var2 = 79.526306;
                            }
                        }
                    } else {
                        var2 = 19.578825;
                    }
                }
            } else {
                if (input[29] < 0.28402486) {
                    var2 = 62.688213;
                } else {
                    var2 = 105.71228;
                }
            }
        } else {
            if (input[12] < 77911.0) {
                if (input[36] < 21.0) {
                    var2 = -43.073746;
                } else {
                    var2 = 55.51368;
                }
            } else {
                var2 = 82.72406;
            }
        }
    } else {
        var2 = -59.756603;
    }
    double var3;
    if (input[37] < 35.0) {
        if (input[34] < 5187.0) {
            if (input[32] < 26.0) {
                if (input[26] < 193.0) {
                    if (input[43] < 1.2928859) {
                        var3 = 0.7467739;
                    } else {
                        if (input[10] >= 57.0) {
                            if (input[10] < 74.0) {
                                var3 = -6.497351;
                            } else {
                                if (input[17] < 10.0) {
                                    var3 = 21.657051;
                                } else {
                                    var3 = 55.666054;
                                }
                            }
                        } else {
                            if (input[43] < 1.5003477) {
                                var3 = 77.33517;
                            } else {
                                if (input[36] < 10.0) {
                                    var3 = -11.312837;
                                } else {
                                    if (input[32] < 24.0) {
                                        if (input[26] < 166.0) {
                                            var3 = 59.796326;
                                        } else {
                                            var3 = 91.985535;
                                        }
                                    } else {
                                        var3 = 21.697313;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    var3 = -1.662145;
                }
            } else {
                var3 = -20.90056;
            }
        } else {
            var3 = 62.556606;
        }
    } else {
        if (input[1] < 36221.0) {
            var3 = -19.216583;
        } else {
            var3 = -36.13776;
        }
    }
    double var4;
    if (input[33] >= 7505.0) {
        if (input[33] < 14546.0) {
            var4 = -14.039748;
        } else {
            var4 = -21.802986;
        }
    } else {
        if (input[42] < 82.0) {
            if (input[23] < 69.0) {
                if (input[35] >= 135.0) {
                    var4 = -24.098433;
                } else {
                    if (input[21] < 3804.131) {
                        var4 = 49.293587;
                    } else {
                        if (input[21] < 6610.401) {
                            var4 = -16.016613;
                        } else {
                            if (input[39] < 33.0) {
                                if (input[11] < 10.0) {
                                    var4 = 18.154234;
                                } else {
                                    var4 = 50.86143;
                                }
                            } else {
                                var4 = 12.657148;
                            }
                        }
                    }
                }
            } else {
                if (input[32] < 21.0) {
                    var4 = 18.880766;
                } else {
                    if (input[2] < 52.0) {
                        var4 = 30.14336;
                    } else {
                        var4 = 65.49698;
                    }
                }
            }
        } else {
            if (input[17] < 50.0) {
                var4 = -28.351648;
            } else {
                var4 = 24.698889;
            }
        }
    }
    return 412.0270916334661 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_EBONLYTHRESHOLD_3D
