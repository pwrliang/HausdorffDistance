
#include <math.h>
#ifndef DECISION_TREE_NUMPOINTSPERCELL_3D
#define DECISION_TREE_NUMPOINTSPERCELL_3D
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
inline double PredictNumPointsPerCell_3D(double * input) {
    double var0;
    if (input[61] < 0.015582935) {
        if (input[32] < 19.0) {
            if (input[61] < 0.0039452827) {
                if (input[17] < 169.0) {
                    if (input[4] < 76.0) {
                        if (input[62] < 1.0154308) {
                            var0 = 1.1;
                        } else {
                            if (input[37] < 54164.0) {
                                var0 = -0.35714287;
                            } else {
                                if (input[2] < 79.0) {
                                    if (input[27] < 137.0) {
                                        var0 = 0.26;
                                    } else {
                                        if (input[26] < 170.0) {
                                            var0 = 0.7;
                                        } else {
                                            var0 = 0.26666668;
                                        }
                                    }
                                } else {
                                    var0 = 0.042857144;
                                }
                            }
                        }
                    } else {
                        var0 = -0.575;
                    }
                } else {
                    if (input[56] < 33717.0) {
                        if (input[0] < 0.46824065) {
                            if (input[37] < 77839.0) {
                                if (input[54] < 57272.0) {
                                    var0 = 0.0;
                                } else {
                                    var0 = -0.86800003;
                                }
                            } else {
                                if (input[26] < 180.0) {
                                    var0 = 0.2;
                                } else {
                                    var0 = -0.5;
                                }
                            }
                        } else {
                            var0 = 0.1;
                        }
                    } else {
                        var0 = -1.775;
                    }
                }
            } else {
                if (input[18] < 48.0) {
                    if (input[25] < 133.0) {
                        var0 = -0.15714286;
                    } else {
                        if (input[0] < 0.28831974) {
                            var0 = 0.46;
                        } else {
                            var0 = 1.8272728;
                        }
                    }
                } else {
                    if (input[54] < 138132.0) {
                        if (input[17] < 664.0) {
                            if (input[19] < 59.0) {
                                if (input[57] < 1338.0) {
                                    var0 = -0.54;
                                } else {
                                    var0 = -0.1;
                                }
                            } else {
                                var0 = -0.90000004;
                            }
                        } else {
                            if (input[22] < 144060.0) {
                                var0 = 0.8333333;
                            } else {
                                if (input[40] < 749.0) {
                                    if (input[40] < 712.0) {
                                        if (input[61] < 0.011702189) {
                                            var0 = 0.13333334;
                                        } else {
                                            var0 = -0.52500004;
                                        }
                                    } else {
                                        var0 = 0.5857143;
                                    }
                                } else {
                                    if (input[3] < 168.62482) {
                                        var0 = -0.22000001;
                                    } else {
                                        var0 = -0.7285714;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[56] < 30159.0) {
                            var0 = 1.35;
                        } else {
                            var0 = 0.06666667;
                        }
                    }
                }
            }
        } else {
            var0 = -2.211111;
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[62] < 1.0758849) {
                var0 = 0.3857143;
            } else {
                if (input[0] < 863.9725) {
                    if (input[47] < 827.0) {
                        var0 = -1.2333333;
                    } else {
                        if (input[23] < 195.3306) {
                            if (input[48] < 94.0) {
                                var0 = -1.0600001;
                            } else {
                                var0 = 0.06666667;
                            }
                        } else {
                            var0 = 0.56666666;
                        }
                    }
                } else {
                    var0 = -1.6375;
                }
            }
        } else {
            if (input[1] < 13626.0) {
                if (input[43] < 5.0) {
                    if (input[50] < 4.0) {
                        var0 = 0.3;
                    } else {
                        var0 = 1.7888889;
                    }
                } else {
                    if (input[10] < 8.0) {
                        if (input[46] < 204.0) {
                            if (input[60] < 10.0) {
                                if (input[49] < 7.0) {
                                    var0 = 0.6666667;
                                } else {
                                    var0 = -1.1;
                                }
                            } else {
                                if (input[6] < 66.0) {
                                    var0 = 1.3000001;
                                } else {
                                    var0 = 0.42;
                                }
                            }
                        } else {
                            if (input[7] < 74.0) {
                                var0 = -1.58;
                            } else {
                                var0 = -0.3;
                            }
                        }
                    } else {
                        if (input[49] < 15.0) {
                            var0 = -1.7545455;
                        } else {
                            var0 = 0.0;
                        }
                    }
                }
            } else {
                if (input[37] < 80088.0) {
                    if (input[56] < 50478.0) {
                        if (input[61] < 0.9591581) {
                            if (input[61] < 0.02758785) {
                                if (input[25] < 143.0) {
                                    if (input[19] < 58.0) {
                                        var0 = -0.725;
                                    } else {
                                        if (input[45] < 147500.0) {
                                            var0 = 0.8411765;
                                        } else {
                                            var0 = -0.08260869;
                                        }
                                    }
                                } else {
                                    if (input[61] < 0.02317072) {
                                        var0 = 1.7222222;
                                    } else {
                                        var0 = 0.58000004;
                                    }
                                }
                            } else {
                                if (input[28] < 0.77193576) {
                                    if (input[20] < 52.0) {
                                        if (input[25] < 62.0) {
                                            var0 = 0.94666666;
                                        } else {
                                            var0 = 1.5123078;
                                        }
                                    } else {
                                        if (input[61] < 0.0389721) {
                                            var0 = 0.06923077;
                                        } else {
                                            var0 = 1.1;
                                        }
                                    }
                                } else {
                                    if (input[39] < 12122.0) {
                                        if (input[62] < 1.3009824) {
                                            var0 = 0.9428572;
                                        } else {
                                            var0 = -0.54545456;
                                        }
                                    } else {
                                        if (input[62] < 1.3338008) {
                                            var0 = -0.27142856;
                                        } else {
                                            var0 = 1.6076924;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[7] < 1381.0) {
                                var0 = -1.2777778;
                            } else {
                                var0 = 1.5;
                            }
                        }
                    } else {
                        if (input[57] < 4474.0) {
                            var0 = -0.033333335;
                        } else {
                            var0 = -0.8600001;
                        }
                    }
                } else {
                    if (input[6] < 5111.0) {
                        if (input[52] < 150178.0) {
                            var0 = 0.73333335;
                        } else {
                            if (input[40] < 19720.0) {
                                var0 = 1.751064;
                            } else {
                                var0 = 0.8600001;
                            }
                        }
                    } else {
                        if (input[47] < 877.0) {
                            var0 = 0.46;
                        } else {
                            if (input[47] < 1509.0) {
                                if (input[28] < 0.8554572) {
                                    if (input[28] < 0.80087143) {
                                        var0 = 0.82;
                                    } else {
                                        var0 = 1.6846154;
                                    }
                                } else {
                                    if (input[58] < 3976.0) {
                                        var0 = 1.58;
                                    } else {
                                        var0 = 0.475;
                                    }
                                }
                            } else {
                                var0 = 0.6333334;
                            }
                        }
                    }
                }
            }
        }
    }
    double var1;
    if (input[61] < 0.015582935) {
        if (input[32] < 19.0) {
            if (input[58] < 51.0) {
                if (input[14] < 3678.0) {
                    if (input[5] < 0.25260195) {
                        var1 = -0.3021021;
                    } else {
                        if (input[54] < 62375.0) {
                            var1 = 0.5756952;
                        } else {
                            var1 = 0.1456369;
                        }
                    }
                } else {
                    if (input[58] < 49.0) {
                        var1 = 0.37825394;
                    } else {
                        var1 = 1.5218182;
                    }
                }
            } else {
                if (input[61] < 0.0038832047) {
                    if (input[17] < 126.0) {
                        if (input[38] < 42336.0) {
                            var1 = -0.40582862;
                        } else {
                            if (input[49] < 18.0) {
                                var1 = 0.13015871;
                            } else {
                                var1 = 0.42540947;
                            }
                        }
                    } else {
                        if (input[56] < 33717.0) {
                            if (input[4] < 135.0) {
                                if (input[57] < 686.0) {
                                    var1 = -0.039995257;
                                } else {
                                    var1 = -0.87721175;
                                }
                            } else {
                                if (input[17] < 639.0) {
                                    if (input[40] < 718.0) {
                                        var1 = -0.27984002;
                                    } else {
                                        var1 = 0.47072002;
                                    }
                                } else {
                                    if (input[57] < 1470.0) {
                                        var1 = -0.63784;
                                    } else {
                                        var1 = -0.16880001;
                                    }
                                }
                            }
                        } else {
                            if (input[17] < 697.0) {
                                var1 = -0.7440001;
                            } else {
                                var1 = -1.7360001;
                            }
                        }
                    }
                } else {
                    if (input[2] < 128.0) {
                        var1 = 1.1417727;
                    } else {
                        if (input[54] < 143531.0) {
                            if (input[45] < 157437.0) {
                                if (input[45] < 149450.0) {
                                    if (input[61] < 0.015445655) {
                                        if (input[24] < 1399428.0) {
                                            var1 = 0.5885714;
                                        } else {
                                            var1 = -0.25612858;
                                        }
                                    } else {
                                        var1 = -0.6742858;
                                    }
                                } else {
                                    if (input[37] < 73306.0) {
                                        if (input[24] < 1483326.0) {
                                            var1 = 1.5564848;
                                        } else {
                                            var1 = 0.62129146;
                                        }
                                    } else {
                                        if (input[3] < 171.0) {
                                            var1 = 0.24562895;
                                        } else {
                                            var1 = -0.5128572;
                                        }
                                    }
                                }
                            } else {
                                if (input[37] < 55415.0) {
                                    var1 = -0.09220783;
                                } else {
                                    var1 = -0.58853406;
                                }
                            }
                        } else {
                            if (input[50] < 13.0) {
                                var1 = 0.24933334;
                            } else {
                                var1 = 1.2839999;
                            }
                        }
                    }
                }
            }
        } else {
            var1 = -1.8180246;
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[46] < 3907.0) {
                if (input[28] < 0.77318484) {
                    if (input[62] < 1.2634743) {
                        var1 = -0.020751707;
                    } else {
                        var1 = -0.9826666;
                    }
                } else {
                    var1 = 0.59659857;
                }
            } else {
                if (input[0] < 616.4713) {
                    if (input[28] < 0.77318424) {
                        var1 = 0.23657143;
                    } else {
                        var1 = -0.44342867;
                    }
                } else {
                    var1 = -1.4626875;
                }
            }
        } else {
            if (input[1] < 13626.0) {
                if (input[24] < 768.0) {
                    if (input[31] < 15.0) {
                        var1 = 0.27185187;
                    } else {
                        var1 = 1.8066667;
                    }
                } else {
                    if (input[50] < 9.0) {
                        if (input[11] < 2.0) {
                            if (input[62] < 1.114576) {
                                var1 = 0.8168889;
                            } else {
                                var1 = -0.5768;
                            }
                        } else {
                            if (input[8] < 14.0) {
                                var1 = -0.2266666;
                            } else {
                                if (input[17] < 10.0) {
                                    var1 = -1.6175455;
                                } else {
                                    var1 = -0.5950455;
                                }
                            }
                        }
                    } else {
                        if (input[48] < 41.0) {
                            var1 = -0.011555557;
                        } else {
                            var1 = 1.3540319;
                        }
                    }
                }
            } else {
                if (input[61] < 0.039482284) {
                    if (input[60] < 64.0) {
                        if (input[5] < 0.17693326) {
                            if (input[25] < 141.0) {
                                if (input[61] < 0.020339133) {
                                    var1 = -0.7096025;
                                } else {
                                    if (input[0] < 0.4417933) {
                                        var1 = 0.16897605;
                                    } else {
                                        var1 = -0.34101865;
                                    }
                                }
                            } else {
                                var1 = 0.6566726;
                            }
                        } else {
                            if (input[0] < 0.47629344) {
                                if (input[62] < 1.0024548) {
                                    if (input[58] < 63.0) {
                                        if (input[28] < 0.18388015) {
                                            var1 = -0.71763694;
                                        } else {
                                            var1 = 0.12650366;
                                        }
                                    } else {
                                        if (input[27] < 137.0) {
                                            var1 = 1.1386951;
                                        } else {
                                            var1 = 0.2423631;
                                        }
                                    }
                                } else {
                                    if (input[23] < 0.4317968) {
                                        var1 = 0.0630847;
                                    } else {
                                        if (input[26] < 161.0) {
                                            var1 = 0.52972454;
                                        } else {
                                            var1 = 1.2843723;
                                        }
                                    }
                                }
                            } else {
                                if (input[1] < 1684993.0) {
                                    if (input[45] < 153459.0) {
                                        var1 = -0.8131326;
                                    } else {
                                        var1 = -0.24445716;
                                    }
                                } else {
                                    var1 = 0.5512;
                                }
                            }
                        }
                    } else {
                        if (input[23] < 0.4688595) {
                            if (input[56] < 26448.0) {
                                var1 = 0.82716733;
                            } else {
                                var1 = 1.78482;
                            }
                        } else {
                            if (input[62] < 1.0069038) {
                                var1 = 0.9301363;
                            } else {
                                var1 = 0.12220951;
                            }
                        }
                    }
                } else {
                    if (input[40] < 225.0) {
                        if (input[25] < 98.0) {
                            if (input[2] < 359.6314) {
                                if (input[25] < 62.0) {
                                    if (input[61] < 0.42481682) {
                                        if (input[3] < 165.06604) {
                                            var1 = 0.024461526;
                                        } else {
                                            var1 = 1.3250909;
                                        }
                                    } else {
                                        if (input[32] < 20.0) {
                                            var1 = 1.4655714;
                                        } else {
                                            var1 = 0.5097778;
                                        }
                                    }
                                } else {
                                    if (input[62] < 1.3975987) {
                                        if (input[20] < 51.0) {
                                            var1 = 1.5650367;
                                        } else {
                                            var1 = 0.42251277;
                                        }
                                    } else {
                                        if (input[3] < 133.11584) {
                                            var1 = 1.1;
                                        } else {
                                            var1 = -0.77555555;
                                        }
                                    }
                                }
                            } else {
                                var1 = -0.2949018;
                            }
                        } else {
                            if (input[46] < 5095.0) {
                                if (input[52] < 219151.0) {
                                    if (input[0] < 616.4713) {
                                        if (input[58] < 2463.0) {
                                            var1 = 0.7055427;
                                        } else {
                                            var1 = -0.14650224;
                                        }
                                    } else {
                                        if (input[46] < 2917.0) {
                                            var1 = 0.3472728;
                                        } else {
                                            var1 = -0.875117;
                                        }
                                    }
                                } else {
                                    if (input[56] < 50478.0) {
                                        if (input[29] < 5111.0) {
                                            var1 = 1.2037209;
                                        } else {
                                            var1 = 0.39468858;
                                        }
                                    } else {
                                        var1 = -0.51409525;
                                    }
                                }
                            } else {
                                if (input[28] < 0.8366955) {
                                    var1 = 0.81271046;
                                } else {
                                    var1 = 1.7106154;
                                }
                            }
                        }
                    } else {
                        if (input[4] < 139.0) {
                            if (input[4] < 138.0) {
                                if (input[6] < 5111.0) {
                                    if (input[37] < 149652.0) {
                                        if (input[46] < 4295.0) {
                                            var1 = 1.2358946;
                                        } else {
                                            var1 = 0.40992904;
                                        }
                                    } else {
                                        var1 = 1.6420902;
                                    }
                                } else {
                                    if (input[47] < 888.0) {
                                        var1 = 0.4193333;
                                    } else {
                                        if (input[58] < 4218.0) {
                                            var1 = 1.4565897;
                                        } else {
                                            var1 = 0.86855334;
                                        }
                                    }
                                }
                            } else {
                                var1 = 0.22835891;
                            }
                        } else {
                            if (input[54] < 142076.0) {
                                var1 = 1.8928111;
                            } else {
                                var1 = 0.67452306;
                            }
                        }
                    }
                }
            }
        }
    }
    double var2;
    if (input[61] < 0.0039452827) {
        if (input[48] < 44.0) {
            if (input[4] < 76.0) {
                if (input[52] < 85370.0) {
                    if (input[47] < 35.0) {
                        if (input[28] < 0.18433216) {
                            if (input[54] < 54601.0) {
                                var2 = -0.3648437;
                            } else {
                                var2 = 0.18532372;
                            }
                        } else {
                            var2 = 0.342648;
                        }
                    } else {
                        if (input[3] < 92.0) {
                            var2 = 1.0575572;
                        } else {
                            var2 = 0.25983575;
                        }
                    }
                } else {
                    if (input[4] < 50.0) {
                        var2 = 0.021079369;
                    } else {
                        var2 = -0.6428978;
                    }
                }
            } else {
                if (input[14] < 69718.0) {
                    if (input[56] < 33717.0) {
                        if (input[23] < 0.44908527) {
                            var2 = -0.73835176;
                        } else {
                            if (input[23] < 0.46054345) {
                                if (input[0] < 0.44448864) {
                                    var2 = 0.26783678;
                                } else {
                                    var2 = -0.40306622;
                                }
                            } else {
                                if (input[4] < 137.0) {
                                    var2 = -0.5742995;
                                } else {
                                    var2 = -0.1744613;
                                }
                            }
                        }
                    } else {
                        var2 = -1.0540334;
                    }
                } else {
                    if (input[54] < 128966.0) {
                        var2 = 0.75819236;
                    } else {
                        if (input[55] < 77925.0) {
                            var2 = -0.04822235;
                        } else {
                            var2 = -0.6137463;
                        }
                    }
                }
            }
        } else {
            var2 = -1.4910634;
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[62] < 1.0758849) {
                var2 = 0.31023508;
            } else {
                if (input[2] < 358.92572) {
                    if (input[57] < 38.0) {
                        var2 = -0.23155677;
                    } else {
                        if (input[61] < 0.28125697) {
                            var2 = -1.4564269;
                        } else {
                            if (input[46] < 3907.0) {
                                var2 = -0.19785382;
                            } else {
                                var2 = -1.0819663;
                            }
                        }
                    }
                } else {
                    if (input[58] < 3292.0) {
                        var2 = 0.16146031;
                    } else {
                        var2 = -0.36581886;
                    }
                }
            }
        } else {
            if (input[61] < 0.027702793) {
                if (input[37] < 48195.0) {
                    if (input[28] < 0.523297) {
                        var2 = -0.36411282;
                    } else {
                        var2 = -1.3502791;
                    }
                } else {
                    if (input[26] < 170.0) {
                        if (input[25] < 131.0) {
                            if (input[27] < 138.0) {
                                var2 = -0.46385065;
                            } else {
                                var2 = 0.006986936;
                            }
                        } else {
                            if (input[62] < 1.0014672) {
                                if (input[17] < 728.0) {
                                    if (input[59] < 76.0) {
                                        if (input[24] < 1314676.0) {
                                            var2 = -0.18899651;
                                        } else {
                                            var2 = 0.5986644;
                                        }
                                    } else {
                                        var2 = -0.406265;
                                    }
                                } else {
                                    var2 = 0.82279724;
                                }
                            } else {
                                if (input[56] < 70870.0) {
                                    if (input[62] < 1.0090925) {
                                        if (input[57] < 1271.0) {
                                            var2 = 0.3964068;
                                        } else {
                                            var2 = 1.6004647;
                                        }
                                    } else {
                                        if (input[61] < 0.01820975) {
                                            var2 = 1.1634909;
                                        } else {
                                            var2 = -0.0612216;
                                        }
                                    }
                                } else {
                                    var2 = -0.20655644;
                                }
                            }
                        }
                    } else {
                        if (input[39] < 16490.0) {
                            if (input[52] < 155912.0) {
                                if (input[3] < 165.0) {
                                    if (input[56] < 24838.0) {
                                        var2 = 0.49199176;
                                    } else {
                                        var2 = -0.26192573;
                                    }
                                } else {
                                    if (input[43] < 50.0) {
                                        if (input[1] < 1516752.0) {
                                            var2 = -0.44162923;
                                        } else {
                                            var2 = 0.14186135;
                                        }
                                    } else {
                                        if (input[19] < 62.0) {
                                            var2 = -0.2535122;
                                        } else {
                                            var2 = -0.6848964;
                                        }
                                    }
                                }
                            } else {
                                var2 = 0.6277096;
                            }
                        } else {
                            if (input[1] < 1442701.0) {
                                if (input[43] < 50.0) {
                                    var2 = 1.1093997;
                                } else {
                                    if (input[18] < 48.0) {
                                        var2 = 0.23397394;
                                    } else {
                                        var2 = -0.34421498;
                                    }
                                }
                            } else {
                                if (input[56] < 31155.0) {
                                    var2 = 1.2644632;
                                } else {
                                    var2 = 0.04248598;
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[40] < 18.0) {
                    if (input[62] < 1.1276892) {
                        if (input[56] < 19.0) {
                            if (input[3] < 0.6303728) {
                                if (input[0] < 10253.7295) {
                                    var2 = 0.7764935;
                                } else {
                                    var2 = -0.06142643;
                                }
                            } else {
                                var2 = -1.2578572;
                            }
                        } else {
                            if (input[17] < 6.0) {
                                if (input[29] < 182.0) {
                                    if (input[17] < 3.0) {
                                        var2 = 0.55580634;
                                    } else {
                                        var2 = 1.1811056;
                                    }
                                } else {
                                    var2 = -0.2745522;
                                }
                            } else {
                                var2 = 1.3942543;
                            }
                        }
                    } else {
                        if (input[8] < 21.0) {
                            if (input[8] < 14.0) {
                                var2 = -0.16633002;
                            } else {
                                var2 = -1.0800086;
                            }
                        } else {
                            if (input[28] < 0.77193576) {
                                if (input[61] < 0.3102258) {
                                    var2 = -0.48165178;
                                } else {
                                    if (input[10] < 16.0) {
                                        if (input[48] < 41.0) {
                                            var2 = 0.54537266;
                                        } else {
                                            var2 = 1.3530325;
                                        }
                                    } else {
                                        var2 = 0.27452636;
                                    }
                                }
                            } else {
                                if (input[61] < 0.19684868) {
                                    var2 = 0.790265;
                                } else {
                                    if (input[52] < 32103.0) {
                                        if (input[50] < 8.0) {
                                            var2 = 0.5693721;
                                        } else {
                                            var2 = -0.2086395;
                                        }
                                    } else {
                                        if (input[28] < 0.77193606) {
                                            var2 = -0.40347496;
                                        } else {
                                            var2 = -0.8708523;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[32] < 5.0) {
                        var2 = -1.0351409;
                    } else {
                        if (input[39] < 12128.0) {
                            if (input[52] < 263582.0) {
                                if (input[2] < 359.6314) {
                                    if (input[61] < 0.03894309) {
                                        if (input[35] < 60139.0) {
                                            var2 = -0.5484283;
                                        } else {
                                            var2 = 0.38687432;
                                        }
                                    } else {
                                        if (input[12] < 2496.0) {
                                            var2 = 0.0;
                                        } else {
                                            var2 = 0.87228423;
                                        }
                                    }
                                } else {
                                    var2 = -0.2432941;
                                }
                            } else {
                                var2 = -0.50227004;
                            }
                        } else {
                            if (input[17] < 901.0) {
                                if (input[16] < 16682.0) {
                                    if (input[2] < 130.0) {
                                        if (input[62] < 1.7374269) {
                                            var2 = 1.6132507;
                                        } else {
                                            var2 = 0.94236404;
                                        }
                                    } else {
                                        if (input[38] < 27372.0) {
                                            var2 = 0.03297035;
                                        } else {
                                            var2 = 0.8929097;
                                        }
                                    }
                                } else {
                                    if (input[61] < 0.03517758) {
                                        if (input[5] < 0.19046782) {
                                            var2 = 0.12165954;
                                        } else {
                                            var2 = -0.49689195;
                                        }
                                    } else {
                                        if (input[62] < 1.0022905) {
                                            var2 = 0.38603482;
                                        } else {
                                            var2 = 1.386607;
                                        }
                                    }
                                }
                            } else {
                                if (input[36] < 190576.0) {
                                    var2 = 1.6622975;
                                } else {
                                    if (input[37] < 149651.0) {
                                        var2 = 0.32824433;
                                    } else {
                                        var2 = 1.1265078;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var3;
    if (input[61] < 0.0039452827) {
        if (input[48] < 44.0) {
            if (input[4] < 76.0) {
                if (input[52] < 85370.0) {
                    if (input[47] < 35.0) {
                        if (input[18] < 18.0) {
                            if (input[2] < 59.0) {
                                var3 = 0.24599722;
                            } else {
                                var3 = 0.011640243;
                            }
                        } else {
                            var3 = -0.11975203;
                        }
                    } else {
                        if (input[3] < 92.0) {
                            var3 = 0.8812976;
                        } else {
                            var3 = 0.21652973;
                        }
                    }
                } else {
                    if (input[4] < 50.0) {
                        var3 = 0.017706681;
                    } else {
                        var3 = -0.5400342;
                    }
                }
            } else {
                if (input[12] < 69684.0) {
                    if (input[55] < 47193.0) {
                        var3 = -0.11275871;
                    } else {
                        if (input[35] < 74049.0) {
                            var3 = -0.8975908;
                        } else {
                            if (input[4] < 138.0) {
                                var3 = -0.54778403;
                            } else {
                                var3 = -0.09015815;
                            }
                        }
                    }
                } else {
                    if (input[50] < 13.0) {
                        var3 = 0.7242313;
                    } else {
                        if (input[40] < 692.0) {
                            var3 = -0.5061528;
                        } else {
                            var3 = -0.052113097;
                        }
                    }
                }
            }
        } else {
            if (input[5] < 0.48860183) {
                var3 = -0.6073109;
            } else {
                var3 = -1.4690022;
            }
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[46] < 3907.0) {
                if (input[28] < 0.77318484) {
                    if (input[50] < 8.0) {
                        var3 = -0.7603542;
                    } else {
                        var3 = 0.13276139;
                    }
                } else {
                    var3 = 0.5602457;
                }
            } else {
                if (input[0] < 616.4713) {
                    if (input[28] < 0.77318424) {
                        var3 = 0.16693538;
                    } else {
                        var3 = -0.38317344;
                    }
                } else {
                    if (input[58] < 2340.0) {
                        var3 = -0.5396907;
                    } else {
                        var3 = -1.1758715;
                    }
                }
            }
        } else {
            if (input[11] >= 2.0) {
                if (input[61] < 0.3247154) {
                    if (input[20] < 10.0) {
                        if (input[43] < 11.0) {
                            var3 = -1.1093844;
                        } else {
                            var3 = -0.3305043;
                        }
                    } else {
                        if (input[25] < 0.9351872) {
                            var3 = 0.6009361;
                        } else {
                            var3 = -0.4328572;
                        }
                    }
                } else {
                    if (input[5] < 0.45358318) {
                        var3 = -0.6385264;
                    } else {
                        if (input[24] < 15762.0) {
                            if (input[46] < 51.0) {
                                var3 = 0.27854607;
                            } else {
                                var3 = 1.2566555;
                            }
                        } else {
                            var3 = -0.088821106;
                        }
                    }
                }
            } else {
                if (input[61] < 0.027702793) {
                    if (input[56] < 63769.0) {
                        if (input[25] < 143.0) {
                            if (input[10] < 16.0) {
                                if (input[60] < 63.0) {
                                    if (input[59] < 74.0) {
                                        if (input[27] < 130.0) {
                                            var3 = -0.27588388;
                                        } else {
                                            var3 = 0.6728899;
                                        }
                                    } else {
                                        if (input[0] < 0.4718401) {
                                            var3 = -0.06662339;
                                        } else {
                                            var3 = -0.5608487;
                                        }
                                    }
                                } else {
                                    if (input[1] < 1432670.0) {
                                        if (input[40] < 749.0) {
                                            var3 = 0.37330732;
                                        } else {
                                            var3 = -0.44025728;
                                        }
                                    } else {
                                        if (input[62] < 1.0023754) {
                                            var3 = 0.23285864;
                                        } else {
                                            var3 = 0.98667985;
                                        }
                                    }
                                }
                            } else {
                                if (input[40] < 728.0) {
                                    var3 = -0.5711896;
                                } else {
                                    var3 = -0.08968754;
                                }
                            }
                        } else {
                            if (input[19] < 66.0) {
                                if (input[4] < 136.0) {
                                    var3 = 1.2220006;
                                } else {
                                    if (input[24] < 1509397.0) {
                                        var3 = -0.031073648;
                                    } else {
                                        if (input[3] < 168.0) {
                                            var3 = 0.34081084;
                                        } else {
                                            var3 = 1.0967122;
                                        }
                                    }
                                }
                            } else {
                                var3 = 0.08979851;
                            }
                        }
                    } else {
                        if (input[57] < 1427.0) {
                            if (input[38] < 39120.0) {
                                var3 = 0.3272793;
                            } else {
                                var3 = -0.34146443;
                            }
                        } else {
                            var3 = -0.7527998;
                        }
                    }
                } else {
                    if (input[39] < 12128.0) {
                        if (input[28] < 0.77193576) {
                            if (input[61] < 0.9591581) {
                                if (input[19] < 62.0) {
                                    if (input[30] < 127.0) {
                                        if (input[43] < 23.0) {
                                            var3 = 0.74401003;
                                        } else {
                                            var3 = 0.16520832;
                                        }
                                    } else {
                                        var3 = -0.600658;
                                    }
                                } else {
                                    if (input[2] < 359.6314) {
                                        if (input[31] < 27.0) {
                                            var3 = 0.0;
                                        } else {
                                            var3 = 1.020104;
                                        }
                                    } else {
                                        if (input[61] < 0.7320926) {
                                            var3 = -0.6856472;
                                        } else {
                                            var3 = 0.9543528;
                                        }
                                    }
                                }
                            } else {
                                if (input[7] < 1381.0) {
                                    var3 = -1.0023274;
                                } else {
                                    var3 = 0.9057847;
                                }
                            }
                        } else {
                            if (input[46] < 5019.0) {
                                if (input[46] < 3993.0) {
                                    if (input[55] < 95862.0) {
                                        if (input[28] < 0.8366985) {
                                            var3 = 0.17145605;
                                        } else {
                                            var3 = 1.0177262;
                                        }
                                    } else {
                                        var3 = -0.5038511;
                                    }
                                } else {
                                    if (input[39] < 12122.0) {
                                        if (input[2] < 359.6314) {
                                            var3 = -1.1562356;
                                        } else {
                                            var3 = -0.2908724;
                                        }
                                    } else {
                                        if (input[39] < 12124.0) {
                                            var3 = 0.47097895;
                                        } else {
                                            var3 = -0.48451176;
                                        }
                                    }
                                }
                            } else {
                                var3 = 1.0493946;
                            }
                        }
                    } else {
                        if (input[17] < 901.0) {
                            if (input[55] < 76689.0) {
                                if (input[62] < 1.0022942) {
                                    if (input[23] < 0.46322903) {
                                        var3 = -0.28177327;
                                    } else {
                                        var3 = 0.7968269;
                                    }
                                } else {
                                    if (input[38] < 27372.0) {
                                        var3 = 0.23105685;
                                    } else {
                                        if (input[25] < 356.3955) {
                                            var3 = 1.232375;
                                        } else {
                                            var3 = 0.76439804;
                                        }
                                    }
                                }
                            } else {
                                if (input[61] < 0.03517758) {
                                    if (input[61] < 0.034091685) {
                                        var3 = 0.09279572;
                                    } else {
                                        var3 = -0.4734052;
                                    }
                                } else {
                                    if (input[61] < 0.5613025) {
                                        if (input[12] < 90827.0) {
                                            var3 = 0.77778846;
                                        } else {
                                            var3 = 0.11014588;
                                        }
                                    } else {
                                        var3 = -0.0012562275;
                                    }
                                }
                            }
                        } else {
                            if (input[37] < 134230.0) {
                                var3 = 1.487193;
                            } else {
                                if (input[37] < 134232.0) {
                                    var3 = 0.46824977;
                                } else {
                                    if (input[58] < 5019.0) {
                                        var3 = 0.44427463;
                                    } else {
                                        var3 = 1.0909742;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var4;
    if (input[61] < 0.015582935) {
        if (input[40] < 138.0) {
            if (input[47] < 707.0) {
                var4 = -0.2351491;
            } else {
                var4 = -1.2625057;
            }
        } else {
            if (input[56] < 70870.0) {
                if (input[26] < 171.0) {
                    if (input[55] < 87868.0) {
                        if (input[0] < 0.45665354) {
                            if (input[57] < 1357.0) {
                                if (input[5] < 0.18959433) {
                                    if (input[50] < 14.0) {
                                        var4 = -0.45133707;
                                    } else {
                                        var4 = 0.0;
                                    }
                                } else {
                                    if (input[5] < 0.23291235) {
                                        if (input[40] < 614.0) {
                                            var4 = 0.714132;
                                        } else {
                                            var4 = 0.22185504;
                                        }
                                    } else {
                                        if (input[12] < 4370.0) {
                                            var4 = 0.18929623;
                                        } else {
                                            var4 = -0.2990476;
                                        }
                                    }
                                }
                            } else {
                                var4 = 0.4533975;
                            }
                        } else {
                            if (input[15] < 46370.0) {
                                if (input[57] < 4838.0) {
                                    var4 = 1.2065414;
                                } else {
                                    var4 = 0.37750238;
                                }
                            } else {
                                var4 = -0.13510107;
                            }
                        }
                    } else {
                        var4 = -0.3349411;
                    }
                } else {
                    if (input[54] < 144998.0) {
                        if (input[25] < 133.0) {
                            if (input[38] < 42233.0) {
                                var4 = 0.0;
                            } else {
                                var4 = 0.5154472;
                            }
                        } else {
                            if (input[61] < 0.003868324) {
                                if (input[18] < 18.0) {
                                    var4 = 0.09040194;
                                } else {
                                    if (input[37] < 77839.0) {
                                        if (input[46] < 39.0) {
                                            var4 = -0.10173161;
                                        } else {
                                            var4 = -0.59455454;
                                        }
                                    } else {
                                        if (input[35] < 80197.0) {
                                            var4 = 0.4110282;
                                        } else {
                                            var4 = -0.3498581;
                                        }
                                    }
                                }
                            } else {
                                if (input[2] < 136.0) {
                                    if (input[3] < 160.0) {
                                        var4 = 0.5762002;
                                    } else {
                                        var4 = -0.043460872;
                                    }
                                } else {
                                    if (input[1] < 1514581.0) {
                                        var4 = -0.4286027;
                                    } else {
                                        var4 = 0.13928337;
                                    }
                                }
                            }
                        }
                    } else {
                        var4 = 0.59771776;
                    }
                }
            } else {
                if (input[23] < 0.4674653) {
                    var4 = -0.21890132;
                } else {
                    var4 = -0.83529246;
                }
            }
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[56] < 5780.0) {
                if (input[61] < 0.37426257) {
                    var4 = -1.1661106;
                } else {
                    var4 = -0.47277266;
                }
            } else {
                if (input[46] < 3523.0) {
                    if (input[47] < 827.0) {
                        var4 = -0.30829132;
                    } else {
                        var4 = 0.71927994;
                    }
                } else {
                    if (input[48] < 125.0) {
                        if (input[28] < 0.77318525) {
                            var4 = -0.7868485;
                        } else {
                            var4 = -0.20043957;
                        }
                    } else {
                        var4 = 0.13531452;
                    }
                }
            }
        } else {
            if (input[18] < 23.0) {
                if (input[30] >= 105.0) {
                    if (input[20] < 9.0) {
                        if (input[23] < 73404.82) {
                            var4 = -0.45169783;
                        } else {
                            var4 = -1.0963771;
                        }
                    } else {
                        var4 = -0.10894759;
                    }
                } else {
                    if (input[48] < 47.0) {
                        if (input[10] >= 8.0) {
                            if (input[24] < 83418.0) {
                                if (input[5] < 0.46844152) {
                                    var4 = -1.0416132;
                                } else {
                                    var4 = -0.036271986;
                                }
                            } else {
                                var4 = 0.5441412;
                            }
                        } else {
                            if (input[6] < 30.0) {
                                if (input[30] < 53.0) {
                                    var4 = -0.70027286;
                                } else {
                                    var4 = 0.3281276;
                                }
                            } else {
                                if (input[5] < 0.6733632) {
                                    if (input[32] < 5.0) {
                                        var4 = 0.08659557;
                                    } else {
                                        if (input[62] < 1.0849594) {
                                            var4 = 0.5662418;
                                        } else {
                                            var4 = 1.0911479;
                                        }
                                    }
                                } else {
                                    var4 = -0.28869802;
                                }
                            }
                        }
                    } else {
                        var4 = 0.93379015;
                    }
                }
            } else {
                if (input[49] < 24.0) {
                    if (input[61] < 0.9591581) {
                        if (input[5] < 0.17899074) {
                            if (input[0] < 0.44617486) {
                                if (input[53] < 134954.0) {
                                    if (input[1] < 1319705.0) {
                                        if (input[25] < 130.0) {
                                            var4 = -0.5573443;
                                        } else {
                                            var4 = 0.37121558;
                                        }
                                    } else {
                                        if (input[61] < 0.38748303) {
                                            var4 = 1.0246786;
                                        } else {
                                            var4 = 0.17817165;
                                        }
                                    }
                                } else {
                                    if (input[1] < 1439241.0) {
                                        if (input[0] < 0.44209707) {
                                            var4 = -0.4713161;
                                        } else {
                                            var4 = -0.15922302;
                                        }
                                    } else {
                                        var4 = 0.28792617;
                                    }
                                }
                            } else {
                                if (input[2] < 145.0) {
                                    var4 = -0.86736584;
                                } else {
                                    var4 = 0.37093112;
                                }
                            }
                        } else {
                            if (input[29] < 5111.0) {
                                if (input[40] < 19.0) {
                                    if (input[58] < 2973.0) {
                                        if (input[62] < 1.1674256) {
                                            var4 = 1.0533694;
                                        } else {
                                            var4 = 0.22936039;
                                        }
                                    } else {
                                        if (input[54] < 159460.0) {
                                            var4 = -0.36282822;
                                        } else {
                                            var4 = 0.6581407;
                                        }
                                    }
                                } else {
                                    if (input[56] < 26558.0) {
                                        if (input[3] < 171.0) {
                                            var4 = 0.26804549;
                                        } else {
                                            var4 = 0.81169933;
                                        }
                                    } else {
                                        if (input[53] < 143411.0) {
                                            var4 = 1.0457257;
                                        } else {
                                            var4 = 0.6057622;
                                        }
                                    }
                                }
                            } else {
                                if (input[61] < 0.45386022) {
                                    if (input[54] < 63416.0) {
                                        var4 = -0.78844273;
                                    } else {
                                        var4 = -0.017163498;
                                    }
                                } else {
                                    var4 = 0.50739044;
                                }
                            }
                        }
                    } else {
                        if (input[58] < 2973.0) {
                            var4 = -0.85971874;
                        } else {
                            var4 = 0.52312624;
                        }
                    }
                } else {
                    if (input[52] < 62110.0) {
                        if (input[55] < 1895.0) {
                            var4 = 0.4112136;
                        } else {
                            if (input[57] < 57.0) {
                                var4 = 0.55217594;
                            } else {
                                var4 = 1.2679017;
                            }
                        }
                    } else {
                        var4 = 0.5649871;
                    }
                }
            }
        }
    }
    double var5;
    if (input[61] < 0.01980223) {
        if (input[40] < 138.0) {
            if (input[1] < 5116.0) {
                var5 = -0.08741128;
            } else {
                var5 = -1.0363731;
            }
        } else {
            if (input[26] < 171.0) {
                if (input[56] < 74801.0) {
                    if (input[55] < 82920.0) {
                        if (input[0] < 0.4601169) {
                            if (input[56] < 23493.0) {
                                if (input[56] < 20096.0) {
                                    if (input[40] < 558.0) {
                                        var5 = -0.111451946;
                                    } else {
                                        if (input[39] < 10631.0) {
                                            var5 = 0.5243381;
                                        } else {
                                            var5 = 0.16726671;
                                        }
                                    }
                                } else {
                                    if (input[28] < 0.18057792) {
                                        var5 = -0.45132264;
                                    } else {
                                        var5 = -0.057626247;
                                    }
                                }
                            } else {
                                if (input[28] < 0.18871462) {
                                    if (input[5] < 0.1800081) {
                                        var5 = 0.8895049;
                                    } else {
                                        if (input[40] < 661.0) {
                                            var5 = 0.0;
                                        } else {
                                            var5 = 0.39003107;
                                        }
                                    }
                                } else {
                                    var5 = -0.08314594;
                                }
                            }
                        } else {
                            if (input[0] < 0.47527987) {
                                var5 = 1.1039901;
                            } else {
                                var5 = 0.3292354;
                            }
                        }
                    } else {
                        if (input[26] < 165.06604) {
                            var5 = 0.3186483;
                        } else {
                            var5 = -0.48148543;
                        }
                    }
                } else {
                    var5 = -0.6396051;
                }
            } else {
                if (input[12] < 82775.0) {
                    if (input[58] < 51.0) {
                        if (input[1] < 58208.0) {
                            var5 = 0.0789067;
                        } else {
                            var5 = 0.5325535;
                        }
                    } else {
                        if (input[14] < 69527.0) {
                            if (input[28] < 0.19523168) {
                                if (input[23] < 0.44356433) {
                                    var5 = -0.60058117;
                                } else {
                                    if (input[27] < 139.0) {
                                        if (input[27] < 138.0) {
                                            var5 = -0.23175104;
                                        } else {
                                            var5 = 0.106606424;
                                        }
                                    } else {
                                        if (input[5] < 0.16930774) {
                                            var5 = -0.046101227;
                                        } else {
                                            var5 = -0.39288342;
                                        }
                                    }
                                }
                            } else {
                                var5 = 0.07431895;
                            }
                        } else {
                            if (input[2] < 133.0) {
                                var5 = 0.47526464;
                            } else {
                                if (input[43] < 52.0) {
                                    if (input[57] < 1535.0) {
                                        if (input[5] < 0.18779947) {
                                            var5 = -0.39629373;
                                        } else {
                                            var5 = 0.013782914;
                                        }
                                    } else {
                                        var5 = 0.101047516;
                                    }
                                } else {
                                    if (input[35] < 76764.0) {
                                        var5 = -0.13649744;
                                    } else {
                                        var5 = 0.39026603;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    var5 = 0.84448147;
                }
            }
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[62] < 1.0758849) {
                var5 = 0.26224273;
            } else {
                if (input[61] < 0.28125697) {
                    var5 = -1.0085304;
                } else {
                    if (input[62] < 1.3489773) {
                        var5 = 0.38615665;
                    } else {
                        if (input[0] < 616.4713) {
                            if (input[40] < 34.0) {
                                var5 = 0.31979242;
                            } else {
                                var5 = -0.39435297;
                            }
                        } else {
                            if (input[61] < 0.67920345) {
                                if (input[46] < 4547.0) {
                                    var5 = -0.71992904;
                                } else {
                                    var5 = 0.13940983;
                                }
                            } else {
                                var5 = -0.9230644;
                            }
                        }
                    }
                }
            }
        } else {
            if (input[18] < 23.0) {
                if (input[24] < 768.0) {
                    if (input[0] < 19802.795) {
                        if (input[31] >= 15.0) {
                            var5 = 1.0587256;
                        } else {
                            var5 = 0.44493443;
                        }
                    } else {
                        var5 = 0.046604004;
                    }
                } else {
                    if (input[47] < 54.0) {
                        if (input[40] < 119.0) {
                            if (input[3] < 0.877775) {
                                var5 = -0.29866657;
                            } else {
                                var5 = -1.0182452;
                            }
                        } else {
                            var5 = 0.0;
                        }
                    } else {
                        if (input[24] < 82814.0) {
                            if (input[5] < 0.38549224) {
                                var5 = -0.6892001;
                            } else {
                                if (input[43] < 12.0) {
                                    if (input[43] < 9.0) {
                                        if (input[49] < 9.0) {
                                            var5 = -0.23889838;
                                        } else {
                                            var5 = 0.42362267;
                                        }
                                    } else {
                                        var5 = -0.7878315;
                                    }
                                } else {
                                    if (input[57] < 3.0) {
                                        var5 = -0.012292137;
                                    } else {
                                        if (input[27] < 1.0) {
                                            var5 = 0.24788284;
                                        } else {
                                            var5 = 0.8151116;
                                        }
                                    }
                                }
                            }
                        } else {
                            var5 = 1.1371983;
                        }
                    }
                }
            } else {
                if (input[49] < 24.0) {
                    if (input[61] < 0.9591581) {
                        if (input[56] < 26060.0) {
                            if (input[35] < 5187.0) {
                                if (input[41] < 18.0) {
                                    if (input[55] < 46432.0) {
                                        if (input[16] < 11301.0) {
                                            var5 = 0.2542193;
                                        } else {
                                            var5 = 1.1851771;
                                        }
                                    } else {
                                        if (input[1] < 1522188.0) {
                                            var5 = -0.9513221;
                                        } else {
                                            var5 = 0.62891936;
                                        }
                                    }
                                } else {
                                    if (input[12] < 69035.0) {
                                        var5 = 0.13550702;
                                    } else {
                                        var5 = 1.3741713;
                                    }
                                }
                            } else {
                                if (input[39] < 1312.0) {
                                    if (input[0] < 0.4504612) {
                                        var5 = -0.91892433;
                                    } else {
                                        var5 = 0.23038642;
                                    }
                                } else {
                                    if (input[15] < 47980.0) {
                                        if (input[46] < 3975.0) {
                                            var5 = 0.39801916;
                                        } else {
                                            var5 = -0.175194;
                                        }
                                    } else {
                                        var5 = -0.6708866;
                                    }
                                }
                            }
                        } else {
                            if (input[52] < 135764.0) {
                                if (input[56] < 32385.0) {
                                    var5 = 1.1718614;
                                } else {
                                    var5 = 0.45288864;
                                }
                            } else {
                                if (input[40] < 19.0) {
                                    if (input[46] < 2999.0) {
                                        var5 = 0.6864763;
                                    } else {
                                        if (input[57] < 4917.0) {
                                            var5 = 0.21085791;
                                        } else {
                                            var5 = -0.5545112;
                                        }
                                    }
                                } else {
                                    if (input[29] < 5111.0) {
                                        if (input[0] < 0.45112288) {
                                            var5 = 0.0;
                                        } else {
                                            var5 = 0.5254068;
                                        }
                                    } else {
                                        if (input[52] < 263582.0) {
                                            var5 = 0.1798642;
                                        } else {
                                            var5 = -0.2963136;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[58] < 2973.0) {
                            var5 = -0.70688;
                        } else {
                            var5 = 0.43593845;
                        }
                    }
                } else {
                    if (input[62] < 1.3975987) {
                        if (input[52] < 22582.0) {
                            var5 = 1.1251011;
                        } else {
                            var5 = 0.5067814;
                        }
                    } else {
                        if (input[28] < 0.8008694) {
                            var5 = -0.016195985;
                        } else {
                            var5 = 0.73224974;
                        }
                    }
                }
            }
        }
    }
    double var6;
    if (input[29] >= 5111.0) {
        if (input[61] < 0.07137578) {
            var6 = -0.8980619;
        } else {
            if (input[28] < 0.77318484) {
                if (input[62] < 1.2634743) {
                    if (input[50] < 8.0) {
                        var6 = -0.29614225;
                    } else {
                        var6 = 0.2453668;
                    }
                } else {
                    if (input[46] < 5367.0) {
                        if (input[61] < 0.4721412) {
                            var6 = -0.8620682;
                        } else {
                            var6 = -0.41324097;
                        }
                    } else {
                        var6 = -0.1177802;
                    }
                }
            } else {
                if (input[48] < 92.0) {
                    if (input[47] < 929.0) {
                        var6 = 0.0;
                    } else {
                        var6 = 0.8849643;
                    }
                } else {
                    if (input[39] < 12122.0) {
                        if (input[28] < 0.7731854) {
                            var6 = -0.06801936;
                        } else {
                            var6 = -0.6174459;
                        }
                    } else {
                        if (input[46] < 4995.0) {
                            var6 = 0.0030477524;
                        } else {
                            var6 = 0.637811;
                        }
                    }
                }
            }
        }
    } else {
        if (input[61] < 0.039482284) {
            if (input[57] < 18.0) {
                var6 = -0.7209682;
            } else {
                if (input[5] < 0.17607194) {
                    if (input[37] < 74073.0) {
                        if (input[3] < 162.0) {
                            var6 = -0.4860992;
                        } else {
                            if (input[26] < 161.0) {
                                var6 = 0.23165277;
                            } else {
                                if (input[4] < 141.0) {
                                    if (input[57] < 1441.0) {
                                        if (input[0] < 0.44448864) {
                                            var6 = -0.26120815;
                                        } else {
                                            var6 = 0.033890456;
                                        }
                                    } else {
                                        var6 = -0.5001832;
                                    }
                                } else {
                                    var6 = 0.05174032;
                                }
                            }
                        }
                    } else {
                        if (input[23] < 0.465193) {
                            if (input[27] < 140.0) {
                                var6 = 0.69292426;
                            } else {
                                var6 = 0.21605325;
                            }
                        } else {
                            var6 = -0.32341534;
                        }
                    }
                } else {
                    if (input[61] < 0.0038832047) {
                        if (input[17] < 152.0) {
                            if (input[26] < 175.0) {
                                if (input[47] < 35.0) {
                                    if (input[56] < 13808.0) {
                                        var6 = -0.11945814;
                                    } else {
                                        var6 = 0.15978518;
                                    }
                                } else {
                                    if (input[26] < 159.0) {
                                        var6 = 0.5492575;
                                    } else {
                                        var6 = 0.064677365;
                                    }
                                }
                            } else {
                                var6 = -0.16343735;
                            }
                        } else {
                            if (input[5] < 0.23823364) {
                                if (input[17] < 559.0) {
                                    if (input[23] < 0.44105378) {
                                        var6 = -0.13023727;
                                    } else {
                                        var6 = 0.29698437;
                                    }
                                } else {
                                    if (input[25] < 138.0) {
                                        if (input[31] < 27.0) {
                                            var6 = -0.064993136;
                                        } else {
                                            var6 = -0.49888316;
                                        }
                                    } else {
                                        if (input[52] < 147688.0) {
                                            var6 = 0.082007185;
                                        } else {
                                            var6 = -0.13808082;
                                        }
                                    }
                                }
                            } else {
                                var6 = -0.5687409;
                            }
                        }
                    } else {
                        if (input[35] < 58574.0) {
                            var6 = -0.4020501;
                        } else {
                            if (input[62] < 1.0024548) {
                                if (input[62] < 0.9967484) {
                                    if (input[37] < 69684.0) {
                                        if (input[61] < 0.023940243) {
                                            var6 = 0.28118938;
                                        } else {
                                            var6 = -0.30378044;
                                        }
                                    } else {
                                        var6 = 0.90424585;
                                    }
                                } else {
                                    if (input[26] < 172.0) {
                                        if (input[60] < 64.0) {
                                            var6 = -0.35219452;
                                        } else {
                                            var6 = 0.09897219;
                                        }
                                    } else {
                                        if (input[28] < 0.18301424) {
                                            var6 = -0.26277494;
                                        } else {
                                            var6 = 0.45448285;
                                        }
                                    }
                                }
                            } else {
                                if (input[40] < 626.0) {
                                    if (input[37] < 68878.0) {
                                        var6 = 1.1620302;
                                    } else {
                                        var6 = 0.17488578;
                                    }
                                } else {
                                    if (input[3] < 171.0) {
                                        if (input[26] < 171.0) {
                                            var6 = 0.26773432;
                                        } else {
                                            var6 = -0.21170948;
                                        }
                                    } else {
                                        if (input[23] < 0.46795508) {
                                            var6 = 0.7730081;
                                        } else {
                                            var6 = 0.043307345;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[57] < 196.0) {
                if (input[61] < 0.37135702) {
                    if (input[62] < 1.1276892) {
                        if (input[5] < 0.48296398) {
                            if (input[7] < 27.0) {
                                var6 = 0.106775604;
                            } else {
                                var6 = -0.90685683;
                            }
                        } else {
                            if (input[26] < 0.7443333) {
                                if (input[2] < 0.71867603) {
                                    var6 = 0.27852586;
                                } else {
                                    var6 = -0.19433628;
                                }
                            } else {
                                if (input[60] < 8.0) {
                                    var6 = 0.9320022;
                                } else {
                                    var6 = 0.36180326;
                                }
                            }
                        }
                    } else {
                        if (input[49] < 23.0) {
                            if (input[43] < 12.0) {
                                var6 = -0.79812694;
                            } else {
                                var6 = -0.1901916;
                            }
                        } else {
                            var6 = 0.17908554;
                        }
                    }
                } else {
                    if (input[56] < 12705.0) {
                        if (input[61] < 0.9591581) {
                            if (input[8] < 21.0) {
                                if (input[46] < 77.0) {
                                    var6 = 0.28552863;
                                } else {
                                    var6 = -0.47978577;
                                }
                            } else {
                                if (input[37] < 22.0) {
                                    var6 = -0.13822943;
                                } else {
                                    if (input[27] < 43.0) {
                                        if (input[52] < 23467.0) {
                                            var6 = 0.75268275;
                                        } else {
                                            var6 = 0.4137662;
                                        }
                                    } else {
                                        var6 = 0.083010785;
                                    }
                                }
                            }
                        } else {
                            if (input[46] < 3233.0) {
                                var6 = -0.41946802;
                            } else {
                                var6 = 0.19532345;
                            }
                        }
                    } else {
                        if (input[61] < 0.7340231) {
                            if (input[56] < 13591.0) {
                                var6 = -0.9434309;
                            } else {
                                var6 = -0.0714125;
                            }
                        } else {
                            var6 = 0.24398346;
                        }
                    }
                }
            } else {
                if (input[42] < 14.0) {
                    var6 = -0.52639955;
                } else {
                    if (input[55] < 44869.0) {
                        if (input[10] < 14.0) {
                            if (input[33] < 9.0) {
                                var6 = 1.2990761;
                            } else {
                                if (input[61] < 0.4030146) {
                                    if (input[27] < 79.0) {
                                        var6 = 1.1065879;
                                    } else {
                                        if (input[58] < 51.0) {
                                            var6 = 0.0049432376;
                                        } else {
                                            var6 = 0.661223;
                                        }
                                    }
                                } else {
                                    var6 = 0.0010099685;
                                }
                            }
                        } else {
                            var6 = -0.029079465;
                        }
                    } else {
                        if (input[55] < 46888.0) {
                            var6 = -0.5771205;
                        } else {
                            if (input[25] < 52.0) {
                                var6 = -0.29034886;
                            } else {
                                if (input[40] < 19.0) {
                                    if (input[28] < 0.77193576) {
                                        var6 = 0.47343406;
                                    } else {
                                        if (input[61] < 0.28261825) {
                                            var6 = 0.37767994;
                                        } else {
                                            var6 = -0.48872837;
                                        }
                                    }
                                } else {
                                    if (input[15] < 38906.0) {
                                        if (input[25] < 132.0) {
                                            var6 = -0.39956507;
                                        } else {
                                            var6 = 0.42194143;
                                        }
                                    } else {
                                        if (input[57] < 4923.0) {
                                            var6 = 0.7913233;
                                        } else {
                                            var6 = 0.37547457;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var7;
    if (input[29] >= 5111.0) {
        if (input[61] < 0.07137578) {
            var7 = -0.7384065;
        } else {
            if (input[28] < 0.77318484) {
                if (input[52] < 38279.0) {
                    if (input[61] < 0.37426257) {
                        var7 = -0.728122;
                    } else {
                        var7 = -0.30768093;
                    }
                } else {
                    if (input[52] < 73258.0) {
                        var7 = 0.3275118;
                    } else {
                        if (input[52] < 195481.0) {
                            var7 = -0.55562645;
                        } else {
                            var7 = 0.0;
                        }
                    }
                }
            } else {
                if (input[57] < 200.0) {
                    if (input[23] < 329.86584) {
                        var7 = -0.054238725;
                    } else {
                        var7 = 0.78437555;
                    }
                } else {
                    if (input[57] < 3552.0) {
                        if (input[46] < 4547.0) {
                            var7 = -0.59746474;
                        } else {
                            var7 = -0.123762965;
                        }
                    } else {
                        if (input[52] < 270253.0) {
                            var7 = 0.44407424;
                        } else {
                            var7 = -0.22432755;
                        }
                    }
                }
            }
        }
    } else {
        if (input[61] < 0.039482284) {
            if (input[57] < 18.0) {
                var7 = -0.5947987;
            } else {
                if (input[60] < 64.0) {
                    if (input[16] < 25223.0) {
                        if (input[57] < 1578.0) {
                            if (input[55] < 74930.0) {
                                if (input[0] < 0.45123652) {
                                    if (input[0] < 0.44617486) {
                                        if (input[3] < 85.90858) {
                                            var7 = 0.19408263;
                                        } else {
                                            var7 = -0.015150427;
                                        }
                                    } else {
                                        if (input[62] < 1.005811) {
                                            var7 = -0.155029;
                                        } else {
                                            var7 = -0.5247034;
                                        }
                                    }
                                } else {
                                    if (input[23] < 0.4317968) {
                                        if (input[61] < 0.015647264) {
                                            var7 = 0.3438316;
                                        } else {
                                            var7 = -0.52930886;
                                        }
                                    } else {
                                        if (input[0] < 0.47318152) {
                                            var7 = 0.45988652;
                                        } else {
                                            var7 = -0.008243545;
                                        }
                                    }
                                }
                            } else {
                                if (input[4] < 138.0) {
                                    if (input[62] < 0.93992597) {
                                        var7 = 0.12758063;
                                    } else {
                                        if (input[17] < 728.0) {
                                            var7 = -0.4968114;
                                        } else {
                                            var7 = -0.09841076;
                                        }
                                    }
                                } else {
                                    if (input[45] < 142128.0) {
                                        if (input[52] < 152101.0) {
                                            var7 = -0.50946254;
                                        } else {
                                            var7 = -0.11143815;
                                        }
                                    } else {
                                        if (input[61] < 0.01980223) {
                                            var7 = -0.029573167;
                                        } else {
                                            var7 = 0.57597893;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[25] < 358.9257) {
                                var7 = 0.7046245;
                            } else {
                                var7 = -0.027375525;
                            }
                        }
                    } else {
                        var7 = -0.8178221;
                    }
                } else {
                    if (input[57] < 1393.0) {
                        if (input[56] < 24573.0) {
                            var7 = -0.11246548;
                        } else {
                            var7 = 0.89966816;
                        }
                    } else {
                        if (input[1] < 1492782.0) {
                            if (input[17] < 646.0) {
                                var7 = 0.010588722;
                            } else {
                                var7 = -0.2988691;
                            }
                        } else {
                            if (input[56] < 31155.0) {
                                if (input[57] < 1499.0) {
                                    var7 = 0.21806781;
                                } else {
                                    var7 = 0.83132595;
                                }
                            } else {
                                if (input[15] < 42379.0) {
                                    var7 = -0.21948066;
                                } else {
                                    var7 = 0.15041249;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[57] < 196.0) {
                if (input[57] < 150.0) {
                    if (input[48] < 89.0) {
                        if (input[60] < 22.0) {
                            if (input[10] >= 8.0) {
                                if (input[5] < 0.46844152) {
                                    var7 = -0.6957485;
                                } else {
                                    if (input[47] < 678.0) {
                                        if (input[17] < 3.0) {
                                            var7 = -0.058580972;
                                        } else {
                                            var7 = 0.6708649;
                                        }
                                    } else {
                                        if (input[48] < 74.0) {
                                            var7 = 0.1426014;
                                        } else {
                                            var7 = -0.63990563;
                                        }
                                    }
                                }
                            } else {
                                if (input[61] < 0.24051426) {
                                    if (input[20] < 6.0) {
                                        var7 = -0.50460875;
                                    } else {
                                        if (input[0] < 26533.898) {
                                            var7 = 0.38503036;
                                        } else {
                                            var7 = -0.15799668;
                                        }
                                    }
                                } else {
                                    if (input[62] < 1.2905569) {
                                        if (input[2] < 0.8122105) {
                                            var7 = 0.16772595;
                                        } else {
                                            var7 = 0.64546394;
                                        }
                                    } else {
                                        if (input[23] < 863.97253) {
                                            var7 = 0.050169345;
                                        } else {
                                            var7 = -0.27222437;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[18] < 24.0) {
                                var7 = 0.31579778;
                            } else {
                                var7 = 0.8964907;
                            }
                        }
                    } else {
                        if (input[62] < 1.2335619) {
                            var7 = 0.2553953;
                        } else {
                            if (input[28] < 0.77193546) {
                                var7 = 0.7728558;
                            } else {
                                var7 = 0.3368996;
                            }
                        }
                    }
                } else {
                    if (input[38] < 2626.0) {
                        if (input[4] < 50.0) {
                            var7 = -0.9002534;
                        } else {
                            var7 = -0.21774743;
                        }
                    } else {
                        if (input[46] < 4663.0) {
                            var7 = -0.39159426;
                        } else {
                            var7 = 0.9072744;
                        }
                    }
                }
            } else {
                if (input[42] < 14.0) {
                    var7 = -0.4386663;
                } else {
                    if (input[55] < 44869.0) {
                        if (input[10] < 14.0) {
                            if (input[33] < 9.0) {
                                if (input[2] < 121.0) {
                                    var7 = 0.5006049;
                                } else {
                                    var7 = 1.240798;
                                }
                            } else {
                                if (input[61] < 0.4030146) {
                                    if (input[27] < 79.0) {
                                        var7 = 0.90740204;
                                    } else {
                                        if (input[58] < 51.0) {
                                            var7 = 0.0041522216;
                                        } else {
                                            var7 = 0.55542743;
                                        }
                                    }
                                } else {
                                    var7 = 0.0008366721;
                                }
                            }
                        } else {
                            var7 = -0.024094418;
                        }
                    } else {
                        if (input[55] < 46888.0) {
                            var7 = -0.47818556;
                        } else {
                            if (input[25] < 52.0) {
                                var7 = -0.23953795;
                            } else {
                                if (input[2] < 140.0) {
                                    if (input[35] < 59194.0) {
                                        var7 = 0.8954121;
                                    } else {
                                        if (input[59] < 76.0) {
                                            var7 = -0.25333413;
                                        } else {
                                            var7 = 0.53832734;
                                        }
                                    }
                                } else {
                                    if (input[58] < 4991.0) {
                                        if (input[48] < 115.0) {
                                            var7 = 0.1647028;
                                        } else {
                                            var7 = 0.77099735;
                                        }
                                    } else {
                                        if (input[56] < 38448.0) {
                                            var7 = -0.026765583;
                                        } else {
                                            var7 = 0.61045593;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var8;
    if (input[61] < 0.007839571) {
        if (input[17] < 664.0) {
            if (input[8] < 27.0) {
                if (input[0] < 0.47165513) {
                    var8 = 0.03313915;
                } else {
                    var8 = 0.80380946;
                }
            } else {
                if (input[36] < 54649.0) {
                    if (input[3] < 75.0) {
                        var8 = -0.6063267;
                    } else {
                        var8 = -0.19102329;
                    }
                } else {
                    if (input[62] < 0.99280995) {
                        var8 = 0.47619984;
                    } else {
                        if (input[26] < 159.0) {
                            var8 = 0.36894187;
                        } else {
                            if (input[39] < 11808.0) {
                                var8 = 0.28682557;
                            } else {
                                if (input[9] < 18.0) {
                                    var8 = 0.17494221;
                                } else {
                                    if (input[56] < 17688.0) {
                                        var8 = 0.03929783;
                                    } else {
                                        if (input[5] < 0.18125793) {
                                            var8 = 0.0;
                                        } else {
                                            var8 = -0.2206028;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[27] < 134.0) {
                var8 = -0.7752523;
            } else {
                if (input[37] < 60976.0) {
                    var8 = 0.074698;
                } else {
                    if (input[55] < 84912.0) {
                        if (input[62] < 1.0070169) {
                            if (input[50] < 15.0) {
                                var8 = -0.15463196;
                            } else {
                                var8 = 0.03181929;
                            }
                        } else {
                            if (input[0] < 0.46458045) {
                                var8 = -0.35080972;
                            } else {
                                var8 = -0.07245819;
                            }
                        }
                    } else {
                        var8 = -0.42193863;
                    }
                }
            }
        }
    } else {
        if (input[29] >= 5551.0) {
            if (input[46] < 3907.0) {
                if (input[47] < 827.0) {
                    var8 = -0.28089964;
                } else {
                    if (input[28] < 0.7731844) {
                        var8 = -0.11330457;
                    } else {
                        var8 = 0.772053;
                    }
                }
            } else {
                if (input[0] < 616.4713) {
                    if (input[28] < 0.77318424) {
                        var8 = 0.1433173;
                    } else {
                        var8 = -0.27842018;
                    }
                } else {
                    if (input[58] < 2340.0) {
                        var8 = -0.069499664;
                    } else {
                        if (input[58] < 5393.0) {
                            var8 = -0.8197547;
                        } else {
                            var8 = -0.16296832;
                        }
                    }
                }
            }
        } else {
            if (input[54] < 4425.0) {
                if (input[59] < 29.0) {
                    if (input[61] < 0.3247154) {
                        if (input[5] < 0.38549224) {
                            var8 = -0.571456;
                        } else {
                            if (input[40] < 2.0) {
                                var8 = -0.36413798;
                            } else {
                                if (input[47] < 136.0) {
                                    if (input[14] < 42.0) {
                                        var8 = 0.0031210661;
                                    } else {
                                        var8 = 0.5733773;
                                    }
                                } else {
                                    var8 = -0.17199974;
                                }
                            }
                        }
                    } else {
                        if (input[60] < 18.0) {
                            if (input[24] < 768.0) {
                                if (input[1] < 1212.0) {
                                    var8 = 0.23247483;
                                } else {
                                    var8 = 0.64570946;
                                }
                            } else {
                                if (input[48] < 43.0) {
                                    if (input[34] >= 2.0) {
                                        if (input[2] < 0.6000889) {
                                            var8 = -0.08360546;
                                        } else {
                                            var8 = -0.43198815;
                                        }
                                    } else {
                                        var8 = 0.20710634;
                                    }
                                } else {
                                    var8 = 0.46075875;
                                }
                            }
                        } else {
                            var8 = 0.6473655;
                        }
                    }
                } else {
                    var8 = -0.68782026;
                }
            } else {
                if (input[35] < 5187.0) {
                    if (input[26] < 88.0) {
                        if (input[28] < 0.33327162) {
                            if (input[33] < 9.0) {
                                if (input[14] < 72176.0) {
                                    var8 = 0.8927555;
                                } else {
                                    var8 = 0.23768793;
                                }
                            } else {
                                if (input[25] < 52.0) {
                                    var8 = -0.5367775;
                                } else {
                                    if (input[17] < 664.0) {
                                        var8 = 0.2091769;
                                    } else {
                                        var8 = 0.73644024;
                                    }
                                }
                            }
                        } else {
                            var8 = -0.48801765;
                        }
                    } else {
                        var8 = 0.81198466;
                    }
                } else {
                    if (input[42] < 30.0) {
                        if (input[0] < 0.4504612) {
                            var8 = -0.78021723;
                        } else {
                            var8 = 0.11403252;
                        }
                    } else {
                        if (input[38] < 77843.0) {
                            if (input[61] < 0.9591581) {
                                if (input[61] < 0.7340231) {
                                    if (input[48] < 77.0) {
                                        if (input[56] < 39284.0) {
                                            var8 = 0.20280936;
                                        } else {
                                            var8 = -0.015579777;
                                        }
                                    } else {
                                        if (input[39] < 12122.0) {
                                            var8 = -0.5401882;
                                        } else {
                                            var8 = 0.11202252;
                                        }
                                    }
                                } else {
                                    if (input[23] < 42.103584) {
                                        var8 = -0.58819056;
                                    } else {
                                        if (input[40] < 19720.0) {
                                            var8 = 0.58846587;
                                        } else {
                                            var8 = -0.106088184;
                                        }
                                    }
                                }
                            } else {
                                if (input[7] < 1381.0) {
                                    if (input[61] < 1.0123856) {
                                        var8 = -0.5759526;
                                    } else {
                                        var8 = -0.17088228;
                                    }
                                } else {
                                    var8 = 0.39462134;
                                }
                            }
                        } else {
                            if (input[47] < 1486.0) {
                                if (input[62] < 1.4662857) {
                                    if (input[55] < 94413.0) {
                                        var8 = 0.54808396;
                                    } else {
                                        var8 = 0.009889794;
                                    }
                                } else {
                                    var8 = 0.75163037;
                                }
                            } else {
                                if (input[47] < 1565.0) {
                                    var8 = -0.39620668;
                                } else {
                                    var8 = 0.38841468;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var9;
    if (input[61] < 0.039482284) {
        if (input[35] < 58574.0) {
            if (input[31] < 27.0) {
                var9 = 0.19434582;
            } else {
                if (input[23] < 0.40789166) {
                    var9 = 0.14203675;
                } else {
                    var9 = -0.5123784;
                }
            }
        } else {
            if (input[39] < 10631.0) {
                if (input[1] < 1308628.0) {
                    var9 = 0.036134496;
                } else {
                    var9 = 0.9272577;
                }
            } else {
                if (input[23] < 0.44356433) {
                    if (input[61] < 0.02317072) {
                        if (input[42] < 61.0) {
                            if (input[5] < 0.18157138) {
                                var9 = -0.060701717;
                            } else {
                                var9 = 0.42009035;
                            }
                        } else {
                            if (input[17] < 664.0) {
                                if (input[55] < 59367.0) {
                                    var9 = -0.18562692;
                                } else {
                                    var9 = -0.49115974;
                                }
                            } else {
                                if (input[56] < 26060.0) {
                                    var9 = 0.010851956;
                                } else {
                                    var9 = -0.220727;
                                }
                            }
                        }
                    } else {
                        var9 = -0.48900366;
                    }
                } else {
                    if (input[58] < 64.0) {
                        if (input[50] < 15.0) {
                            if (input[41] < 49.0) {
                                if (input[0] < 0.47527987) {
                                    if (input[0] < 0.44979313) {
                                        if (input[14] < 57188.0) {
                                            var9 = 0.25605997;
                                        } else {
                                            var9 = -0.07240858;
                                        }
                                    } else {
                                        if (input[17] < 646.0) {
                                            var9 = 0.3390268;
                                        } else {
                                            var9 = 0.8951313;
                                        }
                                    }
                                } else {
                                    if (input[52] < 154389.0) {
                                        var9 = -0.45120057;
                                    } else {
                                        var9 = 0.09029316;
                                    }
                                }
                            } else {
                                if (input[61] < 0.019063905) {
                                    if (input[61] < 0.009615385) {
                                        if (input[8] < 27.0) {
                                            var9 = 0.32774743;
                                        } else {
                                            var9 = -0.0877064;
                                        }
                                    } else {
                                        if (input[26] < 171.0) {
                                            var9 = 0.64992774;
                                        } else {
                                            var9 = 0.07852591;
                                        }
                                    }
                                } else {
                                    if (input[62] < 0.9981704) {
                                        var9 = 0.19267242;
                                    } else {
                                        if (input[56] < 29740.0) {
                                            var9 = -0.4449831;
                                        } else {
                                            var9 = 0.010976192;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[28] < 0.18205085) {
                                if (input[0] < 0.4601169) {
                                    if (input[5] < 0.17791565) {
                                        var9 = 0.047039066;
                                    } else {
                                        var9 = -0.3654515;
                                    }
                                } else {
                                    var9 = 0.24593237;
                                }
                            } else {
                                if (input[40] < 732.0) {
                                    var9 = -0.39719766;
                                } else {
                                    if (input[28] < 0.18844408) {
                                        var9 = -0.16805947;
                                    } else {
                                        var9 = 0.0847308;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[38] < 45014.0) {
                            if (input[3] < 165.0) {
                                var9 = 0.050911363;
                            } else {
                                if (input[38] < 42103.0) {
                                    var9 = 0.42767602;
                                } else {
                                    var9 = 0.98139775;
                                }
                            }
                        } else {
                            if (input[26] < 165.06604) {
                                var9 = 0.38386425;
                            } else {
                                if (input[50] < 13.0) {
                                    if (input[14] < 59728.0) {
                                        var9 = -0.056293745;
                                    } else {
                                        var9 = -0.38434234;
                                    }
                                } else {
                                    if (input[61] < 0.015493562) {
                                        var9 = -0.15926464;
                                    } else {
                                        var9 = 0.63842285;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[3] < 169.0) {
            if (input[29] >= 5551.0) {
                if (input[62] < 1.0758849) {
                    var9 = 0.21974663;
                } else {
                    if (input[61] < 0.28125697) {
                        var9 = -0.61120695;
                    } else {
                        if (input[58] < 2340.0) {
                            if (input[57] < 39.0) {
                                var9 = 0.6365613;
                            } else {
                                var9 = -0.3052437;
                            }
                        } else {
                            if (input[61] < 0.37616688) {
                                var9 = 0.22277546;
                            } else {
                                if (input[0] < 616.4713) {
                                    var9 = 0.0;
                                } else {
                                    if (input[58] < 5393.0) {
                                        var9 = -0.6467709;
                                    } else {
                                        var9 = -0.13689339;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[59] < 1564.0) {
                    if (input[49] < 24.0) {
                        if (input[61] < 0.4696803) {
                            if (input[62] < 1.1276892) {
                                if (input[0] < 0.43404076) {
                                    var9 = -0.3853474;
                                } else {
                                    if (input[40] < 109.0) {
                                        if (input[14] < 42.0) {
                                            var9 = -0.07245659;
                                        } else {
                                            var9 = 0.44071838;
                                        }
                                    } else {
                                        if (input[28] < 0.19285144) {
                                            var9 = 0.40349945;
                                        } else {
                                            var9 = -0.17925975;
                                        }
                                    }
                                }
                            } else {
                                if (input[40] < 147.0) {
                                    if (input[25] < 0.5924962) {
                                        var9 = 0.2574426;
                                    } else {
                                        if (input[52] < 86940.0) {
                                            var9 = -0.545806;
                                        } else {
                                            var9 = -0.020025596;
                                        }
                                    }
                                } else {
                                    if (input[61] < 0.1869674) {
                                        var9 = 0.50395495;
                                    } else {
                                        if (input[62] < 1.1931204) {
                                            var9 = -0.2837024;
                                        } else {
                                            var9 = 0.13121027;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[0] < 0.4732238) {
                                var9 = 0.86250335;
                            } else {
                                if (input[61] < 0.9591581) {
                                    if (input[62] < 1.7983793) {
                                        if (input[62] < 1.5454686) {
                                            var9 = 0.22789769;
                                        } else {
                                            var9 = -0.15808587;
                                        }
                                    } else {
                                        if (input[40] < 19720.0) {
                                            var9 = 0.5383868;
                                        } else {
                                            var9 = -0.089114;
                                        }
                                    }
                                } else {
                                    if (input[58] < 3849.0) {
                                        var9 = -0.41320783;
                                    } else {
                                        var9 = 0.25284767;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[29] >= 36.0) {
                            if (input[48] < 37.0) {
                                var9 = 0.75477475;
                            } else {
                                if (input[62] < 1.0772302) {
                                    var9 = -0.05229788;
                                } else {
                                    if (input[52] < 62110.0) {
                                        if (input[24] < 60356.0) {
                                            var9 = 0.12015686;
                                        } else {
                                            var9 = 0.49793324;
                                        }
                                    } else {
                                        if (input[35] < 221882.0) {
                                            var9 = 0.0;
                                        } else {
                                            var9 = 0.26322785;
                                        }
                                    }
                                }
                            }
                        } else {
                            var9 = -0.091184996;
                        }
                    }
                } else {
                    if (input[58] < 4881.0) {
                        if (input[47] < 1311.0) {
                            if (input[58] < 4569.0) {
                                var9 = 0.11453102;
                            } else {
                                var9 = -0.20937927;
                            }
                        } else {
                            var9 = 0.43944153;
                        }
                    } else {
                        if (input[47] < 1008.0) {
                            if (input[28] < 0.8554572) {
                                var9 = 0.0;
                            } else {
                                var9 = 0.6143676;
                            }
                        } else {
                            var9 = 0.7765079;
                        }
                    }
                }
            }
        } else {
            if (input[62] < 1.1533407) {
                if (input[55] < 82920.0) {
                    if (input[2] < 145.0) {
                        if (input[32] < 25.0) {
                            var9 = 0.81432074;
                        } else {
                            var9 = 0.23656909;
                        }
                    } else {
                        var9 = 0.12388053;
                    }
                } else {
                    var9 = -0.038599014;
                }
            } else {
                if (input[22] < 154818.0) {
                    var9 = -0.68799084;
                } else {
                    var9 = 0.3487514;
                }
            }
        }
    }
    double var10;
    if (input[61] < 0.020339133) {
        if (input[16] < 19797.0) {
            if (input[12] < 82775.0) {
                if (input[2] < 136.0) {
                    if (input[61] < 0.0038832047) {
                        if (input[4] < 76.0) {
                            if (input[4] < 73.0) {
                                if (input[32] < 19.0) {
                                    if (input[26] < 144.0) {
                                        var10 = 0.40823364;
                                    } else {
                                        if (input[47] < 33.0) {
                                            var10 = 0.027811078;
                                        } else {
                                            var10 = -0.25282034;
                                        }
                                    }
                                } else {
                                    var10 = -0.33807504;
                                }
                            } else {
                                var10 = 0.3536055;
                            }
                        } else {
                            if (input[38] < 41977.0) {
                                if (input[59] < 77.0) {
                                    if (input[12] < 6663.0) {
                                        var10 = -0.12875862;
                                    } else {
                                        var10 = -0.45521584;
                                    }
                                } else {
                                    var10 = -0.06475102;
                                }
                            } else {
                                if (input[38] < 45311.0) {
                                    var10 = 0.18130036;
                                } else {
                                    if (input[15] < 36975.0) {
                                        var10 = 0.014089051;
                                    } else {
                                        var10 = -0.19095166;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[39] < 9918.0) {
                            var10 = -0.2615129;
                        } else {
                            if (input[56] < 25570.0) {
                                if (input[0] < 0.4652293) {
                                    if (input[42] < 62.0) {
                                        if (input[57] < 1304.0) {
                                            var10 = 0.1466559;
                                        } else {
                                            var10 = 0.6626062;
                                        }
                                    } else {
                                        var10 = -0.10157883;
                                    }
                                } else {
                                    var10 = 0.727644;
                                }
                            } else {
                                if (input[0] < 0.46316132) {
                                    if (input[5] < 0.18278617) {
                                        if (input[37] < 66058.0) {
                                            var10 = 0.16759382;
                                        } else {
                                            var10 = -0.06284698;
                                        }
                                    } else {
                                        var10 = 0.3209829;
                                    }
                                } else {
                                    if (input[56] < 30159.0) {
                                        var10 = -0.26743278;
                                    } else {
                                        var10 = 0.0058921813;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[57] < 1338.0) {
                        if (input[37] < 70435.0) {
                            var10 = -0.38537896;
                        } else {
                            var10 = -0.0855813;
                        }
                    } else {
                        if (input[26] < 167.0) {
                            if (input[50] < 13.0) {
                                var10 = 0.32735056;
                            } else {
                                var10 = -0.063134275;
                            }
                        } else {
                            if (input[5] < 0.17933306) {
                                var10 = -0.25985762;
                            } else {
                                if (input[40] < 759.0) {
                                    if (input[62] < 1.0020952) {
                                        var10 = -0.11625053;
                                    } else {
                                        if (input[5] < 0.18341935) {
                                            var10 = 0.46440378;
                                        } else {
                                            var10 = 0.018235931;
                                        }
                                    }
                                } else {
                                    if (input[20] < 51.0) {
                                        var10 = -0.20218305;
                                    } else {
                                        var10 = 0.024251899;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                var10 = 0.50488204;
            }
        } else {
            var10 = -0.4075763;
        }
    } else {
        if (input[39] < 12128.0) {
            if (input[28] < 0.77193576) {
                if (input[27] < 132.0) {
                    if (input[26] < 88.0) {
                        if (input[61] < 0.36444244) {
                            if (input[5] < 0.38549224) {
                                if (input[61] < 0.30205673) {
                                    if (input[61] < 0.07137578) {
                                        var10 = -0.526176;
                                    } else {
                                        var10 = -0.058614768;
                                    }
                                } else {
                                    var10 = -0.7847975;
                                }
                            } else {
                                if (input[48] < 14.0) {
                                    var10 = 0.41363525;
                                } else {
                                    if (input[62] < 1.1199349) {
                                        if (input[56] < 18.0) {
                                            var10 = -0.3807468;
                                        } else {
                                            var10 = 0.21430385;
                                        }
                                    } else {
                                        var10 = -0.45332775;
                                    }
                                }
                            }
                        } else {
                            if (input[54] < 159460.0) {
                                if (input[1] < 59383372.0) {
                                    if (input[61] < 0.99021393) {
                                        if (input[40] < 131.0) {
                                            var10 = 0.20267224;
                                        } else {
                                            var10 = -0.17349447;
                                        }
                                    } else {
                                        var10 = -0.2872305;
                                    }
                                } else {
                                    var10 = -0.50778323;
                                }
                            } else {
                                var10 = 0.6646408;
                            }
                        }
                    } else {
                        if (input[16] < 16347.0) {
                            if (input[14] < 72202.0) {
                                if (input[50] < 14.0) {
                                    if (input[28] < 0.27311435) {
                                        if (input[59] < 63.0) {
                                            var10 = -0.311221;
                                        } else {
                                            var10 = 0.30562598;
                                        }
                                    } else {
                                        if (input[0] < 0.46365568) {
                                            var10 = 0.5589291;
                                        } else {
                                            var10 = 0.08860394;
                                        }
                                    }
                                } else {
                                    var10 = 0.7463482;
                                }
                            } else {
                                var10 = -0.1833004;
                            }
                        } else {
                            if (input[62] < 1.0024548) {
                                var10 = 0.2533139;
                            } else {
                                var10 = 0.8085166;
                            }
                        }
                    }
                } else {
                    if (input[56] < 25058.0) {
                        var10 = -0.45313296;
                    } else {
                        if (input[56] < 28868.0) {
                            var10 = 0.31429526;
                        } else {
                            var10 = -0.14305545;
                        }
                    }
                }
            } else {
                if (input[62] < 1.0809538) {
                    if (input[48] < 98.0) {
                        if (input[46] < 2897.0) {
                            var10 = 0.16007778;
                        } else {
                            var10 = 0.7989698;
                        }
                    } else {
                        var10 = -0.2548393;
                    }
                } else {
                    if (input[28] < 0.8366986) {
                        if (input[48] < 55.0) {
                            if (input[47] < 849.0) {
                                var10 = -0.18267815;
                            } else {
                                var10 = -0.773081;
                            }
                        } else {
                            if (input[47] < 822.0) {
                                var10 = -0.67486477;
                            } else {
                                if (input[47] < 871.0) {
                                    var10 = 0.6807402;
                                } else {
                                    if (input[46] < 4559.0) {
                                        if (input[28] < 0.8366955) {
                                            var10 = -0.058888204;
                                        } else {
                                            var10 = 0.38520724;
                                        }
                                    } else {
                                        if (input[46] < 5095.0) {
                                            var10 = -0.6055259;
                                        } else {
                                            var10 = 0.02437355;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        var10 = 0.31777632;
                    }
                }
            }
        } else {
            if (input[17] < 901.0) {
                if (input[50] < 13.0) {
                    if (input[55] < 76689.0) {
                        if (input[62] < 1.0100635) {
                            if (input[5] < 0.18443236) {
                                var10 = -0.2999138;
                            } else {
                                var10 = 0.0959129;
                            }
                        } else {
                            if (input[28] < 0.85545856) {
                                if (input[25] < 356.3955) {
                                    if (input[55] < 61704.0) {
                                        if (input[55] < 51605.0) {
                                            var10 = 0.38619816;
                                        } else {
                                            var10 = -0.031344768;
                                        }
                                    } else {
                                        if (input[62] < 1.0194882) {
                                            var10 = 0.26766372;
                                        } else {
                                            var10 = 0.6746888;
                                        }
                                    }
                                } else {
                                    if (input[46] < 4263.0) {
                                        if (input[52] < 188490.0) {
                                            var10 = -0.44543114;
                                        } else {
                                            var10 = 0.17579275;
                                        }
                                    } else {
                                        if (input[58] < 5127.0) {
                                            var10 = 0.33337578;
                                        } else {
                                            var10 = -0.025260543;
                                        }
                                    }
                                }
                            } else {
                                var10 = 0.7381198;
                            }
                        }
                    } else {
                        if (input[55] < 85922.0) {
                            if (input[0] < 0.4601169) {
                                var10 = 0.14869782;
                            } else {
                                if (input[54] < 164394.0) {
                                    if (input[55] < 77749.0) {
                                        var10 = -0.05242432;
                                    } else {
                                        if (input[15] < 52007.0) {
                                            var10 = -0.5266438;
                                        } else {
                                            var10 = -0.14571683;
                                        }
                                    }
                                } else {
                                    var10 = 0.18848366;
                                }
                            }
                        } else {
                            if (input[17] < 773.0) {
                                if (input[40] < 19717.0) {
                                    if (input[14] < 64370.0) {
                                        if (input[62] < 1.0097301) {
                                            var10 = 0.8102209;
                                        } else {
                                            var10 = 0.28418958;
                                        }
                                    } else {
                                        var10 = 0.0;
                                    }
                                } else {
                                    var10 = -0.13826384;
                                }
                            } else {
                                var10 = -0.19849244;
                            }
                        }
                    }
                } else {
                    if (input[22] < 151050.0) {
                        if (input[61] < 0.038127046) {
                            if (input[25] < 131.0) {
                                var10 = -0.26805636;
                            } else {
                                var10 = 0.06369693;
                            }
                        } else {
                            var10 = 0.5186019;
                        }
                    } else {
                        var10 = 0.75172013;
                    }
                }
            } else {
                if (input[36] < 190576.0) {
                    var10 = 0.60867274;
                } else {
                    if (input[58] < 5770.0) {
                        var10 = -0.06804734;
                    } else {
                        var10 = 0.3628596;
                    }
                }
            }
        }
    }
    double var11;
    if (input[38] < 77843.0) {
        if (input[61] < 0.0038832047) {
            if (input[48] < 44.0) {
                if (input[25] < 132.0) {
                    if (input[62] < 1.0097301) {
                        var11 = 0.4523843;
                    } else {
                        var11 = 0.04950641;
                    }
                } else {
                    if (input[12] < 4370.0) {
                        if (input[6] < 102.0) {
                            var11 = 0.10077008;
                        } else {
                            var11 = -0.13443518;
                        }
                    } else {
                        if (input[25] < 135.0) {
                            var11 = -0.4128273;
                        } else {
                            if (input[62] < 1.1068292) {
                                if (input[17] < 639.0) {
                                    if (input[23] < 0.44105378) {
                                        var11 = -0.1783376;
                                    } else {
                                        if (input[55] < 82920.0) {
                                            var11 = 0.20500207;
                                        } else {
                                            var11 = -0.030840188;
                                        }
                                    }
                                } else {
                                    if (input[61] < 0.0031478927) {
                                        if (input[62] < 1.0039968) {
                                            var11 = -0.072663605;
                                        } else {
                                            var11 = -0.28888732;
                                        }
                                    } else {
                                        var11 = -0.007977987;
                                    }
                                }
                            } else {
                                var11 = -0.23696733;
                            }
                        }
                    }
                }
            } else {
                var11 = -0.38686344;
            }
        } else {
            if (input[30] < 1380.0) {
                if (input[8] < 25.0) {
                    if (input[15] < 59654.0) {
                        if (input[58] < 12.0) {
                            if (input[56] < 19.0) {
                                if (input[61] < 0.36444244) {
                                    var11 = -0.3481497;
                                } else {
                                    var11 = 0.08383896;
                                }
                            } else {
                                if (input[25] < 0.630451) {
                                    var11 = 0.41763744;
                                } else {
                                    var11 = 0.0;
                                }
                            }
                        } else {
                            if (input[23] < 0.44365835) {
                                var11 = 0.28171152;
                            } else {
                                if (input[16] < 28.0) {
                                    if (input[6] < 38.0) {
                                        var11 = -0.31836018;
                                    } else {
                                        var11 = 0.053548463;
                                    }
                                } else {
                                    if (input[62] < 1.0024548) {
                                        var11 = -0.15966313;
                                    } else {
                                        var11 = -0.5175762;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[27] < 139.0) {
                            var11 = -0.032492217;
                        } else {
                            var11 = 0.57849056;
                        }
                    }
                } else {
                    if (input[56] < 50478.0) {
                        if (input[52] < 256459.0) {
                            if (input[62] < 0.9939286) {
                                if (input[14] < 65068.0) {
                                    var11 = -0.036211204;
                                } else {
                                    if (input[1] < 1507594.0) {
                                        if (input[4] < 141.0) {
                                            var11 = 0.6343817;
                                        } else {
                                            var11 = 0.25368607;
                                        }
                                    } else {
                                        var11 = 0.06319984;
                                    }
                                }
                            } else {
                                if (input[62] < 1.0022942) {
                                    if (input[24] < 1177144.0) {
                                        var11 = 0.3854811;
                                    } else {
                                        if (input[57] < 1378.0) {
                                            var11 = -0.31521833;
                                        } else {
                                            var11 = 0.049566757;
                                        }
                                    }
                                } else {
                                    if (input[3] < 172.0) {
                                        if (input[47] < 1452.0) {
                                            var11 = 0.06784468;
                                        } else {
                                            var11 = 0.33666965;
                                        }
                                    } else {
                                        if (input[15] < 37408.0) {
                                            var11 = -0.13636899;
                                        } else {
                                            var11 = 0.34255037;
                                        }
                                    }
                                }
                            }
                        } else {
                            var11 = 0.5366805;
                        }
                    } else {
                        if (input[37] < 38479.0) {
                            var11 = -0.55432403;
                        } else {
                            if (input[52] < 427340.0) {
                                if (input[38] < 34481.0) {
                                    var11 = 0.12392616;
                                } else {
                                    if (input[14] < 70214.0) {
                                        if (input[40] < 19717.0) {
                                            var11 = 0.07723755;
                                        } else {
                                            var11 = -0.10106405;
                                        }
                                    } else {
                                        var11 = -0.23462318;
                                    }
                                }
                            } else {
                                if (input[39] < 34233.0) {
                                    var11 = -0.013373948;
                                } else {
                                    var11 = 0.50608367;
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[62] < 2.2608454) {
                    if (input[57] < 39.0) {
                        var11 = 0.25823435;
                    } else {
                        if (input[59] < 1447.0) {
                            if (input[46] < 5455.0) {
                                if (input[62] < 1.062355) {
                                    var11 = 0.031404115;
                                } else {
                                    if (input[57] < 3998.0) {
                                        if (input[58] < 5393.0) {
                                            var11 = -0.41751105;
                                        } else {
                                            var11 = -0.06849827;
                                        }
                                    } else {
                                        var11 = -0.011888281;
                                    }
                                }
                            } else {
                                var11 = 0.12214848;
                            }
                        } else {
                            var11 = 0.22784916;
                        }
                    }
                } else {
                    var11 = -0.55463094;
                }
            }
        }
    } else {
        if (input[46] < 5367.0) {
            if (input[48] < 62.0) {
                var11 = 0.0;
            } else {
                if (input[6] < 5551.0) {
                    if (input[46] < 3777.0) {
                        var11 = 0.6296472;
                    } else {
                        if (input[56] < 38749.0) {
                            var11 = 0.46096855;
                        } else {
                            var11 = 0.18772532;
                        }
                    }
                } else {
                    var11 = 0.10733681;
                }
            }
        } else {
            var11 = -0.14567344;
        }
    }
    double var12;
    if (input[38] < 77843.0) {
        if (input[46] < 4559.0) {
            if (input[61] < 0.044694055) {
                if (input[31] < 28.0) {
                    if (input[14] < 81897.0) {
                        if (input[56] < 36780.0) {
                            if (input[23] < 0.42974514) {
                                if (input[27] < 137.0) {
                                    var12 = -0.38323212;
                                } else {
                                    var12 = 0.0535953;
                                }
                            } else {
                                if (input[62] < 0.9967484) {
                                    if (input[52] < 141016.0) {
                                        if (input[28] < 0.18157138) {
                                            var12 = 0.68037516;
                                        } else {
                                            var12 = 0.17020373;
                                        }
                                    } else {
                                        if (input[27] < 137.0) {
                                            var12 = 0.32729363;
                                        } else {
                                            var12 = -0.041267283;
                                        }
                                    }
                                } else {
                                    if (input[62] < 1.0024548) {
                                        if (input[26] < 156.0) {
                                            var12 = -0.4806192;
                                        } else {
                                            var12 = -0.055909783;
                                        }
                                    } else {
                                        if (input[24] < 1397405.0) {
                                            var12 = 0.18333304;
                                        } else {
                                            var12 = 0.00033022233;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[55] < 70739.0) {
                                if (input[26] < 171.0) {
                                    var12 = 0.12422409;
                                } else {
                                    var12 = -0.13562755;
                                }
                            } else {
                                if (input[56] < 75833.0) {
                                    if (input[0] < 0.48166844) {
                                        var12 = -0.34826174;
                                    } else {
                                        var12 = -0.086966865;
                                    }
                                } else {
                                    var12 = -0.038285416;
                                }
                            }
                        }
                    } else {
                        var12 = 0.40913638;
                    }
                } else {
                    if (input[8] < 42.0) {
                        var12 = -0.52221555;
                    } else {
                        if (input[0] < 329.86588) {
                            var12 = 0.03620369;
                        } else {
                            var12 = -0.27712652;
                        }
                    }
                }
            } else {
                if (input[27] < 91.0) {
                    if (input[48] < 115.0) {
                        if (input[47] < 1210.0) {
                            if (input[47] < 898.0) {
                                if (input[46] < 3993.0) {
                                    if (input[47] < 887.0) {
                                        if (input[47] < 856.0) {
                                            var12 = 0.033913437;
                                        } else {
                                            var12 = 0.41398343;
                                        }
                                    } else {
                                        if (input[48] < 65.0) {
                                            var12 = -0.4769875;
                                        } else {
                                            var12 = -0.00827446;
                                        }
                                    }
                                } else {
                                    var12 = -0.6664943;
                                }
                            } else {
                                if (input[61] < 0.9591581) {
                                    if (input[28] < 0.77193576) {
                                        var12 = 0.7446152;
                                    } else {
                                        if (input[39] < 12123.0) {
                                            var12 = -0.006036104;
                                        } else {
                                            var12 = 0.24413565;
                                        }
                                    }
                                } else {
                                    var12 = -0.23796566;
                                }
                            }
                        } else {
                            if (input[62] < 1.114576) {
                                var12 = -0.75659174;
                            } else {
                                if (input[59] < 1353.0) {
                                    var12 = -0.34159163;
                                } else {
                                    if (input[39] < 12123.0) {
                                        var12 = 0.57939315;
                                    } else {
                                        var12 = 0.023887178;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[61] < 0.6568583) {
                            var12 = -0.04193001;
                        } else {
                            var12 = 0.4283246;
                        }
                    }
                } else {
                    if (input[28] < 0.1852308) {
                        if (input[40] < 587.0) {
                            if (input[24] < 1143873.0) {
                                var12 = 0.10500008;
                            } else {
                                var12 = 0.42940307;
                            }
                        } else {
                            var12 = -0.06344934;
                        }
                    } else {
                        if (input[56] < 30452.0) {
                            var12 = 0.54087603;
                        } else {
                            var12 = 0.13425973;
                        }
                    }
                }
            }
        } else {
            if (input[62] < 1.783641) {
                if (input[23] < 863.97253) {
                    if (input[46] < 5095.0) {
                        if (input[61] < 0.67172575) {
                            if (input[47] < 1417.0) {
                                if (input[46] < 4683.0) {
                                    var12 = -0.05352931;
                                } else {
                                    var12 = -0.53021365;
                                }
                            } else {
                                var12 = -0.06623217;
                            }
                        } else {
                            var12 = -0.711677;
                        }
                    } else {
                        if (input[7] < 1381.0) {
                            var12 = -0.22760563;
                        } else {
                            var12 = 0.14396566;
                        }
                    }
                } else {
                    var12 = 0.48375472;
                }
            } else {
                if (input[62] < 1.82692) {
                    var12 = 0.7359383;
                } else {
                    if (input[26] < 133.11584) {
                        if (input[62] < 2.2608454) {
                            var12 = -0.053772002;
                        } else {
                            var12 = -0.46589008;
                        }
                    } else {
                        if (input[58] < 3056.0) {
                            var12 = 0.33088708;
                        } else {
                            var12 = 0.019726245;
                        }
                    }
                }
            }
        }
    } else {
        if (input[46] < 5367.0) {
            if (input[48] < 62.0) {
                var12 = 0.0;
            } else {
                if (input[36] < 212544.0) {
                    if (input[6] < 3503.0) {
                        var12 = 0.31239197;
                    } else {
                        var12 = -0.088384554;
                    }
                } else {
                    var12 = 0.4128172;
                }
            }
        } else {
            var12 = -0.12070078;
        }
    }
    double var13;
    if (input[4] < 141.0) {
        if (input[61] < 0.02758785) {
            if (input[14] < 80641.0) {
                if (input[62] < 0.99303746) {
                    if (input[5] < 0.19119279) {
                        if (input[56] < 25315.0) {
                            var13 = 0.61052185;
                        } else {
                            var13 = 0.01010309;
                        }
                    } else {
                        var13 = -0.11243986;
                    }
                } else {
                    if (input[62] < 1.0014672) {
                        if (input[17] < 671.0) {
                            if (input[52] < 124795.0) {
                                var13 = -0.14665486;
                            } else {
                                var13 = 0.062245537;
                            }
                        } else {
                            if (input[54] < 116672.0) {
                                var13 = -0.03838089;
                            } else {
                                var13 = -0.2624017;
                            }
                        }
                    } else {
                        if (input[12] < 80295.0) {
                            if (input[26] < 171.0) {
                                if (input[37] < 54345.0) {
                                    if (input[61] < 0.020127745) {
                                        if (input[31] < 27.0) {
                                            var13 = 0.068206266;
                                        } else {
                                            var13 = -0.17783777;
                                        }
                                    } else {
                                        var13 = -0.39157924;
                                    }
                                } else {
                                    if (input[55] < 82920.0) {
                                        if (input[57] < 1357.0) {
                                            var13 = 0.075100385;
                                        } else {
                                            var13 = 0.450819;
                                        }
                                    } else {
                                        if (input[2] < 140.0) {
                                            var13 = -0.29652128;
                                        } else {
                                            var13 = 0.0;
                                        }
                                    }
                                }
                            } else {
                                if (input[18] < 19.0) {
                                    var13 = 0.06807856;
                                } else {
                                    if (input[57] < 1470.0) {
                                        if (input[45] < 157976.0) {
                                            var13 = -0.21874642;
                                        } else {
                                            var13 = -0.058710348;
                                        }
                                    } else {
                                        var13 = 0.035998013;
                                    }
                                }
                            }
                        } else {
                            var13 = 0.37958726;
                        }
                    }
                }
            } else {
                var13 = -0.2967005;
            }
        } else {
            if (input[37] < 38479.0) {
                if (input[61] < 0.0833566) {
                    if (input[3] < 0.66666514) {
                        var13 = -0.026072294;
                    } else {
                        var13 = -0.50684494;
                    }
                } else {
                    if (input[56] < 48904.0) {
                        if (input[61] < 0.19684868) {
                            if (input[23] < 195.3306) {
                                if (input[35] < 14546.0) {
                                    var13 = 0.6504757;
                                } else {
                                    var13 = 0.2287092;
                                }
                            } else {
                                if (input[47] < 112.0) {
                                    var13 = 0.124184035;
                                } else {
                                    var13 = -0.39912465;
                                }
                            }
                        } else {
                            if (input[61] < 0.24051426) {
                                if (input[12] < 278.0) {
                                    var13 = -0.0024482156;
                                } else {
                                    var13 = -0.44301772;
                                }
                            } else {
                                if (input[62] < 2.2608454) {
                                    if (input[14] >= 72284.0) {
                                        if (input[22] < 154818.0) {
                                            var13 = -0.67211574;
                                        } else {
                                            var13 = 0.004173551;
                                        }
                                    } else {
                                        if (input[12] >= 69527.0) {
                                            var13 = 0.48393783;
                                        } else {
                                            var13 = 0.054605234;
                                        }
                                    }
                                } else {
                                    var13 = -0.36001438;
                                }
                            }
                        }
                    } else {
                        var13 = -0.39590988;
                    }
                }
            } else {
                if (input[50] < 13.0) {
                    if (input[27] < 136.0) {
                        if (input[61] < 0.0385086) {
                            var13 = 0.3487465;
                        } else {
                            if (input[47] < 888.0) {
                                if (input[35] >= 198945.0) {
                                    var13 = -0.31198862;
                                } else {
                                    if (input[61] < 0.18583827) {
                                        if (input[61] < 0.039482284) {
                                            var13 = -0.18676758;
                                        } else {
                                            var13 = 0.25503662;
                                        }
                                    } else {
                                        if (input[39] < 34234.0) {
                                            var13 = -0.3608738;
                                        } else {
                                            var13 = 0.05630066;
                                        }
                                    }
                                }
                            } else {
                                if (input[47] < 923.0) {
                                    var13 = 0.5137567;
                                } else {
                                    if (input[59] < 1149.0) {
                                        if (input[56] < 31629.0) {
                                            var13 = 0.079356566;
                                        } else {
                                            var13 = 0.5754296;
                                        }
                                    } else {
                                        if (input[62] < 1.7856879) {
                                            var13 = 0.13749865;
                                        } else {
                                            var13 = -0.23308466;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[1] < 1636883.0) {
                            if (input[57] < 1295.0) {
                                if (input[40] < 654.0) {
                                    var13 = -0.44024235;
                                } else {
                                    var13 = -0.0490905;
                                }
                            } else {
                                var13 = -0.027235461;
                            }
                        } else {
                            var13 = 0.38771355;
                        }
                    }
                } else {
                    if (input[57] < 1254.0) {
                        var13 = 0.4938481;
                    } else {
                        var13 = 0.07417152;
                    }
                }
            }
        }
    } else {
        if (input[20] < 52.0) {
            if (input[16] < 14499.0) {
                if (input[3] < 168.0) {
                    var13 = -0.12053267;
                } else {
                    if (input[27] < 128.0) {
                        var13 = 0.3827787;
                    } else {
                        var13 = 0.0;
                    }
                }
            } else {
                if (input[40] < 674.0) {
                    var13 = 0.6476594;
                } else {
                    var13 = 0.1844568;
                }
            }
        } else {
            if (input[60] < 64.0) {
                if (input[61] < 0.02317072) {
                    var13 = 0.16399966;
                } else {
                    if (input[28] < 0.23783533) {
                        if (input[3] < 178.0) {
                            var13 = -0.5599222;
                        } else {
                            var13 = -0.12365957;
                        }
                    } else {
                        var13 = 0.024508612;
                    }
                }
            } else {
                if (input[61] < 0.011767868) {
                    var13 = -0.1112285;
                } else {
                    if (input[28] < 0.18662947) {
                        var13 = 0.5944953;
                    } else {
                        if (input[58] < 63.0) {
                            var13 = -0.0930871;
                        } else {
                            var13 = 0.19100952;
                        }
                    }
                }
            }
        }
    }
    double var14;
    if (input[4] < 146.0) {
        if (input[38] < 77843.0) {
            if (input[46] < 4559.0) {
                if (input[48] < 115.0) {
                    if (input[56] < 39946.0) {
                        if (input[59] < 1056.0) {
                            if (input[57] < 12060.0) {
                                if (input[18] < 23.0) {
                                    if (input[24] < 768.0) {
                                        if (input[31] < 15.0) {
                                            var14 = 0.040228557;
                                        } else {
                                            var14 = 0.43253398;
                                        }
                                    } else {
                                        if (input[50] < 9.0) {
                                            var14 = -0.16493672;
                                        } else {
                                            var14 = 0.0012276623;
                                        }
                                    }
                                } else {
                                    if (input[19] < 29.0) {
                                        if (input[60] < 18.0) {
                                            var14 = -0.045487594;
                                        } else {
                                            var14 = 0.5640071;
                                        }
                                    } else {
                                        if (input[42] < 14.0) {
                                            var14 = -0.3116868;
                                        } else {
                                            var14 = 0.043859743;
                                        }
                                    }
                                }
                            } else {
                                var14 = -0.58203787;
                            }
                        } else {
                            if (input[62] < 1.4354582) {
                                if (input[48] < 43.0) {
                                    var14 = -0.19808045;
                                } else {
                                    if (input[23] < 863.97253) {
                                        if (input[56] < 36509.0) {
                                            var14 = 0.33781555;
                                        } else {
                                            var14 = 0.00071159366;
                                        }
                                    } else {
                                        var14 = -0.07432316;
                                    }
                                }
                            } else {
                                var14 = 0.62422323;
                            }
                        }
                    } else {
                        if (input[56] < 40046.0) {
                            var14 = -0.7953153;
                        } else {
                            if (input[48] < 107.0) {
                                if (input[36] < 190576.0) {
                                    if (input[28] < 0.8366985) {
                                        if (input[61] < 0.36284402) {
                                            var14 = 0.0005399177;
                                        } else {
                                            var14 = -0.5937006;
                                        }
                                    } else {
                                        if (input[58] < 4655.0) {
                                            var14 = 0.0;
                                        } else {
                                            var14 = 0.32651436;
                                        }
                                    }
                                } else {
                                    if (input[59] < 1447.0) {
                                        var14 = -0.3251999;
                                    } else {
                                        if (input[28] < 0.85545856) {
                                            var14 = 0.07373246;
                                        } else {
                                            var14 = -0.10890106;
                                        }
                                    }
                                }
                            } else {
                                var14 = 0.43611845;
                            }
                        }
                    }
                } else {
                    if (input[61] < 0.6568583) {
                        var14 = -0.035766374;
                    } else {
                        if (input[62] < 1.2335619) {
                            var14 = 0.119000785;
                        } else {
                            var14 = 0.4171702;
                        }
                    }
                }
            } else {
                if (input[62] < 1.783641) {
                    if (input[23] < 863.97253) {
                        if (input[7] < 1381.0) {
                            if (input[56] < 12756.0) {
                                var14 = -0.12971556;
                            } else {
                                if (input[61] < 0.28351778) {
                                    var14 = -0.23759599;
                                } else {
                                    var14 = -0.65420026;
                                }
                            }
                        } else {
                            if (input[48] < 114.0) {
                                var14 = -0.3238195;
                            } else {
                                if (input[61] < 0.67172575) {
                                    var14 = 0.15302543;
                                } else {
                                    var14 = -0.027232379;
                                }
                            }
                        }
                    } else {
                        var14 = 0.3943013;
                    }
                } else {
                    if (input[62] < 1.82692) {
                        var14 = 0.6022738;
                    } else {
                        if (input[61] < 0.9369428) {
                            if (input[40] < 34.0) {
                                var14 = -0.11279217;
                            } else {
                                var14 = 0.2688333;
                            }
                        } else {
                            var14 = -0.2537335;
                        }
                    }
                }
            }
        } else {
            if (input[46] < 5367.0) {
                if (input[48] < 62.0) {
                    var14 = 0.0;
                } else {
                    if (input[6] < 5551.0) {
                        if (input[46] < 3777.0) {
                            var14 = 0.42000505;
                        } else {
                            if (input[56] < 38749.0) {
                                var14 = 0.3111621;
                            } else {
                                var14 = 0.08176041;
                            }
                        }
                    } else {
                        var14 = 0.014652798;
                    }
                }
            } else {
                var14 = -0.10993303;
            }
        }
    } else {
        var14 = 0.3587531;
    }
    double var15;
    if (input[61] < 0.0038328522) {
        if (input[17] < 659.0) {
            if (input[55] < 72910.0) {
                if (input[2] < 84.0) {
                    if (input[52] < 85370.0) {
                        if (input[10] < 13.0) {
                            if (input[10] < 12.0) {
                                if (input[31] < 27.0) {
                                    var15 = 0.1409128;
                                } else {
                                    if (input[47] < 36.0) {
                                        var15 = 0.052661058;
                                    } else {
                                        var15 = -0.13141595;
                                    }
                                }
                            } else {
                                var15 = -0.1829317;
                            }
                        } else {
                            var15 = 0.29220542;
                        }
                    } else {
                        var15 = -0.16566963;
                    }
                } else {
                    if (input[2] < 133.0) {
                        if (input[26] < 166.0) {
                            var15 = -0.042126957;
                        } else {
                            var15 = -0.32735035;
                        }
                    } else {
                        var15 = 0.0;
                    }
                }
            } else {
                if (input[2] < 138.0) {
                    var15 = 0.30193853;
                } else {
                    var15 = 0.024880296;
                }
            }
        } else {
            if (input[56] < 32385.0) {
                var15 = -0.10925376;
            } else {
                var15 = -0.36610445;
            }
        }
    } else {
        if (input[47] < 36.0) {
            if (input[25] < 52.0) {
                if (input[33] >= 9.0) {
                    var15 = -0.3827395;
                } else {
                    if (input[23] < 0.34359083) {
                        var15 = 0.4560618;
                    } else {
                        if (input[10] >= 4.0) {
                            var15 = -0.19728775;
                        } else {
                            var15 = 0.25844154;
                        }
                    }
                }
            } else {
                if (input[43] < 15.0) {
                    var15 = -0.055317085;
                } else {
                    if (input[3] < 165.06604) {
                        if (input[62] < 1.0944281) {
                            var15 = -0.13943787;
                        } else {
                            var15 = 0.35930562;
                        }
                    } else {
                        var15 = 0.5286864;
                    }
                }
            }
        } else {
            if (input[56] < 19.0) {
                if (input[3] < 0.60514593) {
                    var15 = 0.047208916;
                } else {
                    if (input[51] < 3.0) {
                        var15 = -0.3565848;
                    } else {
                        var15 = -0.052939605;
                    }
                }
            } else {
                if (input[58] < 11.0) {
                    if (input[61] < 0.13745023) {
                        var15 = 0.013918153;
                    } else {
                        var15 = 0.3285321;
                    }
                } else {
                    if (input[8] < 23.0) {
                        if (input[15] < 59654.0) {
                            if (input[4] < 0.98330873) {
                                if (input[1] < 1763.0) {
                                    var15 = -0.1583653;
                                } else {
                                    var15 = 0.09785794;
                                }
                            } else {
                                if (input[23] < 0.4551904) {
                                    var15 = 0.0;
                                } else {
                                    if (input[38] < 50572.0) {
                                        var15 = -0.39654055;
                                    } else {
                                        var15 = -0.08017864;
                                    }
                                }
                            }
                        } else {
                            var15 = 0.1789384;
                        }
                    } else {
                        if (input[0] < 0.4504612) {
                            if (input[0] < 0.44894087) {
                                if (input[26] < 74.0) {
                                    var15 = -0.3662957;
                                } else {
                                    if (input[43] < 47.0) {
                                        if (input[17] < 640.0) {
                                            var15 = 0.086431846;
                                        } else {
                                            var15 = 0.50507325;
                                        }
                                    } else {
                                        if (input[14] < 57188.0) {
                                            var15 = 0.16285019;
                                        } else {
                                            var15 = -0.057637412;
                                        }
                                    }
                                }
                            } else {
                                if (input[28] < 0.18871462) {
                                    if (input[43] < 49.0) {
                                        var15 = -0.24430755;
                                    } else {
                                        var15 = 0.0;
                                    }
                                } else {
                                    var15 = -0.6785264;
                                }
                            }
                        } else {
                            if (input[0] < 0.45118055) {
                                if (input[56] < 24838.0) {
                                    var15 = 0.61475587;
                                } else {
                                    var15 = 0.1295939;
                                }
                            } else {
                                if (input[46] < 4041.0) {
                                    if (input[46] < 3907.0) {
                                        if (input[46] < 3799.0) {
                                            var15 = 0.06904139;
                                        } else {
                                            var15 = -0.2456882;
                                        }
                                    } else {
                                        if (input[54] < 125434.0) {
                                            var15 = 0.5224614;
                                        } else {
                                            var15 = 0.00930535;
                                        }
                                    }
                                } else {
                                    if (input[61] < 0.84336907) {
                                        if (input[61] < 0.83082277) {
                                            var15 = -0.055935103;
                                        } else {
                                            var15 = -0.8864571;
                                        }
                                    } else {
                                        if (input[61] < 0.90844816) {
                                            var15 = 0.54289615;
                                        } else {
                                            var15 = 0.01532678;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var16;
    if (input[46] < 6147.0) {
        if (input[46] < 5611.0) {
            if (input[46] < 4779.0) {
                if (input[38] >= 77843.0) {
                    if (input[35] < 221887.0) {
                        if (input[62] < 1.3489773) {
                            if (input[52] < 273942.0) {
                                var16 = 0.0;
                            } else {
                                var16 = 0.20880714;
                            }
                        } else {
                            var16 = 0.35165414;
                        }
                    } else {
                        var16 = -0.017908325;
                    }
                } else {
                    if (input[56] < 49717.0) {
                        if (input[52] < 256459.0) {
                            if (input[52] < 243974.0) {
                                if (input[54] < 159460.0) {
                                    if (input[52] < 234850.0) {
                                        if (input[28] < 0.85545856) {
                                            var16 = 0.020201517;
                                        } else {
                                            var16 = 0.32443354;
                                        }
                                    } else {
                                        var16 = -0.42903128;
                                    }
                                } else {
                                    if (input[0] < 329.86588) {
                                        var16 = -0.06232447;
                                    } else {
                                        var16 = 0.47996265;
                                    }
                                }
                            } else {
                                var16 = -0.24159682;
                            }
                        } else {
                            var16 = 0.37881804;
                        }
                    } else {
                        if (input[58] < 4793.0) {
                            if (input[47] < 849.0) {
                                if (input[39] < 19513.0) {
                                    if (input[62] < 1.0092515) {
                                        if (input[4] < 138.0) {
                                            var16 = -0.2674536;
                                        } else {
                                            var16 = -0.053414684;
                                        }
                                    } else {
                                        var16 = 0.02404747;
                                    }
                                } else {
                                    var16 = 0.13812004;
                                }
                            } else {
                                if (input[46] < 3499.0) {
                                    if (input[0] < 616.4713) {
                                        var16 = -0.14982636;
                                    } else {
                                        var16 = -0.39705223;
                                    }
                                } else {
                                    var16 = -0.037223857;
                                }
                            }
                        } else {
                            if (input[39] < 34233.0) {
                                var16 = -0.09771981;
                            } else {
                                var16 = 0.5065009;
                            }
                        }
                    }
                }
            } else {
                if (input[46] < 5095.0) {
                    if (input[49] < 24.0) {
                        var16 = -0.3881217;
                    } else {
                        var16 = -0.10734447;
                    }
                } else {
                    if (input[2] < 359.16473) {
                        if (input[40] < 34.0) {
                            var16 = -0.10669439;
                        } else {
                            if (input[46] < 5399.0) {
                                var16 = 0.36744034;
                            } else {
                                var16 = 0.040421568;
                            }
                        }
                    } else {
                        var16 = -0.2685696;
                    }
                }
            }
        } else {
            if (input[61] < 0.7888465) {
                var16 = -0.058336068;
            } else {
                var16 = 0.49455163;
            }
        }
    } else {
        var16 = -0.29062378;
    }
    double var17;
    if (input[4] < 141.0) {
        if (input[5] < 0.17522573) {
            if (input[62] < 1.0095137) {
                if (input[35] < 60139.0) {
                    var17 = 0.39870378;
                } else {
                    if (input[5] < 0.16571127) {
                        if (input[38] < 41977.0) {
                            var17 = -0.046084635;
                        } else {
                            var17 = 0.27141312;
                        }
                    } else {
                        var17 = -0.1651551;
                    }
                }
            } else {
                if (input[1] < 1257084.0) {
                    var17 = -0.12325497;
                } else {
                    var17 = -0.42411456;
                }
            }
        } else {
            if (input[23] < 0.34359083) {
                if (input[30] < 30.0) {
                    var17 = 0.4930098;
                } else {
                    if (input[26] < 89.0) {
                        if (input[15] < 2004.0) {
                            var17 = -0.3357628;
                        } else {
                            if (input[33] < 9.0) {
                                var17 = 0.27564913;
                            } else {
                                var17 = -0.22505645;
                            }
                        }
                    } else {
                        if (input[54] < 55113.0) {
                            if (input[62] < 1.0447392) {
                                var17 = 0.10328732;
                            } else {
                                var17 = 0.44417056;
                            }
                        } else {
                            var17 = -0.09377324;
                        }
                    }
                }
            } else {
                if (input[23] < 0.4326706) {
                    if (input[26] < 74.0) {
                        var17 = -0.45168176;
                    } else {
                        if (input[62] < 1.0758849) {
                            if (input[40] < 688.0) {
                                var17 = -0.2653163;
                            } else {
                                var17 = 0.012622833;
                            }
                        } else {
                            if (input[25] < 77.0) {
                                var17 = 0.24073462;
                            } else {
                                var17 = 0.021056367;
                            }
                        }
                    }
                } else {
                    if (input[32] < 5.0) {
                        if (input[39] < 14.0) {
                            var17 = 0.052634973;
                        } else {
                            var17 = -0.32294178;
                        }
                    } else {
                        if (input[24] < 768.0) {
                            if (input[31] >= 15.0) {
                                var17 = 0.34345195;
                            } else {
                                var17 = 0.065660164;
                            }
                        } else {
                            if (input[35] < 42.0) {
                                var17 = -0.19586486;
                            } else {
                                if (input[46] < 54.0) {
                                    if (input[42] < 56.0) {
                                        var17 = -0.40166482;
                                    } else {
                                        if (input[61] < 0.01980223) {
                                            var17 = -0.064129405;
                                        } else {
                                            var17 = 0.0885174;
                                        }
                                    }
                                } else {
                                    if (input[28] < 0.77193576) {
                                        if (input[32] < 16.0) {
                                            var17 = 0.230358;
                                        } else {
                                            var17 = 0.047919814;
                                        }
                                    } else {
                                        if (input[28] < 0.7719359) {
                                            var17 = -0.36030626;
                                        } else {
                                            var17 = 0.008653324;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[17] < 670.0) {
            if (input[23] < 0.42974514) {
                var17 = -0.37343702;
            } else {
                if (input[40] < 712.0) {
                    var17 = 0.3334286;
                } else {
                    var17 = -0.106869064;
                }
            }
        } else {
            if (input[5] < 0.18935832) {
                if (input[54] < 74556.0) {
                    var17 = 0.42178726;
                } else {
                    if (input[3] < 170.0) {
                        var17 = -0.0073341094;
                    } else {
                        if (input[57] < 1332.0) {
                            var17 = 0.06477731;
                        } else {
                            if (input[5] < 0.18363409) {
                                if (input[61] < 0.018897982) {
                                    var17 = 0.0;
                                } else {
                                    var17 = 0.21264985;
                                }
                            } else {
                                var17 = 0.37735373;
                            }
                        }
                    }
                }
            } else {
                if (input[61] < 0.023701454) {
                    if (input[61] < 0.01980223) {
                        var17 = -0.08299207;
                    } else {
                        var17 = 0.3172692;
                    }
                } else {
                    if (input[40] < 591.0) {
                        var17 = 0.031654663;
                    } else {
                        var17 = -0.2855629;
                    }
                }
            }
        }
    }
    double var18;
    if (input[4] < 146.0) {
        if (input[23] < 0.14445564) {
            var18 = 0.21672189;
        } else {
            if (input[61] < 0.42481682) {
                if (input[57] < 196.0) {
                    if (input[41] < 17.0) {
                        if (input[26] < 0.58704174) {
                            if (input[51] < 3.0) {
                                var18 = -0.24642532;
                            } else {
                                var18 = 0.033545837;
                            }
                        } else {
                            if (input[56] < 18.0) {
                                var18 = -0.1709795;
                            } else {
                                if (input[32] < 5.0) {
                                    var18 = -0.058234375;
                                } else {
                                    if (input[27] < 0.578132) {
                                        var18 = -0.0021769335;
                                    } else {
                                        var18 = 0.2728173;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[59] < 521.0) {
                            if (input[17] < 22.0) {
                                if (input[6] < 140.0) {
                                    var18 = -0.054513853;
                                } else {
                                    var18 = -0.29443195;
                                }
                            } else {
                                var18 = -0.58737785;
                            }
                        } else {
                            if (input[28] < 0.7731841) {
                                var18 = 0.20185788;
                            } else {
                                var18 = -0.23918569;
                            }
                        }
                    }
                } else {
                    if (input[54] < 47132.0) {
                        var18 = 0.3975011;
                    } else {
                        if (input[54] < 55416.0) {
                            if (input[39] < 13760.0) {
                                if (input[3] < 165.06604) {
                                    if (input[57] < 699.0) {
                                        var18 = -0.14410721;
                                    } else {
                                        var18 = -0.4719924;
                                    }
                                } else {
                                    var18 = -0.012942172;
                                }
                            } else {
                                if (input[55] < 48320.0) {
                                    var18 = -0.05744989;
                                } else {
                                    var18 = 0.18237285;
                                }
                            }
                        } else {
                            if (input[25] < 58.0) {
                                var18 = 0.42801234;
                            } else {
                                if (input[61] < 0.36444244) {
                                    if (input[62] < 1.1573554) {
                                        if (input[62] < 1.1498934) {
                                            var18 = 0.0044092345;
                                        } else {
                                            var18 = -0.34564242;
                                        }
                                    } else {
                                        if (input[62] < 1.161465) {
                                            var18 = 0.49140078;
                                        } else {
                                            var18 = 0.11375969;
                                        }
                                    }
                                } else {
                                    if (input[29] < 3503.0) {
                                        if (input[54] < 68017.0) {
                                            var18 = -0.09417427;
                                        } else {
                                            var18 = -0.44588247;
                                        }
                                    } else {
                                        var18 = 0.026580775;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[3] < 165.06604) {
                    if (input[60] < 18.0) {
                        if (input[43] < 5.0) {
                            if (input[61] < 0.569019) {
                                if (input[56] < 39284.0) {
                                    if (input[46] < 3311.0) {
                                        if (input[0] < 2290.0579) {
                                            var18 = -0.002704457;
                                        } else {
                                            var18 = 0.264805;
                                        }
                                    } else {
                                        if (input[3] < 89.0) {
                                            var18 = 0.27883518;
                                        } else {
                                            var18 = 0.6804185;
                                        }
                                    }
                                } else {
                                    var18 = -0.03588574;
                                }
                            } else {
                                if (input[62] < 1.114576) {
                                    var18 = -0.2099543;
                                } else {
                                    if (input[61] < 0.6568583) {
                                        var18 = -0.27961707;
                                    } else {
                                        if (input[61] < 0.7267642) {
                                            var18 = 0.39328325;
                                        } else {
                                            var18 = 0.039105542;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[25] < 0.2478585) {
                                var18 = 0.104754485;
                            } else {
                                var18 = -0.27202475;
                            }
                        }
                    } else {
                        if (input[27] < 59.0) {
                            var18 = 0.3593429;
                        } else {
                            var18 = 0.058233757;
                        }
                    }
                } else {
                    if (input[4] < 137.0) {
                        if (input[41] < 17.0) {
                            var18 = -0.43832928;
                        } else {
                            if (input[25] < 92.45553) {
                                if (input[54] < 65993.0) {
                                    var18 = 0.63012785;
                                } else {
                                    var18 = -0.11809589;
                                }
                            } else {
                                if (input[58] < 2685.0) {
                                    var18 = -0.4264865;
                                } else {
                                    if (input[28] < 0.77193576) {
                                        if (input[23] < 863.9725) {
                                            var18 = 0.47984138;
                                        } else {
                                            var18 = 0.01638504;
                                        }
                                    } else {
                                        if (input[23] < 42.103584) {
                                            var18 = -0.61905766;
                                        } else {
                                            var18 = -0.0046811323;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        var18 = 0.21626897;
                    }
                }
            }
        }
    } else {
        var18 = 0.24951904;
    }
    double var19;
    if (input[46] < 6147.0) {
        if (input[46] < 5611.0) {
            if (input[47] < 1555.0) {
                if (input[38] >= 77844.0) {
                    if (input[47] < 1249.0) {
                        if (input[28] < 0.8008734) {
                            var19 = 0.13930456;
                        } else {
                            var19 = -0.04074646;
                        }
                    } else {
                        var19 = 0.25802144;
                    }
                } else {
                    if (input[47] < 1509.0) {
                        if (input[46] < 4559.0) {
                            if (input[56] < 49717.0) {
                                if (input[54] < 159460.0) {
                                    if (input[52] < 234850.0) {
                                        if (input[28] < 0.85545856) {
                                            var19 = 0.011492246;
                                        } else {
                                            var19 = 0.26995644;
                                        }
                                    } else {
                                        var19 = -0.35279733;
                                    }
                                } else {
                                    if (input[62] < 1.0638986) {
                                        var19 = -0.11089349;
                                    } else {
                                        if (input[39] < 34230.0) {
                                            var19 = 0.3988672;
                                        } else {
                                            var19 = 0.0;
                                        }
                                    }
                                }
                            } else {
                                if (input[58] < 4881.0) {
                                    if (input[47] < 849.0) {
                                        if (input[15] < 45141.0) {
                                            var19 = -0.07806493;
                                        } else {
                                            var19 = 0.11916939;
                                        }
                                    } else {
                                        if (input[57] < 19054.0) {
                                            var19 = -0.058797743;
                                        } else {
                                            var19 = -0.28035745;
                                        }
                                    }
                                } else {
                                    if (input[39] < 34233.0) {
                                        var19 = -0.11481432;
                                    } else {
                                        var19 = 0.3457788;
                                    }
                                }
                            }
                        } else {
                            if (input[61] < 0.8329145) {
                                if (input[28] < 0.8366958) {
                                    if (input[46] < 4807.0) {
                                        var19 = -0.42668977;
                                    } else {
                                        var19 = -0.045453202;
                                    }
                                } else {
                                    if (input[59] < 943.0) {
                                        var19 = 0.2142791;
                                    } else {
                                        var19 = -0.09829769;
                                    }
                                }
                            } else {
                                if (input[46] < 4779.0) {
                                    var19 = 0.3823825;
                                } else {
                                    var19 = -0.11077378;
                                }
                            }
                        }
                    } else {
                        var19 = 0.2247272;
                    }
                }
            } else {
                if (input[62] < 1.8318126) {
                    var19 = -0.34381858;
                } else {
                    if (input[46] < 5367.0) {
                        var19 = 0.09531919;
                    } else {
                        var19 = -0.10857729;
                    }
                }
            }
        } else {
            if (input[61] < 0.7888465) {
                var19 = -0.06905941;
            } else {
                var19 = 0.40708563;
            }
        }
    } else {
        var19 = -0.25001392;
    }
    return 15.5 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_NUMPOINTSPERCELL_3D
