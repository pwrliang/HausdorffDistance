
#include <math.h>
#ifndef DECISION_TREE_MAXHIT_3D
#define DECISION_TREE_MAXHIT_3D
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
inline double PredictMaxHit_3D(double * input) {
    double var0;
    if (input[55] < 10.0) {
        if (input[41] < 46.0) {
            if (input[22] < 14419.246) {
                if (input[0] < 195.33058) {
                    var0 = -11.0077915;
                } else {
                    if (input[16] < 25.0) {
                        if (input[52] < 8.0) {
                            var0 = 13.035166;
                        } else {
                            if (input[55] < 0.3535297) {
                                var0 = 35.173626;
                            } else {
                                var0 = 62.004395;
                            }
                        }
                    } else {
                        var0 = 11.478992;
                    }
                }
            } else {
                if (input[41] < 13.0) {
                    var0 = -17.182539;
                } else {
                    var0 = 9.112858;
                }
            }
        } else {
            if (input[13] < 76564.0) {
                if (input[20] < 73648.0) {
                    if (input[0] < 0.4552213) {
                        if (input[30] < 0.1726872) {
                            var0 = 24.589012;
                        } else {
                            if (input[8] < 0.3201576) {
                                if (input[18] < 29.0) {
                                    var0 = 65.19184;
                                } else {
                                    if (input[51] < 758.0) {
                                        var0 = 29.04048;
                                    } else {
                                        if (input[23] < 1331456.0) {
                                            var0 = 32.507145;
                                        } else {
                                            var0 = 55.79884;
                                        }
                                    }
                                }
                            } else {
                                var0 = 39.12605;
                            }
                        }
                    } else {
                        if (input[53] < 74.0) {
                            var0 = 8.77381;
                        } else {
                            if (input[30] < 0.18433216) {
                                var0 = 59.973812;
                            } else {
                                var0 = 30.602041;
                            }
                        }
                    }
                } else {
                    var0 = 63.88333;
                }
            } else {
                if (input[54] < 64.0) {
                    if (input[17] < 52.0) {
                        if (input[22] < 0.4504612) {
                            var0 = -7.657142;
                        } else {
                            var0 = 18.944899;
                        }
                    } else {
                        var0 = 32.861904;
                    }
                } else {
                    var0 = 56.32857;
                }
            }
        }
    } else {
        if (input[10] < 910.0) {
            if (input[43] < 120384.0) {
                var0 = -25.87294;
            } else {
                if (input[18] < 66.0) {
                    if (input[23] < 1353335.0) {
                        var0 = -4.156302;
                    } else {
                        var0 = 31.773214;
                    }
                } else {
                    if (input[37] < 19489.0) {
                        if (input[44] < 2827.0) {
                            var0 = -18.916883;
                        } else {
                            if (input[44] < 3777.0) {
                                if (input[52] < 3331.0) {
                                    if (input[55] < 171.36662) {
                                        var0 = -4.9350643;
                                    } else {
                                        var0 = 6.640477;
                                    }
                                } else {
                                    var0 = -15.7350645;
                                }
                            } else {
                                var0 = 7.1120887;
                            }
                        }
                    } else {
                        if (input[55] < 215.05327) {
                            var0 = -22.384693;
                        } else {
                            var0 = -12.559524;
                        }
                    }
                }
            }
        } else {
            if (input[2] < -179.23108) {
                if (input[24] < -54.96536) {
                    var0 = 20.937664;
                } else {
                    if (input[53] < 888.0) {
                        var0 = 0.26493585;
                    } else {
                        var0 = -23.960714;
                    }
                }
            } else {
                if (input[45] < 1112.0) {
                    if (input[44] < 3311.0) {
                        var0 = 61.71429;
                    } else {
                        var0 = 41.039684;
                    }
                } else {
                    if (input[44] < 3907.0) {
                        if (input[33] < 19.0) {
                            var0 = 12.483118;
                        } else {
                            var0 = -7.2799997;
                        }
                    } else {
                        if (input[48] < 72752.0) {
                            if (input[46] < 48.0) {
                                var0 = -5.7408156;
                            } else {
                                var0 = 32.826668;
                            }
                        } else {
                            if (input[27] < 78.21388) {
                                var0 = 24.562407;
                            } else {
                                if (input[44] < 4559.0) {
                                    var0 = 25.574026;
                                } else {
                                    var0 = 54.60204;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    double var1;
    if (input[55] < 10.0) {
        if (input[44] < 77.0) {
            if (input[22] < 0.42502642) {
                var1 = 1.0775932;
            } else {
                if (input[13] < 76564.0) {
                    if (input[35] >= 34.0) {
                        if (input[15] < 44601.0) {
                            if (input[19] < 50.0) {
                                if (input[21] < 132300.0) {
                                    if (input[20] < 59246.0) {
                                        if (input[22] < 0.47465277) {
                                            var1 = 34.809742;
                                        } else {
                                            var1 = 55.10229;
                                        }
                                    } else {
                                        var1 = 8.7308445;
                                    }
                                } else {
                                    if (input[44] < 39.0) {
                                        var1 = 26.974852;
                                    } else {
                                        if (input[8] < 0.18278617) {
                                            var1 = 57.750164;
                                        } else {
                                            var1 = 39.182198;
                                        }
                                    }
                                }
                            } else {
                                if (input[24] < 44.0) {
                                    if (input[55] < 3.3200378) {
                                        var1 = 33.076054;
                                    } else {
                                        var1 = 53.345165;
                                    }
                                } else {
                                    var1 = 9.825651;
                                }
                            }
                        } else {
                            var1 = 54.753475;
                        }
                    } else {
                        var1 = 16.09667;
                    }
                } else {
                    if (input[35] < 75629.0) {
                        if (input[2] < 48.0) {
                            var1 = 38.977062;
                        } else {
                            if (input[38] < 658.0) {
                                var1 = 21.171095;
                            } else {
                                var1 = -9.404024;
                            }
                        }
                    } else {
                        if (input[54] < 64.0) {
                            var1 = 26.799519;
                        } else {
                            var1 = 57.113567;
                        }
                    }
                }
            }
        } else {
            if (input[11] < 28.0) {
                if (input[35] >= 198.0) {
                    if (input[11] < 17.0) {
                        if (input[27] < 0.7195917) {
                            var1 = -11.952282;
                        } else {
                            var1 = -24.794052;
                        }
                    } else {
                        var1 = 3.28527;
                    }
                } else {
                    if (input[16] < 6.0) {
                        var1 = 26.332945;
                    } else {
                        var1 = -2.6444726;
                    }
                }
            } else {
                var1 = 41.18602;
            }
        }
    } else {
        if (input[10] < 910.0) {
            if (input[40] < 56.0) {
                var1 = -20.87033;
            } else {
                if (input[0] < 0.4601169) {
                    var1 = 22.541368;
                } else {
                    if (input[38] < 607.0) {
                        if (input[45] < 707.0) {
                            var1 = 6.5834427;
                        } else {
                            if (input[44] < 3777.0) {
                                if (input[55] < 162.21716) {
                                    var1 = -15.535979;
                                } else {
                                    if (input[51] < 16777.0) {
                                        var1 = -9.526519;
                                    } else {
                                        var1 = 2.7630444;
                                    }
                                }
                            } else {
                                var1 = 5.799087;
                            }
                        }
                    } else {
                        if (input[16] < 723.0) {
                            var1 = -18.514801;
                        } else {
                            var1 = -11.236233;
                        }
                    }
                }
            }
        } else {
            if (input[2] < -179.23108) {
                if (input[56] < 590.01465) {
                    if (input[56] < 485.52148) {
                        var1 = -21.09553;
                    } else {
                        var1 = -7.760021;
                    }
                } else {
                    var1 = 14.661391;
                }
            } else {
                if (input[45] < 1112.0) {
                    if (input[51] < 17357.0) {
                        if (input[52] < 2071.0) {
                            var1 = 43.20231;
                        } else {
                            var1 = 22.184992;
                        }
                    } else {
                        var1 = 52.25544;
                    }
                } else {
                    if (input[56] < 406.377) {
                        if (input[55] < 171.36662) {
                            var1 = -14.086057;
                        } else {
                            var1 = 12.37383;
                        }
                    } else {
                        if (input[36] < 212541.0) {
                            if (input[44] < 4115.0) {
                                var1 = -4.293465;
                            } else {
                                if (input[50] < 71470.0) {
                                    if (input[52] < 2884.0) {
                                        var1 = 33.00494;
                                    } else {
                                        var1 = 5.976476;
                                    }
                                } else {
                                    var1 = 42.29864;
                                }
                            }
                        } else {
                            var1 = 37.350384;
                        }
                    }
                }
            }
        }
    }
    double var2;
    if (input[55] < 11.0) {
        if (input[37] >= 29223.0) {
            if (input[1] < 1559769.0) {
                if (input[38] < 753.0) {
                    if (input[30] < 0.19224514) {
                        if (input[50] < 75240.0) {
                            if (input[16] < 658.0) {
                                if (input[38] < 585.0) {
                                    var2 = 45.291187;
                                } else {
                                    if (input[37] < 37647.0) {
                                        var2 = 5.233726;
                                    } else {
                                        if (input[38] < 695.0) {
                                            var2 = 35.439465;
                                        } else {
                                            var2 = 18.819382;
                                        }
                                    }
                                }
                            } else {
                                if (input[30] < 0.18205085) {
                                    var2 = 49.547474;
                                } else {
                                    var2 = 30.052567;
                                }
                            }
                        } else {
                            if (input[22] < 0.46190926) {
                                if (input[50] < 77925.0) {
                                    var2 = -2.6411073;
                                } else {
                                    var2 = 18.518332;
                                }
                            } else {
                                var2 = 36.15761;
                            }
                        }
                    } else {
                        if (input[41] < 50.0) {
                            var2 = -1.1110492;
                        } else {
                            var2 = 29.865147;
                        }
                    }
                } else {
                    if (input[5] < 189.0) {
                        if (input[35] < 74844.0) {
                            var2 = 27.196753;
                        } else {
                            var2 = 47.431858;
                        }
                    } else {
                        var2 = 29.100714;
                    }
                }
            } else {
                if (input[50] < 87868.0) {
                    if (input[38] < 655.0) {
                        var2 = 28.496153;
                    } else {
                        var2 = -10.952845;
                    }
                } else {
                    var2 = 32.06455;
                }
            }
        } else {
            if (input[22] < 14419.246) {
                if (input[13] < 70214.0) {
                    if (input[39] < 45.0) {
                        if (input[27] < 0.99999714) {
                            var2 = 31.954176;
                        } else {
                            if (input[55] < 0.40750378) {
                                var2 = -11.333578;
                            } else {
                                var2 = 19.486685;
                            }
                        }
                    } else {
                        var2 = 34.30322;
                    }
                } else {
                    var2 = -9.422059;
                }
            } else {
                if (input[41] < 13.0) {
                    var2 = -13.591357;
                } else {
                    var2 = 7.480059;
                }
            }
        }
    } else {
        if (input[10] < 910.0) {
            if (input[56] < 249.86397) {
                var2 = -17.244007;
            } else {
                if (input[0] < 0.4601169) {
                    var2 = 13.338582;
                } else {
                    if (input[38] < 136.0) {
                        if (input[30] < 0.7731841) {
                            if (input[48] < 209687.0) {
                                if (input[55] < 190.109) {
                                    var2 = -6.0206013;
                                } else {
                                    var2 = -13.63268;
                                }
                            } else {
                                var2 = 0.5978145;
                            }
                        } else {
                            var2 = 11.717197;
                        }
                    } else {
                        if (input[30] < 0.8366986) {
                            var2 = -14.975398;
                        } else {
                            var2 = -5.5220904;
                        }
                    }
                }
            }
        } else {
            if (input[2] < -179.23108) {
                if (input[24] < -54.96536) {
                    var2 = 16.988346;
                } else {
                    if (input[44] < 4779.0) {
                        var2 = -17.705248;
                    } else {
                        var2 = 1.454887;
                    }
                }
            } else {
                if (input[45] < 1172.0) {
                    if (input[30] < 0.7719367) {
                        var2 = 22.062836;
                    } else {
                        if (input[56] < 551.995) {
                            var2 = 28.23671;
                        } else {
                            var2 = 48.227318;
                        }
                    }
                } else {
                    if (input[51] < 4070.0) {
                        if (input[53] < 1272.0) {
                            if (input[45] < 1338.0) {
                                var2 = 28.870413;
                            } else {
                                if (input[46] < 48.0) {
                                    if (input[45] < 1565.0) {
                                        var2 = -12.252211;
                                    } else {
                                        var2 = 13.271807;
                                    }
                                } else {
                                    var2 = 23.169271;
                                }
                            }
                        } else {
                            var2 = 40.869545;
                        }
                    } else {
                        if (input[48] < 236264.0) {
                            var2 = -12.107623;
                        } else {
                            var2 = 7.945305;
                        }
                    }
                }
            }
        }
    }
    double var3;
    if (input[41] < 46.0) {
        if (input[0] < 195.33058) {
            if (input[56] < 483.59433) {
                if (input[54] < 56.0) {
                    if (input[27] < 74.893845) {
                        var3 = -9.813862;
                    } else {
                        if (input[24] < 85.0) {
                            var3 = -16.44586;
                        } else {
                            var3 = -13.141275;
                        }
                    }
                } else {
                    var3 = -3.0895097;
                }
            } else {
                if (input[45] < 945.0) {
                    var3 = 8.683022;
                } else {
                    var3 = -11.327727;
                }
            }
        } else {
            if (input[2] < -179.14723) {
                if (input[55] < 322.88727) {
                    if (input[53] < 1442.0) {
                        if (input[30] < 0.77318525) {
                            if (input[44] < 3167.0) {
                                var3 = 1.2276622;
                            } else {
                                var3 = -13.800807;
                            }
                        } else {
                            var3 = 7.3667755;
                        }
                    } else {
                        var3 = -16.7164;
                    }
                } else {
                    var3 = 14.017426;
                }
            } else {
                if (input[22] < 14419.246) {
                    if (input[51] < 11423.0) {
                        if (input[51] < 4070.0) {
                            if (input[52] < 8.0) {
                                var3 = 2.7450633;
                            } else {
                                if (input[55] < 177.06439) {
                                    if (input[46] < 12.0) {
                                        var3 = 14.365809;
                                    } else {
                                        if (input[55] < 0.7722962) {
                                            var3 = 43.36604;
                                        } else {
                                            var3 = 23.758446;
                                        }
                                    }
                                } else {
                                    if (input[56] < 584.3728) {
                                        var3 = -0.82436526;
                                    } else {
                                        if (input[56] < 685.4462) {
                                            var3 = 23.773891;
                                        } else {
                                            var3 = 8.025032;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[22] < 863.97253) {
                                var3 = -7.0402846;
                            } else {
                                var3 = 12.87977;
                            }
                        }
                    } else {
                        if (input[52] < 3849.0) {
                            var3 = 22.301548;
                        } else {
                            var3 = 36.829556;
                        }
                    }
                } else {
                    if (input[13] < 158.0) {
                        if (input[5] < 0.5517249) {
                            var3 = -10.966693;
                        } else {
                            var3 = 18.005299;
                        }
                    } else {
                        var3 = -14.906959;
                    }
                }
            }
        }
    } else {
        if (input[55] < 10.0) {
            if (input[1] < 1559769.0) {
                if (input[38] < 600.0) {
                    if (input[8] < 0.18644284) {
                        var3 = 23.093203;
                    } else {
                        var3 = 43.424603;
                    }
                } else {
                    if (input[37] < 37647.0) {
                        if (input[16] < 701.0) {
                            if (input[19] < 49.0) {
                                var3 = 17.256535;
                            } else {
                                var3 = -7.926881;
                            }
                        } else {
                            var3 = 30.33916;
                        }
                    } else {
                        if (input[51] < 686.0) {
                            var3 = 9.229972;
                        } else {
                            if (input[11] < 24.0) {
                                if (input[51] < 1323.0) {
                                    if (input[50] < 75554.0) {
                                        if (input[22] < 0.47465277) {
                                            var3 = 19.487663;
                                        } else {
                                            var3 = 40.83572;
                                        }
                                    } else {
                                        var3 = 0.86933845;
                                    }
                                } else {
                                    if (input[55] < 3.3200378) {
                                        if (input[16] < 736.0) {
                                            var3 = 30.062422;
                                        } else {
                                            var3 = 11.587858;
                                        }
                                    } else {
                                        if (input[35] < 74073.0) {
                                            var3 = 22.794836;
                                        } else {
                                            var3 = 44.523876;
                                        }
                                    }
                                }
                            } else {
                                var3 = 42.156094;
                            }
                        }
                    }
                }
            } else {
                if (input[44] < 52.0) {
                    if (input[41] < 51.0) {
                        var3 = 4.874784;
                    } else {
                        var3 = 34.198444;
                    }
                } else {
                    var3 = -8.7709675;
                }
            }
        } else {
            if (input[30] < 0.1852308) {
                var3 = -12.2121935;
            } else {
                if (input[0] < 0.46727797) {
                    var3 = 26.620438;
                } else {
                    var3 = -2.7070456;
                }
            }
        }
    }
    double var4;
    if (input[55] < 11.0) {
        if (input[37] >= 29223.0) {
            if (input[38] < 757.0) {
                if (input[38] < 658.0) {
                    if (input[21] < 133308.0) {
                        if (input[2] < 60.0) {
                            var4 = 0.4394694;
                        } else {
                            if (input[22] < 0.45503485) {
                                var4 = 17.536263;
                            } else {
                                var4 = 32.974957;
                            }
                        }
                    } else {
                        if (input[50] < 73287.0) {
                            var4 = 40.861755;
                        } else {
                            if (input[54] < 63.0) {
                                var4 = 8.678906;
                            } else {
                                var4 = 33.732944;
                            }
                        }
                    }
                } else {
                    if (input[0] < 0.4552213) {
                        if (input[30] < 0.17668776) {
                            var4 = 5.034914;
                        } else {
                            if (input[24] < 45.0) {
                                if (input[39] < 50.0) {
                                    if (input[23] < 1431991.0) {
                                        var4 = 13.729283;
                                    } else {
                                        var4 = 31.657007;
                                    }
                                } else {
                                    var4 = 1.0867852;
                                }
                            } else {
                                if (input[0] < 0.43918547) {
                                    var4 = 24.4472;
                                } else {
                                    var4 = 38.052044;
                                }
                            }
                        }
                    } else {
                        if (input[0] < 0.46824065) {
                            if (input[48] < 143564.0) {
                                var4 = 1.1246324;
                            } else {
                                var4 = -19.641638;
                            }
                        } else {
                            if (input[1] < 1559769.0) {
                                var4 = 22.612394;
                            } else {
                                var4 = -1.1068144;
                            }
                        }
                    }
                }
            } else {
                if (input[17] < 53.0) {
                    if (input[8] < 0.17663251) {
                        var4 = 17.381403;
                    } else {
                        if (input[51] < 1519.0) {
                            var4 = 34.94009;
                        } else {
                            var4 = 19.794008;
                        }
                    }
                } else {
                    var4 = 9.765513;
                }
            }
        } else {
            if (input[50] < 20.0) {
                var4 = 23.312403;
            } else {
                if (input[30] < 0.74354243) {
                    if (input[10] < 35.0) {
                        if (input[19] < 7.0) {
                            var4 = 24.140944;
                        } else {
                            if (input[0] < 0.45415798) {
                                var4 = 15.802741;
                            } else {
                                var4 = -12.886644;
                            }
                        }
                    } else {
                        if (input[22] < 21734.58) {
                            if (input[44] < 125.0) {
                                var4 = -10.635066;
                            } else {
                                var4 = 18.186863;
                            }
                        } else {
                            var4 = -19.566671;
                        }
                    }
                } else {
                    var4 = 14.244527;
                }
            }
        }
    } else {
        if (input[11] < 28.0) {
            if (input[56] < 483.59433) {
                if (input[55] < 208.34883) {
                    if (input[54] < 61.0) {
                        if (input[45] < 1076.0) {
                            if (input[38] < 19.0) {
                                var4 = -2.5939548;
                            } else {
                                if (input[55] < 38.36554) {
                                    if (input[38] < 552.0) {
                                        var4 = -10.342795;
                                    } else {
                                        var4 = -18.572014;
                                    }
                                } else {
                                    if (input[53] < 521.0) {
                                        if (input[27] < 200.0) {
                                            var4 = -11.908632;
                                        } else {
                                            var4 = -7.1692896;
                                        }
                                    } else {
                                        var4 = -6.0239606;
                                    }
                                }
                            }
                        } else {
                            if (input[0] < 616.4713) {
                                var4 = -10.573521;
                            } else {
                                var4 = -19.715376;
                            }
                        }
                    } else {
                        if (input[0] < 0.46429282) {
                            var4 = 10.973772;
                        } else {
                            var4 = -12.132753;
                        }
                    }
                } else {
                    var4 = 5.208442;
                }
            } else {
                if (input[0] < 863.9725) {
                    if (input[30] < 0.8366958) {
                        if (input[24] < -14.548699) {
                            if (input[55] < 216.46375) {
                                var4 = -7.4646378;
                            } else {
                                var4 = -14.756729;
                            }
                        } else {
                            var4 = 12.228056;
                        }
                    } else {
                        if (input[51] < 15703.0) {
                            var4 = 16.945454;
                        } else {
                            var4 = 2.6830473;
                        }
                    }
                } else {
                    if (input[22] < 329.86588) {
                        var4 = 11.163111;
                    } else {
                        var4 = 31.453444;
                    }
                }
            }
        } else {
            if (input[45] < 1359.0) {
                if (input[5] < 177.0) {
                    var4 = 8.558003;
                } else {
                    if (input[44] < 3673.0) {
                        var4 = 22.837017;
                    } else {
                        var4 = 41.328106;
                    }
                }
            } else {
                if (input[32] < 1381.0) {
                    if (input[48] < 52340.0) {
                        var4 = 10.737802;
                    } else {
                        var4 = -16.922346;
                    }
                } else {
                    var4 = 16.971308;
                }
            }
        }
    }
    return 144.14285714285714 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_MAXHIT_3D
