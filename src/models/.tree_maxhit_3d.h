
#include <math.h>
#ifndef DECISION_TREE_MAXHIT_3D
#define DECISION_TREE_MAXHIT_3D
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
22 File_1_Density
23 File_1_NumPoints
24 File_1_MBR_Dim_0_Range
25 File_1_MBR_Dim_1_Range
26 File_1_MBR_Dim_2_Range
27 File_1_GINI
28 File_1_Cell_P0.99_Value
29 File_1_Cell_P0.95_Value
30 File_1_Cell_P0.75_Value
31 File_1_Cell_P0.5_Value
32 File_1_Cell_P0.25_Value
33 File_1_Cell_P0.1_Value
34 File_1_Cell_P0.99_Count
35 File_1_Cell_P0.95_Count
36 File_1_Cell_P0.75_Count
37 File_1_Cell_P0.5_Count
38 File_1_Cell_P0.25_Count
39 File_1_Cell_P0.1_Count
40 File_1_Dim0_GridSize
41 File_1_Dim1_GridSize
42 File_1_Dim2_GridSize
43 File_1_NonEmptyCells
44 Cell_P0.99_Value
45 Cell_P0.95_Value
46 Cell_P0.75_Value
47 Cell_P0.5_Value
48 Cell_P0.25_Value
49 Cell_P0.1_Value
50 Cell_P0.99_Count
51 Cell_P0.95_Count
52 Cell_P0.75_Count
53 Cell_P0.5_Count
54 Cell_P0.25_Count
55 Cell_P0.1_Count
56 Dim0_GridSize
57 Dim1_GridSize
58 Dim2_GridSize
59 HDLBRatio
60 HDUBRatio

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
inline double PredictMaxHit_3D(double * input) {
    double var0;
    if (input[22] < 0.42709422) {
        var0 = -0.46727282;
    } else {
        if (input[2] < 359.16473) {
            if (input[9] < 16.0) {
                if (input[29] >= 80.0) {
                    var0 = -0.2616858;
                } else {
                    var0 = 0.09358787;
                }
            } else {
                if (input[59] < 0.030330144) {
                    var0 = 0.12264192;
                } else {
                    if (input[22] < 0.45603558) {
                        var0 = -0.051188905;
                    } else {
                        var0 = 0.095688075;
                    }
                }
            }
        } else {
            if (input[60] < 1.616128) {
                var0 = -0.19584355;
            } else {
                var0 = 0.0680097;
            }
        }
    }
    double var1;
    if (input[22] < 0.41695285) {
        var1 = -0.3338566;
    } else {
        if (input[2] < 359.16473) {
            if (input[9] < 16.0) {
                if (input[29] >= 80.0) {
                    var1 = -0.16138642;
                } else {
                    var1 = 0.08336946;
                }
            } else {
                if (input[59] < 0.030330144) {
                    var1 = 0.111827806;
                } else {
                    if (input[3] < 167.0) {
                        var1 = 0.09093741;
                    } else {
                        var1 = -0.011438746;
                    }
                }
            }
        } else {
            if (input[24] < 358.9257) {
                var1 = -0.18204102;
            } else {
                var1 = 0.027946642;
            }
        }
    }
    double var2;
    if (input[22] < 0.41440582) {
        var2 = -0.37473264;
    } else {
        if (input[30] >= 42.0) {
            if (input[60] < 1.2656399) {
                if (input[9] < 21.0) {
                    var2 = -0.2584463;
                } else {
                    var2 = 0.056877907;
                }
            } else {
                if (input[3] < 133.11584) {
                    var2 = 0.091186635;
                } else {
                    if (input[24] < 358.9257) {
                        var2 = -0.11521005;
                    } else {
                        var2 = 0.04647019;
                    }
                }
            }
        } else {
            if (input[59] < 0.03884404) {
                if (input[1] < 1555343.0) {
                    var2 = 0.10660397;
                } else {
                    var2 = 0.03227167;
                }
            } else {
                if (input[4] < 70.0) {
                    var2 = 0.079969056;
                } else {
                    var2 = -0.07965329;
                }
            }
        }
    }
    double var3;
    if (input[22] < 0.40679988) {
        var3 = -0.24988575;
    } else {
        if (input[59] < 0.039522517) {
            if (input[1] < 1555343.0) {
                var3 = 0.09518049;
            } else {
                if (input[17] < 787.0) {
                    var3 = -0.04563908;
                } else {
                    var3 = 0.07573351;
                }
            }
        } else {
            if (input[3] < 133.11584) {
                if (input[22] < 7581.5317) {
                    var3 = 0.07723724;
                } else {
                    var3 = -0.019635595;
                }
            } else {
                if (input[24] < 358.92572) {
                    if (input[17] < 723.0) {
                        var3 = -0.3113716;
                    } else {
                        var3 = 0.019639157;
                    }
                } else {
                    var3 = 0.0363491;
                }
            }
        }
    }
    double var4;
    if (input[22] < 0.40628138) {
        var4 = -0.28483054;
    } else {
        if (input[30] >= 42.0) {
            if (input[60] < 1.2656399) {
                if (input[9] < 21.0) {
                    if (input[60] < 1.0772302) {
                        var4 = -0.034571204;
                    } else {
                        var4 = -0.26793393;
                    }
                } else {
                    var4 = 0.044020034;
                }
            } else {
                if (input[3] < 133.11584) {
                    var4 = 0.07179606;
                } else {
                    if (input[25] < 87.0) {
                        var4 = 0.048105817;
                    } else {
                        var4 = -0.08261729;
                    }
                }
            }
        } else {
            if (input[23] < 1244640.0) {
                if (input[17] >= 159.0) {
                    var4 = -0.093635835;
                } else {
                    var4 = 0.058907658;
                }
            } else {
                var4 = 0.07983046;
            }
        }
    }
    double var5;
    if (input[22] < 0.40628138) {
        var5 = -0.17175223;
    } else {
        if (input[59] < 0.04139111) {
            if (input[1] < 1555343.0) {
                var5 = 0.07785585;
            } else {
                var5 = 0.006671456;
            }
        } else {
            if (input[3] < 168.0) {
                var5 = 0.034015182;
            } else {
                if (input[25] < 159.0) {
                    var5 = -0.2512117;
                } else {
                    var5 = 0.030854512;
                }
            }
        }
    }
    double var6;
    if (input[22] < 0.40628138) {
        var6 = -0.11602813;
    } else {
        if (input[25] < 159.0) {
            if (input[3] < 168.0) {
                if (input[54] < 40082.0) {
                    if (input[22] < 14125.085) {
                        var6 = 0.04833786;
                    } else {
                        var6 = -0.050089497;
                    }
                } else {
                    var6 = -0.08007631;
                }
            } else {
                var6 = -0.14152458;
            }
        } else {
            if (input[9] < 16.0) {
                var6 = -0.03881798;
            } else {
                var6 = 0.06272178;
            }
        }
    }
    double var7;
    if (input[25] < 133.11584) {
        if (input[42] < 15.0) {
            if (input[53] < 73262.0) {
                if (input[60] < 1.0151672) {
                    var7 = 0.106047094;
                } else {
                    if (input[20] < 9.0) {
                        var7 = 0.028526163;
                    } else {
                        var7 = -0.09852477;
                    }
                }
            } else {
                var7 = -0.11149489;
            }
        } else {
            var7 = -0.15679798;
        }
    } else {
        if (input[18] < 48.0) {
            var7 = 0.07215735;
        } else {
            if (input[23] < 1311820.0) {
                var7 = -0.06395447;
            } else {
                var7 = 0.038093366;
            }
        }
    }
    double var8;
    if (input[25] < 155.0) {
        if (input[58] < 51.0) {
            if (input[54] < 42394.0) {
                if (input[48] < 10.0) {
                    var8 = 0.03137734;
                } else {
                    if (input[53] < 41225.0) {
                        var8 = -0.09432811;
                    } else {
                        var8 = 0.073404886;
                    }
                }
            } else {
                var8 = -0.10364226;
            }
        } else {
            var8 = -0.15769535;
        }
    } else {
        if (input[9] < 16.0) {
            var8 = -0.041088965;
        } else {
            var8 = 0.046137974;
        }
    }
    double var9;
    if (input[22] < 0.3806849) {
        if (input[60] < 1.0631542) {
            var9 = 0.04610538;
        } else {
            var9 = -0.18075983;
        }
    } else {
        if (input[2] < 140.0) {
            if (input[15] < 21.0) {
                var9 = -0.03968964;
            } else {
                var9 = 0.045460753;
            }
        } else {
            if (input[3] < 133.11584) {
                var9 = 0.041661244;
            } else {
                if (input[38] < 34232.0) {
                    if (input[55] < 1441.0) {
                        var9 = -0.050761957;
                    } else {
                        var9 = 0.036731865;
                    }
                } else {
                    var9 = -0.13028614;
                }
            }
        }
    }
    double var10;
    if (input[26] < 144.0) {
        if (input[22] < 0.42475474) {
            if (input[24] < 67.0) {
                var10 = -0.14691734;
            } else {
                var10 = 0.027517108;
            }
        } else {
            var10 = 0.015766962;
        }
    } else {
        var10 = 0.09192845;
    }
    double var11;
    if (input[2] < 137.0) {
        if (input[24] < 64.0) {
            if (input[36] >= 144.0) {
                var11 = -0.16626278;
            } else {
                var11 = 0.03719003;
            }
        } else {
            var11 = 0.041278265;
        }
    } else {
        if (input[23] < 1311820.0) {
            var11 = -0.13022633;
        } else {
            var11 = 0.004948827;
        }
    }
    double var12;
    if (input[26] < 144.0) {
        var12 = 0.0066946796;
    } else {
        var12 = 0.07864431;
    }
    return 144.14285714285714 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + 0.008667311 + 0.007006045 + 0.005652688 + 0.004553841 + 0.0036641397 + 0.0029453395 + 0.0023656168);
}

#endif // DECISION_TREE_MAXHIT_3D
