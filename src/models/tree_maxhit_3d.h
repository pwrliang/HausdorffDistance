
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
inline double PredictMaxHit_3D(double * input) {
    double var0;
    if (input[23] < 0.42899838) {
        var0 = -0.32209453;
    } else {
        if (input[2] < 359.16473) {
            if (input[61] < 0.03850133) {
                var0 = 0.12253644;
            } else {
                if (input[3] < 133.11584) {
                    if (input[62] < 1.2144243) {
                        var0 = 0.045648556;
                    } else {
                        var0 = 0.10342413;
                    }
                } else {
                    if (input[28] < 0.18668734) {
                        var0 = 0.005457317;
                    } else {
                        var0 = -0.24417828;
                    }
                }
            }
        } else {
            if (input[62] < 1.5388336) {
                var0 = -0.26681787;
            } else {
                var0 = 0.026094476;
            }
        }
    }
    double var1;
    if (input[23] < 0.42749932) {
        var1 = -0.31252372;
    } else {
        if (input[61] < 0.0389721) {
            var1 = 0.11192989;
        } else {
            if (input[3] < 133.11584) {
                if (input[31] >= 112.0) {
                    var1 = 0.0064691748;
                } else {
                    var1 = 0.08813459;
                }
            } else {
                if (input[25] < 358.9257) {
                    if (input[14] < 90157.0) {
                        var1 = -0.32441914;
                    } else {
                        var1 = -0.01824271;
                    }
                } else {
                    if (input[46] < 3891.0) {
                        var1 = -0.09893816;
                    } else {
                        var1 = 0.06253615;
                    }
                }
            }
        }
    }
    double var2;
    if (input[23] < 0.41938925) {
        var2 = -0.35564336;
    } else {
        if (input[61] < 0.0389721) {
            var2 = 0.101935856;
        } else {
            if (input[3] < 133.11584) {
                var2 = 0.06809879;
            } else {
                if (input[25] < 134.0) {
                    var2 = -0.19740109;
                } else {
                    if (input[59] < 1162.0) {
                        var2 = 0.005428405;
                    } else {
                        var2 = -0.13932554;
                    }
                }
            }
        }
    }
    double var3;
    if (input[23] < 0.42475474) {
        var3 = -0.1950426;
    } else {
        if (input[31] >= 42.0) {
            if (input[10] < 11.0) {
                if (input[62] < 1.2656399) {
                    var3 = -0.22087501;
                } else {
                    if (input[0] < 855.7221) {
                        var3 = -0.039634064;
                    } else {
                        var3 = 0.068268016;
                    }
                }
            } else {
                var3 = 0.070821755;
            }
        } else {
            if (input[61] < 0.027678229) {
                var3 = 0.09609389;
            } else {
                if (input[3] < 172.0) {
                    var3 = 0.07382085;
                } else {
                    var3 = -0.055827852;
                }
            }
        }
    }
    double var4;
    if (input[26] < 133.11584) {
        if (input[0] < 0.48705664) {
            var4 = -0.36831883;
        } else {
            if (input[3] < 89.0) {
                var4 = 0.045292716;
            } else {
                if (input[61] < 0.84336907) {
                    var4 = -0.17987594;
                } else {
                    var4 = 0.057261992;
                }
            }
        }
    } else {
        if (input[1] < 1532078.0) {
            var4 = 0.09030145;
        } else {
            if (input[24] < 1414520.0) {
                var4 = -0.08506371;
            } else {
                if (input[9] < 16.0) {
                    var4 = -0.040382754;
                } else {
                    var4 = 0.063842684;
                }
            }
        }
    }
    double var5;
    if (input[23] < 0.40679988) {
        var5 = -0.1907406;
    } else {
        if (input[61] < 0.0389721) {
            var5 = 0.07526122;
        } else {
            if (input[3] < 133.11584) {
                var5 = 0.042561054;
            } else {
                if (input[23] < 863.9725) {
                    if (input[41] < 51.0) {
                        var5 = -0.12106907;
                    } else {
                        var5 = 0.015634336;
                    }
                } else {
                    var5 = -0.1458851;
                }
            }
        }
    }
    double var6;
    if (input[23] < 0.40679988) {
        var6 = -0.1267094;
    } else {
        if (input[61] < 0.041725196) {
            var6 = 0.06529359;
        } else {
            if (input[3] < 167.0) {
                var6 = 0.023724454;
            } else {
                if (input[25] < 358.9257) {
                    var6 = -0.22893862;
                } else {
                    var6 = 0.06523793;
                }
            }
        }
    }
    double var7;
    if (input[26] < 133.11584) {
        if (input[27] < 50.0) {
            if (input[2] < 135.0) {
                if (input[30] >= 96.0) {
                    var7 = -0.03390403;
                } else {
                    var7 = 0.064768925;
                }
            } else {
                if (input[62] < 1.2656399) {
                    var7 = -0.1375006;
                } else {
                    var7 = 0.015184797;
                }
            }
        } else {
            var7 = -0.16825952;
        }
    } else {
        if (input[9] < 16.0) {
            var7 = -0.04247722;
        } else {
            var7 = 0.0530042;
        }
    }
    double var8;
    if (input[61] < 0.027678229) {
        var8 = 0.052413233;
    } else {
        if (input[20] < 50.0) {
            if (input[10] < 14.0) {
                if (input[48] < 71.0) {
                    if (input[25] < 64.0) {
                        var8 = -0.036404368;
                    } else {
                        var8 = 0.037096366;
                    }
                } else {
                    if (input[47] < 1365.0) {
                        var8 = -0.14396998;
                    } else {
                        var8 = 0.021012463;
                    }
                }
            } else {
                var8 = 0.04708505;
            }
        } else {
            if (input[55] < 74788.0) {
                var8 = -0.2066667;
            } else {
                var8 = 0.024025802;
            }
        }
    }
    double var9;
    if (input[61] < 0.027678229) {
        var9 = 0.044523273;
    } else {
        if (input[23] < 0.45298284) {
            if (input[62] < 0.9969598) {
                var9 = 0.0944272;
            } else {
                if (input[14] < 56325.0) {
                    var9 = -0.049400683;
                } else {
                    var9 = -0.21674684;
                }
            }
        } else {
            var9 = 0.0119814025;
        }
    }
    double var10;
    if (input[37] < 38479.0) {
        if (input[28] < 0.2866716) {
            var10 = -0.1627842;
        } else {
            if (input[28] < 0.33327162) {
                var10 = 0.08102349;
            } else {
                if (input[49] < 8.0) {
                    var10 = 0.053568453;
                } else {
                    if (input[62] < 1.2656399) {
                        var10 = -0.074613474;
                    } else {
                        var10 = 0.012176376;
                    }
                }
            }
        }
    } else {
        if (input[37] >= 134233.0) {
            var10 = -0.04739818;
        } else {
            var10 = 0.03732429;
        }
    }
    double var11;
    if (input[27] < 144.0) {
        var11 = 0.00035880465;
    } else {
        var11 = 0.077365644;
    }
    return 144.14285714285714 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + 0.013003035 + 0.010562757 + 0.008557048 + 0.006916731 + 0.005580616 + var11 + 0.0036880635 + 0.0029649416 + 0.002381643);
}

#endif // DECISION_TREE_MAXHIT_3D
