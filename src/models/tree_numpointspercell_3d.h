
#include <math.h>
#ifndef DECISION_TREE_NUMPOINTSPERCELL_3D
#define DECISION_TREE_NUMPOINTSPERCELL_3D
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
76 SampleRate

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
    double SampleRate;
};

*/
inline double PredictNumPointsPerCell_3D(double * input) {
    double var0;
    if (input[72] < 197.0) {
        var0 = 61.24305;
    } else {
        if (input[72] < 589.0) {
            var0 = 9.0710945;
        } else {
            if (input[71] < 72.8346) {
                var0 = 5.1134214;
            } else {
                if (input[65] < 73.0) {
                    var0 = 2.6605518;
                } else {
                    var0 = 0.9331852;
                }
            }
        }
    }
    double var1;
    if (input[72] < 224.0) {
        var1 = -43.05564;
    } else {
        if (input[70] < 23.375) {
            var1 = 4.0805473;
        } else {
            if (input[64] < 51.0) {
                if (input[68] < -14.875) {
                    var1 = 0.5332665;
                } else {
                    var1 = 2.970409;
                }
            } else {
                if (input[28] < -7.75) {
                    var1 = 0.71087855;
                } else {
                    if (input[27] < 31.0) {
                        var1 = 0.43467757;
                    } else {
                        var1 = -0.21751495;
                    }
                }
            }
        }
    }
    double var2;
    if (input[42] < 587.0) {
        if (input[64] < 51.0) {
            if (input[67] < -11.801) {
                if (input[32] < 29.646) {
                    var2 = -0.13453193;
                } else {
                    var2 = 3.2017288;
                }
            } else {
                if (input[70] < 170.0) {
                    var2 = 6.1823354;
                } else {
                    var2 = 1.8234167;
                }
            }
        } else {
            if (input[11] < 117.0) {
                var2 = 0.7489523;
            } else {
                var2 = -0.19457084;
            }
        }
    } else {
        if (input[37] < 152256.0) {
            if (input[6] < 3793.0) {
                var2 = 0.45746565;
            } else {
                var2 = -3.1784596;
            }
        } else {
            if (input[42] < 701.0) {
                var2 = 0.59088576;
            } else {
                var2 = -0.90197384;
            }
        }
    }
    double var3;
    if (input[74] < 1196208.0) {
        if (input[33] < 49.6073) {
            if (input[69] < 19.68505) {
                var3 = 1.6824477;
            } else {
                if (input[69] < 72.5091) {
                    var3 = -1.3012235;
                } else {
                    var3 = 0.3594124;
                }
            }
        } else {
            if (input[69] < 105.0) {
                var3 = 3.8841312;
            } else {
                if (input[36] < 1301262.0) {
                    var3 = 0.2719663;
                } else {
                    var3 = 2.4276326;
                }
            }
        }
    } else {
        if (input[3] < 52.0) {
            if (input[31] < 175.0) {
                var3 = 0.3148385;
            } else {
                if (input[47] < 69038.0) {
                    var3 = -0.25382087;
                } else {
                    var3 = -1.249792;
                }
            }
        } else {
            var3 = 0.6919984;
        }
    }
    double var4;
    if (input[40] < 7.0) {
        var4 = 2.1765704;
    } else {
        if (input[46] < 41642.0) {
            if (input[31] < 14.2194) {
                var4 = -0.59090257;
            } else {
                if (input[11] < 138.0) {
                    if (input[69] < 124.0) {
                        var4 = 2.0747488;
                    } else {
                        var4 = 0.5346577;
                    }
                } else {
                    if (input[37] < 15180.0) {
                        var4 = -0.43523216;
                    } else {
                        var4 = 0.6728001;
                    }
                }
            }
        } else {
            if (input[32] < 167.0) {
                var4 = 0.5943764;
            } else {
                if (input[31] < 189.0) {
                    if (input[46] < 49314.0) {
                        var4 = -0.9236879;
                    } else {
                        var4 = -0.24928902;
                    }
                } else {
                    var4 = 0.16971377;
                }
            }
        }
    }
    double var5;
    if (input[74] < 1069.0) {
        var5 = 1.7802757;
    } else {
        if (input[33] < 143.0) {
            if (input[44] < 26.0) {
                var5 = -1.134365;
            } else {
                if (input[39] < 10.0) {
                    var5 = 1.5951055;
                } else {
                    if (input[68] < 5.0) {
                        if (input[67] < 52.0) {
                            if (input[28] < 57.0) {
                                var5 = -0.39222372;
                            } else {
                                var5 = 0.38197586;
                            }
                        } else {
                            var5 = 0.9091922;
                        }
                    } else {
                        if (input[30] < 6.0) {
                            var5 = 0.6980495;
                        } else {
                            var5 = -0.65954846;
                        }
                    }
                }
            }
        } else {
            if (input[11] < 127.0) {
                var5 = 1.6861441;
            } else {
                var5 = -0.012265717;
            }
        }
    }
    double var6;
    if (input[31] < 194.0) {
        if (input[43] < 8.0) {
            var6 = -1.2096275;
        } else {
            if (input[40] < 10.0) {
                var6 = 1.3320844;
            } else {
                if (input[68] < 11.0) {
                    if (input[67] < 52.0) {
                        if (input[10] < 14.0) {
                            if (input[28] < 51.0) {
                                var6 = -1.119961;
                            } else {
                                var6 = -0.10009535;
                            }
                        } else {
                            if (input[31] < 173.894) {
                                var6 = 0.28217807;
                            } else {
                                var6 = -0.24157427;
                            }
                        }
                    } else {
                        var6 = 0.5420779;
                    }
                } else {
                    if (input[40] < 27.0) {
                        var6 = -0.6133037;
                    } else {
                        var6 = 1.4045795;
                    }
                }
            }
        }
    } else {
        var6 = 1.2536721;
    }
    double var7;
    if (input[74] < 760.0) {
        var7 = 1.3901047;
    } else {
        if (input[8] < 56.0) {
            if (input[68] < -0.033617) {
                if (input[42] < 3.0) {
                    var7 = 0.5947735;
                } else {
                    var7 = -0.77641577;
                }
            } else {
                var7 = 1.9090058;
            }
        } else {
            if (input[40] < 11.0) {
                var7 = -1.3597963;
            } else {
                if (input[30] < 4.20426) {
                    if (input[68] < 11.0) {
                        if (input[40] < 63.0) {
                            var7 = 0.1798501;
                        } else {
                            var7 = -0.37129626;
                        }
                    } else {
                        var7 = 0.73603994;
                    }
                } else {
                    if (input[74] < 1448604.0) {
                        var7 = -0.7252898;
                    } else {
                        var7 = 0.35395727;
                    }
                }
            }
        }
    }
    double var8;
    if (input[62] < 13.0) {
        if (input[42] < 4.0) {
            var8 = -0.15625748;
        } else {
            var8 = 1.6202669;
        }
    } else {
        if (input[31] < 16.1895) {
            var8 = -0.7548561;
        } else {
            if (input[42] < 3.0) {
                var8 = 1.0305526;
            } else {
                if (input[2] < 9.0) {
                    var8 = -0.67932534;
                } else {
                    if (input[36] < 30746.0) {
                        var8 = 1.062639;
                    } else {
                        if (input[38] < 0.26934326) {
                            if (input[68] < 11.0) {
                                if (input[29] < 41.0) {
                                    var8 = 0.53979385;
                                } else {
                                    var8 = -0.1366357;
                                }
                            } else {
                                var8 = 1.2099063;
                            }
                        } else {
                            var8 = -0.77153003;
                        }
                    }
                }
            }
        }
    }
    double var9;
    if (input[33] < 146.0) {
        if (input[45] < 12.0) {
            var9 = -1.0940905;
        } else {
            if (input[40] < 9.0) {
                var9 = 1.347585;
            } else {
                if (input[66] < 104.0) {
                    if (input[68] < 17.5217) {
                        if (input[47] < 76947.0) {
                            if (input[67] < 45.0) {
                                if (input[39] < 48.0) {
                                    var9 = 0.016874874;
                                } else {
                                    var9 = -0.60566133;
                                }
                            } else {
                                var9 = 0.07963042;
                            }
                        } else {
                            if (input[8] < 45681.0) {
                                var9 = 0.87803566;
                            } else {
                                var9 = -0.004122006;
                            }
                        }
                    } else {
                        var9 = 1.0856813;
                    }
                } else {
                    var9 = -0.7827204;
                }
            }
        }
    } else {
        var9 = 1.0833557;
    }
    double var10;
    if (input[39] < 5.0) {
        var10 = 1.0373747;
    } else {
        if (input[38] < 0.78102773) {
            if (input[4] < 2.0) {
                var10 = 1.0663726;
            } else {
                if (input[45] < 24.0) {
                    var10 = -1.2881137;
                } else {
                    if (input[72] < 752.0) {
                        if (input[62] < 31.0) {
                            if (input[47] < 80.0) {
                                var10 = 0.7191248;
                            } else {
                                if (input[10] < 69952.0) {
                                    if (input[42] < 661.0) {
                                        var10 = -0.605604;
                                    } else {
                                        var10 = 0.08802894;
                                    }
                                } else {
                                    var10 = 0.23980024;
                                }
                            }
                        } else {
                            var10 = -0.9042203;
                        }
                    } else {
                        var10 = 0.6585769;
                    }
                }
            }
        } else {
            var10 = -0.8026371;
        }
    }
    double var11;
    if (input[24] < 6.0) {
        var11 = 0.8987712;
    } else {
        if (input[33] < 35.5) {
            if (input[70] < 98.775) {
                if (input[65] < 28.0) {
                    var11 = -0.024183588;
                } else {
                    var11 = -1.541897;
                }
            } else {
                var11 = 0.32741645;
            }
        } else {
            if (input[71] < 36.8252) {
                var11 = 1.1836215;
            } else {
                if (input[44] < 52.0) {
                    var11 = -0.76618814;
                } else {
                    if (input[67] < 36.3379) {
                        var11 = 0.8694655;
                    } else {
                        if (input[11] < 2450.0) {
                            if (input[4] < 661.0) {
                                var11 = 0.20815633;
                            } else {
                                var11 = -0.105641566;
                            }
                        } else {
                            var11 = -0.24437939;
                        }
                    }
                }
            }
        }
    }
    double var12;
    if (input[40] < 7.0) {
        if (input[42] < 4.0) {
            var12 = -0.25613078;
        } else {
            var12 = 1.4989543;
        }
    } else {
        if (input[75] < 594.0) {
            var12 = -0.68616396;
        } else {
            if (input[75] < 3458.0) {
                if (input[74] < 13038.0) {
                    var12 = -0.001874421;
                } else {
                    var12 = 0.8634786;
                }
            } else {
                if (input[43] < 228.0) {
                    if (input[68] < -0.163693) {
                        var12 = -0.122359686;
                    } else {
                        var12 = -1.2193513;
                    }
                } else {
                    if (input[68] < 32.0) {
                        if (input[40] < 56.0) {
                            var12 = 0.53271717;
                        } else {
                            var12 = -0.05380472;
                        }
                    } else {
                        var12 = 0.66150993;
                    }
                }
            }
        }
    }
    double var13;
    if (input[63] < 4.0) {
        var13 = -0.73064965;
    } else {
        if (input[63] < 13.0) {
            var13 = 1.4818386;
        } else {
            if (input[6] < 17.0) {
                if (input[69] < 66.0) {
                    var13 = 0.90838116;
                } else {
                    var13 = -0.07589065;
                }
            } else {
                if (input[68] < -14.676823) {
                    var13 = -1.0045376;
                } else {
                    if (input[42] < 8.0) {
                        var13 = 0.50645894;
                    } else {
                        if (input[41] < 18.0) {
                            var13 = -0.87604827;
                        } else {
                            if (input[37] < 166400.0) {
                                if (input[38] < 0.17931697) {
                                    var13 = -0.23451495;
                                } else {
                                    var13 = 0.069507;
                                }
                            } else {
                                var13 = 0.57253265;
                            }
                        }
                    }
                }
            }
        }
    }
    double var14;
    if (input[31] < 214.856) {
        if (input[45] < 12.0) {
            var14 = -0.8028986;
        } else {
            if (input[69] < 14.374092) {
                var14 = 0.88611203;
            } else {
                if (input[31] < 9.64565) {
                    var14 = 0.6129294;
                } else {
                    if (input[74] < 7436.0) {
                        var14 = -1.0683635;
                    } else {
                        if (input[58] < 2.0) {
                            if (input[40] < 26.0) {
                                var14 = -0.7089333;
                            } else {
                                if (input[68] < 12.3415) {
                                    if (input[69] < 193.0) {
                                        var14 = -0.0491963;
                                    } else {
                                        var14 = 0.43144685;
                                    }
                                } else {
                                    var14 = 0.6902009;
                                }
                            }
                        } else {
                            var14 = 0.59706944;
                        }
                    }
                }
            }
        }
    } else {
        var14 = 0.85272455;
    }
    double var15;
    if (input[39] < 5.0) {
        var15 = 0.84668505;
    } else {
        if (input[66] < 0.0154215) {
            if (input[67] < -23.1496) {
                if (input[68] < -11.5) {
                    if (input[38] < 0.6637725) {
                        var15 = 0.20683394;
                    } else {
                        var15 = -0.58590186;
                    }
                } else {
                    var15 = 0.8208464;
                }
            } else {
                if (input[41] < 13.0) {
                    var15 = -1.4961308;
                } else {
                    var15 = 0.06519494;
                }
            }
        } else {
            if (input[4] < 13.0) {
                var15 = 1.1414878;
            } else {
                if (input[67] < 84.0) {
                    if (input[68] < 11.0) {
                        var15 = -0.010561201;
                    } else {
                        var15 = 0.76010686;
                    }
                } else {
                    var15 = -0.6413848;
                }
            }
        }
    }
    double var16;
    if (input[62] < 13.0) {
        if (input[49] < 14.0) {
            var16 = -0.48082972;
        } else {
            var16 = 1.2786764;
        }
    } else {
        if (input[67] < -1.99544) {
            if (input[72] < 436.0) {
                var16 = -1.1900657;
            } else {
                if (input[31] < 32.2689) {
                    var16 = 0.2852389;
                } else {
                    var16 = -0.4156251;
                }
            }
        } else {
            if (input[71] < 68.8379) {
                var16 = 0.91508687;
            } else {
                if (input[29] < 48.0) {
                    if (input[68] < 32.0) {
                        var16 = 0.039857294;
                    } else {
                        var16 = 0.7763917;
                    }
                } else {
                    if (input[75] < 128592.0) {
                        var16 = -0.83908015;
                    } else {
                        var16 = -0.010975031;
                    }
                }
            }
        }
    }
    double var17;
    if (input[75] < 168.0) {
        var17 = 0.82113713;
    } else {
        if (input[38] < 0.79260135) {
            if (input[30] < -18.0) {
                var17 = 0.77340335;
            } else {
                if (input[31] < 26.1926) {
                    if (input[38] < 0.57493377) {
                        var17 = -1.0928632;
                    } else {
                        if (input[71] < 52.14) {
                            var17 = -0.5150559;
                        } else {
                            var17 = 0.2877773;
                        }
                    }
                } else {
                    if (input[70] < 65.48465) {
                        var17 = 1.1735333;
                    } else {
                        if (input[26] < 36.0) {
                            if (input[4] < 621.0) {
                                var17 = 0.21311872;
                            } else {
                                var17 = -0.0715855;
                            }
                        } else {
                            var17 = -0.35683787;
                        }
                    }
                }
            }
        } else {
            var17 = -0.6771995;
        }
    }
    double var18;
    if (input[2] < 7.0) {
        if (input[33] < 37.125) {
            if (input[45] < 60.0) {
                var18 = 0.5423929;
            } else {
                var18 = -0.4310256;
            }
        } else {
            var18 = 1.240461;
        }
    } else {
        if (input[59] < 5.0) {
            if (input[7] < 47.0) {
                var18 = 0.26779148;
            } else {
                var18 = -1.5227498;
            }
        } else {
            if (input[59] < 7.0) {
                var18 = 0.53611815;
            } else {
                if (input[8] < 159.0) {
                    var18 = -0.9749102;
                } else {
                    if (input[69] < 159.0) {
                        var18 = 0.5600788;
                    } else {
                        if (input[69] < 193.0) {
                            var18 = -0.03218087;
                        } else {
                            var18 = 0.49117404;
                        }
                    }
                }
            }
        }
    }
    double var19;
    if (input[70] < 6.08729) {
        var19 = 0.8646475;
    } else {
        if (input[70] < 26.3153) {
            var19 = -0.6287634;
        } else {
            if (input[70] < 44.6885) {
                var19 = 0.94814247;
            } else {
                if (input[29] < -9.84265) {
                    if (input[61] < 17.0) {
                        var19 = -0.9827733;
                    } else {
                        var19 = 0.039903086;
                    }
                } else {
                    if (input[3] < 9.0) {
                        var19 = 0.6309908;
                    } else {
                        if (input[40] < 19.0) {
                            var19 = -0.5720746;
                        } else {
                            if (input[42] < 643.0) {
                                if (input[2] < 61.0) {
                                    var19 = -0.05921637;
                                } else {
                                    var19 = 0.69384986;
                                }
                            } else {
                                var19 = -0.05443213;
                            }
                        }
                    }
                }
            }
        }
    }
    return 1.0 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_NUMPOINTSPERCELL_3D
