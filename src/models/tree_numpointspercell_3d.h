
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
    if (input[6] < 5677.0) {
        var0 = -5.6759524;
    } else {
        if (input[41] < 41.0) {
            var0 = 3.198301;
        } else {
            if (input[44] < 45311.0) {
                if (input[6] < 37037.0) {
                    if (input[42] < 660.0) {
                        var0 = -0.043297056;
                    } else {
                        var0 = 0.8038392;
                    }
                } else {
                    if (input[4] < 697.0) {
                        var0 = -0.8775639;
                    } else {
                        if (input[36] < 1497908.0) {
                            var0 = 0.31473455;
                        } else {
                            var0 = -0.39142963;
                        }
                    }
                }
            } else {
                if (input[33] < 139.0) {
                    var0 = 1.4075205;
                } else {
                    if (input[28] < 51.0) {
                        var0 = -0.1617931;
                    } else {
                        var0 = 0.6671475;
                    }
                }
            }
        }
    }
    double var1;
    if (input[29] < 61.0) {
        if (input[74] < 1170092.0) {
            if (input[33] < 139.0) {
                var1 = 0.8911651;
            } else {
                var1 = 1.9614538;
            }
        } else {
            if (input[66] < 54.0) {
                if (input[42] < 776.0) {
                    if (input[33] < 137.0) {
                        var1 = 0.6950267;
                    } else {
                        if (input[42] < 695.0) {
                            if (input[47] < 68764.0) {
                                var1 = -0.08151429;
                            } else {
                                var1 = 0.55057114;
                            }
                        } else {
                            var1 = -0.41896707;
                        }
                    }
                } else {
                    var1 = 0.6788842;
                }
            } else {
                var1 = -0.81133175;
            }
        }
    } else {
        if (input[45] < 35932.0) {
            var1 = 0.17666826;
        } else {
            var1 = -2.8298054;
        }
    }
    double var2;
    if (input[30] < 35.0) {
        if (input[70] < 190.0) {
            var2 = 0.8907763;
        } else {
            if (input[0] < 0.1947205) {
                if (input[66] < 53.0) {
                    if (input[9] < 56395.0) {
                        var2 = -0.36591986;
                    } else {
                        if (input[47] < 56971.0) {
                            var2 = 0.57070345;
                        } else {
                            var2 = 0.043420542;
                        }
                    }
                } else {
                    if (input[33] < 138.0) {
                        var2 = 0.14835294;
                    } else {
                        var2 = -0.6437185;
                    }
                }
            } else {
                var2 = 0.6293667;
            }
        }
    } else {
        if (input[48] < 73519.0) {
            if (input[28] < 70.0) {
                var2 = 0.19794272;
            } else {
                var2 = -0.433429;
            }
        } else {
            var2 = -1.3696336;
        }
    }
    double var3;
    if (input[67] < 87.0) {
        if (input[0] < 0.17253108) {
            var3 = 0.45838752;
        } else {
            if (input[48] < 70023.0) {
                if (input[42] < 656.0) {
                    if (input[48] < 3832.0) {
                        if (input[44] < 29411.0) {
                            var3 = -0.6978604;
                        } else {
                            var3 = 0.08941043;
                        }
                    } else {
                        var3 = 0.0944366;
                    }
                } else {
                    if (input[10] < 68867.0) {
                        var3 = -0.1060392;
                    } else {
                        var3 = -1.0492382;
                    }
                }
            } else {
                if (input[75] < 148614.0) {
                    var3 = 0.5376758;
                } else {
                    if (input[24] < 24.0) {
                        var3 = 0.25286016;
                    } else {
                        var3 = -0.1498244;
                    }
                }
            }
        }
    } else {
        var3 = 0.6236603;
    }
    double var4;
    if (input[70] < 180.0) {
        var4 = 0.49508753;
    } else {
        if (input[38] < 0.28502557) {
            if (input[43] < 9929.0) {
                var4 = 0.7220827;
            } else {
                if (input[39] < 53.0) {
                    if (input[43] < 11343.0) {
                        var4 = -0.3618441;
                    } else {
                        if (input[4] < 772.0) {
                            if (input[48] < 56872.0) {
                                var4 = 0.50750995;
                            } else {
                                if (input[48] < 71372.0) {
                                    if (input[10] < 72172.0) {
                                        var4 = -0.53010815;
                                    } else {
                                        var4 = 0.0020545737;
                                    }
                                } else {
                                    var4 = 0.07705383;
                                }
                            }
                        } else {
                            var4 = 0.61823004;
                        }
                    }
                } else {
                    var4 = -0.34669948;
                }
            }
        } else {
            var4 = -0.59931064;
        }
    }
    double var5;
    if (input[43] < 423.0) {
        var5 = -0.5564379;
    } else {
        if (input[67] < 84.0) {
            if (input[36] < 1497908.0) {
                if (input[28] < 53.0) {
                    if (input[31] < 185.0) {
                        var5 = 0.69584227;
                    } else {
                        if (input[42] < 736.0) {
                            var5 = -0.0022599695;
                        } else {
                            var5 = 0.6858005;
                        }
                    }
                } else {
                    if (input[48] < 5771.0) {
                        var5 = 0.4170282;
                    } else {
                        var5 = -0.113117374;
                    }
                }
            } else {
                if (input[71] < 137.0) {
                    var5 = 0.14995317;
                } else {
                    if (input[66] < 49.0) {
                        var5 = 0.07425893;
                    } else {
                        var5 = -0.57726926;
                    }
                }
            }
        } else {
            var5 = 0.7333263;
        }
    }
    double var6;
    if (input[69] < 124.0) {
        var6 = 0.48924008;
    } else {
        if (input[72] < 34.0) {
            if (input[73] < 59005.0) {
                var6 = 0.52621603;
            } else {
                if (input[63] < 19.0) {
                    var6 = -0.3510472;
                } else {
                    if (input[68] < 4.0) {
                        if (input[38] < 0.19248746) {
                            if (input[45] < 34353.0) {
                                var6 = -0.5273034;
                            } else {
                                var6 = -0.03412708;
                            }
                        } else {
                            var6 = 0.45891672;
                        }
                    } else {
                        if (input[38] < 0.18662947) {
                            var6 = 0.54007405;
                        } else {
                            var6 = -0.013522384;
                        }
                    }
                }
            }
        } else {
            if (input[42] < 149.0) {
                var6 = -0.95213526;
            } else {
                var6 = 0.030127857;
            }
        }
    }
    double var7;
    if (input[28] < 113.0) {
        if (input[42] < 812.0) {
            if (input[75] < 155805.0) {
                if (input[75] < 152880.0) {
                    if (input[73] < 73180.0) {
                        var7 = 0.027108712;
                    } else {
                        var7 = -0.39211854;
                    }
                } else {
                    var7 = 0.3425265;
                }
            } else {
                if (input[4] < 594.0) {
                    if (input[67] < 42.0) {
                        var7 = -0.22472417;
                    } else {
                        var7 = 0.28446743;
                    }
                } else {
                    if (input[67] < 43.0) {
                        var7 = -0.0017573343;
                    } else {
                        var7 = -0.94901735;
                    }
                }
            }
        } else {
            var7 = 0.3245698;
        }
    } else {
        if (input[67] < 45.0) {
            var7 = 0.021968866;
        } else {
            var7 = 0.603291;
        }
    }
    double var8;
    if (input[64] < 40.0) {
        if (input[47] < 4118.0) {
            var8 = 0.5219745;
        } else {
            if (input[45] < 5993.0) {
                var8 = -0.5233795;
            } else {
                if (input[1] < 51.0) {
                    if (input[8] < 45018.0) {
                        if (input[70] < 217.0) {
                            var8 = 0.054568436;
                        } else {
                            var8 = -0.38246784;
                        }
                    } else {
                        if (input[32] < 213.0) {
                            var8 = 0.6172324;
                        } else {
                            var8 = 0.08397407;
                        }
                    }
                } else {
                    if (input[67] < 43.0) {
                        var8 = 0.3097492;
                    } else {
                        if (input[68] < 3.0) {
                            var8 = -0.72695416;
                        } else {
                            var8 = 0.051237393;
                        }
                    }
                }
            }
        }
    } else {
        var8 = -0.4022312;
    }
    double var9;
    if (input[43] < 345.0) {
        var9 = -0.35505518;
    } else {
        if (input[70] < 194.0) {
            if (input[4] < 663.0) {
                var9 = 0.7408942;
            } else {
                var9 = -0.009446747;
            }
        } else {
            if (input[68] < 11.0) {
                if (input[43] < 14635.0) {
                    if (input[29] < 50.0) {
                        if (input[6] < 36888.0) {
                            var9 = 0.40617308;
                        } else {
                            var9 = -0.088371344;
                        }
                    } else {
                        var9 = -0.3385655;
                    }
                } else {
                    if (input[43] < 15343.0) {
                        var9 = 0.53865;
                    } else {
                        var9 = 0.044052977;
                    }
                }
            } else {
                if (input[71] < 133.0) {
                    var9 = -0.033006158;
                } else {
                    var9 = -0.5857273;
                }
            }
        }
    }
    double var10;
    if (input[32] < 219.0) {
        if (input[37] < 165200.0) {
            if (input[23] < 40.0) {
                if (input[32] < 217.0) {
                    if (input[36] < 1557884.0) {
                        if (input[30] < 1.0) {
                            if (input[43] < 13265.0) {
                                if (input[42] < 622.0) {
                                    var10 = 0.12436105;
                                } else {
                                    var10 = -0.30755845;
                                }
                            } else {
                                var10 = 0.300364;
                            }
                        } else {
                            if (input[0] < 0.18955831) {
                                var10 = -0.24753194;
                            } else {
                                var10 = 0.026102528;
                            }
                        }
                    } else {
                        var10 = -0.56327975;
                    }
                } else {
                    var10 = 0.28046113;
                }
            } else {
                var10 = 0.38699338;
            }
        } else {
            var10 = 0.56306225;
        }
    } else {
        var10 = -0.4913295;
    }
    double var11;
    if (input[74] < 1681433.0) {
        if (input[49] < 84447.0) {
            if (input[48] < 82091.0) {
                if (input[38] < 0.16529615) {
                    var11 = -0.34352407;
                } else {
                    if (input[8] < 65589.0) {
                        if (input[2] < 24.0) {
                            if (input[62] < 20.0) {
                                var11 = 0.029346926;
                            } else {
                                var11 = -0.40666795;
                            }
                        } else {
                            if (input[9] < 2587.0) {
                                var11 = 0.6113042;
                            } else {
                                var11 = 0.07345225;
                            }
                        }
                    } else {
                        if (input[38] < 0.18797697) {
                            var11 = 0.011547239;
                        } else {
                            var11 = -0.46888706;
                        }
                    }
                }
            } else {
                var11 = -0.2883581;
            }
        } else {
            var11 = 0.47202727;
        }
    } else {
        var11 = -0.32532066;
    }
    double var12;
    if (input[33] < 106.0) {
        if (input[69] < 188.0) {
            var12 = 0.49801525;
        } else {
            var12 = -0.13923304;
        }
    } else {
        if (input[32] < 209.0) {
            if (input[35] < 6502.0) {
                if (input[47] < 65764.0) {
                    if (input[36] < 77001.0) {
                        var12 = -0.1299541;
                    } else {
                        var12 = 0.43983445;
                    }
                } else {
                    if (input[11] < 2422.0) {
                        var12 = -0.022561496;
                    } else {
                        var12 = -0.45663556;
                    }
                }
            } else {
                var12 = -0.48178235;
            }
        } else {
            if (input[9] < 49935.0) {
                var12 = 0.67375875;
            } else {
                if (input[33] < 137.0) {
                    var12 = 0.26959008;
                } else {
                    var12 = -0.059578475;
                }
            }
        }
    }
    double var13;
    if (input[43] < 327.0) {
        var13 = -0.3746331;
    } else {
        if (input[39] < 20.0) {
            var13 = 0.33852065;
        } else {
            if (input[45] < 4622.0) {
                var13 = -0.37825862;
            } else {
                if (input[32] < 219.0) {
                    if (input[4] < 785.0) {
                        if (input[69] < 184.0) {
                            if (input[0] < 0.18853791) {
                                if (input[4] < 635.0) {
                                    var13 = -0.21813159;
                                } else {
                                    var13 = 0.19412467;
                                }
                            } else {
                                var13 = 0.4801704;
                            }
                        } else {
                            if (input[0] < 0.18588285) {
                                var13 = 0.09628039;
                            } else {
                                var13 = -0.12214034;
                            }
                        }
                    } else {
                        var13 = 0.6736832;
                    }
                } else {
                    var13 = -0.36053845;
                }
            }
        }
    }
    double var14;
    if (input[21] < 17.0) {
        if (input[0] < 0.17253108) {
            var14 = 0.38652334;
        } else {
            if (input[70] < 180.0) {
                if (input[66] < 65.0) {
                    var14 = 0.5463773;
                } else {
                    var14 = -0.015820859;
                }
            } else {
                if (input[44] < 1261.0) {
                    var14 = -0.5172945;
                } else {
                    if (input[28] < 53.0) {
                        if (input[38] < 0.18910873) {
                            if (input[29] < 41.0) {
                                var14 = -0.41951403;
                            } else {
                                var14 = 0.04332532;
                            }
                        } else {
                            var14 = 0.27284375;
                        }
                    } else {
                        if (input[4] < 590.0) {
                            var14 = 0.00637891;
                        } else {
                            var14 = -0.33578297;
                        }
                    }
                }
            }
        }
    } else {
        var14 = -0.25953135;
    }
    double var15;
    if (input[66] < 53.0) {
        if (input[10] < 83363.0) {
            if (input[36] < 1502639.0) {
                if (input[73] < 65237.0) {
                    var15 = 0.32026047;
                } else {
                    var15 = 0.018768122;
                }
            } else {
                if (input[32] < 214.0) {
                    var15 = -0.63991153;
                } else {
                    var15 = 0.0033106392;
                }
            }
        } else {
            var15 = 0.4389352;
        }
    } else {
        if (input[71] < 139.0) {
            if (input[0] < 0.1882564) {
                if (input[33] < 141.0) {
                    var15 = -0.3919242;
                } else {
                    var15 = 0.16765729;
                }
            } else {
                var15 = 0.24251567;
            }
        } else {
            if (input[29] < 48.0) {
                var15 = -0.49983516;
            } else {
                var15 = -0.008236124;
            }
        }
    }
    double var16;
    if (input[38] < 0.16529615) {
        var16 = -0.28169322;
    } else {
        if (input[39] < 53.0) {
            if (input[69] < 191.0) {
                if (input[0] < 0.17290775) {
                    if (input[28] < 50.0) {
                        var16 = -0.49083772;
                    } else {
                        var16 = 0.04152722;
                    }
                } else {
                    if (input[28] < 49.0) {
                        if (input[39] < 49.0) {
                            var16 = 0.024078988;
                        } else {
                            var16 = 0.4365606;
                        }
                    } else {
                        if (input[31] < 188.0) {
                            var16 = 0.023607587;
                        } else {
                            var16 = -0.32862023;
                        }
                    }
                }
            } else {
                var16 = 0.39607656;
            }
        } else {
            if (input[28] < 54.0) {
                var16 = -0.3729274;
            } else {
                var16 = -0.025655134;
            }
        }
    }
    double var17;
    if (input[42] < 71.0) {
        var17 = -0.3016858;
    } else {
        if (input[66] < 112.0) {
            if (input[29] < 44.0) {
                if (input[38] < 0.1862924) {
                    if (input[45] < 39506.0) {
                        if (input[47] < 67903.0) {
                            var17 = -0.095042944;
                        } else {
                            var17 = 0.49550697;
                        }
                    } else {
                        var17 = -0.39500487;
                    }
                } else {
                    if (input[6] < 40648.0) {
                        var17 = -0.097281225;
                    } else {
                        var17 = 0.5218501;
                    }
                }
            } else {
                if (input[29] < 45.0) {
                    var17 = -0.32039884;
                } else {
                    if (input[46] < 63535.0) {
                        var17 = 0.0039518224;
                    } else {
                        var17 = -0.24541095;
                    }
                }
            }
        } else {
            var17 = 0.39500368;
        }
    }
    double var18;
    if (input[32] < 139.0) {
        var18 = 0.2747166;
    } else {
        if (input[2] < 24.0) {
            if (input[46] < 43204.0) {
                var18 = -0.50435704;
            } else {
                var18 = -0.019942317;
            }
        } else {
            if (input[28] < 112.0) {
                if (input[43] < 17374.0) {
                    if (input[67] < 43.0) {
                        if (input[24] < 23.0) {
                            var18 = 0.58202666;
                        } else {
                            var18 = 0.026062716;
                        }
                    } else {
                        if (input[35] < 58422.0) {
                            var18 = 0.16962187;
                        } else {
                            var18 = -0.057987224;
                        }
                    }
                } else {
                    if (input[43] < 18546.0) {
                        var18 = -0.6923053;
                    } else {
                        var18 = 0.015673568;
                    }
                }
            } else {
                var18 = 0.38237008;
            }
        }
    }
    double var19;
    if (input[66] < 120.0) {
        if (input[66] < 64.0) {
            if (input[9] < 79901.0) {
                if (input[28] < 53.0) {
                    if (input[46] < 40995.0) {
                        if (input[0] < 0.17896488) {
                            var19 = 0.012518217;
                        } else {
                            var19 = 0.54317045;
                        }
                    } else {
                        var19 = 0.031696003;
                    }
                } else {
                    if (input[48] < 1711.0) {
                        var19 = 0.33748633;
                    } else {
                        var19 = -0.104598515;
                    }
                }
            } else {
                if (input[6] < 49297.0) {
                    var19 = 0.06420746;
                } else {
                    var19 = -0.4637228;
                }
            }
        } else {
            if (input[10] < 5677.0) {
                var19 = 0.084094666;
            } else {
                var19 = -0.5679218;
            }
        }
    } else {
        var19 = 0.282307;
    }
    return 13.988 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_NUMPOINTSPERCELL_3D
