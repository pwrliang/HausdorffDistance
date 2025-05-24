
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
inline double PredictMaxHit_3D(double * input) {
    double var0;
    if (input[42] < 10.0) {
        if (input[38] < 47.0) {
            if (input[21] < 129277.3) {
                if (input[19] < 24.0) {
                    var0 = 55.246754;
                } else {
                    if (input[11] < 17.0) {
                        if (input[33] >= 198.0) {
                            var0 = -22.812244;
                        } else {
                            if (input[9] < 153.0) {
                                if (input[36] < 7.0) {
                                    var0 = 17.173216;
                                } else {
                                    var0 = -9.573333;
                                }
                            } else {
                                var0 = 35.48312;
                            }
                        }
                    } else {
                        if (input[29] < 0.5099953) {
                            if (input[26] < 211.0) {
                                var0 = -8.462337;
                            } else {
                                var0 = 31.727472;
                            }
                        } else {
                            var0 = 44.278194;
                        }
                    }
                }
            } else {
                var0 = -17.551786;
            }
        } else {
            if (input[1] < 1556976.0) {
                if (input[21] < 0.42733005) {
                    var0 = 24.119482;
                } else {
                    if (input[36] < 53.0) {
                        if (input[21] < 0.456865) {
                            if (input[12] < 76533.0) {
                                if (input[21] < 0.44625092) {
                                    var0 = 61.914967;
                                } else {
                                    if (input[8] < 0.17693326) {
                                        var0 = 22.664936;
                                    } else {
                                        if (input[29] < 0.17970072) {
                                            var0 = 29.527472;
                                        } else {
                                            var0 = 55.440002;
                                        }
                                    }
                                }
                            } else {
                                var0 = 26.973215;
                            }
                        } else {
                            if (input[17] < 49.0) {
                                if (input[21] < 0.4688595) {
                                    var0 = 59.83929;
                                } else {
                                    if (input[29] < 0.18911882) {
                                        var0 = 20.77381;
                                    } else {
                                        if (input[8] < 0.18694437) {
                                            var0 = 28.483118;
                                        } else {
                                            var0 = 57.57403;
                                        }
                                    }
                                }
                            } else {
                                var0 = 66.21755;
                            }
                        }
                    } else {
                        var0 = 30.830612;
                    }
                }
            } else {
                if (input[41] < 151900.0) {
                    if (input[29] < 0.18245564) {
                        var0 = -9.625974;
                    } else {
                        var0 = 29.355844;
                    }
                } else {
                    var0 = 49.46667;
                }
            }
        }
    } else {
        if (input[22] < 1169456.0) {
            if (input[26] < 208.0) {
                var0 = -26.455952;
            } else {
                var0 = -17.753246;
            }
        } else {
            if (input[29] < 0.18839817) {
                if (input[21] < 0.45253512) {
                    var0 = -6.183673;
                } else {
                    var0 = 36.046753;
                }
            } else {
                if (input[14] < 48734.0) {
                    var0 = -15.36;
                } else {
                    var0 = 0.8467541;
                }
            }
        }
    }
    double var1;
    if (input[42] < 10.0) {
        if (input[30] >= 30.0) {
            if (input[10] < 910.0) {
                if (input[36] < 12.0) {
                    if (input[22] < 3022.0) {
                        var1 = 0.013269222;
                    } else {
                        var1 = 31.063215;
                    }
                } else {
                    if (input[43] < 1.4372128) {
                        var1 = -4.222858;
                    } else {
                        var1 = -19.891577;
                    }
                }
            } else {
                var1 = 32.81558;
            }
        } else {
            if (input[35] < 29866.0) {
                if (input[36] < 48.0) {
                    if (input[38] < 5.0) {
                        var1 = 10.018962;
                    } else {
                        var1 = 35.227566;
                    }
                } else {
                    var1 = 2.5173392;
                }
            } else {
                if (input[12] < 79901.0) {
                    if (input[43] < 259.25085) {
                        if (input[42] < 0.5067798) {
                            if (input[2] < 52.0) {
                                var1 = 15.427365;
                            } else {
                                if (input[16] < 27.0) {
                                    var1 = 29.755768;
                                } else {
                                    var1 = 49.084953;
                                }
                            }
                        } else {
                            var1 = 50.56285;
                        }
                    } else {
                        if (input[22] < 1512564.0) {
                            if (input[38] < 50.0) {
                                var1 = 32.220116;
                            } else {
                                var1 = -2.199643;
                            }
                        } else {
                            if (input[23] < 43.0) {
                                if (input[23] < 40.0) {
                                    var1 = 36.413506;
                                } else {
                                    var1 = 54.40812;
                                }
                            } else {
                                var1 = 29.60141;
                            }
                        }
                    }
                } else {
                    if (input[42] < 5.55188) {
                        var1 = 45.38229;
                    } else {
                        if (input[14] < 52009.0) {
                            var1 = -2.0943327;
                        } else {
                            var1 = 21.647158;
                        }
                    }
                }
            }
        }
    } else {
        if (input[22] < 1169456.0) {
            if (input[26] < 203.0) {
                var1 = -21.435362;
            } else {
                var1 = -15.421915;
            }
        } else {
            if (input[0] < 0.45641467) {
                var1 = 14.581477;
            } else {
                if (input[14] < 47067.0) {
                    var1 = -15.524884;
                } else {
                    if (input[35] < 41978.0) {
                        var1 = 13.538648;
                    } else {
                        var1 = -10.14787;
                    }
                }
            }
        }
    }
    double var2;
    if (input[42] < 11.0) {
        if (input[22] < 1246535.0) {
            if (input[20] < 147.0) {
                var2 = 30.59606;
            } else {
                if (input[8] < 0.71798265) {
                    if (input[11] < 18.0) {
                        if (input[0] < 30798.707) {
                            if (input[12] < 49.0) {
                                var2 = -18.4849;
                            } else {
                                var2 = -8.980851;
                            }
                        } else {
                            var2 = 2.307049;
                        }
                    } else {
                        if (input[41] < 125685.0) {
                            if (input[30] < 30.0) {
                                var2 = 31.492659;
                            } else {
                                var2 = -0.9253387;
                            }
                        } else {
                            var2 = -6.002394;
                        }
                    }
                } else {
                    var2 = 27.336039;
                }
            }
        } else {
            if (input[1] < 1570188.0) {
                if (input[42] < 8.0) {
                    if (input[21] < 0.456865) {
                        if (input[8] < 0.17066199) {
                            var2 = 7.2519073;
                        } else {
                            if (input[0] < 0.4551031) {
                                if (input[17] < 25.0) {
                                    var2 = 23.941538;
                                } else {
                                    if (input[22] < 1466032.0) {
                                        var2 = 49.284145;
                                    } else {
                                        var2 = 32.5276;
                                    }
                                }
                            } else {
                                if (input[42] < 2.4341278) {
                                    var2 = 40.16674;
                                } else {
                                    if (input[8] < 0.18746185) {
                                        var2 = -10.041706;
                                    } else {
                                        var2 = 20.346195;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[2] < 65.0) {
                            if (input[36] < 51.0) {
                                var2 = 46.21867;
                            } else {
                                if (input[23] < 44.0) {
                                    if (input[8] < 0.18340074) {
                                        var2 = 30.025562;
                                    } else {
                                        var2 = 48.613483;
                                    }
                                } else {
                                    var2 = 20.68215;
                                }
                            }
                        } else {
                            var2 = 20.045734;
                        }
                    }
                } else {
                    var2 = 15.822825;
                }
            } else {
                if (input[42] < 5.55188) {
                    if (input[5] < 179.7586) {
                        var2 = 10.855027;
                    } else {
                        var2 = 37.673218;
                    }
                } else {
                    if (input[5] < 190.0) {
                        var2 = -12.592006;
                    } else {
                        var2 = 19.674007;
                    }
                }
            }
        }
    } else {
        if (input[36] < 51.0) {
            if (input[42] < 15.0) {
                var2 = -5.1182685;
            } else {
                if (input[23] < 126.0) {
                    var2 = -17.366299;
                } else {
                    var2 = -11.108385;
                }
            }
        } else {
            var2 = 6.758532;
        }
    }
    double var3;
    if (input[42] < 10.0) {
        if (input[30] >= 30.0) {
            if (input[10] >= 910.0) {
                var3 = 24.801144;
            } else {
                if (input[36] < 12.0) {
                    if (input[20] < 147.0) {
                        var3 = 30.652613;
                    } else {
                        if (input[26] < 0.658773) {
                            var3 = 16.254438;
                        } else {
                            var3 = -9.898638;
                        }
                    }
                } else {
                    if (input[31] < 201.0) {
                        if (input[30] < 92.0) {
                            var3 = -14.593284;
                        } else {
                            var3 = 0.6921689;
                        }
                    } else {
                        var3 = -19.778502;
                    }
                }
            }
        } else {
            if (input[21] < 0.42733005) {
                var3 = 3.7104855;
            } else {
                if (input[19] < 76698.0) {
                    if (input[0] < 3131.8567) {
                        if (input[36] < 53.0) {
                            if (input[2] < 48.0) {
                                var3 = 39.47405;
                            } else {
                                if (input[26] < 209.0) {
                                    var3 = 12.363364;
                                } else {
                                    if (input[23] < 40.0) {
                                        var3 = 17.075254;
                                    } else {
                                        if (input[5] < 189.0) {
                                            var3 = 32.641308;
                                        } else {
                                            var3 = 19.576828;
                                        }
                                    }
                                }
                            }
                        } else {
                            var3 = 13.328171;
                        }
                    } else {
                        var3 = 10.874739;
                    }
                } else {
                    if (input[21] < 0.45125267) {
                        if (input[29] < 0.17728063) {
                            var3 = 15.476985;
                        } else {
                            var3 = -18.032978;
                        }
                    } else {
                        if (input[42] < 5.55188) {
                            var3 = 42.804497;
                        } else {
                            if (input[35] < 44825.0) {
                                var3 = 26.449814;
                            } else {
                                var3 = -4.0671506;
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[22] < 1169456.0) {
            if (input[39] < 36.0) {
                var3 = -14.215388;
            } else {
                var3 = -9.343116;
            }
        } else {
            if (input[29] < 0.18839817) {
                if (input[29] < 0.18209875) {
                    var3 = -8.906297;
                } else {
                    var3 = 23.605656;
                }
            } else {
                if (input[8] < 0.19248591) {
                    var3 = -14.08342;
                } else {
                    var3 = -0.1809946;
                }
            }
        }
    }
    double var4;
    if (input[42] < 11.0) {
        if (input[22] < 1246535.0) {
            if (input[19] < 24.0) {
                var4 = 25.528902;
            } else {
                if (input[31] >= 96.0) {
                    var4 = -11.656046;
                } else {
                    if (input[42] < 0.096969485) {
                        var4 = -7.6233277;
                    } else {
                        if (input[14] < 31.0) {
                            if (input[5] < 0.6086972) {
                                var4 = -13.107144;
                            } else {
                                var4 = 6.7303786;
                            }
                        } else {
                            if (input[14] < 160.0) {
                                var4 = 30.724451;
                            } else {
                                if (input[41] < 125685.0) {
                                    var4 = 12.101151;
                                } else {
                                    var4 = -5.7381115;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[1] < 1504409.0) {
                if (input[0] < 0.46758872) {
                    if (input[22] < 1393595.0) {
                        if (input[12] < 70534.0) {
                            if (input[8] < 0.1804688) {
                                var4 = 20.85348;
                            } else {
                                var4 = 36.13859;
                            }
                        } else {
                            var4 = 14.819345;
                        }
                    } else {
                        if (input[33] < 77954.0) {
                            if (input[33] < 76207.0) {
                                if (input[12] < 69018.0) {
                                    if (input[11] < 25.0) {
                                        if (input[18] < 21.0) {
                                            var4 = 14.579056;
                                        } else {
                                            var4 = -4.976471;
                                        }
                                    } else {
                                        var4 = 21.640488;
                                    }
                                } else {
                                    if (input[20] < 147000.0) {
                                        var4 = 34.314266;
                                    } else {
                                        var4 = 19.778143;
                                    }
                                }
                            } else {
                                var4 = -1.767338;
                            }
                        } else {
                            if (input[42] < 2.4341278) {
                                if (input[21] < 0.46853396) {
                                    var4 = 26.70397;
                                } else {
                                    var4 = 13.655782;
                                }
                            } else {
                                var4 = 33.566715;
                            }
                        }
                    }
                } else {
                    if (input[17] < 49.0) {
                        var4 = 23.027609;
                    } else {
                        var4 = 38.60353;
                    }
                }
            } else {
                if (input[21] < 0.45125267) {
                    if (input[23] < 44.0) {
                        var4 = 16.660734;
                    } else {
                        var4 = -23.528725;
                    }
                } else {
                    if (input[35] < 44302.0) {
                        if (input[23] < 41.0) {
                            var4 = 18.063627;
                        } else {
                            var4 = 37.080112;
                        }
                    } else {
                        if (input[38] < 51.0) {
                            var4 = -4.684801;
                        } else {
                            if (input[21] < 0.4688595) {
                                var4 = 6.7628326;
                            } else {
                                var4 = 37.71139;
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[26] < 209.0) {
            if (input[32] < 17.0) {
                var4 = -3.730859;
            } else {
                if (input[33] < 13773.0) {
                    var4 = -11.056988;
                } else {
                    var4 = -16.497892;
                }
            }
        } else {
            if (input[0] < 0.45641467) {
                var4 = 9.971502;
            } else {
                var4 = -7.9752173;
            }
        }
    }
    return 144.14285714285714 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_MAXHIT_3D
