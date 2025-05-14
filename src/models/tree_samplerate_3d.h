
#include <math.h>
#ifndef DECISION_TREE_SAMPLERATE_3D
#define DECISION_TREE_SAMPLERATE_3D
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
};

*/
inline double PredictSampleRate_3D(double * input) {
    double var0;
    if (input[74] < 1196208.0) {
        if (input[38] < 0.22829175) {
            if (input[69] < 129.0) {
                if (input[40] < 30.0) {
                    var0 = 0.0008563199;
                } else {
                    var0 = -0.00020453337;
                }
            } else {
                if (input[69] < 179.0) {
                    if (input[45] < 5163.0) {
                        var0 = 0.00093728484;
                    } else {
                        if (input[29] < 47.0) {
                            var0 = 0.001410012;
                        } else {
                            var0 = 0.0022456597;
                        }
                    }
                } else {
                    if (input[0] < 0.1893898) {
                        if (input[66] < 55.0) {
                            var0 = 0.0011222791;
                        } else {
                            var0 = -0.00010182076;
                        }
                    } else {
                        if (input[28] < 55.0) {
                            var0 = 0.0022184288;
                        } else {
                            var0 = 0.00057728484;
                        }
                    }
                }
            }
        } else {
            if (input[33] < 142.0) {
                if (input[66] < 113.0) {
                    if (input[38] < 0.357069) {
                        if (input[4] < 737.0) {
                            if (input[29] < 96.0) {
                                if (input[40] < 30.0) {
                                    var0 = 0.0027748335;
                                } else {
                                    var0 = 0.001810012;
                                }
                            } else {
                                var0 = 0.0015517621;
                            }
                        } else {
                            var0 = 0.0012127183;
                        }
                    } else {
                        var0 = 0.0007772848;
                    }
                } else {
                    if (input[29] < 46.0) {
                        if (input[4] < 702.0) {
                            var0 = 0.0022827394;
                        } else {
                            var0 = 0.0010850955;
                        }
                    } else {
                        if (input[29] < 50.0) {
                            var0 = 0.00010092118;
                        } else {
                            var0 = 0.0010884289;
                        }
                    }
                }
            } else {
                var0 = 0.00060157495;
            }
        }
    } else {
        if (input[4] < 781.0) {
            if (input[75] < 142443.0) {
                if (input[31] < 189.0) {
                    if (input[73] < 62466.0) {
                        if (input[45] < 32285.0) {
                            if (input[4] < 672.0) {
                                var0 = -0.0007262652;
                            } else {
                                var0 = 0.00038455753;
                            }
                        } else {
                            var0 = 0.0007626237;
                        }
                    } else {
                        if (input[39] < 49.0) {
                            if (input[28] < 49.0) {
                                var0 = 0.000006375729;
                            } else {
                                if (input[43] < 10859.0) {
                                    var0 = -0.000022715178;
                                } else {
                                    var0 = -0.00071206636;
                                }
                            }
                        } else {
                            if (input[38] < 0.18547086) {
                                if (input[68] < 7.0) {
                                    var0 = -0.0007097209;
                                } else {
                                    var0 = -0.00013823781;
                                }
                            } else {
                                if (input[43] < 14930.0) {
                                    var0 = 0.0010617621;
                                } else {
                                    var0 = -0.00037120192;
                                }
                            }
                        }
                    }
                } else {
                    if (input[41] < 49.0) {
                        var0 = -0.00044404296;
                    } else {
                        if (input[48] < 66721.0) {
                            if (input[37] < 150450.0) {
                                var0 = 0.002100921;
                            } else {
                                var0 = 0.000759103;
                            }
                        } else {
                            var0 = 0.00017027908;
                        }
                    }
                }
            } else {
                if (input[68] < 9.0) {
                    if (input[29] < 39.0) {
                        var0 = 0.00015657497;
                    } else {
                        if (input[31] < 194.0) {
                            if (input[39] < 55.0) {
                                if (input[47] < 51676.0) {
                                    if (input[67] < 46.0) {
                                        var0 = -0.0006821979;
                                    } else {
                                        var0 = 0.00030092118;
                                    }
                                } else {
                                    if (input[4] < 749.0) {
                                        if (input[29] < 41.0) {
                                            var0 = -0.00040361224;
                                        } else {
                                            var0 = -0.000707753;
                                        }
                                    } else {
                                        if (input[8] < 52976.0) {
                                            var0 = -0.0005996646;
                                        } else {
                                            var0 = -0.000043054253;
                                        }
                                    }
                                }
                            } else {
                                var0 = -0.000046265213;
                            }
                        } else {
                            var0 = 0.00022092117;
                        }
                    }
                } else {
                    var0 = 0.00011746263;
                }
            }
        } else {
            if (input[66] < 54.0) {
                if (input[68] < 5.0) {
                    if (input[70] < 215.0) {
                        if (input[69] < 187.0) {
                            var0 = -0.00057905423;
                        } else {
                            if (input[38] < 0.18482581) {
                                var0 = -0.00018834478;
                            } else {
                                var0 = 0.0012662791;
                            }
                        }
                    } else {
                        if (input[74] < 1457328.0) {
                            var0 = -0.00028816971;
                        } else {
                            var0 = -0.00071694556;
                        }
                    }
                } else {
                    var0 = 0.0010245576;
                }
            } else {
                var0 = 0.0014202266;
            }
        }
    }
    double var1;
    if (input[68] < 14.0) {
        if (input[75] < 131712.0) {
            if (input[31] < 184.0) {
                if (input[10] < 858.0) {
                    var1 = -0.0006138736;
                } else {
                    var1 = -0.00012643838;
                }
            } else {
                if (input[47] < 63034.0) {
                    if (input[28] < 51.0) {
                        if (input[28] < 49.0) {
                            var1 = 0.00018617428;
                        } else {
                            var1 = -0.0006055032;
                        }
                    } else {
                        if (input[71] < 140.0) {
                            if (input[73] < 59155.0) {
                                var1 = 0.0005656996;
                            } else {
                                var1 = -0.00024694213;
                            }
                        } else {
                            var1 = 0.0012224348;
                        }
                    }
                } else {
                    var1 = 0.0013372417;
                }
            }
        } else {
            if (input[36] < 1582011.0) {
                if (input[32] < 130.0) {
                    var1 = 0.000489387;
                } else {
                    if (input[47] < 55811.0) {
                        if (input[4] < 749.0) {
                            if (input[0] < 0.17950882) {
                                if (input[29] < 46.0) {
                                    if (input[38] < 0.19199656) {
                                        var1 = -0.00048416387;
                                    } else {
                                        var1 = -0.000028692264;
                                    }
                                } else {
                                    var1 = 0.00054677663;
                                }
                            } else {
                                if (input[29] < 43.0) {
                                    var1 = 0.00008437148;
                                } else {
                                    if (input[23] < 30.0) {
                                        if (input[31] < 189.0) {
                                            var1 = -0.0005651323;
                                        } else {
                                            var1 = -0.00012266397;
                                        }
                                    } else {
                                        var1 = 0.000029039476;
                                    }
                                }
                            }
                        } else {
                            var1 = 0.00044038528;
                        }
                    } else {
                        if (input[69] < 182.0) {
                            var1 = 0.00041384675;
                        } else {
                            if (input[73] < 72284.0) {
                                if (input[66] < 54.0) {
                                    if (input[32] < 160.0) {
                                        var1 = 0.00025540238;
                                    } else {
                                        if (input[43] < 14508.0) {
                                            var1 = -0.00042502364;
                                        } else {
                                            var1 = -0.00006415699;
                                        }
                                    }
                                } else {
                                    var1 = 0.00015138717;
                                }
                            } else {
                                if (input[31] < 190.0) {
                                    if (input[5] < 18690.0) {
                                        if (input[71] < 135.0) {
                                            var1 = -0.00025220148;
                                        } else {
                                            var1 = -0.00050531555;
                                        }
                                    } else {
                                        var1 = -0.0002028221;
                                    }
                                } else {
                                    if (input[71] < 143.0) {
                                        if (input[33] < 139.0) {
                                            var1 = -0.00056912826;
                                        } else {
                                            var1 = -0.00020230234;
                                        }
                                    } else {
                                        var1 = 0.00057680067;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[38] < 0.18449563) {
                    if (input[8] < 48645.0) {
                        var1 = 0.0005279866;
                    } else {
                        if (input[28] < 50.0) {
                            if (input[4] < 781.0) {
                                var1 = -0.00038601377;
                            } else {
                                var1 = -0.00071284245;
                            }
                        } else {
                            var1 = -0.00014100867;
                        }
                    }
                } else {
                    if (input[70] < 215.0) {
                        if (input[29] < 40.0) {
                            var1 = 0.00013730633;
                        } else {
                            var1 = 0.0014436299;
                        }
                    } else {
                        if (input[47] < 60772.0) {
                            var1 = 0.00025488468;
                        } else {
                            var1 = -0.00048634122;
                        }
                    }
                }
            }
        }
    } else {
        if (input[6] < 49605.0) {
            if (input[7] < 47837.0) {
                if (input[74] < 998842.0) {
                    if (input[32] < 217.0) {
                        if (input[57] < 1.0) {
                            if (input[35] < 2917.0) {
                                if (input[11] < 1550.0) {
                                    var1 = 0.00088226015;
                                } else {
                                    var1 = -0.00033579112;
                                }
                            } else {
                                if (input[29] < 80.0) {
                                    if (input[41] < 24.0) {
                                        if (input[67] < 107.0) {
                                            var1 = 0.0004781597;
                                        } else {
                                            var1 = 0.0010748105;
                                        }
                                    } else {
                                        if (input[44] < 4567.0) {
                                            var1 = 0.0015081643;
                                        } else {
                                            var1 = 0.00067538314;
                                        }
                                    }
                                } else {
                                    var1 = 0.0021404636;
                                }
                            }
                        } else {
                            var1 = 0.0019036419;
                        }
                    } else {
                        var1 = 0.00013970291;
                    }
                } else {
                    var1 = 0.00005502897;
                }
            } else {
                var1 = -0.0005092299;
            }
        } else {
            if (input[36] < 1601790.0) {
                if (input[71] < 123.0) {
                    var1 = 0.001252079;
                } else {
                    var1 = 0.00223246;
                }
            } else {
                var1 = 0.0009056232;
            }
        }
    }
    double var2;
    if (input[68] < 11.0) {
        if (input[75] < 131712.0) {
            if (input[32] < 209.0) {
                if (input[45] < 33851.0) {
                    var2 = 0.0000040007103;
                } else {
                    var2 = -0.0005435;
                }
            } else {
                if (input[68] < 4.0) {
                    if (input[45] < 32285.0) {
                        var2 = 0.00017667246;
                    } else {
                        var2 = 0.0011914509;
                    }
                } else {
                    if (input[46] < 37961.0) {
                        var2 = 0.00052026234;
                    } else {
                        if (input[67] < 52.0) {
                            var2 = -0.00047649923;
                        } else {
                            var2 = 0.00006495357;
                        }
                    }
                }
            }
        } else {
            if (input[6] < 52733.0) {
                if (input[31] < 190.0) {
                    if (input[32] < 130.0) {
                        var2 = 0.0003114281;
                    } else {
                        if (input[33] < 142.0) {
                            if (input[67] < 47.0) {
                                if (input[41] < 48.0) {
                                    var2 = -0.000067143585;
                                } else {
                                    if (input[71] < 144.0) {
                                        if (input[8] < 67270.0) {
                                            var2 = -0.00033003683;
                                        } else {
                                            var2 = -0.00009136765;
                                        }
                                    } else {
                                        if (input[39] < 49.0) {
                                            var2 = 0.00019708279;
                                        } else {
                                            var2 = -0.00033176766;
                                        }
                                    }
                                }
                            } else {
                                if (input[43] < 12390.0) {
                                    if (input[4] < 638.0) {
                                        var2 = 0.00052619877;
                                    } else {
                                        var2 = -0.00034833996;
                                    }
                                } else {
                                    if (input[49] < 71712.0) {
                                        if (input[11] < 6842.0) {
                                            var2 = -0.00035429202;
                                        } else {
                                            var2 = 0.000057326117;
                                        }
                                    } else {
                                        if (input[0] < 0.18605207) {
                                            var2 = 0.0003817498;
                                        } else {
                                            var2 = -0.00021320977;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[70] < 214.0) {
                                if (input[0] < 0.18363409) {
                                    if (input[38] < 0.18465999) {
                                        var2 = -0.000025198484;
                                    } else {
                                        var2 = 0.0010317789;
                                    }
                                } else {
                                    var2 = -0.00032331733;
                                }
                            } else {
                                if (input[68] < 4.0) {
                                    if (input[75] < 148800.0) {
                                        var2 = -0.0005196662;
                                    } else {
                                        if (input[75] < 161200.0) {
                                            var2 = -0.00008984487;
                                        } else {
                                            var2 = -0.00035722525;
                                        }
                                    }
                                } else {
                                    var2 = 0.000030454717;
                                }
                            }
                        }
                    }
                } else {
                    if (input[69] < 184.0) {
                        var2 = 0.0006233228;
                    } else {
                        if (input[63] < 25.0) {
                            var2 = 0.0004530518;
                        } else {
                            if (input[0] < 0.17874639) {
                                if (input[6] < 37278.0) {
                                    var2 = -0.00035140774;
                                } else {
                                    var2 = -0.0007029927;
                                }
                            } else {
                                if (input[0] < 0.18245564) {
                                    var2 = 0.00056061463;
                                } else {
                                    if (input[4] < 738.0) {
                                        if (input[48] < 68194.0) {
                                            var2 = -0.00008488393;
                                        } else {
                                            var2 = -0.0005136395;
                                        }
                                    } else {
                                        var2 = 0.00014101177;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[44] < 44528.0) {
                    if (input[46] < 42577.0) {
                        var2 = 0.0002320824;
                    } else {
                        var2 = -0.0005224078;
                    }
                } else {
                    if (input[42] < 708.0) {
                        var2 = 0.0013327759;
                    } else {
                        var2 = -0.000023405193;
                    }
                }
            }
        }
    } else {
        if (input[6] < 49605.0) {
            if (input[35] < 75469.0) {
                if (input[38] < 0.20320146) {
                    if (input[30] < 1.0) {
                        if (input[68] < 33.0) {
                            var2 = 0.0004735932;
                        } else {
                            var2 = 0.001404911;
                        }
                    } else {
                        if (input[67] < 60.0) {
                            if (input[28] < 56.0) {
                                var2 = -0.00024167144;
                            } else {
                                var2 = 0.0008523546;
                            }
                        } else {
                            var2 = -0.00049824885;
                        }
                    }
                } else {
                    if (input[39] < 20.0) {
                        if (input[7] < 1337.0) {
                            var2 = -0.00018657085;
                        } else {
                            if (input[33] < 137.0) {
                                if (input[39] < 17.0) {
                                    var2 = 0.0013739254;
                                } else {
                                    var2 = 0.00068098796;
                                }
                            } else {
                                if (input[72] < 49.0) {
                                    var2 = 0.0008530766;
                                } else {
                                    if (input[67] < 89.0) {
                                        var2 = 0.00042200796;
                                    } else {
                                        var2 = -0.00033909635;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[30] < 1.0) {
                            var2 = 0.00042971913;
                        } else {
                            if (input[71] < 106.0) {
                                var2 = 0.0007788154;
                            } else {
                                var2 = 0.0013976655;
                            }
                        }
                    }
                }
            } else {
                if (input[9] < 76210.0) {
                    var2 = -0.00037145236;
                } else {
                    var2 = 0.00044294182;
                }
            }
        } else {
            if (input[0] < 0.19240898) {
                var2 = 0.0005145364;
            } else {
                if (input[25] < 19.0) {
                    var2 = 0.0005569896;
                } else {
                    var2 = 0.0018556;
                }
            }
        }
    }
    double var3;
    if (input[73] < 59195.0) {
        if (input[39] < 13.0) {
            if (input[30] < 4.0) {
                var3 = 0.0013135801;
            } else {
                var3 = -0.0000031131365;
            }
        } else {
            if (input[40] < 25.0) {
                if (input[36] < 1465441.0) {
                    if (input[29] < 53.0) {
                        if (input[39] < 17.0) {
                            var3 = 0.00015977581;
                        } else {
                            var3 = 0.0008936391;
                        }
                    } else {
                        var3 = -0.00012855162;
                    }
                } else {
                    if (input[45] < 1251.0) {
                        var3 = 0.000059449416;
                    } else {
                        var3 = -0.00068216067;
                    }
                }
            } else {
                if (input[69] < 181.0) {
                    if (input[38] < 0.20243937) {
                        if (input[63] < 27.0) {
                            var3 = 0.00070435245;
                        } else {
                            if (input[0] < 0.18394147) {
                                var3 = 0.00031752515;
                            } else {
                                var3 = -0.00045772805;
                            }
                        }
                    } else {
                        if (input[38] < 0.3034366) {
                            if (input[32] < 216.0) {
                                if (input[70] < 149.0) {
                                    var3 = 0.00050792954;
                                } else {
                                    if (input[1] < 21.0) {
                                        var3 = 0.000864613;
                                    } else {
                                        var3 = 0.0015478504;
                                    }
                                }
                            } else {
                                var3 = 0.00036683207;
                            }
                        } else {
                            if (input[3] < 48.0) {
                                var3 = 0.00055348524;
                            } else {
                                var3 = -0.00017307965;
                            }
                        }
                    }
                } else {
                    if (input[29] < 45.0) {
                        if (input[40] < 54.0) {
                            var3 = -0.00010544801;
                        } else {
                            var3 = 0.0009600073;
                        }
                    } else {
                        if (input[48] < 4529.0) {
                            if (input[67] < 53.0) {
                                var3 = -0.00008350242;
                            } else {
                                var3 = 0.00061242154;
                            }
                        } else {
                            var3 = -0.00044523296;
                        }
                    }
                }
            }
        }
    } else {
        if (input[10] < 83406.0) {
            if (input[8] < 57632.0) {
                if (input[46] < 39914.0) {
                    if (input[31] < 188.0) {
                        if (input[70] < 214.0) {
                            if (input[69] < 185.0) {
                                var3 = -0.000074477895;
                            } else {
                                var3 = -0.00045513693;
                            }
                        } else {
                            var3 = 0.0002981531;
                        }
                    } else {
                        if (input[31] < 191.0) {
                            var3 = 0.0010583095;
                        } else {
                            var3 = 0.00022026271;
                        }
                    }
                } else {
                    if (input[69] < 181.0) {
                        var3 = 0.00041090624;
                    } else {
                        if (input[8] < 52733.0) {
                            if (input[29] < 43.0) {
                                if (input[36] < 1341389.0) {
                                    var3 = 0.0008034748;
                                } else {
                                    if (input[62] < 26.0) {
                                        if (input[37] < 157500.0) {
                                            var3 = -0.00009332395;
                                        } else {
                                            var3 = -0.0004183539;
                                        }
                                    } else {
                                        if (input[38] < 0.17374991) {
                                            var3 = -0.0003300516;
                                        } else {
                                            var3 = 0.0007999753;
                                        }
                                    }
                                }
                            } else {
                                if (input[74] < 1366688.0) {
                                    if (input[11] < 2398.0) {
                                        if (input[0] < 0.18074977) {
                                            var3 = -0.00016628059;
                                        } else {
                                            var3 = -0.00045135035;
                                        }
                                    } else {
                                        if (input[69] < 184.0) {
                                            var3 = 0.00016776047;
                                        } else {
                                            var3 = -0.00029104386;
                                        }
                                    }
                                } else {
                                    if (input[68] < 9.0) {
                                        if (input[38] < 0.19598201) {
                                            var3 = -0.00013710497;
                                        } else {
                                            var3 = -0.0003319436;
                                        }
                                    } else {
                                        var3 = 0.00040821658;
                                    }
                                }
                            }
                        } else {
                            if (input[40] < 63.0) {
                                var3 = 0.0006533055;
                            } else {
                                var3 = -0.0004830181;
                            }
                        }
                    }
                }
            } else {
                if (input[74] < 1348649.0) {
                    if (input[66] < 54.0) {
                        if (input[42] < 656.0) {
                            var3 = -0.0005327106;
                        } else {
                            var3 = -0.00022766214;
                        }
                    } else {
                        var3 = -0.0008177928;
                    }
                } else {
                    if (input[70] < 211.0) {
                        var3 = 0.00035497986;
                    } else {
                        if (input[40] < 60.0) {
                            var3 = 0.00027108824;
                        } else {
                            if (input[71] < 136.0) {
                                var3 = 0.00016138738;
                            } else {
                                if (input[70] < 214.0) {
                                    var3 = -0.0004900406;
                                } else {
                                    if (input[46] < 61535.0) {
                                        if (input[31] < 186.0) {
                                            var3 = -0.0002967679;
                                        } else {
                                            var3 = -0.00017445051;
                                        }
                                    } else {
                                        var3 = -0.000414073;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[38] < 0.18465999) {
                if (input[37] < 176800.0) {
                    var3 = -0.00038391617;
                } else {
                    var3 = -0.000033368964;
                }
            } else {
                if (input[44] < 46436.0) {
                    var3 = 0.0009356282;
                } else {
                    var3 = 0.000082578845;
                }
            }
        }
    }
    double var4;
    if (input[46] < 34487.0) {
        if (input[75] < 18400.0) {
            if (input[30] < 42.0) {
                if (input[49] < 4936.0) {
                    if (input[40] < 15.0) {
                        var4 = 0.0007228674;
                    } else {
                        if (input[38] < 0.3034366) {
                            if (input[29] < 41.0) {
                                var4 = 0.00085318804;
                            } else {
                                if (input[33] < 138.0) {
                                    if (input[10] < 64993.0) {
                                        var4 = 0.000101942096;
                                    } else {
                                        var4 = 0.00081747165;
                                    }
                                } else {
                                    if (input[47] < 2365.0) {
                                        var4 = -0.00036749273;
                                    } else {
                                        var4 = 0.00025882572;
                                    }
                                }
                            }
                        } else {
                            if (input[67] < 82.0) {
                                var4 = 0.00044250075;
                            } else {
                                if (input[29] < 44.0) {
                                    var4 = 0.0002536517;
                                } else {
                                    if (input[67] < 93.0) {
                                        var4 = -0.0011434512;
                                    } else {
                                        var4 = -0.000052384526;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[4] < 689.0) {
                        var4 = -0.0008559978;
                    } else {
                        var4 = -0.000024372763;
                    }
                }
            } else {
                if (input[28] < 100.0) {
                    var4 = 0.000030397136;
                } else {
                    var4 = 0.0009226077;
                }
            }
        } else {
            if (input[69] < 182.0) {
                if (input[44] < 7587.0) {
                    var4 = 0.00089487137;
                } else {
                    var4 = 0.00032394528;
                }
            } else {
                var4 = 0.00017374415;
            }
        }
    } else {
        if (input[47] < 67798.0) {
            if (input[4] < 791.0) {
                if (input[31] < 188.0) {
                    if (input[32] < 160.0) {
                        if (input[0] < 0.23693217) {
                            var4 = -0.00025857854;
                        } else {
                            var4 = 0.00068842893;
                        }
                    } else {
                        if (input[69] < 182.0) {
                            if (input[4] < 697.0) {
                                var4 = -0.00063773355;
                            } else {
                                var4 = -0.0002388199;
                            }
                        } else {
                            if (input[37] < 142128.0) {
                                if (input[6] < 41973.0) {
                                    if (input[33] < 143.0) {
                                        if (input[40] < 57.0) {
                                            var4 = 0.00028061442;
                                        } else {
                                            var4 = -0.0001277456;
                                        }
                                    } else {
                                        var4 = 0.00040882287;
                                    }
                                } else {
                                    var4 = 0.00048364417;
                                }
                            } else {
                                if (input[6] < 49071.0) {
                                    if (input[73] < 62845.0) {
                                        var4 = -0.0006854906;
                                    } else {
                                        if (input[73] < 64034.0) {
                                            var4 = 0.00014631652;
                                        } else {
                                            var4 = -0.00019577972;
                                        }
                                    }
                                } else {
                                    var4 = 0.00012950822;
                                }
                            }
                        }
                    }
                } else {
                    if (input[66] < 49.0) {
                        if (input[71] < 139.0) {
                            var4 = -0.00046484466;
                        } else {
                            var4 = -0.00015113625;
                        }
                    } else {
                        if (input[33] < 136.0) {
                            var4 = 0.00082682306;
                        } else {
                            if (input[46] < 38980.0) {
                                var4 = 0.00067092275;
                            } else {
                                if (input[70] < 210.0) {
                                    var4 = 0.00040147893;
                                } else {
                                    if (input[43] < 11260.0) {
                                        var4 = 0.00041116631;
                                    } else {
                                        if (input[42] < 748.0) {
                                            var4 = -0.0002627604;
                                        } else {
                                            var4 = 0.00012455363;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[44] < 49971.0) {
                    if (input[33] < 141.0) {
                        var4 = 0.0001350623;
                    } else {
                        var4 = 0.0009295825;
                    }
                } else {
                    var4 = -0.00020426023;
                }
            }
        } else {
            if (input[31] < 194.0) {
                if (input[68] < 2.0) {
                    if (input[32] < 217.0) {
                        if (input[42] < 610.0) {
                            if (input[42] < 575.0) {
                                var4 = -0.00008424003;
                            } else {
                                var4 = 0.0005161336;
                            }
                        } else {
                            if (input[32] < 149.0) {
                                var4 = -0.00041800924;
                            } else {
                                if (input[38] < 0.1898502) {
                                    if (input[38] < 0.18872815) {
                                        if (input[67] < 48.0) {
                                            var4 = -0.00011515216;
                                        } else {
                                            var4 = 0.00004339513;
                                        }
                                    } else {
                                        var4 = -0.00032968746;
                                    }
                                } else {
                                    var4 = 0.00014618043;
                                }
                            }
                        }
                    } else {
                        if (input[8] < 58166.0) {
                            if (input[46] < 47533.0) {
                                if (input[69] < 187.0) {
                                    var4 = 0.00007472637;
                                } else {
                                    var4 = 0.0007058498;
                                }
                            } else {
                                var4 = -0.00017659125;
                            }
                        } else {
                            var4 = -0.0003091506;
                        }
                    }
                } else {
                    if (input[5] < 18822.0) {
                        if (input[75] < 142100.0) {
                            var4 = -0.0004648746;
                        } else {
                            if (input[46] < 42708.0) {
                                if (input[3] < 41.0) {
                                    var4 = 0.000020391299;
                                } else {
                                    var4 = -0.00018145895;
                                }
                            } else {
                                if (input[69] < 190.0) {
                                    if (input[48] < 73634.0) {
                                        var4 = -0.00026149626;
                                    } else {
                                        var4 = -0.00005536436;
                                    }
                                } else {
                                    var4 = -0.00034025844;
                                }
                            }
                        }
                    } else {
                        var4 = 0.00018193023;
                    }
                }
            } else {
                var4 = -0.0005215885;
            }
        }
    }
    return 0.002222466634232357 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_SAMPLERATE_3D
