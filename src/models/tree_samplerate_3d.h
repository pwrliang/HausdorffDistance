
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
    if (input[74] < 1170092.0) {
        if (input[11] < 222.0) {
            if (input[71] < 37.3208) {
                if (input[31] < 23.62205) {
                    if (input[62] < 8.0) {
                        if (input[70] < 19.24995) {
                            if (input[4] < 8.0) {
                                var0 = -0.000020738491;
                            } else {
                                var0 = 0.0014250302;
                            }
                        } else {
                            if (input[31] < 14.5) {
                                var0 = -0.0010530048;
                            } else {
                                var0 = -0.000102242455;
                            }
                        }
                    } else {
                        if (input[72] < 480.0) {
                            if (input[3] < 13.0) {
                                if (input[29] < -5.728) {
                                    if (input[69] < 19.68505) {
                                        if (input[34] < 124.0) {
                                            var0 = 0.0013511566;
                                        } else {
                                            var0 = 0.0019801212;
                                        }
                                    } else {
                                        if (input[32] < 12.7422) {
                                            var0 = 0.0016760037;
                                        } else {
                                            var0 = 0.00058285793;
                                        }
                                    }
                                } else {
                                    if (input[3] < 8.0) {
                                        if (input[62] < 18.0) {
                                            var0 = 0.0019502388;
                                        } else {
                                            var0 = 0.0003638769;
                                        }
                                    } else {
                                        var0 = -0.000030437519;
                                    }
                                }
                            } else {
                                if (input[37] < 1080.0) {
                                    var0 = -0.0008840606;
                                } else {
                                    var0 = 0.0014250302;
                                }
                            }
                        } else {
                            if (input[0] < 0.48907968) {
                                if (input[64] < 24.0) {
                                    var0 = 0.0005502388;
                                } else {
                                    if (input[72] < 676.0) {
                                        var0 = -0.00026224248;
                                    } else {
                                        var0 = -0.0012958619;
                                    }
                                }
                            } else {
                                if (input[71] < 23.5) {
                                    if (input[27] < 31.0) {
                                        var0 = 0.001963877;
                                    } else {
                                        var0 = 0.00065618457;
                                    }
                                } else {
                                    var0 = -0.00006642781;
                                }
                            }
                        }
                    }
                } else {
                    if (input[37] < 45.0) {
                        var0 = -0.0005130462;
                    } else {
                        if (input[64] < 31.0) {
                            if (input[39] < 7.0) {
                                if (input[0] < 0.6528976) {
                                    if (input[72] < 48.0) {
                                        var0 = 0.000814149;
                                    } else {
                                        if (input[66] < -19.68505) {
                                            var0 = 0.0010613939;
                                        } else {
                                            var0 = 0.001896246;
                                        }
                                    }
                                } else {
                                    if (input[64] < 16.0) {
                                        var0 = 0.0016155578;
                                    } else {
                                        var0 = -0.0002276356;
                                    }
                                }
                            } else {
                                if (input[37] < 1573.0) {
                                    if (input[37] < 162.0) {
                                        var0 = 0.000042670334;
                                    } else {
                                        if (input[67] < -14.1567) {
                                            var0 = 0.00095339736;
                                        } else {
                                            var0 = 0.0016997684;
                                        }
                                    }
                                } else {
                                    var0 = -0.00022496891;
                                }
                            }
                        } else {
                            if (input[32] < 460.007) {
                                if (input[36] < 11025.0) {
                                    if (input[39] < 18.0) {
                                        var0 = 0.0020140603;
                                    } else {
                                        var0 = 0.0013484922;
                                    }
                                } else {
                                    var0 = 0.0012432121;
                                }
                            } else {
                                var0 = 0.0007002388;
                            }
                        }
                    }
                }
            } else {
                if (input[37] < 150.0) {
                    if (input[27] < 22.0) {
                        if (input[59] < 9.0) {
                            if (input[9] < 15.0) {
                                if (input[0] < 0.21668872) {
                                    var0 = -0.00061133335;
                                } else {
                                    if (input[28] < -23.875) {
                                        var0 = -0.0001930048;
                                    } else {
                                        if (input[24] < 12.0) {
                                            var0 = 0.0015746616;
                                        } else {
                                            var0 = 0.00026569774;
                                        }
                                    }
                                }
                            } else {
                                if (input[10] < 24.0) {
                                    if (input[28] < -1.74479) {
                                        var0 = -0.0013567879;
                                    } else {
                                        var0 = -0.0005749697;
                                    }
                                } else {
                                    var0 = 0.00026956247;
                                }
                            }
                        } else {
                            if (input[59] < 12.0) {
                                var0 = -0.0014258788;
                            } else {
                                var0 = -0.0005749697;
                            }
                        }
                    } else {
                        var0 = 0.0011082666;
                    }
                } else {
                    if (input[74] < 415.0) {
                        if (input[46] < 7.0) {
                            var0 = -0.0010295152;
                        } else {
                            var0 = -0.00021133335;
                        }
                    } else {
                        if (input[31] < 2.0) {
                            var0 = -0.00051014766;
                        } else {
                            if (input[67] < -299.94) {
                                if (input[4] < 4.0) {
                                    var0 = -0.001138719;
                                } else {
                                    var0 = 0.0009695624;
                                }
                            } else {
                                if (input[1] < 8.0) {
                                    if (input[58] < 5.0) {
                                        if (input[8] < 9.0) {
                                            var0 = 0.0008100307;
                                        } else {
                                            var0 = 0.0017731279;
                                        }
                                    } else {
                                        var0 = 0.00035115663;
                                    }
                                } else {
                                    if (input[49] < 33223.0) {
                                        if (input[5] < 16545.0) {
                                            var0 = 0.00073481974;
                                        } else {
                                            var0 = -0.00041309447;
                                        }
                                    } else {
                                        var0 = 0.001963877;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[31] < 185.0) {
                if (input[26] < 74.0) {
                    if (input[40] < 13.0) {
                        if (input[40] < 6.0) {
                            if (input[31] < 13.70585) {
                                var0 = -0.000990663;
                            } else {
                                var0 = 0.0006820624;
                            }
                        } else {
                            if (input[2] < 21.0) {
                                if (input[58] < 4.0) {
                                    if (input[21] < 5.0) {
                                        var0 = 0.0019309841;
                                    } else {
                                        var0 = 0.0010250303;
                                    }
                                } else {
                                    var0 = -0.000044674536;
                                }
                            } else {
                                if (input[28] < -4.72548) {
                                    var0 = -0.0007204243;
                                } else {
                                    var0 = 0.00082541537;
                                }
                            }
                        }
                    } else {
                        if (input[67] < 96.0) {
                            if (input[41] < 23.0) {
                                if (input[30] < -4.76162) {
                                    var0 = 0.00038357216;
                                } else {
                                    if (input[0] < 0.2866716) {
                                        var0 = -0.00054381543;
                                    } else {
                                        var0 = -0.0013194472;
                                    }
                                }
                            } else {
                                if (input[31] < 148.0) {
                                    var0 = 0.0010002388;
                                } else {
                                    var0 = -0.0006258788;
                                }
                            }
                        } else {
                            if (input[0] < 0.2113589) {
                                var0 = 0.0010795757;
                            } else {
                                var0 = 0.000061393905;
                            }
                        }
                    }
                } else {
                    if (input[70] < 14.1567) {
                        var0 = 0.000043212083;
                    } else {
                        var0 = -0.0012423216;
                    }
                }
            } else {
                if (input[5] < 62.0) {
                    var0 = -0.00079315156;
                } else {
                    if (input[59] < 10.0) {
                        if (input[43] < 25.0) {
                            if (input[33] < 245.835) {
                                var0 = 0.0005169055;
                            } else {
                                var0 = 0.0015411567;
                            }
                        } else {
                            if (input[49] < 1780.0) {
                                if (input[71] < 91.4153) {
                                    var0 = -0.0008809689;
                                } else {
                                    var0 = 0.000056184574;
                                }
                            } else {
                                var0 = 0.0010613939;
                            }
                        }
                    } else {
                        if (input[46] < 51.0) {
                            var0 = 0.0009835722;
                        } else {
                            if (input[40] < 31.0) {
                                var0 = 0.001963877;
                            } else {
                                var0 = 0.001194646;
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[6] < 6842.0) {
            if (input[37] < 8398.0) {
                if (input[43] < 11895.0) {
                    var0 = -0.00030496894;
                } else {
                    if (input[42] < 710.0) {
                        if (input[44] < 48606.0) {
                            if (input[1] < 17.0) {
                                var0 = 0.0016760037;
                            } else {
                                var0 = 0.000552303;
                            }
                        } else {
                            var0 = -0.000066892346;
                        }
                    } else {
                        if (input[10] < 1127.0) {
                            var0 = 0.0019856978;
                        } else {
                            var0 = 0.0012068484;
                        }
                    }
                }
            } else {
                if (input[47] < 70627.0) {
                    if (input[42] < 659.0) {
                        if (input[1] < 22.0) {
                            var0 = -0.000450663;
                        } else {
                            var0 = -0.0012268923;
                        }
                    } else {
                        if (input[30] < 39.0) {
                            if (input[4] < 166.0) {
                                var0 = -0.00080769695;
                            } else {
                                var0 = -0.00017496974;
                            }
                        } else {
                            if (input[4] < 152.0) {
                                var0 = 0.00048321206;
                            } else {
                                var0 = -0.00020406068;
                            }
                        }
                    }
                } else {
                    if (input[31] < 121.0) {
                        var0 = 0.0015755667;
                    } else {
                        if (input[0] < 0.27885503) {
                            if (input[39] < 50.0) {
                                var0 = -0.0006430945;
                            } else {
                                var0 = 0.00033310763;
                            }
                        } else {
                            var0 = 0.00097556657;
                        }
                    }
                }
            }
        } else {
            if (input[75] < 135936.0) {
                if (input[6] < 51503.0) {
                    if (input[40] < 56.0) {
                        var0 = -0.000007290513;
                    } else {
                        if (input[28] < 51.0) {
                            if (input[4] < 647.0) {
                                if (input[42] < 648.0) {
                                    var0 = -0.0008658788;
                                } else {
                                    var0 = 0.0005413939;
                                }
                            } else {
                                if (input[68] < 6.0) {
                                    var0 = -0.0015462185;
                                } else {
                                    var0 = -0.00087300484;
                                }
                            }
                        } else {
                            if (input[8] < 40759.0) {
                                var0 = -0.00090224243;
                            } else {
                                var0 = -0.001580957;
                            }
                        }
                    }
                } else {
                    var0 = 0.0006271147;
                }
            } else {
                if (input[10] < 79183.0) {
                    if (input[5] < 9753.0) {
                        var0 = -0.0009967879;
                    } else {
                        if (input[71] < 133.0) {
                            var0 = -0.0011804375;
                        } else {
                            var0 = -0.001602546;
                        }
                    }
                } else {
                    if (input[42] < 647.0) {
                        if (input[10] < 81293.0) {
                            var0 = -0.00022951516;
                        } else {
                            var0 = -0.0013270275;
                        }
                    } else {
                        if (input[47] < 55534.0) {
                            if (input[70] < 214.0) {
                                var0 = -0.0011749697;
                            } else {
                                var0 = -0.0006764278;
                            }
                        } else {
                            var0 = -0.0015288133;
                        }
                    }
                }
            }
        }
    }
    double var1;
    if (input[74] < 1170092.0) {
        if (input[71] < 37.3208) {
            if (input[35] < 311.0) {
                if (input[61] < 24.0) {
                    if (input[31] < 22.8751) {
                        if (input[64] < 34.0) {
                            if (input[67] < -15.242901) {
                                if (input[65] < 21.0) {
                                    if (input[38] < 0.36971158) {
                                        var1 = 0.00043226732;
                                    } else {
                                        if (input[33] < 14.9396) {
                                            var1 = 0.000036034202;
                                        } else {
                                            var1 = -0.00080503704;
                                        }
                                    }
                                } else {
                                    if (input[2] < 7.0) {
                                        var1 = -0.000120258126;
                                    } else {
                                        var1 = 0.0010527924;
                                    }
                                }
                            } else {
                                if (input[21] < 3.0) {
                                    var1 = -0.0000001808256;
                                } else {
                                    if (input[36] < 904.0) {
                                        var1 = 0.0014743537;
                                    } else {
                                        if (input[28] < -0.00000367723) {
                                            var1 = 0.00084404426;
                                        } else {
                                            var1 = -0.00029005072;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[41] < 8.0) {
                                var1 = 0.000181076;
                            } else {
                                var1 = -0.0009506491;
                            }
                        }
                    } else {
                        if (input[37] < 60.0) {
                            if (input[73] < 26.0) {
                                var1 = 0.001029129;
                            } else {
                                var1 = -0.0006709524;
                            }
                        } else {
                            if (input[34] < 194.0) {
                                if (input[40] < 6.0) {
                                    if (input[0] < 0.34862795) {
                                        var1 = -0.00011841301;
                                    } else {
                                        var1 = 0.0011643157;
                                    }
                                } else {
                                    if (input[35] < 46.0) {
                                        var1 = 0.0014597141;
                                    } else {
                                        if (input[34] < 125.0) {
                                            var1 = 0.0006642994;
                                        } else {
                                            var1 = 0.001317582;
                                        }
                                    }
                                }
                            } else {
                                if (input[72] < 986.0) {
                                    if (input[6] < 9.0) {
                                        if (input[31] < 59.1302) {
                                            var1 = -0.00043938766;
                                        } else {
                                            var1 = 0.0008553445;
                                        }
                                    } else {
                                        if (input[74] < 2352.0) {
                                            var1 = 0.001105079;
                                        } else {
                                            var1 = 0.00045489016;
                                        }
                                    }
                                } else {
                                    var1 = -0.00040027127;
                                }
                            }
                        }
                    }
                } else {
                    if (input[40] < 7.0) {
                        var1 = 0.0003292674;
                    } else {
                        if (input[42] < 7.0) {
                            if (input[21] < 3.0) {
                                var1 = 0.00084785576;
                            } else {
                                var1 = 0.0014842055;
                            }
                        } else {
                            if (input[38] < 0.6273124) {
                                var1 = 0.0011957845;
                            } else {
                                var1 = 0.00018186071;
                            }
                        }
                    }
                }
            } else {
                if (input[5] < 62.0) {
                    var1 = -0.0007972668;
                } else {
                    if (input[2] < 11.0) {
                        var1 = -0.00048307594;
                    } else {
                        if (input[22] < 8.0) {
                            if (input[34] < 498.0) {
                                var1 = 0.00059500214;
                            } else {
                                var1 = -0.00063535984;
                            }
                        } else {
                            if (input[39] < 13.0) {
                                if (input[21] < 5.0) {
                                    var1 = 0.0014892578;
                                } else {
                                    if (input[63] < 13.0) {
                                        var1 = 0.000074726784;
                                    } else {
                                        if (input[6] < 212.0) {
                                            var1 = 0.00029498778;
                                        } else {
                                            var1 = 0.0016402878;
                                        }
                                    }
                                }
                            } else {
                                var1 = -0.00019468534;
                            }
                        }
                    }
                }
            }
        } else {
            if (input[75] < 162.0) {
                if (input[33] < 146.0) {
                    if (input[6] < 75.0) {
                        if (input[68] < -0.0020148) {
                            var1 = -0.0009726692;
                        } else {
                            if (input[73] < 10.0) {
                                var1 = -0.00082228205;
                            } else {
                                if (input[70] < 77.906) {
                                    if (input[38] < 0.325) {
                                        var1 = 0.00035763567;
                                    } else {
                                        var1 = 0.0014191678;
                                    }
                                } else {
                                    if (input[32] < 19.685) {
                                        var1 = 0.00079336076;
                                    } else {
                                        var1 = -0.0008992984;
                                    }
                                }
                            }
                        }
                    } else {
                        var1 = -0.0010726304;
                    }
                } else {
                    var1 = 0.0008135776;
                }
            } else {
                if (input[62] < 8.0) {
                    if (input[3] < 7.0) {
                        var1 = 0.00030752795;
                    } else {
                        if (input[23] < 15.0) {
                            var1 = 0.0016851;
                        } else {
                            var1 = 0.0010471514;
                        }
                    }
                } else {
                    if (input[32] < 14.4304) {
                        if (input[63] < 28.0) {
                            if (input[64] < 28.0) {
                                if (input[33] < 15.9816) {
                                    var1 = -0.00057351927;
                                } else {
                                    var1 = 0.0005030266;
                                }
                            } else {
                                var1 = -0.0012691317;
                            }
                        } else {
                            if (input[4] < 4.0) {
                                if (input[39] < 13.0) {
                                    var1 = 0.00023845944;
                                } else {
                                    var1 = -0.0006703785;
                                }
                            } else {
                                if (input[22] < 5.0) {
                                    var1 = 0.0011967326;
                                } else {
                                    if (input[36] < 2125.0) {
                                        var1 = -0.000534945;
                                    } else {
                                        var1 = 0.0005750504;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[33] < 28.9161) {
                            if (input[38] < 0.7540465) {
                                if (input[38] < 0.65236545) {
                                    if (input[74] < 7449.0) {
                                        if (input[75] < 280.0) {
                                            var1 = 0.0008791894;
                                        } else {
                                            var1 = 0.0015034451;
                                        }
                                    } else {
                                        if (input[43] < 42.0) {
                                            var1 = -0.00047772922;
                                        } else {
                                            var1 = 0.00059316255;
                                        }
                                    }
                                } else {
                                    if (input[61] < 22.0) {
                                        var1 = 0.00088048633;
                                    } else {
                                        var1 = 0.0016655269;
                                    }
                                }
                            } else {
                                var1 = -0.000060534385;
                            }
                        } else {
                            if (input[38] < 0.5910816) {
                                if (input[66] < 100.0) {
                                    if (input[31] < 23.9965) {
                                        var1 = -0.00045993176;
                                    } else {
                                        if (input[68] < -0.0020148) {
                                            var1 = 0.0016397147;
                                        } else {
                                            var1 = 0.00054826064;
                                        }
                                    }
                                } else {
                                    if (input[29] < 43.0) {
                                        if (input[47] < 2148.0) {
                                            var1 = 0.0002355214;
                                        } else {
                                            var1 = 0.0014152129;
                                        }
                                    } else {
                                        if (input[70] < 172.0) {
                                            var1 = -0.0011140516;
                                        } else {
                                            var1 = 0.00015762752;
                                        }
                                    }
                                }
                            } else {
                                if (input[2] < 23.0) {
                                    if (input[71] < 66.0) {
                                        if (input[34] < 457.0) {
                                            var1 = -0.0010394236;
                                        } else {
                                            var1 = 0.00032351536;
                                        }
                                    } else {
                                        if (input[70] < 1142.8) {
                                            var1 = 0.0007796262;
                                        } else {
                                            var1 = -0.00012867727;
                                        }
                                    }
                                } else {
                                    var1 = -0.0010408404;
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        if (input[6] < 6842.0) {
            if (input[29] < 46.0) {
                var1 = -0.001032378;
            } else {
                if (input[37] < 8398.0) {
                    if (input[28] < 129.0) {
                        if (input[47] < 69310.0) {
                            if (input[2] < 21.0) {
                                var1 = 0.0013434786;
                            } else {
                                var1 = 0.0007397149;
                            }
                        } else {
                            if (input[42] < 690.0) {
                                var1 = -0.000077636236;
                            } else {
                                var1 = 0.0007985457;
                            }
                        }
                    } else {
                        var1 = 0.00007899752;
                    }
                } else {
                    if (input[47] < 75479.0) {
                        if (input[11] < 1329.0) {
                            if (input[7] < 3388.0) {
                                var1 = -0.0009620089;
                            } else {
                                var1 = -0.0003114607;
                            }
                        } else {
                            if (input[32] < 173.0) {
                                var1 = 0.0006617072;
                            } else {
                                if (input[43] < 11659.0) {
                                    var1 = -0.000706846;
                                } else {
                                    if (input[67] < 47.0) {
                                        if (input[71] < 140.0) {
                                            var1 = 0.00019418255;
                                        } else {
                                            var1 = -0.000700394;
                                        }
                                    } else {
                                        var1 = 0.0006587805;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[29] < 82.0) {
                            var1 = 0.0010522271;
                        } else {
                            var1 = 0.000575315;
                        }
                    }
                }
            }
        } else {
            if (input[67] < 55.0) {
                if (input[2] < 67.0) {
                    if (input[33] < 133.0) {
                        if (input[71] < 139.0) {
                            var1 = -0.00087503;
                        } else {
                            var1 = 0.00013612023;
                        }
                    } else {
                        if (input[73] < 58426.0) {
                            var1 = -0.00033116713;
                        } else {
                            if (input[67] < 51.0834) {
                                if (input[71] < 133.0) {
                                    var1 = -0.0005673067;
                                } else {
                                    if (input[46] < 39765.0) {
                                        if (input[5] < 14418.0) {
                                            var1 = -0.00047496255;
                                        } else {
                                            var1 = -0.0010866815;
                                        }
                                    } else {
                                        var1 = -0.0009802075;
                                    }
                                }
                            } else {
                                if (input[9] < 65605.0) {
                                    var1 = -0.000040725226;
                                } else {
                                    var1 = -0.0010375566;
                                }
                            }
                        }
                    }
                } else {
                    var1 = -0.00008515897;
                }
            } else {
                if (input[10] < 66874.0) {
                    var1 = -0.00065476244;
                } else {
                    var1 = 0.0009570689;
                }
            }
        }
    }
    double var2;
    if (input[47] < 41535.0) {
        if (input[36] < 149.0) {
            if (input[41] < 12.0) {
                var2 = -0.00016069294;
            } else {
                var2 = -0.00084005605;
            }
        } else {
            if (input[11] < 222.0) {
                if (input[73] < 8.0) {
                    var2 = -0.0005960841;
                } else {
                    if (input[71] < 184.667) {
                        if (input[21] < 13.0) {
                            if (input[24] < 27.0) {
                                if (input[25] < 29.0) {
                                    if (input[66] < -28.2971) {
                                        if (input[0] < 0.37984648) {
                                            var2 = 0.0000023963769;
                                        } else {
                                            var2 = 0.0009607034;
                                        }
                                    } else {
                                        if (input[67] < 136.0) {
                                            var2 = 0.0003286913;
                                        } else {
                                            var2 = 0.0011107274;
                                        }
                                    }
                                } else {
                                    if (input[62] < 14.0) {
                                        var2 = 0.00037628642;
                                    } else {
                                        if (input[0] < 0.631868) {
                                            var2 = -0.00015084851;
                                        } else {
                                            var2 = -0.0012929305;
                                        }
                                    }
                                }
                            } else {
                                if (input[72] < 510.0) {
                                    if (input[30] < -16.14175) {
                                        var2 = 0.00030144036;
                                    } else {
                                        if (input[66] < -8.8324) {
                                            var2 = 0.00060501305;
                                        } else {
                                            var2 = 0.0011549328;
                                        }
                                    }
                                } else {
                                    if (input[1] < 11.0) {
                                        var2 = 0.0006072046;
                                    } else {
                                        var2 = -0.0004902477;
                                    }
                                }
                            }
                        } else {
                            if (input[71] < 15.1575) {
                                var2 = 0.000638348;
                            } else {
                                if (input[74] < 2122.0) {
                                    var2 = -0.0008471312;
                                } else {
                                    if (input[48] < 1444.0) {
                                        if (input[70] < 191.1251) {
                                            var2 = 0.0007772117;
                                        } else {
                                            var2 = -0.00017590089;
                                        }
                                    } else {
                                        if (input[60] < 26.0) {
                                            var2 = -0.0007324703;
                                        } else {
                                            var2 = 0.00027914802;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[71] < 725.057) {
                            if (input[66] < 50.0) {
                                if (input[4] < 14.0) {
                                    if (input[21] < 12.0) {
                                        if (input[29] < -21.9256) {
                                            var2 = -0.00022680662;
                                        } else {
                                            var2 = -0.0009078669;
                                        }
                                    } else {
                                        var2 = 0.00031610442;
                                    }
                                } else {
                                    var2 = 0.00043942072;
                                }
                            } else {
                                var2 = 0.000551132;
                            }
                        } else {
                            if (input[3] < 13.0) {
                                if (input[1] < 10.0) {
                                    if (input[21] < 5.0) {
                                        if (input[42] < 6.0) {
                                            var2 = 0.00046154516;
                                        } else {
                                            var2 = 0.0012462259;
                                        }
                                    } else {
                                        var2 = 0.00014976604;
                                    }
                                } else {
                                    var2 = 0.0014340897;
                                }
                            } else {
                                var2 = -0.0003398706;
                            }
                        }
                    }
                }
            } else {
                if (input[41] < 22.0) {
                    if (input[75] < 490.0) {
                        if (input[39] < 6.0) {
                            if (input[73] < 20.0) {
                                var2 = 0.0006544289;
                            } else {
                                if (input[32] < 285.304) {
                                    if (input[34] < 965.0) {
                                        var2 = -0.000068402594;
                                    } else {
                                        var2 = -0.0010430194;
                                    }
                                } else {
                                    var2 = 0.00025651077;
                                }
                            }
                        } else {
                            if (input[21] < 4.0) {
                                var2 = 0.0012929118;
                            } else {
                                if (input[4] < 10.0) {
                                    var2 = 0.0009955255;
                                } else {
                                    var2 = 0.00006627204;
                                }
                            }
                        }
                    } else {
                        if (input[64] < 29.0) {
                            if (input[26] < 41.0) {
                                if (input[5] < 168.0) {
                                    var2 = 0.00024012158;
                                } else {
                                    var2 = -0.00044851028;
                                }
                            } else {
                                if (input[25] < 48.0) {
                                    var2 = -0.001304063;
                                } else {
                                    var2 = -0.0004510496;
                                }
                            }
                        } else {
                            if (input[65] < 39.0) {
                                var2 = 0.00049763283;
                            } else {
                                if (input[41] < 13.0) {
                                    if (input[69] < 51.1667) {
                                        var2 = -0.00042744816;
                                    } else {
                                        if (input[74] < 25274.0) {
                                            var2 = 0.0011061145;
                                        } else {
                                            var2 = -0.00008189026;
                                        }
                                    }
                                } else {
                                    if (input[48] < 1554.0) {
                                        if (input[70] < 22.5) {
                                            var2 = -0.00013460957;
                                        } else {
                                            var2 = -0.00087102;
                                        }
                                    } else {
                                        var2 = 0.00018994724;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[28] < 77.0) {
                        if (input[68] < 43.0) {
                            if (input[38] < 0.19462962) {
                                var2 = 0.00042267056;
                            } else {
                                if (input[4] < 102.0) {
                                    var2 = 0.0014078602;
                                } else {
                                    var2 = 0.00092863495;
                                }
                            }
                        } else {
                            var2 = -0.000038083377;
                        }
                    } else {
                        var2 = -0.00044126922;
                    }
                }
            }
        }
    } else {
        if (input[33] < 134.0) {
            if (input[1] < 22.0) {
                if (input[68] < 1.95058) {
                    if (input[67] < 43.1959) {
                        if (input[75] < 163200.0) {
                            if (input[28] < 65.0) {
                                var2 = 0.00034969122;
                            } else {
                                var2 = -0.00049759186;
                            }
                        } else {
                            var2 = 0.0006556106;
                        }
                    } else {
                        if (input[42] < 616.0) {
                            var2 = 0.00022066418;
                        } else {
                            if (input[27] < 30.0) {
                                var2 = 0.0013330614;
                            } else {
                                var2 = 0.0007635728;
                            }
                        }
                    }
                } else {
                    if (input[42] < 715.0) {
                        if (input[74] < 1397961.0) {
                            if (input[10] < 2134.0) {
                                var2 = 0.000623214;
                            } else {
                                var2 = -0.0003995882;
                            }
                        } else {
                            var2 = -0.0007361937;
                        }
                    } else {
                        var2 = 0.0006925493;
                    }
                }
            } else {
                if (input[75] < 146461.0) {
                    if (input[29] < 51.0) {
                        var2 = 0.0006974882;
                    } else {
                        var2 = -0.00041671397;
                    }
                } else {
                    if (input[23] < 20.0) {
                        var2 = -0.000794067;
                    } else {
                        var2 = -0.00009720901;
                    }
                }
            }
        } else {
            if (input[67] < 53.2296) {
                if (input[74] < 1179491.0) {
                    var2 = 0.00018668789;
                } else {
                    if (input[32] < 156.0) {
                        var2 = 0.000041435007;
                    } else {
                        if (input[37] < 173400.0) {
                            if (input[40] < 57.0) {
                                var2 = -0.00092009077;
                            } else {
                                if (input[74] < 1291970.0) {
                                    if (input[5] < 12800.0) {
                                        var2 = 0.00028230716;
                                    } else {
                                        if (input[41] < 49.0) {
                                            var2 = -0.00075532607;
                                        } else {
                                            var2 = -0.00024771705;
                                        }
                                    }
                                } else {
                                    if (input[25] < 20.0) {
                                        if (input[47] < 65605.0) {
                                            var2 = -0.00091783487;
                                        } else {
                                            var2 = -0.0006204572;
                                        }
                                    } else {
                                        if (input[29] < 80.3774) {
                                            var2 = -0.0005723427;
                                        } else {
                                            var2 = -0.00013396093;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[4] < 851.0) {
                                var2 = 0.00020608974;
                            } else {
                                var2 = -0.00061250787;
                            }
                        }
                    }
                }
            } else {
                if (input[29] < 43.0) {
                    var2 = 0.00083116785;
                } else {
                    if (input[41] < 49.0) {
                        var2 = -0.00095982396;
                    } else {
                        var2 = -0.00002219725;
                    }
                }
            }
        }
    }
    double var3;
    if (input[47] < 41535.0) {
        if (input[41] < 31.0) {
            if (input[0] < 0.85056216) {
                if (input[71] < 69.5896) {
                    if (input[60] < 24.0) {
                        if (input[74] < 7204.0) {
                            if (input[33] < 54.3126) {
                                if (input[28] < -7.991) {
                                    if (input[34] < 1128.0) {
                                        if (input[21] < 4.0) {
                                            var3 = -0.000051358453;
                                        } else {
                                            var3 = 0.0006036759;
                                        }
                                    } else {
                                        if (input[72] < 186.0) {
                                            var3 = -0.00090611883;
                                        } else {
                                            var3 = 0.00020565744;
                                        }
                                    }
                                } else {
                                    if (input[69] < 8.04651) {
                                        if (input[66] < -0.0265766) {
                                            var3 = 0.00076228933;
                                        } else {
                                            var3 = -0.00015871503;
                                        }
                                    } else {
                                        if (input[61] < 23.0) {
                                            var3 = -0.00025948256;
                                        } else {
                                            var3 = 0.00060199446;
                                        }
                                    }
                                }
                            } else {
                                if (input[6] < 203.0) {
                                    if (input[37] < 48.0) {
                                        var3 = -0.00041203827;
                                    } else {
                                        if (input[33] < 124.0) {
                                            var3 = 0.0008989748;
                                        } else {
                                            var3 = 0.0003719332;
                                        }
                                    }
                                } else {
                                    if (input[61] < 11.0) {
                                        var3 = 0.0002909044;
                                    } else {
                                        var3 = -0.00071318203;
                                    }
                                }
                            }
                        } else {
                            if (input[33] < 719.999) {
                                if (input[72] < 315.0) {
                                    if (input[6] < 17.0) {
                                        var3 = 0.000806407;
                                    } else {
                                        if (input[40] < 14.0) {
                                            var3 = 0.00046451352;
                                        } else {
                                            var3 = 0.000017386268;
                                        }
                                    }
                                } else {
                                    if (input[38] < 0.56551886) {
                                        var3 = -0.0006952406;
                                    } else {
                                        if (input[28] < 66.0) {
                                            var3 = -0.000007961694;
                                        } else {
                                            var3 = 0.0007104628;
                                        }
                                    }
                                }
                            } else {
                                var3 = -0.00078753073;
                            }
                        }
                    } else {
                        if (input[1] < 8.0) {
                            if (input[44] < 28.0) {
                                var3 = -0.00010639826;
                            } else {
                                var3 = 0.0008377203;
                            }
                        } else {
                            var3 = 0.0010238198;
                        }
                    }
                } else {
                    if (input[70] < 0.000000476837) {
                        var3 = -0.0009999494;
                    } else {
                        if (input[58] < 3.0) {
                            if (input[33] < 6.89384) {
                                var3 = -0.00053992245;
                            } else {
                                if (input[48] < 1118.0) {
                                    if (input[75] < 75.0) {
                                        var3 = -0.00044543427;
                                    } else {
                                        if (input[23] < 8.0) {
                                            var3 = -0.00008914811;
                                        } else {
                                            var3 = 0.00040104656;
                                        }
                                    }
                                } else {
                                    if (input[0] < 0.34757558) {
                                        if (input[33] < 138.0) {
                                            var3 = 0.0004595047;
                                        } else {
                                            var3 = -0.00021229831;
                                        }
                                    } else {
                                        var3 = -0.0008263979;
                                    }
                                }
                            }
                        } else {
                            if (input[28] < -9.34375) {
                                if (input[44] < 26.0) {
                                    if (input[67] < -77.906) {
                                        var3 = 0.00016281244;
                                    } else {
                                        if (input[22] < 7.0) {
                                            var3 = -0.00078539096;
                                        } else {
                                            var3 = -0.00022472453;
                                        }
                                    }
                                } else {
                                    var3 = -0.0012618918;
                                }
                            } else {
                                if (input[4] < 5.0) {
                                    if (input[48] < 22.0) {
                                        var3 = -0.00012070197;
                                    } else {
                                        var3 = 0.0008726446;
                                    }
                                } else {
                                    var3 = -0.00046351465;
                                }
                            }
                        }
                    }
                }
            } else {
                if (input[25] < 13.0) {
                    if (input[60] < 7.0) {
                        var3 = 0.0012236357;
                    } else {
                        var3 = 0.00066820614;
                    }
                } else {
                    var3 = 0.00004704701;
                }
            }
        } else {
            if (input[42] < 607.0) {
                if (input[24] < 21.0) {
                    if (input[70] < 184.0) {
                        var3 = 0.00073957;
                    } else {
                        var3 = 0.000014600136;
                    }
                } else {
                    var3 = 0.0010012907;
                }
            } else {
                var3 = -0.0002505502;
            }
        }
    } else {
        if (input[33] < 133.0) {
            if (input[4] < 80.0) {
                if (input[37] < 2842.0) {
                    var3 = -0.00016270454;
                } else {
                    if (input[39] < 52.0) {
                        if (input[30] < 49.0) {
                            var3 = 0.00020330763;
                        } else {
                            var3 = 0.000663186;
                        }
                    } else {
                        var3 = 0.00092623953;
                    }
                }
            } else {
                if (input[31] < 117.921) {
                    var3 = 0.00050053705;
                } else {
                    if (input[70] < 212.0) {
                        if (input[33] < 124.0) {
                            var3 = -0.0002325113;
                        } else {
                            var3 = 0.00067268714;
                        }
                    } else {
                        if (input[67] < 43.0) {
                            if (input[36] < 136334.0) {
                                var3 = -0.00013936158;
                            } else {
                                var3 = 0.0004958362;
                            }
                        } else {
                            if (input[74] < 1396696.0) {
                                if (input[4] < 159.0) {
                                    var3 = -0.00046121163;
                                } else {
                                    var3 = 0.00026908537;
                                }
                            } else {
                                if (input[69] < 188.686) {
                                    var3 = -0.0007595659;
                                } else {
                                    var3 = -0.00025332544;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            if (input[21] < 9.0) {
                var3 = 0.00039052498;
            } else {
                if (input[6] < 52728.0) {
                    if (input[68] < 12.0) {
                        if (input[75] < 126665.0) {
                            if (input[67] < 51.0834) {
                                var3 = -0.0004624794;
                            } else {
                                var3 = -0.0010354866;
                            }
                        } else {
                            if (input[67] < 53.2296) {
                                if (input[31] < 190.358) {
                                    if (input[6] < 1852.0) {
                                        var3 = -0.000040746105;
                                    } else {
                                        if (input[7] < 30196.0) {
                                            var3 = -0.0006702031;
                                        } else {
                                            var3 = -0.00036177793;
                                        }
                                    }
                                } else {
                                    if (input[69] < 184.0) {
                                        var3 = 0.00052720524;
                                    } else {
                                        if (input[42] < 645.0) {
                                            var3 = 0.000053778484;
                                        } else {
                                            var3 = -0.00041856486;
                                        }
                                    }
                                }
                            } else {
                                if (input[2] < 61.0) {
                                    var3 = -0.0003493687;
                                } else {
                                    var3 = 0.0008472318;
                                }
                            }
                        }
                    } else {
                        var3 = 0.000044994973;
                    }
                } else {
                    if (input[74] < 1311573.0) {
                        var3 = 0.00069855095;
                    } else {
                        if (input[5] < 20903.0) {
                            if (input[1] < 51.0) {
                                var3 = 0.0004086808;
                            } else {
                                var3 = -0.00026706196;
                            }
                        } else {
                            if (input[11] < 90579.0) {
                                var3 = -0.00032801458;
                            } else {
                                var3 = -0.0007224618;
                            }
                        }
                    }
                }
            }
        }
    }
    double var4;
    if (input[1] < 26.0) {
        if (input[21] < 10.0) {
            if (input[30] < -20.2723) {
                if (input[74] < 1866.0) {
                    if (input[7] < 18.0) {
                        var4 = 0.00043872345;
                    } else {
                        if (input[69] < 19.68505) {
                            var4 = 0.0007709417;
                        } else {
                            var4 = 0.0012858701;
                        }
                    }
                } else {
                    if (input[73] < 58.0) {
                        var4 = -0.0006608983;
                    } else {
                        if (input[46] < 45.0) {
                            var4 = 0.0008633756;
                        } else {
                            if (input[33] < 172.856) {
                                if (input[71] < 603.5) {
                                    if (input[34] < 206.0) {
                                        if (input[65] < 43.0) {
                                            var4 = 0.000570236;
                                        } else {
                                            var4 = -0.00020862064;
                                        }
                                    } else {
                                        var4 = 0.0009366314;
                                    }
                                } else {
                                    var4 = -0.00024713995;
                                }
                            } else {
                                var4 = -0.00054018473;
                            }
                        }
                    }
                }
            } else {
                if (input[37] < 13464.0) {
                    if (input[72] < 27.0) {
                        if (input[66] < 14.1274) {
                            if (input[72] < 19.0) {
                                var4 = -0.00022647588;
                            } else {
                                var4 = -0.0009995186;
                            }
                        } else {
                            var4 = 0.000433465;
                        }
                    } else {
                        if (input[69] < 114.0) {
                            if (input[72] < 61.0) {
                                if (input[4] < 2.0) {
                                    var4 = -0.00023538624;
                                } else {
                                    if (input[11] < 21.0) {
                                        var4 = 0.0010805558;
                                    } else {
                                        if (input[26] < 19.0) {
                                            var4 = -0.00018338978;
                                        } else {
                                            var4 = 0.000748809;
                                        }
                                    }
                                }
                            } else {
                                if (input[31] < 20.0) {
                                    if (input[68] < -45.6068) {
                                        var4 = 0.0009102532;
                                    } else {
                                        if (input[2] < 9.0) {
                                            var4 = -0.00021744065;
                                        } else {
                                            var4 = 0.00018556106;
                                        }
                                    }
                                } else {
                                    if (input[0] < 0.7011046) {
                                        if (input[26] < 7.0) {
                                            var4 = -0.0002236712;
                                        } else {
                                            var4 = 0.0004069161;
                                        }
                                    } else {
                                        if (input[25] < 13.0) {
                                            var4 = 0.00047122742;
                                        } else {
                                            var4 = -0.00043837898;
                                        }
                                    }
                                }
                            }
                        } else {
                            if (input[70] < 37.10705) {
                                var4 = -0.0006121276;
                            } else {
                                if (input[66] < 127.0) {
                                    if (input[68] < -68.593) {
                                        if (input[61] < 12.0) {
                                            var4 = -0.0008443271;
                                        } else {
                                            var4 = -0.00017604473;
                                        }
                                    } else {
                                        if (input[63] < 12.0) {
                                            var4 = 0.0004927337;
                                        } else {
                                            var4 = -0.0000205591;
                                        }
                                    }
                                } else {
                                    if (input[4] < 4.0) {
                                        var4 = 0.000059707847;
                                    } else {
                                        var4 = 0.0007641841;
                                    }
                                }
                            }
                        }
                    }
                } else {
                    var4 = 0.0007013017;
                }
            }
        } else {
            if (input[34] < 266.0) {
                if (input[21] < 16.0) {
                    if (input[74] < 4148.0) {
                        if (input[29] < -43.07445) {
                            var4 = -0.0003820923;
                        } else {
                            if (input[58] < 3.0) {
                                var4 = 0.0003775507;
                            } else {
                                var4 = 0.0008493266;
                            }
                        }
                    } else {
                        if (input[47] < 164.0) {
                            var4 = -0.00071598974;
                        } else {
                            if (input[33] < 112.0) {
                                if (input[30] < 32.0) {
                                    if (input[21] < 12.0) {
                                        var4 = -0.0005314466;
                                    } else {
                                        var4 = 0.00035283106;
                                    }
                                } else {
                                    if (input[32] < 173.0) {
                                        var4 = 0.0008499307;
                                    } else {
                                        var4 = 0.00026930045;
                                    }
                                }
                            } else {
                                if (input[3] < 23.0) {
                                    if (input[29] < 64.0) {
                                        var4 = 0.00057186256;
                                    } else {
                                        var4 = -0.00011781876;
                                    }
                                } else {
                                    if (input[38] < 0.18249917) {
                                        var4 = 0.00011841357;
                                    } else {
                                        if (input[74] < 1461280.0) {
                                            var4 = -0.00089370424;
                                        } else {
                                            var4 = -0.00017152402;
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    if (input[66] < -2.67173) {
                        var4 = 0.00015364763;
                    } else {
                        var4 = 0.00094985584;
                    }
                }
            } else {
                if (input[70] < 9.95751) {
                    var4 = 0.00039745655;
                } else {
                    if (input[64] < 46.0) {
                        if (input[60] < 4.0) {
                            var4 = 0.00020991468;
                        } else {
                            if (input[42] < 5.0) {
                                if (input[34] < 770.0) {
                                    var4 = -0.0010356558;
                                } else {
                                    var4 = -0.00018663384;
                                }
                            } else {
                                var4 = -0.0012431558;
                            }
                        }
                    } else {
                        var4 = 0.00016533313;
                    }
                }
            }
        }
    } else {
        if (input[67] < 125.0) {
            if (input[58] < 7.0) {
                if (input[3] < 10.0) {
                    if (input[72] < 200.0) {
                        var4 = -0.00012445451;
                    } else {
                        if (input[29] < -0.00000137091) {
                            var4 = -0.0011892885;
                        } else {
                            var4 = -0.0005524547;
                        }
                    }
                } else {
                    if (input[73] < 63054.0) {
                        if (input[28] < 52.0) {
                            if (input[66] < 122.0) {
                                if (input[64] < 63.0) {
                                    if (input[67] < 58.0) {
                                        if (input[44] < 31703.0) {
                                            var4 = -0.00006947463;
                                        } else {
                                            var4 = 0.0006191739;
                                        }
                                    } else {
                                        if (input[40] < 26.0) {
                                            var4 = 0.0010543462;
                                        } else {
                                            var4 = 0.0002667693;
                                        }
                                    }
                                } else {
                                    var4 = -0.0006032712;
                                }
                            } else {
                                var4 = -0.0005037209;
                            }
                        } else {
                            if (input[8] < 39765.0) {
                                if (input[4] < 587.0) {
                                    var4 = -0.0002721931;
                                } else {
                                    var4 = 0.000791053;
                                }
                            } else {
                                if (input[71] < 113.0) {
                                    if (input[28] < 54.0) {
                                        var4 = -0.0008310964;
                                    } else {
                                        var4 = 0.0007518676;
                                    }
                                } else {
                                    if (input[68] < 9.0) {
                                        if (input[0] < 0.18168657) {
                                            var4 = -0.00060703006;
                                        } else {
                                            var4 = 0.00009913612;
                                        }
                                    } else {
                                        var4 = -0.0012472852;
                                    }
                                }
                            }
                        }
                    } else {
                        if (input[28] < 57.0) {
                            if (input[28] < 47.0) {
                                if (input[74] < 1332034.0) {
                                    var4 = -0.0006922658;
                                } else {
                                    if (input[67] < 48.0) {
                                        if (input[75] < 158100.0) {
                                            var4 = -0.00040208944;
                                        } else {
                                            var4 = -0.00024612856;
                                        }
                                    } else {
                                        var4 = 0.0000056987765;
                                    }
                                }
                            } else {
                                if (input[31] < 190.358) {
                                    if (input[37] < 172380.0) {
                                        if (input[46] < 38960.0) {
                                            var4 = -0.0004888939;
                                        } else {
                                            var4 = -0.00020958726;
                                        }
                                    } else {
                                        if (input[42] < 691.0) {
                                            var4 = 0.00037092148;
                                        } else {
                                            var4 = -0.00014667968;
                                        }
                                    }
                                } else {
                                    if (input[75] < 143820.0) {
                                        var4 = 0.0006274847;
                                    } else {
                                        if (input[29] < 44.0) {
                                            var4 = 0.000090119574;
                                        } else {
                                            var4 = -0.00041717847;
                                        }
                                    }
                                }
                            }
                        } else {
                            var4 = 0.0003107015;
                        }
                    }
                }
            } else {
                var4 = 0.00060967164;
            }
        } else {
            if (input[8] < 58643.0) {
                var4 = 0.0009032218;
            } else {
                var4 = 0.000053547283;
            }
        }
    }
    return 0.004681166585694882 + (var0 + var1 + var2 + var3 + var4);
}

#endif // DECISION_TREE_SAMPLERATE_3D
