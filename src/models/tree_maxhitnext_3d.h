
#include <math.h>
#ifndef DECISION_TREE_MAXHITNEXT_3D
#define DECISION_TREE_MAXHITNEXT_3D
/*
0 A_Density
1 A_GiniIndex
2 A_GridSize_0
3 A_GridSize_1
4 A_GridSize_2
5 A_MaxPoints
6 A_NonEmptyCells
7 A_NumPoints
8 B_Density
9 B_GiniIndex
10 B_GridSize_0
11 B_GridSize_1
12 B_GridSize_2
13 B_MaxPoints
14 B_NonEmptyCells
15 B_NumPoints
16 CMax2
17 ComparedPairs
18 Density
19 EBTime
20 Hits1
21 NumInputPoints
22 NumOutputPoints
23 NumPointsPerCell
24 NumTermPoints
25 RTTime
26 SampleRate

struct Input {
    double A_Density;
    double A_GiniIndex;
    double A_GridSize[3];
    double A_MaxPoints;
    double A_NonEmptyCells;
    double A_NumPoints;
    double B_Density;
    double B_GiniIndex;
    double B_GridSize[3];
    double B_MaxPoints;
    double B_NonEmptyCells;
    double B_NumPoints;
    double CMax2;
    double ComparedPairs;
    double Density;
    double EBTime;
    double Hits1;
    double NumInputPoints;
    double NumOutputPoints;
    double NumPointsPerCell;
    double NumTermPoints;
    double RTTime;
    double SampleRate;
};

*/
inline double PredictMaxHitNext_3D(double * input) {
    double var0;
    if (input[16] < 57.0) {
        if (input[22] < 441837.0) {
            if (input[16] < 34.0) {
                if (input[7] < 107328.0) {
                    var0 = 7.8925796;
                } else {
                    var0 = 15.199036;
                }
            } else {
                if (input[8] < 0.46190926) {
                    var0 = 12.45752;
                } else {
                    var0 = 4.0305104;
                }
            }
        } else {
            var0 = 21.180313;
        }
    } else {
        if (input[16] < 99.0) {
            if (input[22] < 30003.0) {
                if (input[22] < 980.0) {
                    var0 = -1.7227892;
                } else {
                    var0 = 9.344723;
                }
            } else {
                var0 = -6.0579724;
            }
        } else {
            if (input[22] < 14482.0) {
                var0 = -5.342511;
            } else {
                var0 = -6.6048684;
            }
        }
    }
    double var1;
    if (input[16] < 57.0) {
        if (input[22] < 441837.0) {
            if (input[20] < 3163345.0) {
                if (input[25] < 4.649) {
                    var1 = 9.697127;
                } else {
                    var1 = 4.044351;
                }
            } else {
                if (input[22] < 370749.0) {
                    var1 = 16.451225;
                } else {
                    var1 = 5.717666;
                }
            }
        } else {
            if (input[14] < 82543.0) {
                var1 = 19.855434;
            } else {
                var1 = 12.917397;
            }
        }
    } else {
        if (input[16] < 69.0) {
            if (input[22] < 2513.0) {
                var1 = 6.406933;
            } else {
                var1 = -5.187156;
            }
        } else {
            if (input[22] < 30003.0) {
                var1 = -3.937622;
            } else {
                var1 = -5.128142;
            }
        }
    }
    double var2;
    if (input[16] < 57.0) {
        if (input[22] < 441837.0) {
            if (input[9] < 0.18824393) {
                if (input[9] < 0.18461692) {
                    var2 = 6.994297;
                } else {
                    var2 = 12.045103;
                }
            } else {
                if (input[22] < 160472.0) {
                    var2 = 6.9096293;
                } else {
                    var2 = -1.6053356;
                }
            }
        } else {
            if (input[25] < 4.301) {
                var2 = 9.065384;
            } else {
                var2 = 16.098635;
            }
        }
    } else {
        if (input[16] < 75.0) {
            if (input[22] < 14482.0) {
                var2 = 6.2945957;
            } else {
                var2 = -3.8900692;
            }
        } else {
            if (input[9] < 0.15698235) {
                var2 = 0.1226541;
            } else {
                var2 = -3.7803113;
            }
        }
    }
    double var3;
    if (input[16] < 57.0) {
        if (input[0] < 0.22023438) {
            if (input[2] < 20.0) {
                var3 = -2.9336426;
            } else {
                var3 = 6.176704;
            }
        } else {
            if (input[16] < 36.0) {
                if (input[2] < 49.0) {
                    var3 = 6.4467974;
                } else {
                    var3 = 12.17446;
                }
            } else {
                if (input[8] < 0.47007447) {
                    var3 = 7.677255;
                } else {
                    var3 = -2.6705575;
                }
            }
        }
    } else {
        if (input[16] < 99.0) {
            if (input[22] < 30003.0) {
                var3 = 0.9096593;
            } else {
                var3 = -3.0655334;
            }
        } else {
            if (input[25] < 18.392) {
                var3 = -3.2957814;
            } else {
                var3 = -2.304692;
            }
        }
    }
    double var4;
    if (input[16] < 59.0) {
        if (input[7] < 1420471.0) {
            if (input[16] < 42.0) {
                if (input[25] < 0.177) {
                    var4 = -1.4392294;
                } else {
                    var4 = 5.375692;
                }
            } else {
                if (input[4] < 48.0) {
                    var4 = -8.8101225;
                } else {
                    var4 = 2.8914938;
                }
            }
        } else {
            if (input[0] < 0.4505519) {
                var4 = 15.855651;
            } else {
                var4 = 5.7850847;
            }
        }
    } else {
        if (input[22] < 473.0) {
            if (input[18] < 0.8790229) {
                var4 = -3.7560487;
            } else {
                var4 = 0.73608214;
            }
        } else {
            if (input[22] < 30003.0) {
                var4 = -1.1666735;
            } else {
                var4 = -2.7062285;
            }
        }
    }
    double var5;
    if (input[16] < 59.0) {
        if (input[22] < 968.0) {
            if (input[20] < 2766890.0) {
                if (input[25] < 5.004) {
                    var5 = 3.9039452;
                } else {
                    var5 = -6.1010528;
                }
            } else {
                if (input[22] < 404.0) {
                    var5 = 6.8034897;
                } else {
                    var5 = 14.971535;
                }
            }
        } else {
            if (input[16] < 34.0) {
                var5 = 4.797502;
            } else {
                var5 = -0.32771546;
            }
        }
    } else {
        if (input[16] < 75.0) {
            if (input[10] < 51.0) {
                var5 = 1.1301607;
            } else {
                var5 = -3.5765312;
            }
        } else {
            if (input[18] < 0.8727557) {
                var5 = -2.1373894;
            } else {
                var5 = -1.0433543;
            }
        }
    }
    double var6;
    if (input[16] < 36.0) {
        if (input[6] < 75951.0) {
            if (input[16] < 8.0) {
                if (input[0] < 0.24196646) {
                    var6 = -2.8436391;
                } else {
                    var6 = 2.8537498;
                }
            } else {
                if (input[22] < 878.0) {
                    var6 = 9.61004;
                } else {
                    var6 = 2.3483145;
                }
            }
        } else {
            var6 = 11.469205;
        }
    } else {
        if (input[16] < 75.0) {
            if (input[22] < 14482.0) {
                if (input[22] < 449.0) {
                    var6 = -2.1636393;
                } else {
                    var6 = 10.18735;
                }
            } else {
                var6 = -1.8472294;
            }
        } else {
            if (input[9] < 0.15698235) {
                var6 = 1.2298468;
            } else {
                var6 = -1.6410128;
            }
        }
    }
    double var7;
    if (input[16] < 59.0) {
        if (input[22] < 416309.0) {
            if (input[22] < 783.0) {
                if (input[16] < 14.0) {
                    var7 = 0.325837;
                } else {
                    var7 = 6.1482873;
                }
            } else {
                if (input[18] < 0.84514403) {
                    var7 = 2.6718037;
                } else {
                    var7 = -2.4038963;
                }
            }
        } else {
            if (input[25] < 4.206) {
                var7 = 1.6762526;
            } else {
                var7 = 8.336716;
            }
        }
    } else {
        if (input[15] < 1538823.0) {
            if (input[16] < 75.0) {
                var7 = 0.7681939;
            } else {
                var7 = -1.2621231;
            }
        } else {
            if (input[22] < 968.0) {
                var7 = -4.337157;
            } else {
                var7 = -1.2909803;
            }
        }
    }
    double var8;
    if (input[16] < 36.0) {
        if (input[6] < 75951.0) {
            if (input[8] < 0.46149337) {
                if (input[22] < 40998.0) {
                    var8 = 1.7356008;
                } else {
                    var8 = 7.157843;
                }
            } else {
                if (input[4] < 50.0) {
                    var8 = 1.6475384;
                } else {
                    var8 = -4.6724253;
                }
            }
        } else {
            var8 = 8.769437;
        }
    } else {
        if (input[7] < 1361188.0) {
            if (input[20] < 2333970.0) {
                if (input[11] < 62.0) {
                    var8 = -1.0790813;
                } else {
                    var8 = 5.4289656;
                }
            } else {
                var8 = -2.7432086;
            }
        } else {
            if (input[0] < 0.4363274) {
                var8 = 3.2184937;
            } else {
                var8 = -0.71740776;
            }
        }
    }
    double var9;
    if (input[16] < 36.0) {
        if (input[2] < 49.0) {
            if (input[8] < 0.46096665) {
                if (input[9] < 0.17247567) {
                    var9 = -4.131616;
                } else {
                    var9 = 3.7440114;
                }
            } else {
                if (input[7] < 1266538.0) {
                    var9 = 1.5899911;
                } else {
                    var9 = -3.5312688;
                }
            }
        } else {
            if (input[7] < 1314433.0) {
                var9 = 10.874915;
            } else {
                var9 = 3.4620035;
            }
        }
    } else {
        if (input[22] < 132436.0) {
            if (input[16] < 75.0) {
                var9 = 2.0412443;
            } else {
                var9 = -0.69234055;
            }
        } else {
            if (input[25] < 5.69) {
                var9 = -3.4745548;
            } else {
                var9 = -1.1144456;
            }
        }
    }
    double var10;
    if (input[16] < 34.0) {
        if (input[22] < 40998.0) {
            if (input[22] < 788.0) {
                if (input[16] < 14.0) {
                    var10 = -0.38103375;
                } else {
                    var10 = 7.5518885;
                }
            } else {
                if (input[25] < 4.354) {
                    var10 = -1.6749375;
                } else {
                    var10 = -8.61122;
                }
            }
        } else {
            if (input[15] < 1576111.0) {
                var10 = 6.69806;
            } else {
                var10 = 2.3300626;
            }
        }
    } else {
        if (input[22] < 38.0) {
            if (input[10] < 49.0) {
                var10 = -0.90979654;
            } else {
                var10 = -5.7782035;
            }
        } else {
            if (input[22] < 14482.0) {
                var10 = 0.24732606;
            } else {
                var10 = -0.959849;
            }
        }
    }
    double var11;
    if (input[16] < 59.0) {
        if (input[20] < 2856258.0) {
            if (input[25] < 5.541) {
                if (input[9] < 0.1901689) {
                    var11 = 2.061142;
                } else {
                    var11 = -1.9776793;
                }
            } else {
                if (input[18] < 0.8397175) {
                    var11 = -1.4891735;
                } else {
                    var11 = -9.195418;
                }
            }
        } else {
            if (input[22] < 888.0) {
                var11 = 5.7409744;
            } else {
                var11 = 0.92135304;
            }
        }
    } else {
        if (input[14] < 4671.0) {
            if (input[9] < 0.21668872) {
                var11 = -0.39736173;
            } else {
                var11 = -1.5367895;
            }
        } else {
            if (input[25] < 0.855) {
                var11 = 3.052591;
            } else {
                var11 = -0.43995973;
            }
        }
    }
    double var12;
    if (input[16] < 59.0) {
        if (input[18] < 0.49737906) {
            if (input[12] < 51.0) {
                if (input[2] < 18.0) {
                    var12 = -1.8077122;
                } else {
                    var12 = 2.32404;
                }
            } else {
                var12 = -4.7827616;
            }
        } else {
            if (input[18] < 0.846888) {
                if (input[9] < 0.17427662) {
                    var12 = -2.7230632;
                } else {
                    var12 = 3.6750648;
                }
            } else {
                var12 = 0.24305959;
            }
        }
    } else {
        if (input[22] < 490.0) {
            if (input[16] < 98.0) {
                var12 = -3.4656007;
            } else {
                var12 = -0.57299197;
            }
        } else {
            if (input[22] < 546.0) {
                var12 = 3.7352352;
            } else {
                var12 = -0.32067448;
            }
        }
    }
    double var13;
    if (input[16] < 34.0) {
        if (input[18] < 0.87479734) {
            if (input[22] < 40998.0) {
                if (input[22] < 788.0) {
                    var13 = 0.8342186;
                } else {
                    var13 = -3.3501453;
                }
            } else {
                if (input[15] < 1555343.0) {
                    var13 = 4.747215;
                } else {
                    var13 = -0.6639693;
                }
            }
        } else {
            if (input[15] < 1600127.0) {
                var13 = 4.6664515;
            } else {
                var13 = 8.288441;
            }
        }
    } else {
        if (input[1] < 0.1709954) {
            if (input[1] < 0.16681667) {
                var13 = -1.0955333;
            } else {
                var13 = 5.4976254;
            }
        } else {
            if (input[16] < 35.0) {
                var13 = -2.699703;
            } else {
                var13 = -0.30314678;
            }
        }
    }
    double var14;
    if (input[18] < 0.878111) {
        if (input[9] < 0.18844242) {
            if (input[25] < 6.59) {
                if (input[7] < 1381586.0) {
                    var14 = 0.46514222;
                } else {
                    var14 = 4.1845794;
                }
            } else {
                if (input[25] < 6.905) {
                    var14 = -3.411616;
                } else {
                    var14 = 0.019830178;
                }
            }
        } else {
            if (input[18] < 0.84514403) {
                var14 = -0.36482686;
            } else {
                var14 = -2.3129256;
            }
        }
    } else {
        if (input[3] < 61.0) {
            if (input[7] < 1487411.0) {
                var14 = 6.168214;
            } else {
                var14 = -1.5640478;
            }
        } else {
            if (input[0] < 0.46425146) {
                var14 = 3.5908687;
            } else {
                var14 = -1.3468827;
            }
        }
    }
    double var15;
    if (input[16] < 36.0) {
        if (input[7] < 1420471.0) {
            if (input[7] < 1318434.0) {
                if (input[20] < 3159446.0) {
                    var15 = 0.38403544;
                } else {
                    var15 = 7.562263;
                }
            } else {
                if (input[0] < 0.45652142) {
                    var15 = -5.219044;
                } else {
                    var15 = 2.1095355;
                }
            }
        } else {
            if (input[0] < 0.4537567) {
                var15 = 8.075867;
            } else {
                if (input[15] < 1598372.0) {
                    var15 = -2.4074774;
                } else {
                    var15 = 4.583835;
                }
            }
        }
    } else {
        if (input[11] < 67.0) {
            if (input[22] < 105.0) {
                var15 = -1.7702217;
            } else {
                var15 = -0.04123909;
            }
        } else {
            var15 = -3.9418957;
        }
    }
    double var16;
    if (input[16] < 146.0) {
        if (input[11] < 68.0) {
            if (input[22] < 160472.0) {
                if (input[1] < 0.16614307) {
                    var16 = -2.7282064;
                } else {
                    var16 = 1.1426004;
                }
            } else {
                if (input[8] < 0.46149337) {
                    var16 = 0.62983626;
                } else {
                    var16 = -1.2543216;
                }
            }
        } else {
            var16 = -3.4587708;
        }
    } else {
        if (input[25] < 18.392) {
            if (input[1] < 0.1715446) {
                if (input[7] < 1345927.0) {
                    var16 = -0.8497061;
                } else {
                    var16 = -2.0059817;
                }
            } else {
                var16 = -0.66423935;
            }
        } else {
            if (input[0] < 0.4551251) {
                var16 = 0.83337194;
            } else {
                var16 = -0.61608624;
            }
        }
    }
    double var17;
    if (input[1] < 0.19140181) {
        if (input[18] < 0.878111) {
            if (input[3] < 58.0) {
                if (input[20] < 3186190.0) {
                    var17 = 0.24469534;
                } else {
                    var17 = 8.466629;
                }
            } else {
                if (input[8] < 0.4677079) {
                    var17 = 0.08163834;
                } else {
                    var17 = -1.6665081;
                }
            }
        } else {
            if (input[22] < 136654.0) {
                if (input[25] < 8.042) {
                    var17 = 7.064018;
                } else {
                    var17 = 0.34256718;
                }
            } else {
                var17 = 0.3042688;
            }
        }
    } else {
        if (input[15] < 1640658.0) {
            if (input[22] < 8486.0) {
                var17 = -0.98445225;
            } else {
                var17 = 0.43364435;
            }
        } else {
            var17 = -3.9542263;
        }
    }
    double var18;
    if (input[25] < 0.177) {
        var18 = -3.237362;
    } else {
        if (input[25] < 0.266) {
            var18 = 3.9531593;
        } else {
            if (input[1] < 0.19088592) {
                if (input[1] < 0.18955559) {
                    if (input[9] < 0.17826946) {
                        var18 = -1.1841928;
                    } else {
                        var18 = 0.3519726;
                    }
                } else {
                    if (input[7] < 1473295.0) {
                        var18 = 9.632564;
                    } else {
                        var18 = 0.17896283;
                    }
                }
            } else {
                if (input[14] < 73849.0) {
                    if (input[0] < 0.22023438) {
                        var18 = -1.9767641;
                    } else {
                        var18 = 0.1613534;
                    }
                } else {
                    if (input[0] < 0.4846747) {
                        var18 = -2.490009;
                    } else {
                        var18 = 0.25397742;
                    }
                }
            }
        }
    }
    double var19;
    if (input[16] < 34.0) {
        if (input[22] < 40998.0) {
            if (input[22] < 788.0) {
                if (input[12] < 51.0) {
                    var19 = 2.8367732;
                } else {
                    var19 = -2.417176;
                }
            } else {
                if (input[25] < 4.354) {
                    var19 = -1.1408451;
                } else {
                    var19 = -6.8105288;
                }
            }
        } else {
            if (input[22] < 68647.0) {
                var19 = 7.084268;
            } else {
                var19 = 1.6459345;
            }
        }
    } else {
        if (input[0] < 0.4363274) {
            if (input[2] < 50.0) {
                var19 = -0.16066445;
            } else {
                var19 = 4.607793;
            }
        } else {
            if (input[0] < 0.4428271) {
                var19 = -2.3789558;
            } else {
                var19 = -0.1425373;
            }
        }
    }
    return 53.879858657243815 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_MAXHITNEXT_3D
