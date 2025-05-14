
#include <math.h>
#ifndef DECISION_TREE_MAXHITINIT_3D
#define DECISION_TREE_MAXHITINIT_3D
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
16 Density
17 GiniIndex
18 HDLowerBound
19 HDUpperBound
20 MaxPoints
21 NonEmptyCells
22 NumPointsPerCell
23 SampleRate

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
    double Density;
    double GiniIndex;
    double HDLowerBound;
    double HDUpperBound;
    double MaxPoints;
    double NonEmptyCells;
    double NumPointsPerCell;
    double SampleRate;
};

*/
inline double PredictMaxHitInit_3D(double * input) {
    double var0;
    if (input[1] < 0.20912473) {
        if (input[18] < 16.0) {
            if (input[8] < 0.4661079) {
                if (input[8] < 0.4578863) {
                    var0 = 0.43963033;
                } else {
                    var0 = 3.0619771;
                }
            } else {
                if (input[8] < 0.4682928) {
                    var0 = -4.33835;
                } else {
                    var0 = -0.5322811;
                }
            }
        } else {
            if (input[10] < 13.0) {
                var0 = -4.8736515;
            } else {
                var0 = 4.0699344;
            }
        }
    } else {
        if (input[7] < 44519.0) {
            if (input[8] < 0.45281994) {
                var0 = -11.2389145;
            } else {
                var0 = -5.9699173;
            }
        } else {
            if (input[19] < 193.02072) {
                var0 = -7.746595;
            } else {
                var0 = -2.1046896;
            }
        }
    }
    double var1;
    if (input[1] < 0.19955413) {
        if (input[18] < 7.0) {
            if (input[17] < 0.21414842) {
                if (input[6] < 80293.0) {
                    var1 = 1.3244518;
                } else {
                    var1 = 7.148557;
                }
            } else {
                if (input[8] < 0.44471714) {
                    var1 = -4.395377;
                } else {
                    var1 = -0.45378456;
                }
            }
        } else {
            if (input[18] < 101.0) {
                var1 = 2.126291;
            } else {
                var1 = -6.2074714;
            }
        }
    } else {
        if (input[7] < 44519.0) {
            if (input[3] < 17.0) {
                var1 = -3.9056065;
            } else {
                var1 = -9.01474;
            }
        } else {
            if (input[19] < 241.2509) {
                var1 = -1.7737817;
            } else {
                var1 = -7.332296;
            }
        }
    }
    double var2;
    if (input[7] < 1289242.0) {
        if (input[15] < 1503899.0) {
            if (input[3] < 23.0) {
                if (input[14] < 70626.0) {
                    var2 = -7.0164795;
                } else {
                    var2 = -1.5338633;
                }
            } else {
                if (input[8] < 0.3378809) {
                    var2 = -5.6378927;
                } else {
                    var2 = 0.9290714;
                }
            }
        } else {
            if (input[15] < 1532078.0) {
                var2 = -11.18737;
            } else {
                var2 = -3.7420323;
            }
        }
    } else {
        if (input[16] < 0.86256117) {
            if (input[15] < 1538698.0) {
                var2 = 2.073932;
            } else {
                var2 = -1.6260465;
            }
        } else {
            if (input[15] < 1493349.0) {
                var2 = -2.486809;
            } else {
                var2 = 1.7674035;
            }
        }
    }
    double var3;
    if (input[1] < 0.20912473) {
        if (input[8] < 0.4661079) {
            if (input[18] < 101.0) {
                if (input[17] < 0.21679285) {
                    var3 = 0.47421423;
                } else {
                    var3 = 2.517831;
                }
            } else {
                var3 = -6.6187744;
            }
        } else {
            if (input[19] < 254.97647) {
                if (input[7] < 1410701.0) {
                    var3 = 2.8844879;
                } else {
                    var3 = -1.362968;
                }
            } else {
                var3 = -1.586502;
            }
        }
    } else {
        if (input[9] < 0.17380732) {
            if (input[9] < 0.17033537) {
                var3 = -0.14965105;
            } else {
                var3 = 4.9418635;
            }
        } else {
            if (input[9] < 0.18274362) {
                var3 = -6.094343;
            } else {
                var3 = -2.5595586;
            }
        }
    }
    double var4;
    if (input[7] < 1289242.0) {
        if (input[14] < 78117.0) {
            if (input[19] < 189.93947) {
                if (input[0] < 0.29325816) {
                    var4 = -6.8425074;
                } else {
                    var4 = -1.5441434;
                }
            } else {
                if (input[7] < 37466.0) {
                    var4 = -5.248672;
                } else {
                    var4 = 0.3859659;
                }
            }
        } else {
            if (input[7] < 95815.0) {
                var4 = -0.5095674;
            } else {
                var4 = -6.032385;
            }
        }
    } else {
        if (input[0] < 0.49583834) {
            if (input[17] < 0.23313276) {
                var4 = 0.51363045;
            } else {
                var4 = 2.6030066;
            }
        } else {
            if (input[11] < 59.0) {
                var4 = -7.1294074;
            } else {
                var4 = -0.9625887;
            }
        }
    }
    double var5;
    if (input[7] < 1300621.0) {
        if (input[15] < 1502233.0) {
            if (input[12] < 51.0) {
                if (input[19] < 204.09311) {
                    var5 = -2.9692237;
                } else {
                    var5 = 0.25102076;
                }
            } else {
                if (input[0] < 0.35773146) {
                    var5 = 7.2778506;
                } else {
                    var5 = 1.4077082;
                }
            }
        } else {
            if (input[7] < 95815.0) {
                var5 = -0.51886415;
            } else {
                var5 = -4.722317;
            }
        }
    } else {
        if (input[7] < 1329451.0) {
            if (input[17] < 0.21596994) {
                var5 = -0.66772574;
            } else {
                var5 = 6.0701547;
            }
        } else {
            if (input[3] < 58.0) {
                var5 = -4.425021;
            } else {
                var5 = 0.58429337;
            }
        }
    }
    double var6;
    if (input[19] < 189.93947) {
        if (input[13] < 34.0) {
            if (input[0] < 0.2877747) {
                var6 = -4.3201375;
            } else {
                if (input[8] < 0.40679353) {
                    var6 = 3.799265;
                } else {
                    var6 = -0.50133574;
                }
            }
        } else {
            if (input[16] < 0.33731893) {
                if (input[11] < 25.0) {
                    var6 = 1.2811573;
                } else {
                    var6 = -4.459744;
                }
            } else {
                var6 = -7.7381973;
            }
        }
    } else {
        if (input[18] < 16.0) {
            if (input[17] < 0.21432234) {
                var6 = 1.2731614;
            } else {
                var6 = -0.62608093;
            }
        } else {
            if (input[1] < 0.18738392) {
                var6 = 3.3840673;
            } else {
                var6 = -0.26586032;
            }
        }
    }
    double var7;
    if (input[6] < 58422.0) {
        if (input[19] < 241.2509) {
            if (input[19] < 235.74138) {
                if (input[19] < 228.65475) {
                    var7 = -0.3816018;
                } else {
                    var7 = -4.229168;
                }
            } else {
                var7 = 5.1854286;
            }
        } else {
            if (input[0] < 0.42983738) {
                if (input[3] < 34.0) {
                    var7 = -5.2569222;
                } else {
                    var7 = -9.3686495;
                }
            } else {
                var7 = -0.975316;
            }
        }
    } else {
        if (input[6] < 60554.0) {
            if (input[9] < 0.19021797) {
                var7 = 0.5525665;
            } else {
                var7 = 7.967289;
            }
        } else {
            if (input[7] < 1289242.0) {
                var7 = -2.850232;
            } else {
                var7 = 0.39853814;
            }
        }
    }
    double var8;
    if (input[10] < 13.0) {
        if (input[7] < 1419295.0) {
            var8 = -6.922531;
        } else {
            var8 = -1.4978577;
        }
    } else {
        if (input[8] < 0.46734184) {
            if (input[2] < 18.0) {
                if (input[4] < 18.0) {
                    var8 = 0.39117864;
                } else {
                    var8 = -4.112521;
                }
            } else {
                if (input[1] < 0.16178523) {
                    var8 = -4.135345;
                } else {
                    var8 = 0.75854254;
                }
            }
        } else {
            if (input[6] < 75629.0) {
                if (input[1] < 0.16979574) {
                    var8 = 1.8988028;
                } else {
                    var8 = -2.2750528;
                }
            } else {
                if (input[6] < 82543.0) {
                    var8 = 3.470593;
                } else {
                    var8 = -2.5124023;
                }
            }
        }
    }
    double var9;
    if (input[3] < 66.0) {
        if (input[0] < 0.48824966) {
            if (input[19] < 189.93947) {
                if (input[12] < 17.0) {
                    var9 = -5.311803;
                } else {
                    var9 = -0.9738777;
                }
            } else {
                if (input[10] < 48.0) {
                    var9 = 1.207056;
                } else {
                    var9 = -0.3827621;
                }
            }
        } else {
            if (input[15] < 1420803.0) {
                var9 = -7.136876;
            } else {
                var9 = 0.64849347;
            }
        }
    } else {
        if (input[19] < 260.14227) {
            if (input[16] < 0.771177) {
                var9 = 3.253983;
            } else {
                var9 = -5.0822124;
            }
        } else {
            if (input[19] < 263.7385) {
                var9 = 7.6069956;
            } else {
                var9 = 2.3451347;
            }
        }
    }
    double var10;
    if (input[4] < 13.0) {
        var10 = -4.4987106;
    } else {
        if (input[6] < 80293.0) {
            if (input[1] < 0.1867623) {
                if (input[18] < 41.0) {
                    if (input[12] < 48.0) {
                        var10 = -3.6756368;
                    } else {
                        var10 = 0.4004769;
                    }
                } else {
                    var10 = 2.5321357;
                }
            } else {
                if (input[12] < 17.0) {
                    var10 = -5.2608676;
                } else {
                    var10 = -0.5203603;
                }
            }
        } else {
            if (input[6] < 82010.0) {
                if (input[17] < 0.21846864) {
                    var10 = 5.9663234;
                } else {
                    var10 = -1.7939177;
                }
            } else {
                if (input[3] < 65.0) {
                    var10 = -2.058769;
                } else {
                    var10 = 1.2338889;
                }
            }
        }
    }
    double var11;
    if (input[18] < 101.0) {
        if (input[17] < 0.23313276) {
            if (input[15] < 185137.0) {
                if (input[7] < 1481988.0) {
                    if (input[0] < 0.3787928) {
                        var11 = -5.675882;
                    } else {
                        var11 = 3.0615041;
                    }
                } else {
                    var11 = -9.189672;
                }
            } else {
                if (input[18] < 16.0) {
                    var11 = -0.16586426;
                } else {
                    var11 = 2.3435764;
                }
            }
        } else {
            if (input[14] < 2398.0) {
                if (input[11] < 25.0) {
                    var11 = 0.0459657;
                } else {
                    var11 = -5.884156;
                }
            } else {
                if (input[1] < 0.29926172) {
                    var11 = 3.477865;
                } else {
                    var11 = -3.8988914;
                }
            }
        }
    } else {
        var11 = -4.527206;
    }
    double var12;
    if (input[9] < 0.17380732) {
        if (input[8] < 0.4398957) {
            if (input[0] < 0.45674828) {
                if (input[0] < 0.4432806) {
                    var12 = 0.76991373;
                } else {
                    var12 = -3.126378;
                }
            } else {
                if (input[10] < 49.0) {
                    var12 = -0.3572944;
                } else {
                    var12 = 7.514618;
                }
            }
        } else {
            var12 = 5.2084627;
        }
    } else {
        if (input[9] < 0.1756505) {
            if (input[19] < 253.01581) {
                if (input[11] < 58.0) {
                    var12 = -4.2075324;
                } else {
                    var12 = -0.32115278;
                }
            } else {
                var12 = -6.75237;
            }
        } else {
            if (input[3] < 66.0) {
                var12 = -0.11894249;
            } else {
                var12 = 2.462836;
            }
        }
    }
    double var13;
    if (input[4] < 13.0) {
        var13 = -3.5640168;
    } else {
        if (input[8] < 0.46734184) {
            if (input[8] < 0.4560275) {
                if (input[3] < 18.0) {
                    var13 = -5.4541554;
                } else {
                    if (input[7] < 59206.0) {
                        var13 = 5.5691314;
                    } else {
                        var13 = -0.14043008;
                    }
                }
            } else {
                if (input[2] < 53.0) {
                    var13 = 1.7811215;
                } else {
                    var13 = -3.6265862;
                }
            }
        } else {
            if (input[8] < 0.4682928) {
                if (input[0] < 0.45336494) {
                    var13 = -3.070884;
                } else {
                    var13 = -6.6860957;
                }
            } else {
                if (input[1] < 0.2909872) {
                    var13 = 0.022196421;
                } else {
                    var13 = -3.9497707;
                }
            }
        }
    }
    double var14;
    if (input[0] < 0.49583834) {
        if (input[6] < 80293.0) {
            if (input[10] < 48.0) {
                if (input[15] < 1302851.0) {
                    if (input[0] < 0.45553112) {
                        var14 = 0.70177984;
                    } else {
                        var14 = -1.4737197;
                    }
                } else {
                    var14 = 2.2859402;
                }
            } else {
                if (input[1] < 0.17874639) {
                    var14 = 0.35073718;
                } else {
                    var14 = -0.95394516;
                }
            }
        } else {
            if (input[18] < 4.0) {
                var14 = 7.503278;
            } else {
                if (input[7] < 1513004.0) {
                    var14 = 6.312015;
                } else {
                    var14 = -0.0012609329;
                }
            }
        }
    } else {
        if (input[1] < 0.20274064) {
            var14 = -5.256313;
        } else {
            var14 = 1.4546497;
        }
    }
    double var15;
    if (input[8] < 0.49709308) {
        if (input[18] < 101.0) {
            if (input[14] < 81337.0) {
                if (input[19] < 260.99426) {
                    if (input[6] < 71517.0) {
                        var15 = 0.46319857;
                    } else {
                        var15 = -1.031088;
                    }
                } else {
                    if (input[19] < 263.55453) {
                        var15 = 3.6372402;
                    } else {
                        var15 = -1.25267;
                    }
                }
            } else {
                if (input[15] < 1604940.0) {
                    if (input[6] < 6860.0) {
                        var15 = 1.2051085;
                    } else {
                        var15 = -5.2247443;
                    }
                } else {
                    var15 = 0.22261456;
                }
            }
        } else {
            var15 = -4.6378455;
        }
    } else {
        if (input[0] < 0.44912454) {
            var15 = -1.4062786;
        } else {
            var15 = 4.473931;
        }
    }
    double var16;
    if (input[0] < 0.462561) {
        if (input[0] < 0.4592676) {
            if (input[0] < 0.45691958) {
                if (input[16] < 0.8680702) {
                    var16 = 0.025431696;
                } else {
                    var16 = -2.7549016;
                }
            } else {
                if (input[9] < 0.19105472) {
                    var16 = 6.096936;
                } else {
                    var16 = -1.6773901;
                }
            }
        } else {
            if (input[1] < 0.1865496) {
                var16 = 1.0744184;
            } else {
                var16 = -7.5650125;
            }
        }
    } else {
        if (input[19] < 254.56236) {
            if (input[0] < 0.48824966) {
                var16 = 2.4654667;
            } else {
                var16 = -2.7805831;
            }
        } else {
            if (input[17] < 0.21542752) {
                var16 = 1.1926184;
            } else {
                var16 = -1.9058468;
            }
        }
    }
    double var17;
    if (input[3] < 65.0) {
        if (input[7] < 1659287.0) {
            if (input[18] < 95.0) {
                if (input[17] < 0.27038977) {
                    if (input[13] < 55.0) {
                        var17 = 0.015209787;
                    } else {
                        var17 = -2.0881555;
                    }
                } else {
                    var17 = 1.7714981;
                }
            } else {
                var17 = -3.8569055;
            }
        } else {
            var17 = -5.464912;
        }
    } else {
        if (input[1] < 0.195686) {
            if (input[15] < 1311471.0) {
                var17 = 4.7746387;
            } else {
                if (input[16] < 0.826876) {
                    var17 = -3.2094285;
                } else {
                    var17 = 2.287309;
                }
            }
        } else {
            if (input[1] < 0.1970009) {
                var17 = -3.939523;
            } else {
                var17 = 1.3550284;
            }
        }
    }
    double var18;
    if (input[1] < 0.16178523) {
        if (input[4] < 51.0) {
            var18 = 0.2888382;
        } else {
            var18 = -5.814136;
        }
    } else {
        if (input[18] < 1.0) {
            if (input[3] < 60.0) {
                if (input[21] < 106801.0) {
                    var18 = -0.8227706;
                } else {
                    var18 = 5.4021635;
                }
            } else {
                if (input[6] < 70013.0) {
                    var18 = -8.012045;
                } else {
                    var18 = -0.36287662;
                }
            }
        } else {
            if (input[18] < 3.0) {
                if (input[14] < 81337.0) {
                    var18 = 3.637338;
                } else {
                    var18 = -1.8546575;
                }
            } else {
                if (input[18] < 6.0) {
                    var18 = -1.122002;
                } else {
                    var18 = 0.40783593;
                }
            }
        }
    }
    double var19;
    if (input[19] < 189.93947) {
        if (input[9] < 0.3278451) {
            if (input[17] < 0.27038977) {
                if (input[19] < 173.19064) {
                    var19 = 0.6917797;
                } else {
                    var19 = -3.864148;
                }
            } else {
                var19 = 3.337408;
            }
        } else {
            var19 = -4.2934704;
        }
    } else {
        if (input[7] < 44519.0) {
            if (input[6] < 1025.0) {
                var19 = 1.940684;
            } else {
                if (input[6] < 1291.0) {
                    var19 = -3.1865964;
                } else {
                    var19 = -6.219568;
                }
            }
        } else {
            if (input[6] < 2185.0) {
                if (input[19] < 226.05087) {
                    var19 = 7.280092;
                } else {
                    var19 = -3.0864208;
                }
            } else {
                var19 = 0.076223455;
            }
        }
    }
    return 127.58666666666667 + (var0 + var1 + var2 + var3 + var4 + var5 + var6 + var7 + var8 + var9 + var10 + var11 + var12 + var13 + var14 + var15 + var16 + var17 + var18 + var19);
}

#endif // DECISION_TREE_MAXHITINIT_3D
