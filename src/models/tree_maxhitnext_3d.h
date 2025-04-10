
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
23 NumTermPoints
24 RTTime

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
    double NumTermPoints;
    double RTTime;
};

*/
inline double PredictMaxHitNext_3D(double * input) {
    double var0;
    if (input[16] <= 56.5) {
        if (input[22] <= 441383.0) {
            if (input[16] <= 33.5) {
                if (input[21] <= 104810.0) {
                    var0 = 93.21428571428571;
                } else {
                    if (input[6] <= 73555.5) {
                        var0 = 125.12820512820512;
                    } else {
                        var0 = 183.33333333333334;
                    }
                }
            } else {
                if (input[8] <= 0.4618832617998123) {
                    var0 = 117.24137931034483;
                } else {
                    if (input[24] <= 6.019999980926514) {
                        var0 = 36.36363636363637;
                    } else {
                        var0 = 98.28571428571429;
                    }
                }
            }
        } else {
            var0 = 163.7037037037037;
        }
    } else {
        if (input[16] <= 98.5) {
            if (input[22] <= 43265.5) {
                var0 = 58.91304347826087;
            } else {
                var0 = 23.358778625954198;
            }
        } else {
            var0 = 24.36426116838488;
        }
    }
    return var0;
}

#endif // DECISION_TREE_MAXHITNEXT_3D
