
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
};

*/
inline double PredictMaxHitInit_3D(double * input) {
    double var0;
    if (input[1] <= 0.20997854322195053) {
        if (input[18] <= 15.5) {
            if (input[8] <= 0.4660695195198059) {
                var0 = 132.93413173652695;
            } else {
                var0 = 121.97802197802197;
            }
        } else {
            if (input[10] <= 12.5) {
                var0 = 102.0;
            } else {
                if (input[18] <= 100.5) {
                    var0 = 149.68085106382978;
                } else {
                    var0 = 100.0;
                }
            }
        }
    } else {
        if (input[7] <= 44550.5) {
            var0 = 83.79310344827586;
        } else {
            if (input[19] <= 193.12427520751953) {
                var0 = 86.85714285714286;
            } else {
                if (input[21] <= 90919.5) {
                    if (input[4] <= 16.5) {
                        var0 = 172.0;
                    } else {
                        var0 = 123.27868852459017;
                    }
                } else {
                    var0 = 98.29787234042553;
                }
            }
        }
    }
    return var0;
}

#endif // DECISION_TREE_MAXHITINIT_3D
