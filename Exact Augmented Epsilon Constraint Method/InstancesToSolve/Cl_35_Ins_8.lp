\ Model Cl_35_Ins_8.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 98 x[0] + 91 x[1] + 90 x[2] + 86 x[3] + 83 x[4]
   + 80 x[5] + 68 x[6] + 66 x[7] + 64 x[8] + 60 x[9] + 60 x[10] + 59 x[11]
   + 58 x[12] + 53 x[13] + 51 x[14] + 50 x[15] + 44 x[16] + 39 x[17]
   + 39 x[18] + 34 x[19] + 23 x[20] + 20 x[21] + 20 x[22] + 19 x[23]
   + 14 x[24] + 14 x[25] + 12 x[26] + 10 x[27] + 10 x[28] + 8 x[29]
   + 7 x[30] + 6 x[31] + 6 x[32] + 3 x[33] + 3 x[34] = 0
 obj2_Exp: - y[1] + 88 x[0] + 39 x[1] + 77 x[2] + 21 x[3] + 42 x[4]
   + 61 x[5] + 4 x[6] + 86 x[7] + 18 x[8] + 97 x[9] + 9 x[10] + 75 x[11]
   + 77 x[12] + 7 x[13] + 41 x[14] + 55 x[15] + 22 x[16] + 11 x[17]
   + 40 x[18] + 80 x[19] + 36 x[20] + 46 x[21] + 27 x[22] + 4 x[23]
   + 55 x[24] + 76 x[25] + 10 x[26] + 8 x[27] + 13 x[28] + 46 x[29]
   + 10 x[30] + 97 x[31] + 62 x[32] + 63 x[33] + 3 x[34] = 0
 Budget: 108 x[0] + 101 x[1] + 100 x[2] + 96 x[3] + 93 x[4] + 90 x[5]
   + 78 x[6] + 76 x[7] + 74 x[8] + 70 x[9] + 70 x[10] + 69 x[11] + 68 x[12]
   + 63 x[13] + 61 x[14] + 60 x[15] + 54 x[16] + 49 x[17] + 49 x[18]
   + 44 x[19] + 33 x[20] + 30 x[21] + 30 x[22] + 29 x[23] + 24 x[24]
   + 24 x[25] + 22 x[26] + 20 x[27] + 20 x[28] + 18 x[29] + 17 x[30]
   + 16 x[31] + 16 x[32] + 13 x[33] + 13 x[34] <= 804
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34]
Generals
 y[0] y[1]
End
