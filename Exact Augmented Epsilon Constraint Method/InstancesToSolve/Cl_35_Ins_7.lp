\ Model Cl_35_Ins_7.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 97 x[0] + 93 x[1] + 84 x[2] + 82 x[3] + 82 x[4]
   + 74 x[5] + 72 x[6] + 71 x[7] + 68 x[8] + 67 x[9] + 62 x[10] + 60 x[11]
   + 54 x[12] + 51 x[13] + 51 x[14] + 50 x[15] + 48 x[16] + 41 x[17]
   + 41 x[18] + 35 x[19] + 33 x[20] + 30 x[21] + 29 x[22] + 28 x[23]
   + 27 x[24] + 23 x[25] + 22 x[26] + 21 x[27] + 17 x[28] + 13 x[29]
   + 10 x[30] + 9 x[31] + 8 x[32] + 7 x[33] + 3 x[34] = 0
 obj2_Exp: - y[1] + 16 x[0] + 45 x[1] + 32 x[2] + 48 x[3] + 69 x[4]
   + 8 x[5] + 39 x[6] + 84 x[7] + 59 x[8] + 63 x[9] + 2 x[10] + 36 x[11]
   + 63 x[12] + 17 x[13] + 50 x[14] + 14 x[15] + 89 x[16] + 25 x[17]
   + 33 x[18] + 2 x[19] + 19 x[20] + 55 x[21] + 85 x[22] + 78 x[23]
   + 91 x[24] + 33 x[25] + 16 x[26] + 59 x[27] + 30 x[28] + 43 x[29]
   + 2 x[30] + 21 x[31] + 2 x[32] + 98 x[33] + 94 x[34] = 0
 Budget: 107 x[0] + 103 x[1] + 94 x[2] + 92 x[3] + 92 x[4] + 84 x[5]
   + 82 x[6] + 81 x[7] + 78 x[8] + 77 x[9] + 72 x[10] + 70 x[11] + 64 x[12]
   + 61 x[13] + 61 x[14] + 60 x[15] + 58 x[16] + 51 x[17] + 51 x[18]
   + 45 x[19] + 43 x[20] + 40 x[21] + 39 x[22] + 38 x[23] + 37 x[24]
   + 33 x[25] + 32 x[26] + 31 x[27] + 27 x[28] + 23 x[29] + 20 x[30]
   + 19 x[31] + 18 x[32] + 17 x[33] + 13 x[34] <= 829
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34]
Generals
 y[0] y[1]
End
