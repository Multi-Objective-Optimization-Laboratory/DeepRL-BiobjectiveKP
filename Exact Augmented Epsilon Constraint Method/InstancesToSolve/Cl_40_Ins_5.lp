\ Model Cl_40_Ins_5.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 93 x[0] + 92 x[1] + 92 x[2] + 83 x[3] + 74 x[4]
   + 69 x[5] + 65 x[6] + 65 x[7] + 64 x[8] + 64 x[9] + 62 x[10] + 56 x[11]
   + 54 x[12] + 54 x[13] + 53 x[14] + 53 x[15] + 52 x[16] + 52 x[17]
   + 51 x[18] + 48 x[19] + 45 x[20] + 36 x[21] + 35 x[22] + 33 x[23]
   + 33 x[24] + 32 x[25] + 31 x[26] + 29 x[27] + 25 x[28] + 25 x[29]
   + 25 x[30] + 24 x[31] + 14 x[32] + 13 x[33] + 12 x[34] + 11 x[35]
   + 9 x[36] + 7 x[37] + 6 x[38] + 2 x[39] = 0
 obj2_Exp: - y[1] + 41 x[0] + 11 x[1] + 85 x[2] + 65 x[3] + 11 x[4]
   + 69 x[5] + 12 x[6] + x[7] + 36 x[8] + 79 x[9] + 61 x[10] + 76 x[11]
   + 88 x[12] + 89 x[13] + 23 x[14] + 7 x[15] + 93 x[16] + 54 x[17]
   + 92 x[18] + 6 x[19] + 24 x[20] + 76 x[21] + 41 x[22] + 60 x[23]
   + 83 x[24] + 14 x[25] + 80 x[26] + 48 x[27] + 63 x[28] + 97 x[29]
   + 39 x[30] + 42 x[31] + 41 x[32] + 8 x[33] + 20 x[34] + 93 x[35]
   + 80 x[36] + 38 x[37] + 28 x[38] + 26 x[39] = 0
 Budget: 103 x[0] + 102 x[1] + 102 x[2] + 93 x[3] + 84 x[4] + 79 x[5]
   + 75 x[6] + 75 x[7] + 74 x[8] + 74 x[9] + 72 x[10] + 66 x[11] + 64 x[12]
   + 64 x[13] + 63 x[14] + 63 x[15] + 62 x[16] + 62 x[17] + 61 x[18]
   + 58 x[19] + 55 x[20] + 46 x[21] + 45 x[22] + 43 x[23] + 43 x[24]
   + 42 x[25] + 41 x[26] + 39 x[27] + 35 x[28] + 35 x[29] + 35 x[30]
   + 34 x[31] + 24 x[32] + 23 x[33] + 22 x[34] + 21 x[35] + 19 x[36]
   + 17 x[37] + 16 x[38] + 12 x[39] <= 932
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34] x[35] x[36] x[37]
 x[38] x[39]
Generals
 y[0] y[1]
End
