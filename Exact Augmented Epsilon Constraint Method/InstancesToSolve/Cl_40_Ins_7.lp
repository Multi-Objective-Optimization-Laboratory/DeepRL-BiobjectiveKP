\ Model Cl_40_Ins_7.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 97 x[0] + 95 x[1] + 93 x[2] + 93 x[3] + 89 x[4]
   + 79 x[5] + 79 x[6] + 78 x[7] + 74 x[8] + 72 x[9] + 69 x[10] + 68 x[11]
   + 67 x[12] + 66 x[13] + 65 x[14] + 62 x[15] + 61 x[16] + 60 x[17]
   + 56 x[18] + 54 x[19] + 53 x[20] + 50 x[21] + 48 x[22] + 48 x[23]
   + 46 x[24] + 43 x[25] + 42 x[26] + 42 x[27] + 41 x[28] + 39 x[29]
   + 25 x[30] + 23 x[31] + 21 x[32] + 21 x[33] + 18 x[34] + 14 x[35]
   + 9 x[36] + 6 x[37] + 5 x[38] + x[39] = 0
 obj2_Exp: - y[1] + 45 x[0] + 19 x[1] + 26 x[2] + 73 x[3] + 91 x[4]
   + 71 x[5] + 77 x[6] + 93 x[7] + x[8] + 61 x[9] + 28 x[10] + 5 x[11]
   + 26 x[12] + 18 x[13] + 13 x[14] + 27 x[15] + 19 x[16] + 27 x[17]
   + 18 x[18] + 78 x[19] + 91 x[20] + 49 x[21] + 65 x[22] + 12 x[23]
   + 34 x[24] + 79 x[25] + 17 x[26] + 11 x[27] + 78 x[28] + 11 x[29]
   + 8 x[30] + 26 x[31] + 69 x[32] + 24 x[33] + 82 x[34] + 9 x[35]
   + 5 x[36] + 67 x[37] + 23 x[38] + 93 x[39] = 0
 Budget: 107 x[0] + 105 x[1] + 103 x[2] + 103 x[3] + 99 x[4] + 89 x[5]
   + 89 x[6] + 88 x[7] + 84 x[8] + 82 x[9] + 79 x[10] + 78 x[11] + 77 x[12]
   + 76 x[13] + 75 x[14] + 72 x[15] + 71 x[16] + 70 x[17] + 66 x[18]
   + 64 x[19] + 63 x[20] + 60 x[21] + 58 x[22] + 58 x[23] + 56 x[24]
   + 53 x[25] + 52 x[26] + 52 x[27] + 51 x[28] + 49 x[29] + 35 x[30]
   + 33 x[31] + 31 x[32] + 31 x[33] + 28 x[34] + 24 x[35] + 19 x[36]
   + 16 x[37] + 15 x[38] + 11 x[39] <= 652
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34] x[35] x[36] x[37]
 x[38] x[39]
Generals
 y[0] y[1]
End
