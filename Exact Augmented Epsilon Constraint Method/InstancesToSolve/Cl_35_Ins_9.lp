\ Model Cl_35_Ins_9.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 93 x[0] + 92 x[1] + 90 x[2] + 87 x[3] + 77 x[4]
   + 75 x[5] + 72 x[6] + 71 x[7] + 66 x[8] + 64 x[9] + 59 x[10] + 58 x[11]
   + 57 x[12] + 55 x[13] + 53 x[14] + 50 x[15] + 48 x[16] + 48 x[17]
   + 44 x[18] + 42 x[19] + 40 x[20] + 40 x[21] + 38 x[22] + 36 x[23]
   + 35 x[24] + 32 x[25] + 28 x[26] + 27 x[27] + 27 x[28] + 27 x[29]
   + 26 x[30] + 26 x[31] + 24 x[32] + 20 x[33] + 18 x[34] = 0
 obj2_Exp: - y[1] + 93 x[0] + 95 x[1] + 8 x[2] + 62 x[3] + 80 x[4]
   + 90 x[5] + 74 x[6] + 44 x[7] + 30 x[8] + 29 x[9] + 74 x[10] + 51 x[11]
   + 37 x[12] + 70 x[13] + 53 x[14] + 75 x[15] + 42 x[16] + 21 x[17]
   + 99 x[18] + 5 x[19] + 60 x[20] + 88 x[21] + 38 x[22] + 12 x[23]
   + 3 x[24] + 72 x[25] + 96 x[26] + 75 x[27] + 36 x[28] + 68 x[29]
   + 52 x[30] + 65 x[31] + 9 x[32] + 87 x[33] + 36 x[34] = 0
 Budget: 103 x[0] + 102 x[1] + 100 x[2] + 97 x[3] + 87 x[4] + 85 x[5]
   + 82 x[6] + 81 x[7] + 76 x[8] + 74 x[9] + 69 x[10] + 68 x[11] + 67 x[12]
   + 65 x[13] + 63 x[14] + 60 x[15] + 58 x[16] + 58 x[17] + 54 x[18]
   + 52 x[19] + 50 x[20] + 50 x[21] + 48 x[22] + 46 x[23] + 45 x[24]
   + 42 x[25] + 38 x[26] + 37 x[27] + 37 x[28] + 37 x[29] + 36 x[30]
   + 36 x[31] + 34 x[32] + 30 x[33] + 28 x[34] <= 755
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24] x[25]
 x[26] x[27] x[28] x[29] x[30] x[31] x[32] x[33] x[34]
Generals
 y[0] y[1]
End
