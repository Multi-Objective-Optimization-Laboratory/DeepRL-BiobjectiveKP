\ Model Cl_25_Ins_6.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 98 x[0] + 95 x[1] + 95 x[2] + 93 x[3] + 90 x[4]
   + 88 x[5] + 86 x[6] + 82 x[7] + 70 x[8] + 70 x[9] + 70 x[10] + 61 x[11]
   + 56 x[12] + 54 x[13] + 42 x[14] + 37 x[15] + 34 x[16] + 33 x[17]
   + 31 x[18] + 30 x[19] + 28 x[20] + 26 x[21] + 26 x[22] + 24 x[23]
   + x[24] = 0
 obj2_Exp: - y[1] + 31 x[0] + 31 x[1] + 56 x[2] + 45 x[3] + 66 x[4]
   + 19 x[5] + 78 x[6] + x[7] + 47 x[8] + 96 x[9] + 3 x[10] + 43 x[11]
   + 89 x[12] + 8 x[13] + 70 x[14] + 61 x[15] + 79 x[16] + 22 x[17]
   + 17 x[18] + 93 x[19] + 28 x[20] + 70 x[21] + 68 x[22] + 97 x[23]
   + 28 x[24] = 0
 Budget: 108 x[0] + 105 x[1] + 105 x[2] + 103 x[3] + 100 x[4] + 98 x[5]
   + 96 x[6] + 92 x[7] + 80 x[8] + 80 x[9] + 80 x[10] + 71 x[11] + 66 x[12]
   + 64 x[13] + 52 x[14] + 47 x[15] + 44 x[16] + 43 x[17] + 41 x[18]
   + 40 x[19] + 38 x[20] + 36 x[21] + 36 x[22] + 34 x[23] + 11 x[24]
   <= 618
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24]
Generals
 y[0] y[1]
End
