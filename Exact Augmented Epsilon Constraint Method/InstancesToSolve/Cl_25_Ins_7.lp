\ Model Cl_25_Ins_7.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 98 x[0] + 96 x[1] + 85 x[2] + 82 x[3] + 73 x[4]
   + 72 x[5] + 67 x[6] + 56 x[7] + 54 x[8] + 52 x[9] + 52 x[10] + 51 x[11]
   + 45 x[12] + 44 x[13] + 40 x[14] + 39 x[15] + 34 x[16] + 32 x[17]
   + 28 x[18] + 21 x[19] + 15 x[20] + 13 x[21] + 4 x[22] + 3 x[23]
   + 2 x[24] = 0
 obj2_Exp: - y[1] + 60 x[0] + 94 x[1] + 2 x[2] + 3 x[3] + 23 x[4] + 11 x[5]
   + 6 x[6] + 15 x[7] + 88 x[8] + 53 x[9] + 76 x[10] + 36 x[11] + 60 x[12]
   + 23 x[13] + 48 x[14] + 26 x[15] + 35 x[16] + 87 x[17] + 77 x[18]
   + 63 x[19] + 81 x[20] + 78 x[21] + 10 x[22] + 42 x[23] + 35 x[24] = 0
 Budget: 108 x[0] + 106 x[1] + 95 x[2] + 92 x[3] + 83 x[4] + 82 x[5]
   + 77 x[6] + 66 x[7] + 64 x[8] + 62 x[9] + 62 x[10] + 61 x[11] + 55 x[12]
   + 54 x[13] + 50 x[14] + 49 x[15] + 44 x[16] + 42 x[17] + 38 x[18]
   + 31 x[19] + 25 x[20] + 23 x[21] + 14 x[22] + 13 x[23] + 12 x[24]
   <= 579
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24]
Generals
 y[0] y[1]
End
