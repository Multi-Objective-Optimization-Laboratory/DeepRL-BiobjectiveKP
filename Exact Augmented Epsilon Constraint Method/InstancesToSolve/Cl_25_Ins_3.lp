\ Model Cl_25_Ins_3.lp
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
 
Subject To
 obj1: y[0] = 0
 obj2: y[1] = 0
 obj1_Exp: - y[0] + 99 x[0] + 96 x[1] + 88 x[2] + 88 x[3] + 84 x[4]
   + 81 x[5] + 75 x[6] + 66 x[7] + 59 x[8] + 50 x[9] + 50 x[10] + 40 x[11]
   + 35 x[12] + 31 x[13] + 31 x[14] + 30 x[15] + 24 x[16] + 23 x[17]
   + 23 x[18] + 13 x[19] + 11 x[20] + 8 x[21] + 7 x[22] + 5 x[23] + x[24]
   = 0
 obj2_Exp: - y[1] + 96 x[0] + 60 x[1] + 74 x[2] + 91 x[3] + 96 x[4]
   + 76 x[5] + 98 x[6] + 94 x[7] + 35 x[8] + 20 x[9] + 86 x[10] + 16 x[11]
   + 77 x[12] + 37 x[13] + 58 x[14] + 11 x[15] + 23 x[16] + 55 x[17]
   + 87 x[18] + 23 x[19] + 39 x[20] + 11 x[21] + 29 x[22] + 80 x[23]
   + 11 x[24] = 0
 Budget: 109 x[0] + 106 x[1] + 98 x[2] + 98 x[3] + 94 x[4] + 91 x[5]
   + 85 x[6] + 76 x[7] + 69 x[8] + 60 x[9] + 60 x[10] + 50 x[11] + 45 x[12]
   + 41 x[13] + 41 x[14] + 40 x[15] + 34 x[16] + 33 x[17] + 33 x[18]
   + 23 x[19] + 21 x[20] + 18 x[21] + 17 x[22] + 15 x[23] + 11 x[24]
   <= 382
Bounds
Binaries
 x[0] x[1] x[2] x[3] x[4] x[5] x[6] x[7] x[8] x[9] x[10] x[11] x[12] x[13]
 x[14] x[15] x[16] x[17] x[18] x[19] x[20] x[21] x[22] x[23] x[24]
Generals
 y[0] y[1]
End
