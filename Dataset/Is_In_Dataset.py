import os
filepath = 'D:\文件\北大\github\ProteinBackboneDesign\Dataset\PDBDataset_raw_chain'
k = 0
lst_1 = []
lst_2 = []
lst_3 = []
with open('D:\文件\北大\github\ProteinBackboneDesign\Dataset\cullpdb_pc30.0_res0.0-2.5_len40-10000_R0.25_Xray_d2021_02_28_chains14717.txt', 'r') as file_to_read: 
    for i in file_to_read:
        lst_1.append(i.split()[0][:4])

for i in os.listdir(filepath):
    lst_2.append(i[:4])

for i in lst_1:
    if i not in lst_2:
        lst_3.append(i)

print(lst_3)