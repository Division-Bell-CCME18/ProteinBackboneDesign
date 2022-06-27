import numpy as np

list = []
cnt = 0

with open('1.txt','r') as file:
    while True:
            lines = file.readline()
            if not lines:
                break
            lines = lines.split()
            try:
                list.append(float(lines[0]))
            except:
                pass
            cnt += 1
            if (cnt % 50) == 0:
                # print(np.mean(list))
                print(np.mean(list), np.std(list, ddof=1))
                list = []
           



