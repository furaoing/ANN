import random
data=list()
sample_num=100

for i in range(sample_num):
        data.append([0.9])

target=list()
for i in range(sample_num):
        if data[i][0]>0.4*1:
                target.append(1)
        else:
                target.append(0)
