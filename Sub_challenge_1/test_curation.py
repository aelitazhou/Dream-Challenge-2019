import numpy as np
from sklearn import preprocessing
from math import sqrt
import csv
import matplotlib.pyplot as plt

with open('SubCh1_TestData.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    A = []
    for i in csv_reader:
        A.append(i[2:5546])
        AA = np.array(A)
        AAA = AA[1:201, :]
C1 = '24HR'
C2 = '6HR'
C3 = 'DHA'
C4 = 'UT'
A1 = []
A2 = []
A4 = []
A5 = []
for i in range(len(AAA)):
    if AAA[i][0] == C1:
        A1.append(AAA[i])
        AA1 = np.array(A1)
for j in range(len(AAA)):
    if AAA[j][0] == C2:
        A2.append(AAA[j])
        AA2 = np.array(A2)
print(AA1[:, 3:5544].shape, AA2[:, 0:5543].shape)
A3 = np.column_stack((AA2[:, 0:5543], AA1[:, 3:5544]))
print(A3.shape)
label = []
for k in range(len(A3)):
    if A3[k][1] == C3:
        A4.append(A3[k])
        AA4 = np.array(A4)
    if A3[k][1] == C4:
        A5.append(A3[k])
        AA5 = np.array(A5)
F4 = AA4[:, 3:11083].astype(np.float)
F5 = AA5[:, 3:11083].astype(np.float)
print(AA4.shape, AA5.shape)
F = F4 - F5
F_scaled = preprocessing.scale(F)
np.save('Ft_scaled.npy', F_scaled)
