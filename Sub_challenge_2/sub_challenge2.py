from __future__ import division
import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
from math import sqrt
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle


#data reading
with open('SubCh2_TrainingData.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    A = []
    for i in csv_reader:    #read by rows(i)
        A.append(i)   #col 2 up to the last row
        AA = np.array(A)
print(AA.shape)
np.save('origin.npy', AA)   #[1044,4957]
with open('SubCh2_TestData.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    B = []
    for i in csv_reader:    #read by rows(i)
        B.append(i)   #col 2 up to the last row
    BB = np.array(B)
print(BB.shape)
np.save('Test.npy', BB)   #[289,4968]

Train = np.load('origin.npy')
Test = np.load('Test.npy')
a = Train[0, :][4:4956]    #(4952,) name of genes
b = Test[0, :][7:-1]     #(4960,) name of genes
c = []
for i in range(4952):
   if b[i+8] == a[i]:
       c.append(a[i])
c = np.array(c)
print(c.shape)   #(4952,)there are only 8 columns in test different from train
d = 0
if 'NA' in b:
    d = d+1
print(d)    #no NA in test




#mean value of each column
Train = np.load('origin.npy')
Train = Train[1:1044, 4:4956]
A = Train
a = []
b = []
c = []
d = []
for i in range(4952):
    for j in range(1043):
        if not 'NA' in A[:, i]:
            b = A[:, i]
        elif A[:, i][j] == 'NA':      #ith column, jth row
            c.append(j)
            b = np.delete(A[:, i], c)
    b = np.array(b)
    b = b.astype(np.float)
    #print(np.mean(b))
    d.append(np.mean(b))
d = np.array(d)
np.save('mean of columns.npy', d)   #[4952,]



#delete NA from train
A = np.load('origin.npy')
A = A[1:1044, 4:4956]
B = np.load('mean of columns.npy')
for i in range(4952):
    for j in range(1043):
        if A[:, i][j] == 'NA':
            A[:, i][j] = B[i]
print(A.shape)
np.save('deleteNA.npy', A)




#correlation
Train = np.load('origin.npy')
Test = np.load('Test.npy')
A = np.load('deleteNA.npy')
A = np.column_stack((Train[1:1044, 2], A))     #(1043, 4953)
B = np.load('origin.npy')[1:1044, -1]      #(1043,)
C = np.column_stack((Test[1:289, 2], Test[1:289, 15:4967]))    #(288, 4953)
na = []
for i in range(1043):
    if B[i] == 'NA':
        na.append(i)
A = np.delete(A, na, axis=0)   #(1034, 4953)
B = np.delete(B, na, axis=0)   #(1034,)
D = np.row_stack((A, C))       #(1322, 4953)
D = preprocessing.scale(D)
np.save('Train+Test_F_scaled.npy', D)

A = np.load('Train+Test_F_scaled.npy')
a = []
for i in range(1322):
    for j in range(1322):
        rho, pval = stats.spearmanr(A[i, :], A[j, :])
        a.append(rho)
a = np.array(a)
a = np.reshape(a, (1322, 1322))
print(a.shape)
np.save('Feature.npy', a)
np.save('Label.npy', B)

def cross_val(X, Y, iter):
    batch_size = int(len(X)/iter)
    best_prc = 0
    best_c = 0
    # print(X.shape,Y.shape)
    c_list = [0.0001,0.001,0.01,0.1,1,5,10,15,100,1000]
    # c_list = [10]
    for c in c_list:
        score = 0
        for i in range(iter):
            # Y_predict = []
            # P_predict = []
            colums = list(range(0,batch_size*i)) + list(range(batch_size*(i+1),len(X)))
            # row = list(range(0,batch_size*i)) + list(range(batch_size*(i+1),len(X)))
            X_test = X[batch_size*i:batch_size*(i+1),colums]
            Y_test = Y[batch_size*i:batch_size*(i+1)]
            # print(X.shape, Y_test)
            X_train = X[colums]
            X_train = X_train[:,colums]
            Y_train = Y[colums]
            # print(X_train.shape,Y_train.shape, X_test.shape,Y_test.shape)

            clf = svm.SVC(gamma='auto',kernel='precomputed', probability=True, C=c, max_iter=1000000)
            clf.fit(X_train, Y_train)
            # Y_predict.append(y[0])
            X_predict = clf.predict(X_test)
            P_predict = clf.predict_proba(X_test)
            # P_predict.append(p[0,1])
            # P_predict = np.asarray(P_predict)
            AUPRC = average_precision_score(Y_test, P_predict[:,1])
            # print(X_predict)
            score += AUPRC
        score = score / iter
        if score > best_prc:
            best_c = c
            best_prc = score
    return best_prc, best_c



F = np.load('Feature.npy')
X = F[:1034, :1034]   #(1034, 1322)
XX = F[1034:, :1034]    #test set

L = np.load('Label.npy')
Y = []
for i in range(1034):
    if L[i] == 'Fast':
        Y.append(1)
    else:
        Y.append(0)
Y = np.array(Y)


# X, Y = shuffle(X, Y)
print(X.shape,Y.shape)
best_prc, best_c = cross_val(X, Y, 5)
print(best_prc, best_c)

clf = svm.SVC(gamma='auto',kernel='precomputed', probability=True, C=best_c, max_iter=1000000)
clf.fit(X, Y)
# Y_predict.append(y[0])
XX_predict = clf.predict(XX)
PP_predict = clf.predict_proba(XX)[:,1]
# print(XX_predict)
result_list = [["Isolate","Predicted_Categorical_Clearance","Probability"]]
with open('SubCh2_TestData.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)


    name = next(csv_reader)[0].split('.')
    isolate = name[0]
    hours = name[1]
    treat = name[2]
    i = 0
    current = isolate
    test_class = []
    test_proba = []

    count = 0
    sum = 0
    fast_count = 0
    if treat == "UT" and hours == '24HR':
    # if 1:
        count = 1
        sum = PP_predict[i]
        if XX_predict[i] == 1:
            fast_count = 1
        else:
            fast_count = 0
    for line in csv_reader:
        name = line[0].split('.')
        isolate = name[0]
        treat = name[2]
        i += 1
        if current != isolate:
            temp = [current]
            if sum/count > 0.5:
                test_class.append(1)
                temp.append('FAST')
                test_proba.append(sum/count)
                temp.append(str("{0:.2f}".format(sum/count)))
            else:
                test_class.append(0)
                temp.append('SLOW')
                test_proba.append(2 * sum/count)
                temp.append(str("{0:.2f}".format(sum/count)))
            result_list.append(temp)

            current = isolate
            if treat == "UT" and hours == '24HR':
            # if 1:
                count = 1
                sum = PP_predict[i]
                if XX_predict[i] == 1:
                    fast_count = 1
                else:
                    fast_count = 0
            else:
                count = 0
                sum = 0
                fast_count = 0

        else:
            if treat == "UT" and hours == '24HR':
            # if 1:
                count += 1
                sum += PP_predict[i]
                if XX_predict[i] == 1:
                    fast_count += 1

temp = [current]
if sum/count > 0.5:
    test_class.append(1)
    temp.append('FAST')
    test_proba.append(2*(sum/count - 0.5))
    temp.append(str("{0:.2f}".format(sum/count)))
else:
    test_class.append(0)
    temp.append('SLOW')
    test_proba.append(2 * sum/count)
    temp.append(str("{0:.2f}".format(sum/count)))
result_list.append(temp)

print(test_class)
print(test_proba)
print(result_list)

w = open("SubCh2_Submission_final.txt",'w+')
for line in result_list:
    w.write('\t'.join(x for x in line) + '\n')

w.close()