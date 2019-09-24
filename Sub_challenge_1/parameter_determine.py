import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import scipy.stats as stats


def myrange():
    for n in 0.01, 0.1, 1, 10, 100, 1000:
        yield n


def custom_score_function(y_true, y_pred):
    for i in range(len(y_pred)-1):
        if y_pred[i] != y_pred[i+1]:
            a, b = stats.spearmanr(y_true, y_pred)
            return a
    return -0.9


X = np.load('F_scaled.npy')
Y = np.load('L_scaled.npy')
Z = np.load('Ft_scaled.npy')
a = []
b = []
for i in range(46):
    for j in myrange():
        kpca = KernelPCA(n_components=i+4, kernel="rbf", fit_inverse_transform=True, gamma=j)
        newX = kpca.fit_transform(X)
        newX = newX.astype('float64')
        Y = Y.astype('float64')
        for k in myrange():
            ridge = Ridge(alpha=k)
            scores = cross_val_score(ridge, newX, Y, cv=5, scoring='neg_mean_squared_error')
            scores = scores.mean()
            newX_train, newX_test, Y_train, Y_test = train_test_split(newX, Y, test_size=0.1, random_state=0)
            clf = ridge.fit(newX_train, Y_train)
            Y_predict1 = clf.predict(newX_test)
            Y_predict2 = clf.predict(newX_train)
            Y_test = np.array(Y_test).astype(float)
            Y_train = np.array(Y_train).astype(float)
            a.append(scores)
            b.append((i, j, k))
print(b[a.index(max(a))])