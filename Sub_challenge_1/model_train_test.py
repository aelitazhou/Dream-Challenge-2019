import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import Ridge
import scipy.stats as stats

X = np.load('F_scaled.npy')
Y = np.load('L_scaled.npy')
Z = np.load('Ft_scaled.npy')

kpca = KernelPCA(n_components=49, kernel="rbf", fit_inverse_transform=True, gamma=0.01)
newX = kpca.fit_transform(X)
newX = newX.astype('float64')
newZ = kpca.fit_transform(Z)
newZ = newZ.astype('float64')
Y = Y.astype('float64')

ridge = Ridge(alpha=0.1)
newX_train, newX_test, Y_train, Y_test = train_test_split(newX, Y, test_size=0.1, random_state=0)
clf = ridge.fit(newX_train, Y_train)

Z_predict = clf.predict(newZ)
mean = 1.5703147058823528
W = mean*np.ones(50)
ZZ_predict = Z_predict + W

Test_pred = []
for i in range(0, 50, 2):
    Test_pred.append((ZZ_predict[i] + ZZ_predict[i+1])/2)
    T_pred = np.array(Test_pred)
print(T_pred)
