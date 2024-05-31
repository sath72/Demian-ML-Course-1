import numpy as np
import matplotlib.pyplot as plt
from libs.ML_Course_libs.lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2) # reduced display precision on numpy arrays

'''
x=np.arange(0,20,1)
y=1+x**2
X=x.reshape(-1, 1)
'''
'''
x=np.arange(0,20,1)
y=1+x**2
X=x**2 #added engineered feature
X=X.reshape(-1, 1)
'''

'''
x=np.arange(0,20,1)
y=1+x**2
X=np.c_[x,x**2, x**3]   #added engineered feature
#X=X.reshape(-1, 1)


print(X) #confirm the initial data creation

model_w, model_b=run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)

plt.scatter(x,y, marker='x', c='r', label="Actual Value");plt.title("no feature engineering")
plt.plot(x,X@model_w+model_b, label="Predicted Value");plt.xlabel("X");plt.ylabel("y")
plt.legend();plt.show()
'''

'''
#create target data
x=np.arange(1,20,1)
y=x**2

#engineering features
X=np.c_[x, x**2, x**3]
X_feature=['x','x^2','x^3']

fig,ax=plt.subplots(1,3, figsize=(12,3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i],y)
    ax[i].set_xlabel(X_feature[i])
ax[0].set_ylabel("y")
plt.show()
'''
'''
x=np.arange(1,20,1)
y=x**2

X=np.c_[x, x**2, x**3] #feature engineering
X=zscore_normalize_features(X) #zscore normalizing

model_w, model_b=run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-1)

plt.scatter(x,y, marker='x', c='r', label="Actual Value");plt.title("Normalizaed x, X^2, X^3 feature")
plt.plot(x,X@model_w+model_b, label="Predicted Value");plt.xlabel("X");plt.ylabel("y")
plt.legend();plt.show()
'''

#Complex Function
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X)

model_w,model_b = run_gradient_descent_feng(X, y, iterations=1000000, alpha = 1e-1)

plt.scatter(x, y, marker='x', c='r', label="Actual Value"); plt.title("Normalized x x**2, x**3 feature")
plt.plot(x,X@model_w + model_b, label="Predicted Value"); plt.xlabel("x"); plt.ylabel("y"); plt.legend(); plt.show()



