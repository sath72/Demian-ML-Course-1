import copy, math
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# data is stored in numpy array/matrix
print(f"X Shape: {X_train.shape}, X Type:{type(X_train)})")
print(X_train)
print(f"y Shape: {y_train.shape}, y Type:{type(y_train)})")
print(y_train)

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


def predict_single_loop(x, w, b):
    """
    single predict using linear regression

    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):  model parameter

    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(0,n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict_single_loop(x_vec, w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")


def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters
      b (scalar):             model parameter

    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

def compute_cost(X, y, w, b):

    m=X.shape[0]
    cost=0.0
    for i in range(0,m):
        f_wb_i=np.dot(X[i],w)+b
        cost=cost+(f_wb_i-y[i])**2
    cost=cost/(2/m)
    return cost

cost=compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

def compute_gradient(X, y, w, b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0

    for i in range(0,m):
        err=(np.dot(X[i],w)+b)-y[i]
        for j in range(0,n):
            dj_dw[j]=dj_dw[j]+err*X[i,j]
        dj_db=dj_db+err
        dj_dw=dj_dw/m
        dj_db=dj_db/m

    return dj_db, dj_dw

tmp_dj_db, tmp_dj_dw=compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b : {tmp_dj_db}')
print(f'dj_dw at initial w, b : {tmp_dj_dw}')

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history=[]
    w=copy.deepcopy(w_in)
    b=b_in

    for i in range(0, num_iters):
        dj_db, dj_dw=gradient_function(X, y, w, b)

        w=w-alpha*dj_dw
        b=b-alpha*dj_db

        if i<10000:
            J_history.append(cost_function(X,y,w,b))

        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")

    return w, b, J_history

initial_w=np.zeros_like(w_init)
initial_b=0

iterations=1000
alpha=5.0e-7

w_final, b_final, J_hist=gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha,iterations)
print(f"b, w found by gradient descent: {b_final:0.2f}, {w_final}")
m,_=X_train.shape
for i in range(0,m):
    print(f"Prediction : {np.dot(X_train[i],w_final)+b_final}, Target value : {y_train[i]}")

