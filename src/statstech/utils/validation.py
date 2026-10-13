import numpy as np

def check_array(X):
    X = np.asarray(X)
    n = X.ndim
    if n == 1:
        raise ValueError('Esse array é 1D, precisa ser 2D. Sugestão: '
        'usar X.reshape(-1, 1) se for uma feature só')
    if n == 0 or n>2:
        raise ValueError('O array não é 2D')
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError('O array está vazio')
    if not np.isfinite(X).all():
        raise ValueError('O array contém Nan ou algum valor infinito')
    return X

def check_X_y(X, y):
    X = check_array(X)
    y = np.asarray(y)
    n = y.ndim
    if n != 1:
        raise ValueError('y não é 1D')
    if X.shape[0] != y.shape[0]:
        raise ValueError('X e y tem tamanhos diferentes')
    if not np.isfinite(y).all():
        raise ValueError('y contém Nan ou algum valor infinito')
    return X, y

def check_is_fitted(X):
    pass
