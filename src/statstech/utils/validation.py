import numpy as np

class NotFittedError(ValueError, AttributeError):
    """ Erro para quando o modelo é usado antes do fit """

def _is_numeric(a):
    """ Diz se o array é numérico (inteiro, float ou bool) """
    return np.issubdtype(a.dtype, np.number) or np.issubdtype(
        a.dtype, np.bool_)

def _check_finite(a, name, allow_nan=False):
    """ Erro se um array tiver NaN ou infinito."""
    if allow_nan:
        if np.isinf(a).any():
            raise ValueError(f"{name} contém valores infinitos.")
    elif not np.isfinite(a).all():
        raise ValueError(f"{name} contém NaN ou valores infinitos.")

def check_array(X, allow_nan=False, ensure_numeric=True):
    ''' Verifica de X é um array 2D e converte para array se for lista'''
    X = np.asarray(X)
    n = X.ndim
    if n == 1:
        raise ValueError('Esse array é 1D, precisa ser 2D. Sugestão: '
        'usar X.reshape(-1, 1) se for uma feature só')
    if n == 0 or n>2:
        raise ValueError(f'O array não é 2D, tem {X.ndim} dimensões')
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f'O array está vazio, formato {X.shape}')
    if _is_numeric(X):
        _check_finite(X, 'X', allow_nan)
    elif ensure_numeric:
        raise ValueError(f'X tem dtype {X.dtype}, precisa ser numérico')
    return X

def check_X_y(X, y):
    ''' 
    Verifica se X é um array 2D, se y é um array 1D e se têm o mesmo tamanho.
    Converte para array se for lista 
    '''
    X = check_array(X)
    y = np.asarray(y)
    n = y.ndim
    if n == 2 and y.shape[1] == 1:
        raise ValueError("y tem formato (n, 1), mas precisa ser 1D. Sugestão: use y.ravel()")
    if n != 1:
        raise ValueError(f'y deve ser 1D, mas tem {y.ndim} dimensões')
    if X.shape[0] != y.shape[0]:
        raise ValueError(f'X e y tem tamanhos diferentes: {X.shape[0]} e {y.shape[0]}')
    if _is_numeric:
        _check_finite(y, 'y')
    elif y.dtype == object and any(v is none or (isinstance(v, float) and np.isnan(v) for v in y)):
        raise ValueError('y contém Nan ou None')
    return X, y

def check_is_fitted(model, attributes=None):
    ''' Verifica se o modelo foi treinado'''
    name = type(model).__name__
    if attributes is None:
        fitted = [a for a in vars(model)
            if a.endswith("_") and not a.startswith("__")]
        if not fitted:
            raise NotFittedError(f'{name} ainda não foi treinado')
    if isinstance(attributes, str):
        attributes = [attributes]
    missing = [a for a in attributes if not hasattr(model, a)]
    if missing:
        raise NotFittedError(f"{name} ainda não foi treinado, faltam {missing} ")