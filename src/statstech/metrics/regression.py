import numpy as np
import numpy.typing as npt

def _validate_regression_inputs(
        y_true: npt.ArrayLike, 
        y_pred: npt.ArrayLike,
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Valida e converte os arrays de valores reais e previstos.
    
    Parameters
    ----------
    y_true: array-like
        Array com os valores reais do alvo.
    y_pred: array-like
        Array com os valores previstos pelo modelo.


    Returns
    -------
    y_true: numpy.ndarray
        Array unidimensional validado, com os valores float64 reais. 
    y_pred: numpy.ndarray
        Array unidimensional validado, com os valores float64 previstos. 

    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError("Os arrays devem ter o mesmo formato.")

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Os arrays devem ser unidimensionais.")

    if y_pred.size == 0:
        raise ValueError("Os arrays devem não estar vazios.")

    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("Os arrays não podem conter NaN ou infinito.")

    return y_true, y_pred


def residual_sum_squares(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula a Soma dos Quadrados dos Resíduos.

    Parameters
    ----------
    y_true: array-like 
        Array com os valores reais do alvo.
    y_pred: array-like 
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    rss: float 
        Soma das diferenças ao quadrado entre os valores previstos e reais.

    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    rss = float(np.sum((y_true - y_pred) ** 2))
    return rss

def mean_squared_error(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula o Erro Quadrático Médio.

    Parameters
    ----------
    y_true: array-like 
        Array com os valores reais do alvo.
    y_pred: array-like 
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    mse: float 
        Média dos quadrados das diferenças entre os valores reais e previstos.

    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """    
    total_error = residual_sum_squares(y_true, y_pred)
    n = len(y_true)
    mse = float(total_error/n)
    return mse
    
def mean_absolute_error(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula o Erro Absoluto Médio.

    Parameters
    ----------
    y_true: array-like 
        Array com os valores reais do alvo.
    y_pred: array-like 
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    mae: float 
        Média do valor absoluto das diferenças entre os valores reais e previstos.

    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """    
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return mae

def root_mean_squared_error(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula a raíz do Erro Quadrático Médio.

    Parameters
    ----------
    y_true: array-like 
        Array com os valores reais do alvo.
    y_pred: array-like 
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    rmse: float 
        A raíz quadrada da média dos quadrados das diferenças entre os valores reais e previstos.

    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """        
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse**0.5
    return rmse

def r2_score(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Calcula o coeficiente de determinação (R²).

    Parameters
    ----------
    y_true: array-like 
        Array com os valores reais do alvo.
    y_pred: array-like 
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    r2_score: float 
        Coeficiente de determinação, que indica o quanto da variabilidade
        dos valores reais é explicada pelas previsões do modelo.
        Se todos os valores de y_true forem iguais, zerando o TSS, o valor 1 é retornado.  
    
    Raises
    ------
    ValueError
        Se os arrays tiverem formatos diferentes.
        Se os arrays não forem unidimensionais.
        Se os arrays estiverem vazios.
        Se os arrays contiverem NaN ou infinito.
    """
    y_true, y_pred = _validate_regression_inputs(y_true, y_pred)            
    rss = residual_sum_squares(y_true, y_pred)
    media = np.mean(y_true) 
    tss = np.sum((media-y_true)**2)
    if tss != 0:
        r2_score = float(1-rss/tss)
        return r2_score
    if tss == 0:
        return 1.0

