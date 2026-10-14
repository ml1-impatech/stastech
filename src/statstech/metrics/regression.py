import numpy as np
import numpy.typing as npt

def RSS(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calcula a Soma dos Quadrados dos Resíduos.

    Parameters
    ----------
    y_true: numpy.ndarray 
        Array com os valores reais do alvo.
    y_pred: numpy.ndarray
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    rss: float 
        Soma das diferenças ao quadrado entre os valores previstos e reais.

    Raises
    ------
    ValueError
        Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Os arrays reais e previstos devem ter o mesmo comprimento.") 
    if len(y_true) == 0:
        raise ValueError("Os arrays devem não estar vazios.")

    rss = float(np.sum((y_true - y_pred) ** 2))
    return rss

def MSE(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calcula o Erro Quadrático Médio.

    Parameters
    ----------
    y_true: numpy.ndarray 
        Array com os valores reais do alvo.
    y_pred: numpy.ndarray
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    rss: float 
        Média dos quadrados das diferenças entre os valores reais e previstos.

    Raises
    ------
    ValueError
        Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
    """    
    total_error = RSS(y_true, y_pred)
    n = len(y_true)
    mse = float(total_error/n)
    return mse
    
def MAE(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calcula o Erro Absoluto Médio.

    Parameters
    ----------
    y_true: numpy.ndarray 
        Array com os valores reais do alvo.
    y_pred: numpy.ndarray
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    mae: float 
        Média do valor absoluto das diferenças entre os valores reais e previstos.

    Raises
    ------
    ValueError
        Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
    """    
    if len(y_true) != len(y_pred):
        raise ValueError("Os arrays reais e previstos devem ter o mesmo comprimento.")
    if len(y_true) == 0:
        raise ValueError("Os arrays devem estar não vazios.")
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return mae

def RMSE(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calcula a raíz do Erro Quadrático Médio.

    Parameters
    ----------
    y_true: numpy.ndarray 
        Array com os valores reais do alvo.
    y_pred: numpy.ndarray
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    rss: float 
        A raíz quadrada da média dos quadrados das diferenças entre os valores reais e previstos.

    Raises
    ------
    ValueError
        Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
    """        
    mse = MSE(y_true, y_pred)
    rmse = mse**0.5
    return rmse

def R2_score(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    """Calcula o coeficiente de determinação (R²).

    Parameters
    ----------
    y_true: numpy.ndarray 
        Array com os valores reais do alvo.
    y_pred: numpy.ndarray
        Array com os valores previstos pelo modelo.
    
    Returns
    -------
    r2_score: float 
        Coeficiente de determinação, que indica o quanto da variabilidade
        dos valores reais é explicada pelas previsões do modelo.    
    
    Raises
    ------
    ValueError
        Se os arrays tiverem comprimentos diferentes ou estiverem vazios.
        Se todos os valores de y_true forem iguais, zerando o TSS e tornando R² indefinido.
    """            
    rss = RSS(y_true, y_pred)
    media = np.mean(y_true) 
    tss = np.sum((media-y_true)**2)
    if tss != 0:
        r2_score = float(1-rss/tss)
        return r2_score
    else:
        raise ValueError("R² é indefinido quando TSS = 0")

