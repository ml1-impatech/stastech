from collections.abc import Callable

def RSS(X: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Residual Sum of Squares
    
    Computes the sum of the squared differences between the actual
    target values and the values predicted by the function f.
    
    Parameters
    ----------
    X: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in X.
    f: callable
        Function that takes an observation from X and returns a predicted value.
        
    Returns
    -------
    rss: float 
        Sum of the squared differences between predicted and actual values.
    
    Raises
    ------
    ValueError
        If the lists of input observations and target values does not have
        same lenth or are empty.
    """
    if len(X) != len(y):
        raise ValueError("The lists of input observations and target values" \
        " must have the same length.") 
    if len(X) == 0:
        raise ValueError("The input lists must not be empty")
    errors = []
    n = len(X)
    for i in range(n):
        y_pred = f(X[i])
        y_true = y[i]
        sqrd_error = (y_pred - y_true)**2
        errors.append(sqrd_error)
    rss = sum(errors)
    return rss

def MSE(X: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Mean Squared Error (MSE).
    
    Computes the RSS and divides it by the number of observations,
    obtaining the avarege of the squared prediction errors.
    
    Parameters
    ----------
    X: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in X.
    f: callable
        Function that takes an observation from X and returns a predicted value.
    
    Returns
    -------
    mse: float 
        Avarege of the squared differences between predicted and actual values.
    
    Raises
    ------
    ValueError
        If the lists of input observations and target values does not have
        same lenth or are empty.
    """
    total_error = RSS(X, y, f)
    n = len(X)
    mse = total_error/n
    return mse
    
def MAE(X: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Mean Absolute Error (MAE).
    
    Computes the avarege of absolute diffences between the values,
    predicted by the function f and the actual target values.
    
    Parameters
    ----------
    X: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in X.
    f: callable
        Function that takes an observation from X and returns a predicted value.
    
    Returns
    -------
    mae: float 
        Avarege of the absolute prediction errors.
    
    Raises
    ------
    ValueError
        If the lists of input observations and target values does not have
        same lenth or are empty.
    """
    if len(X) != len(y):
        raise ValueError("The lists of input observations and target values" \
        " must have the same length.")
    if len(X) == 0:
        raise ValueError("The input lists must not be empty")
    errors = []
    n = len(X)
    for i in range(n):
        abs_error = abs(f(X[i]) - y[i])
        errors.append(abs_error)
    total = sum(errors)
    mae = total/n
    return mae

def RMSE(X: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Root Mean Absoltue Error (MAE).
    
    Computes the square root MSE, expressing the prediction error
    in the same unit of measurement as the actual predicted values.
    
    Parameters
    ----------
    X: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in X.
    f: callable
        Function that takes an observation from X and returns a predicted value.
    
    Returns
    -------
    rmse: float 
        Square root of the mean squared prediction errors.
    
    Raises
    ------
    ValueError
        If the lists of input observations and target values does not have
        same lenth or are empty.
    """
    mse = MSE(X, y, f)
    rmse = mse**0.5
    return rmse

def R2_score(X: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """
    Calculates the coefficient of determination (R²).
    
    Compares the model's residual sum of squares with the total sum
    of squares, which measures the variation of the actual target values
    around their mean. The result indicates how well the model performs 
    compared to using the mean as prediction.
    
    Parameters
    ----------
    X: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in X.
    f: callable
        Function that takes an observation from X and returns a predicted value.
    
    Returns
    -------
    r2_score: float 
        Coefficient of determination R². A value 1 indicates a perfect fit,
        a value 0 indicates performance equivalent to predicting the mean,
        and a negative value indicates worse performance than predicting the mean.

    Raises
    ------
    ValueError
        If all actual target values are identical, making TSS equal to zero
        and R² undefined.
        If the lists of input observations and target values does not have
        same lenth or are empty.
    """
    rss = RSS(X, y, f)
    n = len(X)
    media = sum(y)/n 
    errors = []
    for i in range(n):
        y_pred = media
        y_true = y[i]
        sqrd_error = (y_true - y_pred)**2
        errors.append(sqrd_error)
    tss = sum(errors)
    if tss != 0:
        r2_score = 1-rss/tss
        return r2_score
    else:
        raise ValueError("R² is undefined when TSS = 0")

