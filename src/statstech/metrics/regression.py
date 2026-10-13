from collections.abc import Callable

def RSS(x: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Residual Sum of Squares
    
    Computes the sum of the squared differences between the actual
    target values and the values predicted by the function f.
    
    Parameters
    ----------
    x: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in x.
    f: callable
        Function that takes an observation from x and returns a predicted value.
        
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
    if len(x) != len(y):
        raise ValueError("The lists of input observations and target values" \
        " must have the same length.") 
    if len(x) == 0:
        raise ValueError("The input lists must not be empty")
    errors = []
    n = len(x)
    for i in range(n):
        y_pred = f(x[i])
        y_true = y[i]
        sqrd_error = (y_pred - y_true)**2
        errors.append(sqrd_error)
    rss = sum(errors)
    return rss

def MSE(x: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Mean Squared Error (MSE).
    
    Computes the RSS and divides it by the number of observations,
    obtaining the avarege of the squared prediction errors.
    
    Parameters
    ----------
    x: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in x.
    f: callable
        Function that takes an observation from x and returns a predicted value.
    
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
    total_error = RSS(x, y, f)
    n = len(x)
    mse = total_error/n
    return mse
    
def MAE(x: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Mean Absolute Error (MAE).
    
    Computes the avarege of absolute diffences between the values,
    predicted by the function f and the actual target values.
    
    Parameters
    ----------
    x: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in x.
    f: callable
        Function that takes an observation from x and returns a predicted value.
    
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
    if len(x) != len(y):
        raise ValueError("The lists of input observations and target values" \
        " must have the same length.")
    if len(x) == 0:
        raise ValueError("The input lists must not be empty")
    errors = []
    n = len(x)
    for i in range(n):
        abs_error = abs(f(x[i]) - y[i])
        errors.append(abs_error)
    total = sum(errors)
    mae = total/n
    return mae

def RMSE(x: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """Calculates the Root Mean Absoltue Error (MAE).
    
    Computes the square root MSE, expressing the prediction error
    in the same unit of measurement as the actual predicted values.
    
    Parameters
    ----------
    x: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in x.
    f: callable
        Function that takes an observation from x and returns a predicted value.
    
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
    mse = MSE(x, y, f)
    rmse = mse**0.5
    return rmse

def R2_score(x: list[list[float]], y: list[float], f: Callable[[list[float]], float]) -> float:
    """
    Calculates the coefficient of determination (R²).
    
    Compares the model's residual sum of squares with the total sum
    of squares, which measures the variation of the actual target values
    around their mean. The result indicates how well the model performs 
    compared to using the mean as prediction.
    
    Parameters
    ----------
    x: list[list[float]]
        List of observations, where each observation contains the input
        features used by the prediction function.
    y: list[float]
        List of actual target values corresponding to the observations in x.
    f: callable
        Function that takes an observation from x and returns a predicted value.
    
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
    rss = RSS(x, y, f)
    n = len(x)
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