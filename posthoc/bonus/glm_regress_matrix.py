import numpy as np

def glm_regress_matrix(input_mtx, regressor=None, polynomial_fit=0, censor=None):
    """
    Performs GLM regression for each column of a MxN input matrix (input_mtx) independently
    with a MxK regressor matrix (regressor). 

    Parameters
    ----------
    input_mtx : np.ndarray
        M x N matrix, where M is dimension of a sample, N is number of samples.

    regressor : np.ndarray, optional
        M x K matrix, where M is dimension of a sample, K is number of regressors.

    polynomial_fit : int, optional
        -1/0/1 (default is 0). If polynomial_fit is set to -1, no regressors are added.
        If set to 0, a Mx1 vector of ones is prepended to the regressor matrix. 
        If set to 1, a Mx1 vector of ones and a Mx1 vector from linspace(-1, 1, M) 
        is prepended to the regressor matrix.

    censor : np.ndarray, optional
        Mx1 vector with 0 and 1 (default is None), 1 means kept frames, 0 means removed frames. 
        If provided, censored frames will be excluded from the input and regressor matrices.

    Returns
    -------
    resid_mtx : np.ndarray
        M x N matrix of residuals after GLM regression.

    coef_mtx : np.ndarray
        Coefficient matrix after GLM regression.

    std_mtx : np.ndarray
        Standard deviation of the residual matrix.

    retrend_mtx : np.ndarray
        Matrix used to add back the linear trend, or empty if regressor is provided.
    """
    # Default for regressor if None
    if regressor is None:
        regressor = np.empty((input_mtx.shape[0], 0))

    # Check censor input
    if censor is not None:
        if censor.ndim != 1:
            raise ValueError("Input argument 'censor' should be a column vector")
    
    # Check if there are no regressors
    if (regressor.size == 0) and (polynomial_fit == -1):
        raise ValueError("ERROR: No regressor, quitting!")

    # Construct GLM regressors
    Y = input_mtx
    X = regressor

    if polynomial_fit == 1:
        X = np.column_stack((np.linspace(-1, 1, Y.shape[0]), X))
    if polynomial_fit in [0, 1]:
        X = np.column_stack((np.ones(Y.shape[0]), X))

    # Least squares GLM regression
    if censor is not None:
        # If censor vector is not empty
        censor_mask = censor.astype(bool)
        Y_censor = Y[censor_mask]
        X_censor = X[censor_mask]
        b = np.linalg.inv(X_censor.T @ X_censor) @ (X_censor.T @ Y_censor)
    else:
        # If censor vector is empty
        b = np.linalg.inv(X.T @ X) @ (X.T @ Y)

    # Output coefficient matrix, residual matrix, and retrend_matrix
    resid_mtx = Y - X @ b
    coef_mtx = b
    std_mtx = np.std(resid_mtx, axis=0)

    # Retrend matrix
    retrend_mtx = X @ b if regressor.size == 0 else np.empty((Y.shape[0], 0))

    return resid_mtx, coef_mtx, std_mtx, retrend_mtx