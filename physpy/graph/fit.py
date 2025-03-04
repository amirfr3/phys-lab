import numpy as np
import pandas as pd
from scipy.odr import Model, ODR, RealData

from .stats import calc_stats, format_output

# === Types of fits - add custom fits as necessary === #


def function(A, x):
    pass  # Define your function. See some examples below.


def linear(A, x):
    return A[1] * x + A[0]


def _linear_inverse(A, y):
    return y/A[1] - A[0]/A[1]


def polynomial(A, x):
    return A[2] * x**2 + A[1] * x + A[0]


def optics(A, x):
    return A[1] * x / (x - A[1]) + A[0]


def exponential(A, x):
    return A[2] * np.exp(A[1] * x) + A[0]


def sinusoidal(A, x):
    return A[3] * np.sin(A[1] * x + A[2]) + A[0]


INVERSE_FUNCTION = {linear: _linear_inverse}


# === Fit function === #


def odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y):
    model = Model(fit_func)
    odr_data = RealData(x, y, sx=delta_x, sy=delta_y)
    odr = ODR(data=odr_data, model=model, beta0=initial_guesses)
    output = odr.run()

    fit_params = output.beta
    fit_params_error = output.sd_beta
    fit_cov = output.cov_beta
    return fit_params, fit_params_error, fit_cov, output


def find_outliers(residuals, errors):
    outliers = filter(
        lambda x: x[1] > 3,
        [
            (i, abs(residual / (error * 2)))
            for i, residual, error in zip(residuals.index, residuals, errors)
        ],
    )

    return sorted(outliers, key=lambda x: x[1])


def get_columns(data, columns):
    """Columns should be |X|delta_X|Y|delta_Y|"""
    x = data[columns[0]]
    delta_x = data[columns[1]]
    y = data[columns[2]]
    delta_y = data[columns[3]]

    return x, delta_x, y, delta_y


def fit_curve(
    fit_func, initial_guesses, table_or_file_path, sheet_idx=None, columns=(0, 1, 2, 3)
):
    try:
        if sheet_idx is None:
            raise ValueError("Please specify sheet name or index")
        data = pd.read_excel(table_or_file_path, sheet_name=sheet_idx)
    except ValueError:
        data = table_or_file_path

    # Change column integers to column names.
    columns = [data.columns.values[c] if isinstance(c, int) else c for c in columns]
    x, delta_x, y, delta_y = get_columns(data, columns)

    fit_params, fit_params_error, fit_cov, output = odr_fit(
        fit_func, initial_guesses, x, delta_x, y, delta_y
    )

    residuals, degrees_of_freedom, chi2_red, p_val, x_residuals = calc_stats(
        x, y, fit_func, fit_params, output
    )

    fit_results_str = format_output(
        fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom
    )
    # output.pprint() # More verbose

    processed_data = {
        "data": data,
        "columns": columns,
        "x": x,
        "y": y,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "fit_func": fit_func,
        "fit_params": fit_params,
        "fit_params_error": fit_params_error,
        "fit_cov": fit_cov,
        "output": output,
        "residuals": residuals,
        "x_residuals": x_residuals,
        "dof": degrees_of_freedom,
        "chi2_red": chi2_red,
        "p_val": p_val,
        "fit_results": fit_results_str,
        "outliers": find_outliers(residuals, delta_y),
    }

    return processed_data


def extract_fit_param(fit_data, param_idx, ext_func=None, ext_err_func=None):
    fit_param = fit_data["fit_params"][param_idx]
    fit_param_err = fit_data["fit_params_error"][param_idx]
    rel_param_err_percent = (abs(fit_param_err / fit_param)) * 100
    return (
        fit_param,
        fit_param_err,
        rel_param_err_percent,
        np.sqrt(2 / fit_data["dof"]),
        fit_data["p_val"],
    )
