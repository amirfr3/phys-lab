import math
import scipy
import numpy as np


def calc_stats(x, y, fit_func, fit_params, output):
    residuals = y - fit_func(fit_params, x)
    degrees_of_freedom = len(x) - len(fit_params)
    chi2 = output.sum_square
    chi2_red = chi2 / degrees_of_freedom
    p_val = scipy.stats.chi2.sf(chi2, degrees_of_freedom)
    return residuals, degrees_of_freedom, chi2_red, p_val


def format_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    output = []
    for i in range(len(fit_params)):
        output.append(
            f"a[{i}]: {fit_params[i]} \u00B1 {fit_params_error[i]} ({(abs(fit_params_error[i]/fit_params[i]))*100}% error)"
        )
    output.append(
        f"chi squared reduced = {chi2_red:.5f} \u00B1 {np.sqrt(2/degrees_of_freedom)}"
    )
    output.append(f"p-probability = {p_val:.5e}")
    output.append(f"DOF = {degrees_of_freedom}")

    return "\n".join(output)


def nsigma(v1, v2):
    """
    v1, v2 are tuples of the form (value, uncertainty)
    """
    return abs(v1[0] - v2[0]) / math.sqrt(v1[1] ** 2 + v2[1]**2)

