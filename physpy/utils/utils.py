import numpy as np


def get_value_error(value, error):
    return f"{value} ± {error} ({error / value * 100}%)"


def calc_inst_error(res):
    return res/np.sqrt(12)
