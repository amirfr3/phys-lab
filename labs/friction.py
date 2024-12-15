import math
from pprint import pprint

import pandas as pd

import sympy
from sympy.parsing.latex import parse_latex

# from physpy.graph.graph import make_graph
# from physpy.graph.fit import fit_curve, extract_fit_param, linear, polynomial
# from physpy.equation.equation import (
#     get_uncertainties,
#     calculate_indirect_error_formula,
#     calculate_value_with_uncertainty,
# )
import physpy


DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Friction\results"
RESULTS_FOLDER = (
    r"C:\Users\flami\OneDrive\School\SemA\LabA\Friction\results\data_processing"
)


def main():
    sympy.init_printing()

    datasheet = f"{DATASHEET_FOLDER}\\friction.xlsx"
    should_show = True
    # calibrate = True

    # Part A - Calculate friction coefficient

    # Perform linear fit for all sets of measurements
    # Linear fit 
    v_0, v_f, g, x = sympy.symbols("v_0,v_f,g,x")
    mu_k_expr = sympy.Eq(sympy.symbols("mu_k"), -1 * (v_f ** 2 - v_0 ** 2) / 2*g*x)
    print(sympy.latex(mu_k_expr))
    delta_mu_k_expr = physpy.equation.calculate_indirect_error_formula(mu_k_expr)
    print(sympy.latex(delta_mu_k_expr))
    return

    mu_k_values_with_uncertainties = {
        "v_0": (1, 1),
        "v_f": (1, 1),
        "g": (1, 1),
        "x": (1, 1),
    }
    print(calculate_value_with_uncertainty(mu_k_values_with_uncertainties))

    # check out maabara? https://github.com/c2man/maabara
    # fit_data = make_graph(
    #     "מסת כדור כתלות ברדיוס כדור בשלישית",
    #     datasheet,
    #     "mass of radius",
    #     linear,
    #     (0, 33.5),
    #     show=should_show,
    # )


if __name__ == "__main__":
    main()
