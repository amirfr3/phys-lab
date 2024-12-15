import math
import pandas as pd
from pprint import pprint

import sympy
from sympy.parsing.latex import parse_latex

from ..graph.graph import make_graph
from ..graph.fit import fit_curve, extract_fit_param, linear, polynomial
from ..equation.equation import (
    get_uncertainties,
    calculate_indirect_error_formula,
    calculate_value_with_uncertainty,
)


DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Friction\results"
RESULTS_FOLDER = (
    r"C:\Users\flami\OneDrive\School\SemA\LabA\Friction\results\data_processing"
)


def main():
    sympy.init_printing()

    datasheet = f"{DATASHEET_FOLDER}\\friction.xlsx"
    should_show = True
    # calibrate = True

    mu_k_symbols = sympy.symbols("v_0,v_f,g,x")
    mu_k_expr = sympy.parsing.latex.parse_latex(r"\mu_k = -\frac{v_f^2 - v_0^2}{2gx}")
    print(mu_k_expr)
    delta_mu_k_expr = calculate_indirect_error_formula(mu_k_expr)
    print(delta_mu_k_expr)

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
