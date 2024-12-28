import math
from pprint import pprint

import pandas as pd

import sympy
# from sympy.parsing.latex import parse_latex

import physpy
import sympy.printing


DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Friction\results"
RESULTS_FOLDER = (
    r"C:\Users\flami\OneDrive\School\SemA\LabA\Friction\results\data_processing"
)


def main():
    sympy.init_printing()

    # print(physpy.equation.calculate_indirect_error_formula(sympy.Eq(sympy.symbols("v"), sympy.symbols("v_0")**2 - sympy.symbols("v_f")**2)))

    datasheet = f"{DATASHEET_FOLDER}\\friction.xlsx"
    should_show = True
    # calibrate = True

    # Part A - Calculate friction coefficient

    # Perform linear fit for all sets of measurements
    # Linear fit velocity
    # fit_data = physpy.graph.make_graph(
    #     "מיקום עגלה כתלות בזמן - שער ראשון",
    #     datasheet,
    #     "part a velocity",
    #     physpy.graph.fit.linear,
    #     (0, 0),
    #     show=should_show,
    # )
    # v_0_val = physpy.graph.extract_fit_param(fit_data, 1)

    # fit_data = physpy.graph.make_graph(
    #     "מיקום עגלה כתלות בזמן - שער שני",
    #     datasheet,
    #     "part a final velocity",
    #     physpy.graph.fit.linear,
    #     (0, 0),
    #     show=should_show,
    # )
    # v_f_val = physpy.graph.extract_fit_param(fit_data, 1)

    # Calculate friction coefficient
    v_0, v_f, g, x = sympy.symbols("v_0,v_f,g,x")
    mu_k_expr = sympy.Eq(sympy.symbols("mu_k"), -1 * (v_f ** 2 - v_0 ** 2) / 2*g*x)
    delta_mu_k_expr = physpy.equation.calculate_indirect_error_formula(mu_k_expr)
    print(f"{physpy.equation.latexify(mu_k_expr)}, {physpy.equation.latexify(delta_mu_k_expr)}")

    # Part B - Check energy conservation
    g_theo, dg_theo = 9.81, 0.1

    h, M, m, g, v = sympy.symbols("h M m g v")
    dM, dm = physpy.equation.get_uncertainties([M, m])
    fit_p = sympy.symbols("a")

    # fit_p_expr = ((M + m) * v**2) / (2 * m * g)
    fit_p_expr = (1/(2*g)) * (1 + M/m) * v**2
    g_expr = sympy.Eq(g, ((fit_p_expr * g) / fit_p) / v**2)
    delta_g_expr = physpy.equation.calculate_indirect_error_formula(g_expr)
    print(f"{physpy.equation.latexify(g_expr)}, {physpy.equation.latexify(delta_g_expr)}")

    m_subs = {
        m: 0.12359,
        M: 0.530785,
        dm: 6.45497224368061E-06,
        dM: 0.0000122474487139237,
    }
    g_expr = g_expr.subs(m_subs)
    delta_g_expr = delta_g_expr.subs(m_subs)

    vh_table = pd.read_excel(datasheet, "part b velocity")

    # Parabolic fit velocity
    fit_data = physpy.graph.make_graph(
        "מהירות עגלה כתלות במהירות - התאמה פרבולית",
        vh_table,
        0,
        physpy.graph.fit.polynomial,
        (0, 0, 0),
        show=should_show,
    )
    a_g_p, da_g_p = physpy.graph.extract_fit_param(fit_data, 2)[:2]
    a_g_p_subs = {
        fit_p: a_g_p,
        physpy.equation.get_uncertainties(fit_p): da_g_p,
    }

    g_p = g_expr.rhs.subs(a_g_p_subs)
    dg_p = delta_g_expr.rhs.subs(a_g_p_subs)
    print(f"Polynomial: Calculated g={g_p}+-{dg_p} ({(dg_p / g_p) * 100}%) (N_sigma={physpy.graph.nsigma((g_theo, dg_theo), (g_p, dg_p))})")

    v2 = pd.Series(
        [v**2 for v in vh_table["Velocity [m/sec]"]],
        name="Velocity^2 [m^2/sec^2]"
    )
    dv2 = pd.Series(
        [2*v*dv for v, dv in zip(vh_table["Velocity [m/sec]"], vh_table["delta_Velocity [m/sec]"])],
        name="delta_Velocity^2 [m^2/sec^2]"
    )
    v2h_table = pd.concat(
        [v2, dv2, vh_table["Height [m]"], vh_table["delta_Height [m]"]],
        axis=1
    )

    # Perform linear fit for measurements
    # Linear fit velocity
    fit_data = physpy.graph.make_graph(
        "מהירות עגלה כתלות במהירות - התאמה לינארית",
        v2h_table,
        0,
        physpy.graph.fit.linear,
        (0, 0),
        show=should_show,
    )
    a_g_l, da_g_l = physpy.graph.extract_fit_param(fit_data, 1)[:2]
    a_g_l_subs = {
        fit_p: a_g_l,
        physpy.equation.get_uncertainties(fit_p): da_g_l,
    }

    g_l = g_expr.rhs.subs(a_g_l_subs)
    dg_l = delta_g_expr.rhs.subs(a_g_l_subs)
    print(f"Linear: Calculated g={g_l}+-{dg_l} ({(dg_l / g_l) * 100}%) (N_sigma={physpy.graph.nsigma((g_theo, dg_theo), (g_l, dg_l))})")


if __name__ == "__main__":
    main()
