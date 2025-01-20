import physpy
import physpy.utils
import math

DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Optics\results"
DATASHEET_PATH = DATASHEET_FOLDER + r"\optics.xlsx"
RESULTS_FOLDER = DATASHEET_FOLDER + r"\data_processing"


def main():
    physpy.sympy.init_printing()
    physpy.graph.single_picture_graphs(True)

    fit_tables, param_table = physpy.table.parse_data(DATASHEET_PATH)
    symbols_table = physpy.equation.get_symbols_table(param_table)

    # Part A - Calculate focal length of chosen lens
    symbols_table.x_s.set_expr(
        (symbols_table.x_s_max.sym + symbols_table.x_s_min.sym) / 2,
        delta_expr=(symbols_table.x_s_max.sym - symbols_table.x_s_min.sym)
        / physpy.np.sqrt(12),
    )
    symbols_table.x_s.calculate_value(symbols_table.values())
    print(symbols_table.x_s)

    # TODO: Why is this backwards actually?
    # symbols_table.v.set_expr(symbols_table.x_s.sym - symbols_table.x_l.sym)
    symbols_table.v.set_expr(symbols_table.x_l.sym - symbols_table.x_s.sym)
    symbols_table.v.calculate_value(symbols_table.values())
    print(symbols_table.v)

    symbols_table.u.set_expr(symbols_table.l_b.sym - symbols_table.l_c.sym)
    symbols_table.u.calculate_value(symbols_table.values())
    print(symbols_table.u)

    u = symbols_table.u.sym
    v = symbols_table.v.sym

    # Step 1 - Calculate according to assumption that u -> inf
    symbols_table.f1.set_expr(v)
    symbols_table.f1.calculate_value(symbols_table.values())
    print(symbols_table.f1)

    # Step 2 - Calculate according to real u value
    symbols_table.f2.set_expr(u * v / (u + v))
    symbols_table.f2.calculate_value(symbols_table.values())
    print(symbols_table.f2)

    # Step 3 - Compare!
    print(
        f"N_Sigma: {physpy.graph.nsigma((symbols_table.f1.val, symbols_table.f1.delta_val), (symbols_table.f2.val, symbols_table.f2.delta_val))}"
    )


if __name__ == "__main__":
    main()
