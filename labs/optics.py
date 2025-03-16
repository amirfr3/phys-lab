import physpy
import physpy.utils

DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Optics\results"
DATASHEET_PATH = DATASHEET_FOLDER + r"\optics.xlsx"
RESULTS_FOLDER = DATASHEET_FOLDER + r"\data_processing"


def calculate_uv_tables(table):
    symbols = physpy.equation.get_symbols_table(
        symbols_str="X_O X_L X_S u v h_img H_obj"
    )
    symbols_dict = {
        symbols.x_o.sym: "X_O [m]",
        symbols.x_o.delta_sym: "delta_X_O [m]",
        symbols.x_l.sym: "X_L [m]",
        symbols.x_l.delta_sym: "delta_X_L [m]",
        symbols.x_s.sym: "X_S [m]",
        symbols.x_s.delta_sym: "delta_X_S [m]",
        symbols.h_img.sym: "h_img [m]",
        symbols.h_img.delta_sym: "delta_h_img [m]",
        symbols.h_obj.sym: "H_obj [m]",
        symbols.h_obj.delta_sym: "delta_H_obj [m]",
    }

    u_col, delta_u_col = physpy.table.convert_value(
        table,
        symbols_dict,
        "u [m]",
        symbols.x_l.sym - symbols.x_o.sym,
    )

    v_col, delta_v_col = physpy.table.convert_value(
        table,
        symbols_dict,
        "v [m]",
        symbols.x_s.sym - symbols.x_l.sym,
    )

    uv_table = physpy.pd.concat([u_col, delta_u_col, v_col, delta_v_col], axis=1)
    uv_symbols_dict = {
        symbols.v.sym: "v [m]",
        symbols.v.delta_sym: "delta_v [m]",
        symbols.u.sym: "u [m]",
        symbols.u.delta_sym: "delta_u [m]",
    }

    inv_u_col, delta_inv_u_col = physpy.table.convert_value(
        uv_table,
        uv_symbols_dict,
        "1/u [1/m]",
        1 / symbols.u.sym,
    )

    inv_v_col, delta_inv_v_col = physpy.table.convert_value(
        uv_table,
        uv_symbols_dict,
        "1/v [1/m]",
        1 / symbols.v.sym,
    )

    inv_uv_table = physpy.pd.concat(
        [inv_u_col, delta_inv_u_col, inv_v_col, delta_inv_v_col], axis=1
    )

    vu_col, delta_vu_col = physpy.table.convert_value(
        uv_table,
        uv_symbols_dict,
        "v/u [~]",
        symbols.v.sym / symbols.u.sym,
    )

    h_table = physpy.pd.concat(
        [vu_col, delta_vu_col, table["h_img [m]"], table["delta_h_img [m]"]], axis=1
    )

    return uv_table, inv_uv_table, h_table


def main():
    physpy.sympy.init_printing()
    physpy.graph.single_picture_graphs(True)

    fit_tables, param_table = physpy.table.parse_data(DATASHEET_PATH)
    symbols_table = physpy.equation.get_symbols_table(param_table=param_table)

    # Part A - Calculate focal length of chosen lens
    symbols_table.x_s.set_expr(
        (symbols_table.x_s_max.sym + symbols_table.x_s_min.sym) / 2,
        delta_expr=physpy.sympy.sqrt(
            ((symbols_table.x_s_max.sym - symbols_table.x_s_min.sym) / 10) ** 2
            + 0.5 * symbols_table.x_s_max.delta_sym**2
        ),  # / physpy.np.sqrt(12),
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
        f"N_Sigma: {physpy.graph.nsigma((symbols_table.f1.value, symbols_table.f1.delta), (symbols_table.f2.value, symbols_table.f2.delta))}"
    )

    # Part B - Calculate experimental U, V
    optics_table = fit_tables["optics"]
    optics_table = physpy.table.convert_units(
        optics_table, "H_obj [cm]", "H_obj [m]", lambda x: x / 100
    )
    optics_table = physpy.table.convert_units(
        optics_table, "h_img [cm]", "h_img [m]", lambda x: x / 100
    )
    optics_table = physpy.table.convert_units(
        optics_table, "delta_H_obj [cm]", "delta_H_obj [m]", lambda x: x / 100
    )
    optics_table = physpy.table.convert_units(
        optics_table, "delta_h_img [cm]", "delta_h_img [m]", lambda x: x / 100
    )

    uv_table, inv_uv_table, h_table = calculate_uv_tables(optics_table)
    uv_table = uv_table[1:]
    inv_uv_table = inv_uv_table[1:]
    h_table = h_table[1:]

    uv_table = physpy.table.add_relative_error_to_table(uv_table, "u [m]")
    uv_table = physpy.table.add_relative_error_to_table(uv_table, "v [m]")

    inv_uv_table = physpy.table.add_relative_error_to_table(inv_uv_table, "1/u [1/m]")
    inv_uv_table = physpy.table.add_relative_error_to_table(inv_uv_table, "1/v [1/m]")

    h_table = physpy.table.add_relative_error_to_table(h_table, "v/u [~]")
    h_table = physpy.table.add_relative_error_to_table(h_table, "h_img [m]")
    print(uv_table)
    print(inv_uv_table)
    print(h_table)

    # TODO: Fix graph names
    # Part B1 - Fit v against u, optics fit
    fit_data = physpy.graph.make_graph(
        "מרחק הדמות כתלות במרחק העצם - התאמה הופכית",
        uv_table,
        None,
        physpy.graph.fit.optics,
        (0, symbols_table.f1.value),
        output_folder=RESULTS_FOLDER,
    )
    f3 = physpy.graph.extract_fit_param(fit_data, 1)[:2]
    print(physpy.utils.get_value_error(*f3))
    print(
        f"NSIGMA: {physpy.graph.nsigma(f3, (symbols_table.f1.value, symbols_table.f1.delta))}"
    )
    return
    # Part B2 - Fit v against u, linear fit
    inv_fit_data = physpy.graph.make_graph(
        "מרחק הדמות כתלות במרחק העצם - התאמה לינארית",
        inv_uv_table,
        None,
        physpy.graph.fit.linear,
        (1 / symbols_table.f1.value, -1),
        output_folder=RESULTS_FOLDER,
    )
    a = physpy.graph.extract_fit_param(inv_fit_data, 0)[:2]
    b = physpy.graph.extract_fit_param(inv_fit_data, 1)[:2]
    f4 = (1 / a[0], a[1] / (a[0] ** 2))
    print(physpy.utils.get_value_error(*f4))
    print(
        f"NSIGMA: {physpy.graph.nsigma(f4, (symbols_table.f1.value, symbols_table.f1.delta))}"
    )
    print(f"SLOPE NSIGMA: {physpy.graph.nsigma(b, (-1, 0))}")

    print(f"NSIGMA BETWEEN CALC VALUES: {physpy.graph.nsigma(f3, f4)}")

    # Part B3 - Fit h against v/u
    h_fit_data = physpy.graph.make_graph(
        "גודל דמות כתלות ביחס מרחק דמות למרחק עצם",
        h_table,
        None,
        physpy.graph.fit.linear,
        (0, -1 * symbols_table.h.value),
        output_folder=RESULTS_FOLDER,
    )
    a = physpy.graph.extract_fit_param(h_fit_data, 1)[:2]
    # TODO: Why is it positive?
    # Oh it's okay actually, the image is reversed. The H parameter should be negative bc it was measured upside down.
    # h = (-1*a[0], a[1])
    h = a
    print(physpy.utils.get_value_error(*h))
    print(
        f"NSIGMA: {physpy.graph.nsigma(h, (symbols_table.h.value, symbols_table.h.delta))}"
    )
    # print(f"NSIGMA: {physpy.graph.nsigma(h, (symbols_table.h.value+0.001, symbols_table.h.delta))}")


if __name__ == "__main__":
    main()
