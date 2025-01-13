import math
from pprint import pprint

import pandas as pd

import sympy

# from sympy.parsing.latex import parse_latex

import physpy
import sympy.printing


DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Circuits\results"
DATASHEET_PATH = DATASHEET_FOLDER + r"\circuits.xlsx"
RESULTS_FOLDER = DATASHEET_FOLDER + r"\data_processing"


def calculate_error(value, std_dev, measurement_count, scale, measure, accuracy):
    measure_percent = measure / 100
    accuracy_percent = accuracy / 100

    inst_err = value * measure_percent + scale * accuracy_percent
    stat_err = std_dev / math.sqrt(measurement_count)
    total_err = math.sqrt(inst_err**2 + stat_err**2)

    return (value, total_err)


MEASUREMENT_COUNT = 100
# Scale, Measure, Accuracy (according to datasheet)
AMPEREMETER_INFO = (100, 0.01, 0.004)  # 100mA
VOLTMETER_INFO = (10, 0.0015, 0.0004)  # 10V
RESIST_INFO = (100, 0.003, 0.003)  # 100 Ohm


def convert_value_stddev_current(table, col):
    valerrs = []
    table = physpy.graph.convert_units(
        table, f"stddev {col} [uA]", f"stddev {col} [mA]", lambda x: x / 1000
    )

    value_col = table[f"{col} [mA]"]
    stddev_col = table[f"stddev {col} [mA]"]
    valerrs.append(value_col)

    errs = []
    for value, stddev in zip(value_col, stddev_col):
        value, error = calculate_error(
            value, stddev, MEASUREMENT_COUNT, *AMPEREMETER_INFO
        )
        errs.append(error)
    valerrs.append(pd.Series(errs, name=f"delta_{col} [mA]"))

    return valerrs


def convert_value_stddev_voltage(table, col):
    valerrs = []
    table = physpy.graph.convert_units(
        table, f"stddev {col} [uV]", f"stddev {col} [V]", lambda x: x / 1000000
    )

    value_col = table[f"{col} [V]"]
    stddev_col = table[f"stddev {col} [V]"]
    valerrs.append(value_col)

    errs = []
    for value, stddev in zip(value_col, stddev_col):
        value, error = calculate_error(
            value, stddev, MEASUREMENT_COUNT, *VOLTMETER_INFO
        )
        errs.append(error)
    valerrs.append(pd.Series(errs, name=f"delta_{col} [V]"))

    return valerrs


def convert_mA_cols_to_A(table, col):
    table = physpy.graph.convert_units(
        table, f"{col} [mA]", f"{col} [A]", lambda x: x / 1000
    )
    table = physpy.graph.convert_units(
        table, f"delta_{col} [mA]", f"delta_{col} [A]", lambda x: x / 1000
    )
    return table


def convert_voltage_current_table_to_resistance(table, voltage_col, current_col):
    r_s, v_s, i_s = sympy.symbols("R V I")
    dv_s, di_s = physpy.equation.get_uncertainties([v_s, i_s])
    r_expr = sympy.Eq(r_s, v_s / i_s)
    dr_expr = physpy.equation.calculate_indirect_error_formula(r_expr)

    table = table.copy()
    v_col = table[f"{voltage_col} [V]"]
    delta_v_col = table[f"delta_{voltage_col} [V]"]
    i_col = table[f"{current_col} [A]"]
    delta_i_col = table[f"delta_{current_col} [A]"]

    r_vals = []
    dr_vals = []
    for v, i, dv, di in zip(v_col, i_col, delta_v_col, delta_i_col):
        # r = v / i
        # dr = r * math.sqrt((dv / v) ** 2 + (di / i) ** 2)
        r_vals.append(float(r_expr.subs({v_s: v, i_s: i}).rhs.evalf()))
        dr_vals.append(float(dr_expr.subs({v_s: v, i_s: i, dv_s: dv, di_s: di}).rhs.evalf()))

    r_series = pd.Series(r_vals, name="Resistance [Ohm]")
    dr_series = pd.Series(dr_vals, name="delta_Resistance [Ohm]")
    return pd.concat([r_series, dr_series], axis=1)


def get_value_error(value, error):
    return f"{value} ± {error} ({error / value * 100}%)"


def generate_ohms_law_graphs(table, output_folder, should_show, label_suffix=""):
    fit_data = physpy.graph.make_graph(
        "המתח על הנגד הנמדד כתלות בזרם בו" + label_suffix,
        table,
        None,
        physpy.graph.fit.linear,
        (0, 0),
        output_folder=output_folder,
        show=should_show,
    )
    resistance = physpy.graph.extract_fit_param(fit_data, 1)[:2]

    fit_data = physpy.graph.make_graph(
        "הזרם בנגד הנמדד כתלות במתח עליו" + label_suffix,
        physpy.graph.flip_table_axis(table),
        None,
        physpy.graph.fit.linear,
        (0, 0),
        output_folder=output_folder,
        show=should_show,
    )
    inv_resistance = physpy.graph.extract_fit_param(fit_data, 1)
    inv_resistance = (1 / inv_resistance[0], inv_resistance[1] / inv_resistance[0] ** 2)

    return resistance, inv_resistance


def add_relative_error_to_table(table, col, delta_col=""):
    if delta_col == "":
        delta_col = f"delta_{col}"

    table = table.copy()
    table[f"Relative Error {col[:col.find("[")] + "(%)"}"] = (
        table[delta_col] / table[col]
    ) * 100
    return table


def main():
    sympy.init_printing()

    datasheet = DATASHEET_PATH
    results = RESULTS_FOLDER
    should_show = True

    remove_outliers = True
    log_table = False

    # Part 0 - Choose correct circuit
    diff_table = physpy.graph.read_table(datasheet, "CircuitChoosing").iloc[:, :8]

    valerrs = []
    for col in ["I_a", "I_b"]:
        valerrs.extend(convert_value_stddev_current(diff_table, col))

    for col in ["V_a", "V_b"]:
        valerrs.extend(convert_value_stddev_voltage(diff_table, col))

    diff_table = pd.concat(valerrs, axis=1)
    diff_table = convert_mA_cols_to_A(diff_table, "I_a")
    diff_table = convert_mA_cols_to_A(diff_table, "I_b")
    diff_table = add_relative_error_to_table(diff_table, "I_a [A]")
    diff_table = add_relative_error_to_table(diff_table, "I_b [A]")
    diff_table = add_relative_error_to_table(diff_table, "V_a [V]")
    diff_table = add_relative_error_to_table(diff_table, "V_b [V]")

    print(diff_table)
    print(
        get_value_error(
            diff_table["I_a [A]"].mean(), diff_table["delta_I_a [A]"].mean()
        )
    )
    print(
        get_value_error(
            diff_table["I_b [A]"].mean(), diff_table["delta_I_b [A]"].mean()
        )
    )
    print(
        get_value_error(
            diff_table["V_a [V]"].mean(), diff_table["delta_V_a [V]"].mean()
        )
    )
    print(
        get_value_error(
            diff_table["V_b [V]"].mean(), diff_table["delta_V_b [V]"].mean()
        )
    )
    print(
        (
            abs(diff_table["I_a [A]"].mean() - diff_table["I_b [A]"].mean())
            / (diff_table["I_a [A]"].mean() + diff_table["I_b [A]"].mean())
        )
        * 100
    )
    print(
        (
            abs(diff_table["V_a [V]"].mean() - diff_table["V_b [V]"].mean())
            / (diff_table["V_a [V]"].mean() + diff_table["V_b [V]"].mean())
        )
        * 100
    )

    # Part A - Calculate resistance of R1
    ohm_table = physpy.graph.read_table(datasheet, "OhmsLawData").iloc[:, :4]
    valerrs = []
    valerrs.extend(convert_value_stddev_current(ohm_table, "Current"))
    valerrs.extend(convert_value_stddev_voltage(ohm_table, "Voltage"))
    ohm_table = pd.concat(valerrs, axis=1)
    ohm_table = convert_mA_cols_to_A(ohm_table, "Current")

    # TODO: Calculate theoretical resistance according to table?
    direct_resistance = (98.32894, 0.006048797)
    print(f"Direct resistance of R1: {get_value_error(*direct_resistance)})")

    if remove_outliers:
        # Remove outliers
        ohm_table = ohm_table[1:]
        label_suffix = " - ללא מדידות חריגות"
    else:
        label_suffix = ""

    resistance, inv_resistance = generate_ohms_law_graphs(
        ohm_table, results, should_show, label_suffix=label_suffix
    )
    print(
        f"Resistance of R1: {get_value_error(*resistance)} - N_sigma = {physpy.graph.nsigma(direct_resistance, resistance)}"
    )
    print(
        f"Resistance of R1 (from inverse): {get_value_error(*inv_resistance)} - N_sigma = {physpy.graph.nsigma(direct_resistance, inv_resistance)}"
    )

    if log_table:
        log_ohm_table = physpy.graph.flip_table_axis(ohm_table).copy()
        # log_ohm_table["Voltage [V]"] = log_ohm_table["Voltage [V]"].apply(math.log)
        log_ohm_table["Current [A]"] = log_ohm_table["Current [A]"].apply(math.exp)
        fit_data = physpy.graph.make_graph(
            "המתח על הנגד כתלות בזרם בו - לוגריתמי" + label_suffix,
            log_ohm_table,
            None,
            physpy.graph.fit.linear,
            (0, 0),
            output_folder=results,
            show=should_show,
        )

        res_table = convert_voltage_current_table_to_resistance(
            ohm_table, "Voltage", "Current"
        )
        res_table["Resistance [Ohm]"] = res_table["Resistance [Ohm]"].apply(math.log)
        resistance_ohm_table = pd.concat(
            [
                pd.Series(range(0, 600, 60), name="Time [sec]"),
                pd.Series([3] * 10, name="delta_Time [sec]"),
                # ohm_table["Voltage [V]"],
                # ohm_table["delta_Voltage [V]"],
                res_table,
            ],
            axis=1,
        )

        physpy.graph.make_graph(
            "התנגדות מעגל כתלות בזמן" + label_suffix,
            resistance_ohm_table,
            None,
            physpy.graph.fit.linear,
            (0, 0),
            output_folder=results,
            show=should_show,
        )

    ohm_table = add_relative_error_to_table(ohm_table, "Current [A]")
    ohm_table = add_relative_error_to_table(ohm_table, "Voltage [V]")
    ohm_table_output = pd.concat(
        [
            ohm_table["Current [A]"],
            ohm_table["delta_Current [A]"],
            ohm_table["Relative Error Current (%)"],
            ohm_table["Voltage [V]"],
            ohm_table["delta_Voltage [V]"],
            ohm_table["Relative Error Voltage (%)"],
        ],
        axis=1,
    )
    ohm_table_output.to_excel(results + r"\ohm_table.xlsx")

    # Part B - Calculate resistivity of wire
    # theo_resistivity = (1.1, 0)
    theo_resistivity = (0.5, 0)
    # theo_resistance_per_meter = (21.3, 0.5)

    resistivity_table = physpy.graph.read_table(datasheet, "ResistanceLengthData").iloc[
        :, :6
    ]
    length_table = resistivity_table.iloc[:, :2]
    resistivity_table = resistivity_table.iloc[:, 2:]
    valerrs = []
    valerrs.extend(convert_value_stddev_current(resistivity_table, "I"))
    valerrs.extend(convert_value_stddev_voltage(resistivity_table, "V"))
    voltage_current_table = pd.concat(valerrs, axis=1)
    voltage_current_table = convert_mA_cols_to_A(voltage_current_table, "I")
    voltage_current_table = add_relative_error_to_table(voltage_current_table, "V [V]")
    voltage_current_table = add_relative_error_to_table(voltage_current_table, "I [A]")
    resistivity_table = convert_voltage_current_table_to_resistance(
        voltage_current_table, "V", "I"
    )

    resistivity_table = pd.concat([length_table, resistivity_table], axis=1)
    resistivity_table = add_relative_error_to_table(resistivity_table, "Length [m]")
    resistivity_table = add_relative_error_to_table(
        resistivity_table, "Resistance [Ohm]"
    )

    resistivity_table_output = pd.concat(
        [
            resistivity_table["Length [m]"],
            resistivity_table["delta_Length [m]"],
            resistivity_table["Relative Error Length (%)"],
            voltage_current_table["V [V]"],
            voltage_current_table["delta_V [V]"],
            voltage_current_table["Relative Error V (%)"],
            voltage_current_table["I [A]"],
            voltage_current_table["delta_I [A]"],
            voltage_current_table["Relative Error I (%)"],
            resistivity_table["Resistance [Ohm]"],
            resistivity_table["delta_Resistance [Ohm]"],
            resistivity_table["Relative Error Resistance (%)"],
        ],
        axis=1,
    )
    resistivity_table_output.to_excel(results + r"\resistivity_table.xlsx")

    """
    if remove_outliers:    
        # Remove outliers (1st and 8th measurements)
        resistivity_table = resistivity_table.drop([1, 8])
        label_suffix = " - ללא מדידות חריגות"
    else:
        label_suffix = ""
    """
    label_suffix = ""

    fit_data = physpy.graph.make_graph(
        "התנגדות מעגל נגד תיל כתלות באורך התיל" + label_suffix,
        resistivity_table,
        None,
        physpy.graph.fit.linear,
        (theo_resistivity[0], 0),
        output_folder=results,
        show=should_show,
    )
    resistance_per_meter = physpy.graph.extract_fit_param(fit_data, 1)

    wire_diameter = (0.263, 0.000288675134594813)  # mm
    wire_radius = (wire_diameter[0] / 2, wire_diameter[1] / 2)  # mm
    wire_area = (
        math.pi * (wire_radius[0] ** 2),
        2 * math.pi * wire_radius[0] * wire_radius[1],
    )  # mm^2

    resistivity = (
        resistance_per_meter[0] * wire_area[0],
        resistance_per_meter[0]
        * wire_area[0]
        * math.sqrt(
            (resistance_per_meter[1] / resistance_per_meter[0]) ** 2
            + (wire_area[1] / wire_area[0]) ** 2
        ),
    )

    print(
        f"Resistance per meter: {get_value_error(resistance_per_meter[0], resistance_per_meter[1])} [Ohm/m]"# - N_sigma = {physpy.graph.nsigma(theo_resistance_per_meter, resistance_per_meter)}"
    )
    print(
        f"Diameter {get_value_error(*wire_diameter)} , Area: {get_value_error(*wire_area)} [mm^2]"
    )
    print(
        f"Resistivity: {get_value_error(*resistivity)} [Ohm*mm^2/m] - N_sigma = {physpy.graph.nsigma(theo_resistivity, resistivity)}"
    )


if __name__ == "__main__":
    main()
