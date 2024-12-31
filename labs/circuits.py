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
    total_err = math.sqrt(inst_err ** 2 + stat_err ** 2)

    return (value, total_err)


MEASUREMENT_COUNT = 100
# Scale, Measure, Accuracy (according to datasheet)
AMPEREMETER_INFO = (100, 0.01, 0.004) # 100mA
VOLTMETER_INFO = (10, 0.0015, 0.0004) # 10V
RESIST_INFO = (100, 0.003, 0.003) # 100 Ohm


def convert_value_stddev_current(table, col):
    valerrs = []
    table = physpy.graph.convert_units(table, f"stddev {col} [uA]", f"stddev {col} [mA]", lambda x: x / 1000)

    value_col = table[f"{col} [mA]"]
    stddev_col = table[f"stddev {col} [mA]"]
    valerrs.append(value_col)

    errs = []
    for value, stddev in zip(value_col, stddev_col):
        value, error = calculate_error(value, stddev, MEASUREMENT_COUNT, *AMPEREMETER_INFO)
        errs.append(error)
    valerrs.append(pd.Series(errs, name=f"delta_{col} [mA]"))

    return valerrs


def convert_value_stddev_voltage(table, col):
    valerrs = []
    table = physpy.graph.convert_units(table, f"stddev {col} [uV]", f"stddev {col} [V]", lambda x: x / 1000000)

    value_col = table[f"{col} [V]"]
    stddev_col = table[f"stddev {col} [V]"]
    valerrs.append(value_col)

    errs = []
    for value, stddev in zip(value_col, stddev_col):
        value, error = calculate_error(value, stddev, MEASUREMENT_COUNT, *VOLTMETER_INFO)
        errs.append(error)
    valerrs.append(pd.Series(errs, name=f"delta_{col} [V]"))

    return valerrs

def convert_mA_cols_to_A(table, col):
    table = physpy.graph.convert_units(table, f"{col} [mA]", f"{col} [A]", lambda x: x / 1000)
    table = physpy.graph.convert_units(table, f"delta_{col} [mA]", f"delta_{col} [A]", lambda x: x / 1000)
    return table

def main():
    sympy.init_printing()

    datasheet = DATASHEET_PATH
    results = RESULTS_FOLDER
    should_show = True

    # Part 0 - Choose correct circuit
    diff_table = physpy.graph.read_table(datasheet, "CircuitChoosing").iloc[:, :8]
    # for col in ["I_a", "I_b"]:
    #     diff_table = physpy.graph.convert_units(diff_table, f"stddev {col} [uA]", f"stddev {col} [mA]", lambda x: x / 1000)
    # for col in ["V_a", "V_b"]:
    #     diff_table = physpy.graph.convert_units(diff_table, f"stddev {col} [uV]", f"stddev {col} [V]", lambda x: x / 1000000)


    valerrs = []
    for col in ["I_a", "I_b"]:
        valerrs.extend(convert_value_stddev_current(diff_table, col))
        # value_col = diff_table[f"{col} [mA]"]
        # stddev_col = diff_table[f"stddev {col} [mA]"]
        # valerrs.append(value_col)

        # errs = []
        # for value, stddev in zip(value_col, stddev_col):
        #     value, error = calculate_error(value, stddev, MEASUREMENT_COUNT, *AMPEREMETER_INFO)
        #     errs.append(error)
        # valerrs.append(pd.Series(errs, name=f"delta_{col} [mA]"))

    for col in ["V_a", "V_b"]:
        valerrs.extend(convert_value_stddev_voltage(diff_table, col))
        # value_col = diff_table[f"{col} [V]"]
        # stddev_col = diff_table[f"stddev {col} [V]"]
        # valerrs.append(value_col)

        # errs = []
        # for value, stddev in zip(value_col, stddev_col):
        #     value, error = calculate_error(value, stddev, MEASUREMENT_COUNT, *VOLTMETER_INFO)
        #     errs.append(error)
        # valerrs.append(pd.Series(errs, name=f"delta_{col} [V]"))

    diff_table = pd.concat(valerrs, axis=1)
    diff_table = convert_mA_cols_to_A(diff_table, "I_a")
    diff_table = convert_mA_cols_to_A(diff_table, "I_b")
    # diff_table = physpy.graph.convert_units(diff_table, "I_a [mA]", "I_a [A]", lambda x: x / 1000)
    # diff_table = physpy.graph.convert_units(diff_table, "delta_I_a [mA]", "delta_I_a [A]", lambda x: x / 1000)
    # diff_table = physpy.graph.convert_units(diff_table, "I_b [mA]", "I_b [A]", lambda x: x / 1000)
    # diff_table = physpy.graph.convert_units(diff_table, "delta_I_b [mA]", "delta_I_b [A]", lambda x: x / 1000)

    # Part A - Calculate resistance of R1
    ohm_table = physpy.graph.read_table(datasheet, "OhmsLawData").iloc[:, :4]
    valerrs = []
    valerrs.extend(convert_value_stddev_current(ohm_table, "Current"))
    valerrs.extend(convert_value_stddev_voltage(ohm_table, "Voltage"))
    ohm_table = pd.concat(valerrs, axis=1)
    ohm_table = convert_mA_cols_to_A(ohm_table, "Current")

    # TODO: UNITS!!!
    # TODO: Make from value,stddev and convert to table with value,error,relerr

    # ohm_table = physpy.graph.convert_units(ohm_table, "Current [mA]", "Current [A]", lambda x: x / 1000)
    # ohm_table = physpy.graph.convert_units(ohm_table, "delta_Current [mA]", "delta_Current [A]", lambda x: x / 1000)
    # print(ohm_table)

    inverse_ohm_table = physpy.graph.flip_table_axis(ohm_table)
    print(inverse_ohm_table)

    # TODO: Calculate theoretical resistance according to table?
    direct_resistance = (98.32894, 0.006048797)

    fit_data = physpy.graph.make_graph(
        "המתח על המעגל כתלות בזרם",
        ohm_table,
        None,
        physpy.graph.fit.linear,
        (0, 0),
        output_folder=results,
        show=should_show,
    )
    resistance = physpy.graph.extract_fit_param(fit_data, 1)[:2]
    print(f"Resistance of R1: {resistance} - N_sigma = {physpy.graph.nsigma(direct_resistance, resistance)}")

    fit_data = physpy.graph.make_graph(
        "הזרם במעגל כתלות במתח",
        inverse_ohm_table,
        None,
        physpy.graph.fit.linear,
        (0, 0),
        output_folder=results,
        show=should_show,
    )
    inv_resistance = physpy.graph.extract_fit_param(fit_data, 1)
    inv_resistance = (1/inv_resistance[0], inv_resistance[1]/inv_resistance[0]**2)
    print(f"Resistance of R1 (from inverse): {inv_resistance} - N_sigma = {physpy.graph.nsigma(direct_resistance, inv_resistance)}")

    # Part B - Calculate resistivity of wire
    wire_diameter = (0.263, 0.000288675134594813) #mm
    wire_radius = (wire_diameter[0] / 2, wire_diameter[1] / 2) #mm
    wire_area = (math.pi * wire_radius[0] ** 2, 2 * math.pi * wire_radius[0] * wire_radius[1]) #mm^2

    resistivity_table = physpy.graph.read_table(datasheet, "ResistanceLength").iloc[:, :4]

    # TODO: UNITS!!!

    fit_data = physpy.graph.make_graph(
        "התנגדות מעגל הכולל נגד תיל כתלות באורך התיל",
        resistivity_table,
        None,
        physpy.graph.fit.linear,
        (0, 0),
        output_folder=results,
        show=should_show,
    )
    resistivity_per_area = physpy.graph.extract_fit_param(fit_data, 1)
    print(f"Resistivity per area: {resistivity_per_area}")
    resistivity = (resistivity_per_area[0] * wire_area[0], resistivity_per_area[0] * wire_area[0] * math.sqrt((resistivity_per_area[1] / resistivity_per_area[0]) ** 2 + (wire_area[1] / wire_area[0]) ** 2))
    print(f"Resistivity: {resistivity}")


if __name__ == "__main__":
    main()
