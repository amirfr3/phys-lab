import math
from pprint import pprint

import os
import pandas as pd
import numpy as np
import scipy as sp

import sympy

# from sympy.parsing.latex import parse_latex

import physpy
import sympy.printing


DATASHEET_FOLDER = r"C:\Users\flami\OneDrive\School\SemA\LabA\Harmonic\results"
DATASHEET_PATH = DATASHEET_FOLDER + r"\harmonic.xlsx"
RESULTS_FOLDER = DATASHEET_FOLDER + r"\data_processing"


DX = physpy.graph.calc_inst_error(0.001)
DT = physpy.graph.calc_inst_error(0.0001)


# def calculate_error(value, std_dev, measurement_count, scale, measure, accuracy):
#     # measure_percent = measure / 100
#     # accuracy_percent = accuracy / 100

#     inst_err = value * measure_percent + scale * accuracy_percent
#     stat_err = std_dev / math.sqrt(measurement_count)
#     total_err = math.sqrt(inst_err**2 + stat_err**2)

#     return (value, total_err)


def read_params(datasheet):
    # Add sheet "Params" to table
    params = pd.read_excel(datasheet, sheet_name="Params")

    # Add values to global namespace
    for name, series in params.items():
        globals()["P_" + name] = tuple(series.to_numpy()[:2])  # )


def get_value_error(value, error):
    return f"{value} ± {error} ({error / value * 100}%)"


def add_relative_error_to_table(table, col, delta_col=""):
    if delta_col == "":
        delta_col = f"delta_{col}"

    table = table.copy()
    table[f"Relative Error {col[:col.find("[")] + "(%)"}"] = (
        table[delta_col] / table[col]
    ) * 100
    return table


def calculate_spring_constant_and_uncertainty(m, x_i, x_f, g):
    k_s, m_s, x_i_s, x_f_s, g_s = physpy.equation.get_symbols_with_uncertainties(
        "k m x_i x_f g"
    )

    k_expr = sympy.Eq(k_s[0], m_s[0] * g_s[0] / (x_f_s[0] - x_i_s[0]))
    dk_expr = physpy.equation.calculate_indirect_error_formula(k_expr)

    # Pass values to value expression and values with uncertainties to uncertainty expression
    k = k_expr.subs(
        {
            m_s[0]: m[0],
            x_i_s[0]: x_i[0],
            x_f_s[0]: x_f[0],
            g_s[0]: g[0],
        }
    ).rhs
    dk = dk_expr.subs(
        {
            m_s[0]: m[0],
            x_i_s[0]: x_i[0],
            x_f_s[0]: x_f[0],
            g_s[0]: g[0],
            m_s[1]: m[1],
            x_i_s[1]: x_i[1],
            x_f_s[1]: x_f[1],
            g_s[1]: g[1],
        }
    ).rhs

    return k, dk


def remove_outliers(xt_table):
    # 1. Remove first few entries due to rounding error in the program
    # xt_table = xt_table[5:]
    return xt_table


def parse_position_data(filename):
    data = [i.split("\t") for i in open(filename).read().splitlines()[2:]]
    t = []
    dt = []
    x = []
    dx = []
    for i in data:
        t.append(np.float64(i[0]))
        dt.append(DT)
        x.append(np.float64(i[1]))
        dx.append(DX)

    xt_table = pd.concat(
        [
            pd.Series(t, name="Time [sec]"),
            pd.Series(dt, name="delta_Time [sec]"),
            pd.Series(x, name="Position [m]"),
            pd.Series(dx, name="delta_Position [m]"),
        ],
        axis=1,
    )

    xt_table = remove_outliers(xt_table)
    optimize_sine_fit(xt_table)
    return xt_table


def optimize_sine_fit(xt_table):
    import numpy as np
    from scipy.optimize import leastsq
    import pylab as plt

    data = xt_table["Position [m]"]
    N = len(data)  # number of data points
    t = np.linspace(0, 4 * np.pi, N)
    # f = 1.15247 # Optional!! Advised not to use
    # data = 3.0*np.sin(f*t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

    do_guess = False
    do_fft = False

    if do_guess:
        guess_mean = np.mean(data)
        guess_std = 3 * np.std(data) / (2**0.5) / (2**0.5)
        guess_phase = 0
        guess_freq = 1
        guess_amp = 1

        # we'll use this to plot our first estimate. This might already be good enough for you
        data_first_guess = guess_std * np.sin(t + guess_phase) + guess_mean

        # Define the function to optimize, in this case, we want to minimize the difference
        # between the actual data and our "guessed" parameters
        optimize_func = lambda x: x[0] * np.sin(x[1] * t + x[2]) + x[3] - data
        est_amp, est_freq, est_phase, est_mean = leastsq(
            optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean]
        )[0]

        # recreate the fitted curve using the optimized parameters
        data_fit = est_amp * np.sin(est_freq * t + est_phase) + est_mean

        # recreate the fitted curve using the optimized parameters

        fine_t = np.arange(0, max(t), 0.1)
        data_fit = est_amp * np.sin(est_freq * fine_t + est_phase) + est_mean

        plt.plot(t, data, ".")
        plt.plot(t, data_first_guess, label="first guess")
        plt.plot(fine_t, data_fit, label="after fitting")
        plt.legend()
        plt.show()

        return guess_mean, guess_std, guess_phase, guess_freq, guess_amp

    if do_fft:
        mfft = np.fft.fft(data)
        imax = np.argmax(np.absolute(mfft))
        mask = np.zeros_like(mfft)
        mask[[imax]] = 1
        mfft *= mask
        fdata = np.fft.ifft(mfft)

        plt.plot(t, data, ".")
        plt.plot(t, fdata, ".", label="FFT")
        plt.legend()
        plt.show()

        return None


def generate_oscillation_graph(
    measurement_filename, guess=(0, 3.7, 0, 0), should_show=False
):
    pos_table = parse_position_data(measurement_filename)

    _, part, num = measurement_filename.split("\\")[-1].split("_")

    part_str = {"a": "א'", "b": "ב'"}[part]
    num_str = num[::-1]

    graph_name = f"גרף תנודה - חלק {part_str} מדידה {num_str}"

    # NOTE: This is NOT shown by default
    fit_data = physpy.graph.make_graph(
        graph_name,
        pos_table,
        None,
        physpy.graph.fit.sinusoidal,
        guess,
        output_folder=RESULTS_FOLDER,
        show=should_show,
    )

    # while fit_data['']
    # print(f"{measurement_filename}:\n{fit_data['fit_results']}\n")
    # if fit_data['p_val'] < 0.05:
    #     print("LOW P!")
    # if fit_data['p_val'] > 0.95:
    #     print('HIGH P!')
    return fit_data


def get_guesses(datasheet, sheet_name):
    guesses_table = pd.read_excel(datasheet, sheet_name=sheet_name)

    # Reorder parameters according to fit function and convert period to frequency
    guesses_list = list(
        zip(
            guesses_table["D"],
            ((2 * math.pi) / guesses_table["B"]),
            guesses_table["C"],
            guesses_table["A"],
        )
    )
    return guesses_list


def get_oscillation_data(guesses_datasheet, guesses_datasheet_name, folder_path):
    guesses = get_guesses(guesses_datasheet, guesses_datasheet_name)

    # Assumed to be ordered by number ascending
    fit_data = []
    for filename, guess in zip(os.listdir(folder_path), guesses):
        # if filename.endswith(".txt"):
        fit_data.append(
            generate_oscillation_graph(os.path.join(folder_path, filename), guess=guess)
        )

    return fit_data


def calculate_squared_period_and_uncertainty(fit_data_list):
    # Fit data contains frequency rather than period, so we need to convert it
    p_l = []
    dp_l = []

    for fit_data in fit_data_list:
        # Calculate period from frequency
        f, df = physpy.graph.extract_fit_param(fit_data, 1)[:2]
        p = (2 * math.pi) / f
        dp = (2 * math.pi) * df / (f**2)

        # Square period and its uncertainty
        p = p**2
        dp = 2 * p * dp

        # TODO: Print more info from fit data?

        p_l.append(p)
        dp_l.append(dp)

    # Return period^2 and uncertainty as numpy series
    return pd.Series(p_l, name="Period^2 [sec^2]"), pd.Series(
        dp_l, name="delta_Period^2 [sec^2]"
    )


def generate_period_graph(fit_data_list, table, x_col):
    p_s, dp_s = calculate_squared_period_and_uncertainty(fit_data_list)

    fit_table = pd.concat(
        [
            table[x_col],
            table["delta_" + x_col],
            p_s,
            dp_s,
        ],
        axis=1,
    )

    i = {
        "Length [m]": 0,
        "Mass [kg]": 1,
    }[x_col]

    graph_name = [
        "ריבוע זמן מחזור מטוטלת מתמטית כתלות באורך החוט",
        "ריבוע זמן מחזור מסה תלויה על קפיץ כתלות במסה",
    ][i]

    initial_guess = [
        (0, 4*(math.pi**2)/9.81),
        (0, 4*(math.pi**2)/25),
    ][i]
    
    fit_data = physpy.graph.make_graph(
        graph_name,
        fit_table,
        None,
        physpy.graph.fit.linear,
        initial_guess,
        output_folder=RESULTS_FOLDER,
        show=True,
    )

    fit_table = physpy.table.add_relative_error_to_table(fit_table, x_col, f"delta_{x_col}")
    fit_table = physpy.table.add_relative_error_to_table(fit_table, "Period^2 [sec^2]", f"delta_Period^2 [sec^2]")
    fit_table.to_excel(RESULTS_FOLDER + f"\\{graph_name}.xlsx")

    return fit_data


# def sum_of_squared_error(params):
#     # import warnings
#     # warnings.filterwarnings("ignore")

#     val = physpy.graph.fit.sinusoidal(


# def generate_phase_guess():
#     parameter_bounds = []
#     parameter_bounds.append(0.001, 0.01)
#     parameter_bounds.append(math.sqrt(1.2), math.sqrt(2))
#     parameter_bounds.append(-math.pi/2, math.pi/2)
#     parameter_bounds.append(0.15,0.175)

#     result = sp.optimize.differential_evolution(sum_of_squared_error, parameter_bounds, seed=3)
#     return result.x


def extract_harmonic_fit_param(fit_data):
    slope, dslope = physpy.graph.extract_fit_param(fit_data, 1)[:2]
    return (4 * (math.pi**2) / slope, 8 * (math.pi**2) * dslope / (slope**2))


def main():
    sympy.init_printing()

    datasheet = DATASHEET_PATH
    physpy.graph.single_picture_graphs(True)

    # Read params sheet into global namespace
    read_params(datasheet)

    # get_oscillation_data(r"C:\Users\flami\Downloads\1")
    # get_oscillation_data(r"C:\Users\flami\Downloads\2")

    # Part 1 - Calculate oscillation period of simple pendulum
    # pendulum_oscillation_data = get_oscillation_data(
    #     datasheet,
    #     "Fit_Simple",
    #     r"C:\Users\flami\OneDrive\School\SemA\LabA\Harmonic\results\measurements\part_a",
    # )
    pendulum_oscillation_data = get_oscillation_data(
        datasheet,
        "Fit_Simple_Nimo",
        r"C:\Users\flami\OneDrive\School\SemA\LabA\Harmonic\results\measurements\part_a",
    )
    pendulum_table = physpy.graph.read_table(datasheet, "Fit_Simple_Nimo")
    pendulum_table = physpy.graph.convert_units(
        pendulum_table, "Length [cm]", "Length [m]", lambda x: x / 100
    )
    pendulum_table = physpy.graph.convert_units(
        pendulum_table, "delta_Length [cm]", "delta_Length [m]", lambda x: x / 100
    )
    pendulum_fit_data = generate_period_graph(
        pendulum_oscillation_data, pendulum_table, "Length [m]"
    )

    g_theo = P_g
    print(f"Theoretical gravity constant: {get_value_error(*g_theo)}")

    g_exp = extract_harmonic_fit_param(pendulum_fit_data)
    print(
        f"Experimental gravity constant: {get_value_error(*g_exp)} (Expected: {get_value_error(*g_theo)})"
    )
    print(f"N_sigma - {physpy.graph.nsigma(g_exp, g_theo)}")

    # Part 2B - Calculate oscillation period of spring
    spring_oscillation_data = get_oscillation_data(
        datasheet,
        "Fit_Spring",
        r"C:\Users\flami\OneDrive\School\SemA\LabA\Harmonic\results\measurements\part_b",
    )
    spring_table = physpy.graph.read_table(datasheet, "Fit_Spring")
    spring_fit_data = generate_period_graph(
        spring_oscillation_data, spring_table, "Mass [kg]"
    )

    # Compare spring constant to expected value
    k_theo = calculate_spring_constant_and_uncertainty(P_dx_m, P_x1, P_x0, P_g)
    print(f"Theoretical spring constant: {get_value_error(*k_theo)}")

    k_exp = extract_harmonic_fit_param(spring_fit_data)
    print(
        f"Experimental spring constant: {get_value_error(*k_exp)} (Expected: {get_value_error(*k_theo)})"
    )
    print(f"N_sigma - {physpy.graph.nsigma(k_exp, k_theo)}")

    # Part 3 - Calculate oscillation period of physical pendulum
    # TODO: Copy the above parts to this one if this is relevant


if __name__ == "__main__":
    main()
