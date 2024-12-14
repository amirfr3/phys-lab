import pandas as pd

import math

from pprint import pprint

from ..graph.graph import make_graph
from ..graph.fit import fit_curve, extract_fit_param, linear, exponential


def build_xy_table(ball_sheet_file, ball_sheet_name, ball_diameter):
    # Calculate position as function of time

    data = pd.read_excel(ball_sheet_file, sheet_name=ball_sheet_name)
    t, x, y = data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2]
    n = len(data.index)
    # y = pd.Series([10 - i for i in y], name="y")

    # TODO: Fix X, Y, find bad data points
    # TODO: Fix error calculation

    tweak = 1
    dxy = math.sqrt(
        (tweak * ball_diameter / math.sqrt(12)) ** 2 + 0.028867513**2
    )  # Resolution error of diameter
    # dy = 0.0066 * 5 / math.sqrt(12)
    dxy_s = pd.Series([dxy] * n, name="delta_y [cm]")
    dt = (1 / 30) / math.sqrt(12)  # Resolution error of 30 FPS
    dt_s = pd.Series([dt] * n, name="delta_t [sec]")

    return pd.concat([t, dt_s, x, dxy_s, y, dxy_s], axis=1)


def build_v_table(y_table):
    t, dt, y, dy = (
        y_table.iloc[:, 0],
        y_table.iloc[:, 1],
        y_table.iloc[:, 2],
        y_table.iloc[:, 3],
    )
    n = len(y_table.index)

    v = pd.Series(
        [((y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])) for i in range(1, n - 1)],
        name="v [cm/sec]",
        index=range(1, n - 1),
    )
    tweak = 1
    dv = pd.Series(
        [
            tweak
            * (
                math.sqrt(2 * (dy[i] ** 2) + 2 * (v[i] * dt[i]) ** 2)
                / (t[i + 1] - t[i - 1])
            )
            for i in range(1, n - 1)
        ],
        name="delta_v [cm/sec]",
        index=range(1, n - 1),
    )
    return pd.concat([t[1 : n - 1], dt[1 : n - 1], v, dv], axis=1).reset_index(
        drop=True
    )


def main():
    datasheet = (
        r"C:\Users\flami\OneDrive\School\SemA\LabA\Viscosity\results\viscosity.xlsx"
    )
    should_show = True
    mass_graph = False
    vf_graph = False
    linear_fit = False
    exponential_fit = False
    find_tau = True

    if mass_graph:
        fit_data = make_graph(
            "מסת כדור כתלות ברדיוס כדור בשלישית",
            datasheet,
            "mass of radius",
            linear,
            (0, 33.5),
            show=should_show,
        )

        fit_data = make_graph(
            "מסת כדור כתלות ברדיוס כדור בשלישית - ללא מדידות חריגות",
            datasheet,
            "mass of radius (2)",
            linear,
            (0, 33.5),
            show=should_show,
        )

    if vf_graph:
        fit_data = make_graph(
            "מהירות סופית כפונקציה של רדיוס כדור בריבוע",
            datasheet,
            "vf of radius",
            linear,
            (0, 0),
            show=should_show,
        )

        fit_data = make_graph(
            "מהירות סופית כפונקציה של רדיוס כדור בריבוע - ללא מדידות חריגות",
            datasheet,
            "vf of radius (2)",
            linear,
            (0, 0),
            show=should_show,
        )

    ball_tables = [f"Ball {i+1}" for i in range(6)]
    ball_diameters = [0.5494, 0.4993, 0.4496, 0.3994, 0.3489, 0.296]
    v_f = [8.773966532, 7.252713534, 6.026764669, 4.885188811, 3.823164947, 2.924066617]
    v_0 = []
    datasheet_part2 = r"C:\Users\flami\OneDrive\School\SemA\LabA\Viscosity\results\Viscosity Data.xlsx"

    v_f_values = []
    tau_values = []
    values = []

    for table, diameter, v_f_i, i in zip(ball_tables, ball_diameters, v_f, range(6)):
        xy_table = build_xy_table(datasheet_part2, table, diameter)
        t, dt, x, dx, y, dy = (
            xy_table.iloc[:, 0],
            xy_table.iloc[:, 1],
            xy_table.iloc[:, 2],
            xy_table.iloc[:, 3],
            xy_table.iloc[:, 4],
            xy_table.iloc[:, 5],
        )

        n = len(xy_table.index)  # More correct way to get table length?
        x_table = pd.concat([t, dt, x, dx], axis=1)
        y_table = pd.concat([t, dt, y, dy], axis=1)

        # TODO: Rotate X positions to find bad data points?
        # fit_data = make_graph(
        #     f"מיקום אופקי כתלות בזמן עבור כדור מספר {i+1}",
        #     x_table,
        #     0,
        #     linear,
        #     (0,0),
        #     show=should_show,
        # )

        # Get terminal velocity from linear fit of movement at terminal velocity
        if linear_fit:
            y_lin_table = y_table.copy(deep=True)
            y_lin_fit = fit_curve(linear, (0, 0), y_lin_table)
            dropped_idx = 0
            while len(y_lin_table.index) >= 1 and y_lin_fit["p_val"] < 0.05:
                # print(f"P-Value of %{y_lin_fit["p_val"]*100:.02} at {dropped_idx}")
                y_lin_table = y_lin_table.drop(y_lin_table.index[:1])
                dropped_idx += 1
                y_lin_fit = fit_curve(linear, (0, 0), y_lin_table)
            print(
                f"P-Value of %{y_lin_fit["p_val"]*100:.02} at {dropped_idx} ({y_table.iloc[:, 0][dropped_idx]} sec)"
            )

            y_lin_fit = make_graph(
                f"גובה כתלות בזמן עבור כדור מספר {i+1} - תנועה במהירות סופית",
                y_lin_table,
                0,
                linear,
                (0, 0),
                show=should_show,
            )
            v_f_0 = extract_fit_param(y_lin_fit, 1)
            print(
                f"Terminal velocity of {table}: {v_f_0[0]}±{v_f_0[1]} ({v_f_0[2]}%) [cm/sec] - p-val ({v_f_0[4]})"
            )
            v_f_guess = v_f_0[0]
            v_f = (v_f_0,)  # TODO: Should we make more estimates?
            v_f_values.append(v_f[0])

        # Calculate finite difference for speed
        v_table = build_v_table(y_table)

        # From arbitrary linear point
        # TODO: Find automatically instead
        # Look for linear portion
        # diff_v = pd.Series([abs(((v[i]-v[i-1])/(t[i]-t[i-1])) - ((v[i+1]-v[i])/(t[i+1]-t[i]))) for i in range (2, n-2)], name="diff_v", index=range(2,n-2))

        # Linear fit for velocity graph (useless?)
        if linear_fit:
            lin_idx = dropped_idx
            v_table_lin = v_table.drop(v_table.index[:lin_idx])
            # print(v_table_lin)

            v_lin_fit = make_graph(
                f"Velocity - {table} - Linear"[::-1],
                v_table_lin,
                0,
                linear,
                (0, 0.2),
                show=should_show,
            )

        if exponential_fit:
            exp_idx = 3  # Some weird data points where it started off going backwards
            v_table_exp = v_table.drop(v_table.index[:exp_idx])
            # print(v_table_exp)

            tweak = 1
            tau_guess = (-1 * tweak) / (126.8092437 * ((diameter / 2) ** 2))

            v_exp_fit = make_graph(
                f"מהירות כתלות בזמן עבור כדור {i+1} - תנועה בתאוצה",
                v_table_exp,
                0,
                exponential,
                # (v_f_guess, tau_guess, -1*v_f_guess),
                (v_f_i, tau_guess, -1 * v_f_i),
                show=should_show,
            )

        # Find V_f from fits
        # v_f_1 = extract_fit_param(v_lin_fit, 0)
        # v_f_2 = extract_fit_param(v_exp_fit, 0)
        # v_f_3 = extract_fit_param(v_exp_fit, 2) # This is (v_0 - v_f) - we assume v_0 = 0 so we fix this accordingly
        # v_f_3 = list(v_f_3)
        # v_f_3[0] = (lambda v: -1*v)(v_f_3[0])
        # v_f_3 = tuple(v_f_3)
        # v_f = (v_f_0, v_f_1, v_f_2, v_f_3)

        # Find Tau from fits
        # tau = []

        # tau_0 = extract_fit_param(v_exp_fit, 1)
        # tau_0 = list(tau_0)
        # tau_0[1] = (lambda tau, delta_tau: delta_tau/(tau**2))(tau_0[0], tau_0[1])
        # tau_0[0] = (lambda tau: -1/tau)(tau_0[0])
        # tau_0 = tuple(tau_0)
        # tau.append(tau_0)

        # tau_values.append(tuple(tau)[0])
        # continue # TODO: Don't skip value search

        # Find Tau from value search
        # v= v_f_guess
        v = v_f_i
        v_0 = 0
        exp_val = -1 * (v + ((v_0 - v) / math.e))
        # t_col = v_table.iloc[:, 0]
        # v_col = v_table.iloc[:, 2]

        closest = v_table.iloc[(v_table["v [cm/sec]"] - exp_val).abs().argsort()[:3]]
        # print(closest)
        closest_time = closest["t [sec]"].values[0]
        print(
            f"Looking for {exp_val}, found {closest['v [cm/sec]'].values[0]} at {closest_time}"
        )
        # print(closest_time)
        # tau.append(closest_time)

        # tau = tuple(tau)
        tau_values.append(closest_time)

        # values.append({
        #     "diameter": diameter,
        #     "v_f": v_f,
        #     "tau": tau,
        # })
    values = list(zip(ball_diameters, v_f_values, tau_values))
    pprint(values)

    dr = 0.0000144337567297406  # Delta_r_inst

    r_2 = pd.Series([], name="r^2 [cm^2]")
    dr_2 = pd.Series([], name="d_r^2 [cm^2]")
    v_f = pd.Series([], name="V_f [cm/sec]")
    dv_f = pd.Series([], name="d_V_f [cm/sec]")
    tau = pd.Series([], name="tau [sec^-1]")
    dtau = pd.Series([], name="d_tau [sec^-1]")

    # for v in values:
    for d_i, v_f_i, tau_i in values:
        # r_i = v["diameter"]/2
        r_i = d_i / 2

        r_2 = pd.concat([r_2, pd.Series([r_i**2])])
        dr_2 = pd.concat([dr_2, pd.Series([2 * r_i * dr])])

        # v_f_i = v["v_f"][0]

        v_f = pd.concat([v_f, pd.Series([-1 * v_f_i[0]])])
        dv_f = pd.concat([dv_f, pd.Series([v_f_i[1]])])

        # tau_i = v["tau"][1]

        tau = pd.concat([tau, pd.Series([tau_i[0]])])
        dtau = pd.concat([dtau, pd.Series([tau_i[1]])])

    vf_table = pd.concat([r_2, dr_2, v_f, dv_f], axis=1)
    tau_table = pd.concat([r_2, dr_2, tau, dtau], axis=1)

    # vf_fit = make_graph(
    #     "מהירות סופית כפונקציה של רדיוס כדור בריבוע",
    #     vf_table,
    #     0,
    #     linear,
    #     (0, 0),
    #     show = True,
    # )
    # eta_vf = extract_fit_param(vf_fit, 1)
    # print(f"Viscosity Slope, Delta Slope for V_f: {eta_vf[0]} +- {eta_vf[1]} ({eta_vf[2]})")

    # tau_fit = make_graph(
    #     "סקלת זמן כפונקציה של רדיוס כדור בריבוע",
    #     tau_table,
    #     0,
    #     linear,
    #     (0, 0),
    #     show = True,
    # )
    # eta_tau = extract_fit_param(tau_fit, 1)
    # print(f"Viscosity Slope, Delta Slope for Tau: {eta_tau[0]} +- {eta_tau[1]} ({eta_tau[2]})")

    return


if __name__ == "__main__":
    main()
