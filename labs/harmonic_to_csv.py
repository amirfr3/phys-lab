import os
import pandas as pd
import numpy as np


DX = 0.001 / np.sqrt(12)
DT = 0.0001 / np.sqrt(12)


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

    return pd.concat(
        [
            pd.Series(t, name="Time [sec]"),
            pd.Series(dt, name="delta_Time [sec]"),
            pd.Series(x, name="Position [m]"),
            pd.Series(dx, name="delta_Position [m]"),
        ],
        axis=1,
    )


def get_oscillation_data(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            parse_position_data(os.path.join(folder_path, filename)).to_excel(folder_path + "\\" + filename.split(".")[0] + ".xlsx")


def main():
    get_oscillation_data(r"C:\Users\flami\Downloads\1")


if __name__ == "__main__":
    main()
