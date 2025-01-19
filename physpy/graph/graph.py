import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .fit import fit_curve
from typing import Optional

_SINGLE_PICTURE_GRAPHS = False

def single_picture_graphs(b: bool):
    global _SINGLE_PICTURE_GRAPHS
    _SINGLE_PICTURE_GRAPHS = b

def _suffix(s):
    return ' {s}' if s is not None else ''


def build_plot_with_residuals(data, plot_name, xsuffix: Optional[str]=None, ysuffix: Optional[str]=None, show_x_residuals=False):
    plt.close("all")
    if _SINGLE_PICTURE_GRAPHS:
        fig1, axs = plt.subplots(1, 2, figsize=(15, 6))
        figs = [fig1]
    else:
        figs = []
        axs = []
        fig1, ax1 = plt.subplots(1, 1, figsize=(8,6))
        figs.append(fig1)
        axs.append(ax1)
        fig1, ax1 = plt.subplots(1, 1, figsize=(8,6))
        figs.append(fig1)
        axs.append(ax1)

    if show_x_residuals:
        fig2, ax2 = plt.subplots(1, 1, figsize=(7, 6))
        figs.append(fig2)

    plt.style.use("classic")

    for fig in figs:
        fig.patch.set_facecolor("white")

    for ax in axs:
        ax.set_facecolor("white")
    if show_x_residuals:
        ax2.set_facecolor("white")

    x_fit = np.linspace(min(data["x"]), max(data["x"]), 10 * len(data["x"]))
    y_fit = data["fit_func"](data["fit_params"], x_fit)
    axs[0].errorbar(
        data["x"],
        data["y"],
        xerr=data["delta_x"],
        yerr=data["delta_y"],
        fmt=".b",
        label="Data",
        ecolor="gray",
    )  # Change the label if needed
    axs[0].plot(
        x_fit, y_fit, label="Fit", c="r", alpha=0.5
    )  # Change the label if needed

    # If you want to plot multiple functions, change here the relevant parameters (x, y, xerr, yerr, label). Otherwise, uncomment the 2 next lines:
    # axs[0].errorbar(data["x"] + 0.2, data["y"] + 0.3, xerr=data["delta_x"], yerr=data["delta_y"], fmt='.g', label='Data', ecolor='gray')
    # axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

    axs[0].set_title(plot_name)  # Add here the full title for the fit
    axs[0].set_xlabel(
        f'{data["columns"][0]}' + _suffix(xsuffix)
    )  # Change x-axis label if needed
    axs[0].set_ylabel(
        f'{data["columns"][2]}' + _suffix(ysuffix)
    )  # Change y-axis label if needed

    axs[0].grid(True)
    # axs[0].legend()

    axs[1].errorbar(
        data["x"],
        data["residuals"],
        xerr=data["delta_x"],
        yerr=data["delta_y"],
        fmt=".b",
        label="Data",
        ecolor="gray",
    )
    #axs[1].hlines(0, min(data["x"]), max(data["x"]), colors="r", linestyles="dashed")
    axs[1].axhline(0, color="r", linestyle="dashed")

    axs[1].set_title(
        " - גרף שארים"[::-1] + plot_name
    )  # Add here the full title for the residuals
    axs[1].set_xlabel(
        f'{data["columns"][0]}' + _suffix(xsuffix))  # Change column names if needed
    axs[1].set_ylabel(
        f'{data["columns"][2].split()[0]} - fit({data["columns"][0].split()[0]}) {data["columns"][2].split()[1]}' + _suffix(ysuffix)[1:]
    )  # Change column names if needed

    axs[1].grid(True)
    # axs[1].legend()
    if show_x_residuals:
        if data['x_residuals'] is None:
            raise TypeError('No inverse function for the chosen fit function. consider defining it and adding it to to INVERSE_FUNCTION dict.')

        ax2.errorbar(
            data["y"],
            data["x_residuals"],
            xerr=data["delta_y"],
            yerr=data["delta_x"],
            fmt=".b",
            label="Data",
            ecolor="gray",
        )
        #ax2.hlines(0, min(data["y"]), max(data["y"]), colors="r", linestyles="dashed")
        ax2.axhline(0, color="r", linestyle="dashed")

        ax2.set_title(
            " - גרף שארים בציר x"[::-1] + plot_name
        )  # Add here the full title for the residuals
        ax2.set_xlabel(
            f'{data["columns"][2]}' + _suffix(ysuffix)
        )  # Change column names if needed
        ax2.set_ylabel(
            f'{data["columns"][0].split()[0]} - fit^-1({data["columns"][2].split()[0]}) {data["columns"][0].split()[1]}' + _suffix(xsuffix)[1:]
        )  # Change column names if needed

        ax2.grid(True)

    plt.tight_layout()
    return plt


def read_table(
    file_path,
    sheet_idx,
):
    data = pd.read_excel(file_path, sheet_name=sheet_idx)
    return data


def convert_units(
    table,
    src_col_name,
    dst_col_name,
    conversion_func,
):
    dst_col = table[src_col_name].apply(conversion_func).rename(dst_col_name)
    table[src_col_name] = dst_col
    table.rename(columns={src_col_name: dst_col_name}, inplace=True)
    return table


def flip_table_axis(table):
    table = table[[table.columns[2], table.columns[3], table.columns[0], table.columns[1]]]
    return table


def make_graph(
    graph_title,
    table_or_file_path,
    sheet_idx,
    fit_func,
    initial_guesses,
    output_folder=None,
    show=True,
    debug_show=False,
    columns=(0,1,2,3),
    xsuffix: Optional[str]=None,
    ysuffix: Optional[str]=None,
    show_x_residuals=False
):
    """
    graph_title: Title for graph (RTL)
    table_or_file_path: DataFrame or Replace with your file path: r"/content/<your file name>.xlsx"
    sheet_idx: Replace with your sheet number
    fit_func: Choose your fit function name
    initial_guesses: Define the initial guesses for the parameters in list "A" (make sure they are the same length, and in the same order!)
    """

    # Reverse Hebrew RTL
    graph_title_rtl = graph_title[::-1]
    processed_data = fit_curve(fit_func, initial_guesses, table_or_file_path, sheet_idx, columns=columns)

    plt = build_plot_with_residuals(processed_data, graph_title_rtl, xsuffix=xsuffix, ysuffix=ysuffix, show_x_residuals=show_x_residuals)

    if not output_folder:
        output_folder = "." # Default to current directory
    with open(f"{output_folder}\\{graph_title}_stats.txt", "w") as f:
        f.write(processed_data["fit_results"])
    plt.savefig(f"{output_folder}\\{graph_title}.png")

    if show:
        if debug_show:
            print(
                f"=== EXAMPLE DATA FOR {graph_title_rtl} ===\n{processed_data['data'][:5]}\n================="
            )
            print(processed_data["fit_results"])
        plt.show()

    if processed_data['extreme_measurments']:
        print("**EXTREME MEASUMENTS**\n")
        for m in processed_data['extreme_measurments']:
            print(f"{m[0]}: {m[1]}")

    return processed_data


def calc_inst_error(res):
    return res/np.sqrt(12)
