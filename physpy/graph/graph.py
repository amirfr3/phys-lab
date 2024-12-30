import numpy as np
import matplotlib.pyplot as plt
from .fit import fit_curve


def build_plot_with_residuals(data, plot_name):
    plt.close("all")
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plt.style.use("classic")

    fig.patch.set_facecolor("white")
    for ax in axs:
        ax.set_facecolor("white")

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
    # axs[0].errorbar(x + 0.2, y + 0.3, xerr=delta_x, yerr=delta_y, fmt='.g', label='Data', ecolor='gray')
    # axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

    axs[0].set_title(plot_name)  # Add here the full title for the fit
    axs[0].set_xlabel(
        f'{data["data"].columns[[data["columns"][0]]][0]}'
    )  # Change x-axis label if needed
    axs[0].set_ylabel(
        f'{data["data"].columns[[data["columns"][2]]][0]}'
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
    axs[1].hlines(0, min(data["x"]), max(data["x"]), colors="r", linestyles="dashed")

    axs[1].set_title(
        " - גרף שארים"[::-1] + plot_name
    )  # Add here the full title for the residuals
    axs[1].set_xlabel(
        f'{data["data"].columns[[data["columns"][0]]][0]}'
    )  # Change column names if needed
    axs[1].set_ylabel(
        f'{data["data"].columns[[data["columns"][2]]][0]} - fit({data["data"].columns[[data["columns"][0]]][0]})'
    )  # Change column names if needed

    axs[1].grid(True)
    # axs[1].legend()

    plt.tight_layout()
    return plt


def make_graph(
    graph_title,
    table_or_file_path,
    sheet_idx,
    fit_func,
    initial_guesses,
    show=True,
    debug_show=False,
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
    processed_data = fit_curve(fit_func, initial_guesses, table_or_file_path, sheet_idx)

    plt = build_plot_with_residuals(processed_data, graph_title_rtl)

    with open(f"{graph_title}_stats.txt", "w") as f:
        f.write(processed_data["fit_results"])
    plt.savefig(f"{graph_title}.png")

    if show:
        if debug_show:
            print(
                f"=== EXAMPLE DATA FOR {graph_title_rtl} ===\n{processed_data['data'][:5]}\n================="
            )
        print(processed_data["fit_results"])
        plt.show()

    return processed_data

