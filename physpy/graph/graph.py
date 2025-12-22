import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from physpy.graph.fit import fit_curve
from typing import Optional


_SINGLE_PICTURE_GRAPHS = False
_LATEX_WRAP = False
_FILETYPE = "png"
_GRAPH_SCALE = 5
_LABEL_FONT_SIZE = 20

def set_graph_file_type(t: str):
    global _FILETYPE, _GRAPH_SCALE, _LABEL_FONT_SIZE
    _FILETYPE=t
    if _FILETYPE not in ['svg', 'eps']:
        _GRAPH_SCALE = 5
        _LABEL_FONT_SIZE = 20
    else:
        _GRAPH_SCALE = 1
        _LABEL_FONT_SIZE = 15


def single_picture_graphs(b: bool):
    global _SINGLE_PICTURE_GRAPHS
    _SINGLE_PICTURE_GRAPHS = b


def latex_labels(b: bool):
    global _LATEX_WRAP
    _LATEX_WRAP = b


def _latex_wrap(s):
    if _LATEX_WRAP:
        return '$' + s + '$'
    return s


def _is_hebrew(s):
    return any(c in s for c in 'אבגדהוזחטיכלמנסעפצקרשת')


def _scatter_data(figure, data_x, data_y, error_x, error_y):
    figure.add_trace(go.Scatter(x=data_x, y=data_y,
                mode='markers', 
                error_x=dict(type='data',
                             array=error_x,
                             color='grey'), 
                error_y=dict(type='data',
                             array=error_y,
                             color='grey'),
                marker=dict(color='blue')))


def _plot_layout(figure, plot_title, x_title, y_title):
    figure.update_layout(
        plot_bgcolor='white',
        showlegend=False,
        title=dict(
            text=plot_title,
            x=0.5,
            xanchor='center',
            font=dict(
                size=30
            )
        ),
        xaxis=dict(
            title=dict(
                text=x_title,
                font=dict(size=_LABEL_FONT_SIZE)),
            showgrid=True,
            linecolor='grey',
            linewidth=2,
            ticks='outside',
            gridcolor='lightgrey',
            zerolinecolor='lightgrey',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text=y_title,
                font=dict(size=_LABEL_FONT_SIZE)),
            showgrid=True,
            linecolor='grey',
            linewidth=2,
            ticks='outside',
            gridcolor='lightgrey',
            zerolinecolor='lightgrey',
            zerolinewidth=1
        ),
        margin=dict(
            l=20, r=20, t=20, b=20,
        )
    )


def _fit_plot(data, plot_name: Optional[str]=None, xsuffix: Optional[str]=None, ysuffix: Optional[str]=None):
    fit_figure = go.Figure()

    x_fit = np.linspace(min(data['x']), max(data['x']), 10 * len(data['x']))

    y_fit = data["fit_func"](data["fit_params"], x_fit)
    fit_figure.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='lines', line=dict(color='red')))
    _scatter_data(fit_figure, data['x'], data['y'], data['delta_x'], data['delta_y'])

    sep = '\\,' if _LATEX_WRAP else ' '
    x_label_suffix = sep.join(((data['columns'][0].split()[1] if len(data['columns'][0].split()) > 1 else ''),
        (xsuffix if xsuffix is not None else '')))
    x_label = data['columns'][0].split()[0]
    y_label_suffix = sep.join(((data['columns'][2].split()[1] if len(data['columns'][2].split()) > 1 else ''),
        (ysuffix if ysuffix is not None else '')))
    y_label = data['columns'][2].split()[0]

    _plot_layout(fit_figure, plot_name, 
                 _latex_wrap(f'{x_label}{sep}{x_label_suffix}'), 
                 _latex_wrap(f'{y_label}{sep}{y_label_suffix}'))

    return fit_figure


def _residual_plot(data, data_x, residuals, error_x, error_y, plot_name:Optional[str]=None, xsuffix: Optional[str]=None, ysuffix: Optional[str]=None, 
                    fit_func_name:Optional[str]='fit', title_suffix:Optional[str]=' - Residuals', title_suffix_hebrew:Optional[str]=" - גרף שארים"):
    residual_figure = go.Figure()

    _scatter_data(residual_figure, data_x, residuals, error_x, error_y)

    # Add zero line
    residual_figure.add_hline(y=0, line_width=3, line_dash="dash", line_color="red")

    if plot_name is not None:
        plot_name = plot_name + title_suffix_hebrew if _is_hebrew(plot_name) else plot_name + title_suffix

    x_idx, y_idx = (0, 2) if data['x'].name == data_x.name else (2, 0)

    sep = '\\,' if _LATEX_WRAP else ' '
    x_label_suffix = sep.join(((data['columns'][x_idx].split()[1] if len(data['columns'][x_idx].split()) > 1 else ''),
        (xsuffix if xsuffix is not None else '')))
    x_label = data['columns'][x_idx].split()[0]
    y_label_suffix = sep.join(((data['columns'][y_idx].split()[1] if len(data['columns'][y_idx].split()) > 1 else ''),
        (ysuffix if ysuffix is not None else '')))
    y_label = data['columns'][y_idx].split()[0]
    
    _plot_layout(residual_figure, plot_name, 
        _latex_wrap(f'{x_label}{sep}{x_label_suffix}'),
        _latex_wrap(f'{y_label} - {fit_func_name}({x_label}){sep}{y_label_suffix}')
    )

    return residual_figure


def build_plot_with_residuals(data, plot_name:Optional[str]= None, xsuffix: Optional[str]=None, ysuffix: Optional[str]=None, x_residuals=False):
    # For now, only single pic graphs
    fit_figure = _fit_plot(data, plot_name, xsuffix, ysuffix)
    residual_figure = _residual_plot(data, data['x'], data['residuals'], data['delta_x'], data['delta_y'], plot_name)
    if x_residuals:
        if data['x_residuals'] is None:
            raise TypeError('No inverse function for the chosen fit function. consider defining it and adding it to to INVERSE_FUNCTION dict.')

        x_residual_figure = _residual_plot(data, data['y'], data['x_residuals'], data['delta_y'], data['delta_x'], plot_name, 
                                            fit_func_name='fit^1', title_suffix='- X Axis Residuals')
        return fit_figure, residual_figure, x_residual_figure

    return fit_figure, residual_figure


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
    output_folder='.',
    show=True,
    debug_show=False,
    columns=(0,1,2,3),
    xsuffix: Optional[str]=None,
    ysuffix: Optional[str]=None,
    show_x_residuals=False,
    print_outliers=True,
    graph_filename="fit"
):
    """
    graph_title: Title for graph (RTL)
    table_or_file_path: DataFrame or Replace with your file path: r"/content/<your file name>.xlsx"
    sheet_idx: Replace with your sheet number
    fit_func: Choose your fit function name
    initial_guesses: Define the initial guesses for the parameters in list "A" (make sure they are the same length, and in the same order!)
    """

    # Reverse Hebrew RTL
    if graph_title is not None:
        if _is_hebrew(graph_title):
            #graph_title_rtl = graph_title[::-1]
            graph_title_rtl = graph_title
        else:
            graph_title_rtl = graph_title
    else:
        graph_title_rtl = None
    processed_data = fit_curve(fit_func, initial_guesses, table_or_file_path, sheet_idx, columns=columns)

    figures = build_plot_with_residuals(processed_data, graph_title_rtl, xsuffix=xsuffix, ysuffix=ysuffix, 
                                    x_residuals=show_x_residuals)

    if output_folder is not None:
        graph_filename = graph_title.replace(' ', '_') if graph_title is not None else graph_filename
        with open(os.path.join(output_folder, f"{graph_filename}_stats.txt"), "w") as f:
            f.write(processed_data["fit_results"])
        for i, fig in enumerate(figures):
            fig.write_image(os.path.join(output_folder, f"{graph_filename}_{i}.{_FILETYPE}"), scale=_GRAPH_SCALE)
            #fig.savefig(os.path.join(output_folder,f"{graph_filename}_{i}.svg"), bbox_inches='tight')
        pd.concat((processed_data['x'], processed_data['delta_x'], 
                  processed_data['y'], processed_data['delta_y']), axis=1)\
        .to_csv(os.path.join(output_folder, f'{graph_filename}_fit_data.csv'), index=False)

    if show:
        if debug_show:
            print(
                f"=== EXAMPLE DATA FOR {graph_title_rtl} ===\n{processed_data['data'][:5]}\n================="
            )
            print(processed_data["fit_results"])
        processed_data['show_figures'] = [fig.show() for fig in figures]

    if processed_data['outliers'] and print_outliers:
        print("**OUTLIERS**")
        for m in processed_data['outliers']:
            print(f"{m[0]}: {m[1]}")
        print()

    return processed_data


def calc_inst_error(res):
    return res/np.sqrt(12)
