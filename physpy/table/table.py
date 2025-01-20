import pandas as pd
from dataclasses import dataclass

@dataclass
class Parameter:
    name: str
    val: float
    delta_val: float
    units: str

def _parse_params(param_table):
    params = {}
    for param in param_table.columns.values:
        # Find errors - Currently disabled
        #error=None
        #for p in param_table.columns.values:
        #    if p.startwith("d"+param.split[0]):
        #        error = p

        # This column is just table headers!
        if param == "Name":
            continue

        # params[param.split()[0]] = param_table[param][0] if len(param_table[param]) == 1 else param_table[param]
        params[param] = Parameter(param, *param_table[param])

    return params


def parse_data(filepath):
    """
    Returns tuple of all fit tables in sheet and parameters namespace.
    """
    fit_tables = {}
    params = None
    with pd.ExcelFile(filepath) as xl:
        for sheet in xl.sheet_names:
            if sheet.lower().endswith('_fit'):
                fit_tables[sheet.replace('_fit', '')] = pd.read_excel(xl, sheet)
            elif sheet.lower().startswith('fit_'):
                fit_tables[sheet.replace('fit_', '')] = pd.read_excel(xl, sheet)
            elif sheet.lower() == 'params':
                params = _parse_params(pd.read_excel(xl, sheet))

    return fit_tables, params


def add_relative_error_to_table(table, col, delta_col=""):
    if delta_col == "":
        delta_col = f"delta_{col}"

    table = table.copy()
    table[f"Relative Error {col[:col.find("[")] + "(%)"}"] = (
        table[delta_col] / table[col]
    ) * 100
    return table


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

