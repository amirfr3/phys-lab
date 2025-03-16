import pandas as pd
from dataclasses import dataclass
from ..equation.equation import calculate_indirect_error_formula, get_symbols_with_uncertainties
import sympy

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
            if sheet.lower() == 'params':
                params = _parse_params(pd.read_excel(xl, sheet))
                continue

            FIT_TAGS = ['_fit', '_Fit', 'fit_', 'Fit_']

            for fit_tag in FIT_TAGS:
                if fit_tag in sheet:
                    fit_tables[sheet.replace(fit_tag, '').lower()] = pd.read_excel(xl, sheet)
                    break

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


def convert_value(
    table,
    symbol_dict,
    col_name,
    expr,
    delta_expr=None,
):
    v_l = []
    dv_l = []

    # Generate dummy variables for expressions
    v_sym, dv_sym = get_symbols_with_uncertainties("v")
    expr = sympy.Eq(v_sym, expr)
    if delta_expr:
        delta_expr = sympy.Eq(dv_sym, expr)
    if not delta_expr:
        delta_expr = calculate_indirect_error_formula(expr)

    # Reduce table to relevant columns
    # table = table[symbol_dict.keys()]

    # col_dict = {
    #     key: table[symbol_dict[key]] for key in symbol_dict.keys()
    # }

    for row in [s for _, s in table.iterrows()]:
        val_dict = {}
        for key in symbol_dict.keys():
            val_dict[key] = row[symbol_dict[key]]
        
        v_l.append(float(expr.rhs.subs(val_dict)))
        dv_l.append(float(delta_expr.rhs.subs(val_dict)))

    return pd.Series(v_l, name=col_name), pd.Series(dv_l, name=f"delta_{col_name}")
