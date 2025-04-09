import pandas as pd
import re

def _parse_params(param_table):
    if param_table is None:
        return
    params = {}
    for param in filter(lambda n: not n.startswith('Unnamed'), param_table.columns.values):
        params[param] = param_table[param][0] if len(param_table[param]) == 1 else tuple(p[1] for p in param_table[param].items())

    return params

def parse_data(filepath):
    fit_tables = {}
    params = None
    fit_regex = re.compile('(_fit)|(fit_)', re.IGNORECASE)
    with pd.ExcelFile(filepath) as xl:
        for sheet in xl.sheet_names:
            if sheet.lower().endswith('_fit') or sheet.lower().startswith('fit_'):
                fit_tables[re.sub(fit_regex, '', sheet)] = pd.read_excel(xl, sheet)
            if sheet.lower() == 'params':
                params = pd.read_excel(xl, sheet)

    return fit_tables, _parse_params(params)
