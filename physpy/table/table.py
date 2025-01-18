import pandas as pd


def _parse_params(param_table):
    params = {}
    for param in param_table.columns.values:
        # Find errors - Currently disabled
        #error=None
        #for p in param_table.columns.values:
        #    if p.startwith("d"+param.split[0]):
        #        error = p
        params[param.split()[0]] = param_table[param][0] if len(param_table[param]) == 1 else param_table[param]

    return params

def parse_data(filepath):
    fit_tables = {}
    params = None
    with pd.ExcelFile(filepath) as xl:
        for sheet in xl.sheet_names:
            if sheet.endswith('_fit'):
                fit_tables[sheet.replace('_fit', '')] = pd.read_excel(xl, sheet)
        if sheet.lower() == 'params':
            params = pd.read_excel(xl, sheet)

    return fit_tables, _parse_params(params)
