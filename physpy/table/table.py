import pandas as pd


def _parse_params(param_table):
    if param_table is None:
        return
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
            print(sheet)
            if sheet.lower().endswith('_fit') or sheet.lower().startswith('fit_'):
                fit_tables[sheet.replace('_fit', '')] = pd.read_excel(xl, sheet)
            if sheet.lower() == 'params':
                params = pd.read_excel(xl, sheet)

    return fit_tables, _parse_params(params)
