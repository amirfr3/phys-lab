import pandas as pd
import re
import csv


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


def parse_param_csv(filepath):
    params = {}
    with open(filepath, mode='r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            print(row)
            params[row['name']] = {
                'value': float(row['value']),
                'uncert': float(row['uncert']),
                'unit': row['unit']
            }
    return params


def rename(table, old_columns, new_columns):
    return table.rename(columns={old:new for old, new in zip(old_columns, new_columns)})