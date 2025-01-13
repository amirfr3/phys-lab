import pandas as pd


def _read_params(params):
    # Add values to global namespace
    for name, series in params.items():
        globals()["P_" + name] = tuple(series.to_numpy()[:2])  # )


def parse_data(filepath):
    fit_tables = {}
    params = None
    with pd.ExcelFile(filepath) as xl:
        for sheet in xl.sheet_names:
            if sheet.endswith('_fit'):
                fit_tables[sheet.replace('_fit', '')] = pd.read_excel(xl, sheet)
        if sheet.lower() == 'params':
            params = pd.read_excel(xl, sheet)

    _read_params(params)
    return fit_tables
