# pip install uncertainties docx math2docx
import uncertainties
#import docx
#import math2docx


def add_math_to_tbl(row, name, dimension, value, error=None):
    row.cells[0].vertical_alignment = docx.enum.table.WD_ALIGN_VERTICAL.CENTER
    row.cells[1].vertical_alignment = docx.enum.table.WD_ALIGN_VERTICAL.CENTER
    par = row.cells[1].add_paragraph()
    if error is None:
        v = f"{uncertainties.ufloat(0, value):.2uf}"
        v = v[v.index("-") + 1:]
        p = ""
    else:
        v = f"{uncertainties.ufloat(value, error):.2uf}".replace("+/-", "\\pm ")
        p = f"{uncertainties.ufloat(0, abs((error/value)*100)):.2uf}"
        p = p[p.index("-") + 1:]
        p = f"\\ \\ \\ ({p}\\%)"
    if dimension is None:
        dimension = ""
    else:
        dimension = f"[{dimension}]"
    math2docx.add_math(par, f"{name} = {v} \\ {dimension} {p}")
