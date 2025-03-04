import sympy
from uncertainties import ufloat
import math
import re

def _get_uncertainty(symbol):
    return sympy.symbols(f"Delta_{symbol.name}")
    # return sympy.parsing.latex.parse_latex(f"\\Delta{{{symbol.name}}}")


def get_uncertainties(symbols):
    if hasattr(symbols, '__iter__'):
        return (_get_uncertainty(symbol) for symbol in symbols)

    # Singular
    return _get_uncertainty(symbols)


def get_symbols_with_uncertainties(symbols_str):
    symbols = sympy.symbols(symbols_str)
    uncertainties = get_uncertainties(symbols)

    # Return each symbol with its uncertainty
    return zip(symbols, uncertainties) 


class UnsupportedOperationError(KeyError):
    pass


def calculate_indirect_error_sym(expr):
    assert expr.func is sympy.core.symbol.Symbol

    return get_uncertainties(expr)


def calculate_indirect_error_add(expr):
    assert expr.func is sympy.core.add.Add

    return sympy.sqrt(sum([_calculate_indirect_error(s)**2 for s in expr.args]))


def calculate_indirect_error_pow(expr):
    assert expr.func is sympy.core.power.Pow

    # TODO: Recurse
    base, power = expr.args
    assert power > 1

    return expr.diff(base) * _calculate_indirect_error(base)


def calculate_indirect_error_mul(expr):
    assert expr.func is sympy.core.mul.Mul

    # TODO: Change Pow operations to absolute value
    e_unc = {e: _calculate_indirect_error(e) for e in expr.args}

    return expr * sympy.sqrt(sum([(e_unc[e]/e)**2 for e in expr.args]))


def _calculate_indirect_error(expr):
    if expr.is_number:
        return expr

    if sympy.core.numbers.NegativeOne in expr.args:
        # Try absolute value and return to sender
        print(expr)
        return _calculate_indirect_error(-1*expr)

    # TODO: Handle integers
    OPS = {
        sympy.core.symbol.Symbol: calculate_indirect_error_sym,
        sympy.core.add.Add: calculate_indirect_error_add,
        sympy.core.power.Pow: calculate_indirect_error_pow,
        sympy.core.mul.Mul: calculate_indirect_error_mul,
    }

    try:
        return OPS[expr.func](expr)

    except KeyError:
        raise UnsupportedOperationError("Unsupported operation in expression")


# TODO: Maybe receive list of parameters as well
# Assumed to be equality
def calculate_indirect_error_formula(expr):
    assert expr.is_Equality, "Expression must be an equality"
    # Basic version
    return sympy.Eq(
        get_uncertainties(expr.lhs),
        sympy.sqrt(sum(
            [(expr.rhs.diff(s)*get_uncertainties(s))**2 for s in expr.rhs.free_symbols]
        ))
    )

    return sympy.Eq(get_uncertainties(expr.lhs), _calculate_indirect_error(expr.rhs))


def latexify(expr):
    latex_str = sympy.printing.latex(expr)

    latex_str = latex_str.replace(
        r"Delta_",
        r"Delta"
    )

    # TODO: Fix subscripts

    return latex_str


def _to_rounded_value_str(value, error):
    s = f"{ufloat(value, error):.2u}"
    if '(' in s:
        # Exponent factoring
        v, e, f = re.split(r'\+/-|\(|\)', s)[1:]
    else:
        v, e = s.split('+/-')
        f = ""
    return v, e, f


def _to_rounded_number_str(value):
    v, e, f = _to_rounded_value_str(value, 10**math.floor(math.log(abs(value), 10)))
    return v, f


def _latexify_value(name, value: str, error: str, factor, relative_error: str, relative_error_factor, unit):
    latex_str = f'{name} = \\SI' + f'{{{value}({error}){factor}}}' + '{' + (unit if unit is not None else '') + '}' 
    if relative_error is not None:
        latex_str += '\\,' + f'(\\num{{{relative_error}{relative_error_factor}}}\\%)'
    return latex_str


def latexify_and_round_value(name, value, error=0, unit=None, relative_error=True):
    # Currently need to supply the latex unit yourself.
    if value == 0 and error == 0:
        v, e, f = 0, 0, ""
    elif error == 0:
        v, f = _to_rounded_number_str(value)
        e = 0
    else:
        v, e, f = _to_rounded_value_str(value, error)
    p, pf = None, None
    if relative_error and v != 0:
        p, pf = _to_rounded_number_str(abs((error/value)*100))
    return _latexify_value(name, v, e, f, p, pf, unit)


def latexify_and_round_fit_params(fit_data, units=None):
    latex_str = ""
    if units is None:
        units = list()
    units += [None]*(len(fit_data['fit_params'])-len(units))

    for i, (param, error, unit) in enumerate(zip(fit_data['fit_params'], fit_data['fit_params_error'], units)):
        latex_str += latexify_and_round_value(f'a_{i}', param, error, unit=unit) + '\n'
    
    latex_str += latexify_and_round_value('\\chi^2_{red}', fit_data['chi2_red'], math.sqrt(2/fit_data['dof']), relative_error=False) + '\n'
    latex_str += latexify_and_round_value('P_{prob}', fit_data['p_val'], relative_error=False) + '\n'

    return latex_str

def latexify_nsigma(nsigma, val1=None, val2=None):
    values = ""
    if val1 is not None:
        if val2 is None:
            raise ValueError("Need both value names")
        values = f"({val1},\\:{val2})"
        
    return latexify_and_round_value("N_{\\sigma}" + values,  nsigma, relative_error=False)

def calculate_value_with_uncertainty(expr, val_dict):
    # Get symbols
    # Get uncertainties symbols
    # Separate val_dict into values dict and uncertainties dict
    # Combine them
    # Calculate expr, sub values dict into it
    # Calculate uncertainty formula, sub values and uncertainties combined into it

    # sympy.sqrt(sum([(expr.diff(s)*sympy.symbols(f"d{s.name}"))**2 for s in expr.free_symbols]))
    # sigma*sympy.sqrt(sum([(sympy.symbols(f"d{s.name}")/s)**2 for s in sigma.free_symbols]))
    # (d_sigma - d_sigma_2).simplify()
    pass
