import sympy
# import uncertainties


def _get_uncertainty(symbol):
    return sympy.symbols(f"Delta_{symbol.name}")
    # return sympy.parsing.latex.parse_latex(f"\\Delta{{{symbol.name}}}")


def get_uncertainties(symbols):
    if hasattr(symbols, '__iter__'):
        return (_get_uncertainty(symbol) for symbol in symbols)

    # Singular
    return _get_uncertainty(symbols)


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
