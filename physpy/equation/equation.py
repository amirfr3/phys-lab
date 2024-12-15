import sympy
import uncertainties


def get_uncertainties(symbols):
    if hasattr(symbols, '__iter__'):
        return sympy.symbols([f"delta_{symbol.name}" for symbol in symbols])
    else:
        symbol = symbols
        return sympy.symbols(f"delta_{symbol.name}")


class UnsupportedOperationError(KeyError):
    pass


def calculate_indirect_error_sym(expr):
    assert expr.func is sympy.core.symbol.Symbol

    return get_uncertainties(expr)


def calculate_indirect_error_pow(expr):
    assert expr.func is sympy.core.symbol.Symbol

    # TODO: Recurse
    base, power = expr.args
    assert power > 1

    return sympy.Eq(
        get_uncertainties(expr.lhs),
        power * base ** (power-1)
    )


def calculate_indirect_error_mul(expr):
    assert expr.func is sympy.core.mul.Mul

    # TODO: Change Pow operations to absolute value
    e_unc = {e: calculate_indirect_error_formula(e).rhs for e in expr.args}

    return sympy.Eq(
        get_uncertainties(expr.lhs),
        expr.lhs * sympy.sqrt(sum([(e_unc[e]/e)**2 for e in expr.args]))
    )


# TODO: Maybe receive list of parameters as well
# Assumed to be equality
def calculate_indirect_error_formula(expr):
    # Basic version
    return sympy.Eq(
        get_uncertainties(expr.lhs),
        sympy.sqrt(sum(
            [(expr.rhs.diff(s)*get_uncertainties(s))**2 for s in expr.rhs.free_symbols]
        ))
    )

    # TODO: Handle integers
    OPS = {
        sympy.core.symbol.Symbol: calculate_indirect_error_sym,
        sympy.core.power.Pow: calculate_indirect_error_pow,
        sympy.core.mul.Mul: calculate_indirect_error_mul,
    }

    try:
        return OPS[expr.func](expr)

    except KeyError:
        raise UnsupportedOperationError("Unsupported operation in expression")


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
