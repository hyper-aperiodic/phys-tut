"""
LaTeX formatting helpers for generated numbers, so templates never render
'1x', '-1x', '+ -3x' or '\\frac{...}{1}'.

Templates build question text by slotting random integers into LaTeX. Pasting
a number in directly (f"{a}x") goes wrong whenever the number is 1, -1,
negative or a fraction. These helpers write the number the way a person would.

Use them whenever a generated number multiplies a variable or sits in a
fraction. Anything produced by SymPy's own latex() is already formatted
correctly and doesn't need them.
"""
from sympy import latex


def coef(n, var=''):
    """Coefficient n times var: coef(1,'x') -> 'x', coef(-1,'x') -> '-x', coef(3,'x') -> '3x'.
    n may be an int or a SymPy Rational.

    Further examples:
        coef(Rational(1, 2), 'x^2')  -> '\\frac{1}{2}x^2'
        coef(5)                      -> '5'   (no variable: just the number)
        coef(-1, r'\\sin')           -> '-\\sin'
    """
    # A bare number (no variable to attach to) is printed as-is, including 1.
    if not var:
        return latex(n)
    # Unit coefficients are never written in maths: 1x -> x, -1x -> -x.
    if n == 1:
        return var
    if n == -1:
        return f'-{var}'
    # latex() rather than str() so Rationals come out as \frac{p}{q}.
    return f'{latex(n)}{var}'


def poly(*terms):
    """Join (coeff, var) pairs into a LaTeX sum, handling signs and unit coefficients.
    E.g. poly((-3,'x^2y'),(1,'yz')) -> '-3x^2y + yz'

    - Terms with a zero coefficient are dropped entirely.
    - The first term carries its sign directly ('-3x'). Later terms are joined
      with ' + ' or ' - ', so you never get '+ -3x'.
    - If every term is zero the result is '0'.
    """
    parts = []
    for n, v in terms:
        if n == 0:
            continue
        # Format the magnitude only; the sign is handled by the joiner below.
        t = coef(abs(n), v)
        parts.append(f'-{t}' if (n < 0 and not parts) else   # first term, negative
                     f'- {t}' if n < 0 else                   # later term, negative
                     f'+ {t}' if parts else t)                # later term positive / first term positive
    return ' '.join(parts) or '0'


def frac(num, den):
    """num/den as LaTeX, dropping a denominator of 1.

    E.g. frac(r'n\\pi x', 2) -> '\\frac{n\\pi x}{2}', but frac(r'n\\pi x', 1) -> 'n\\pi x'.
    num and den may be strings or numbers; they are inserted as-is.
    """
    if den == 1:
        return str(num)
    return rf'\frac{{{num}}}{{{den}}}'
