"""
ODE question templates.
Covers: separable (f(x), g(y), f(x)g(y), homogeneous f(x,y)), linear first-order, 2nd-order constant coefficients, Laplace.

Templates in this module, in sidebar order:
    Classifying ODEs              state the order, linear or not, homogeneous or not
    Separable: dy/dx = f(x)       integrate directly
    Separable: dy/dx = g(y)       divide by g(y), integrate, exponentiate
    Separable: dy/dx = f(x)g(y)   separate the variables, integrate both sides
    Separable: dy/dx = f(x, y)    homogeneous: substitute y = ux, which makes it separable
    2nd-Order Homogeneous         a y'' + b y' + c y = 0, via the characteristic equation
    2nd-Order Non-Homogeneous     y'' + b y' + c y = f(x), undetermined coefficients, split into
                                  the lecture's four cases for f(x): polynomial, constant,
                                  exponential, trigonometric (chosen on the main page)
    (Laplace Transforms           commented out at the bottom; not yet enabled)

Notation: derivatives are always shown in Leibniz form (dy/dx, d^2y/dx^2), never
primes, and trig functions always have bracketed arguments, e.g. sin(x).
"""
import random

from sympy import (symbols, Function, dsolve, solve, diff, simplify, latex,
                   exp, sin, cos, sqrt, log, integrate, Integer, Rational, I, factor, expand, Symbol,
                   laplace_transform, inverse_laplace_transform, apart, Poly, Mul)
from ..core.types import QuestionTemplate, Step
from ..core.fmt import coef, poly, frac

# x is the independent variable; s is the Laplace variable (Laplace template only).
x, s = symbols('x s')
# y as an unknown *function* y(x), for SymPy's dsolve (the 2nd-order templates).
y = Function('y')


# y and its derivatives in Leibniz notation, indexed by order
# (_DY[0] = y, _DY[1] = dy/dx, _DY[2] = d^2y/dx^2, _DY[3] = d^3y/dx^3)
_DY = ("y", r"\frac{dy}{dx}", r"\frac{d^2y}{dx^2}", r"\frac{d^3y}{dx^3}")


def _ode_str(a, b, c):
    """Format a y'' + b y' + c y (in Leibniz notation), handling signs and unity coefficients.

    E.g. _ode_str(1, -3, 2) -> '\\frac{d^2y}{dx^2} - 3\\frac{dy}{dx} + 2y'
    (no "= 0": the caller appends the right-hand side).
    """
    return poly((a, _DY[2]), (b, _DY[1]), (c, _DY[0]))


# ── Classifying ODEs ──────────────────────────────────────────────────────────
# Build a random ODE from a term bank; each term is (int coeff, LaTeX body, SymPy
# expr in D = y, dy/dx, d2y/dx2, d3y/dx3 as plain symbols). Order, linearity and homogeneity
# are then read off the SymPy expressions.
#
# The random parameters decide what the answer should be (order n, linear or
# not, homogeneous or not), and the builder picks terms to match. The solver
# then re-derives the classification from the SymPy expressions, independently
# of those flags, and raises an error if the two disagree. So the answer shown
# is always the one the maths gives, not just the one intended.
#
# Why plain symbols instead of y(x) and its derivatives: treating y, y', y'' as
# independent symbols d0, d1, d2 makes the tests simple:
#   - order:      the highest k for which some term contains d_k
#   - linear:     every second partial derivative with respect to the d's is zero
#                 (y^2, y y', sin(y) and (y')^2 all fail this)
#   - homogeneous: every term contains at least one d_k (no pure-x forcing term)

D = symbols('d0:4')   # d0..d3 stand for y, dy/dx, d^2y/dx^2, d^3y/dx^3
_ORDINALS = ("zeroth", "first", "second", "third")   # indexed by order, for the summary line

# Coefficient functions a linear term may carry: (LaTeX prefix, SymPy expression).
# x-dependent coefficients like x^2 or sin(x) keep the term linear in y, which
# is a common point of confusion and worth practising.
_X_COEFFS = (("", Integer(1)),("x", x), ("x^2", x**2), (r"\sin(x)", sin(x)), ("e^{x}", exp(x)))

# Nonlinear terms that aren't squared derivatives: (order, LaTeX body, SymPy expr).
# `order` is the highest derivative the term contains, so a term is only used
# when it doesn't exceed the ODE's intended order.
_NONLINEAR = (  # (order, body, expr); squared derivatives come from _squared()
    (0, "y^2", D[0]**2), (0, r"\sin(y)", sin(D[0])), (0, "e^{y}", exp(D[0])),
    (1, "y" + _DY[1], D[0] * D[1]),
    (2, "y" + _DY[2], D[0] * D[2]),
)

# Forcing terms: functions of x alone, which make an ODE inhomogeneous.
# ("", 1) is a constant forcing term; its value comes from the random coefficient.
_FORCING = (("x^2", x**2), (r"\sin(x)", sin(x)), ("e^{x}", exp(x)), (r"\cos(2x)", cos(2 * x)), ("", Integer(1)))

# Integer coefficients for non-leading terms: -4..4 excluding 0.
_NONZERO = [c for c in range(-4, 5) if c != 0]


def _squared(k):
    """(d^k y/dx^k)^2 as a (body, expr) pair.

    Squared derivatives are there to test the difference between order and
    degree: (dy/dx)^2 is nonlinear, but the ODE is still first order.
    """
    return r"\left(" + _DY[k] + r"\right)^2", D[k]**2


def _classify_build(p):
    """2 or 3 terms in total: the highest derivative, plus a nonlinear term and/or
    forcing term as required, topped up with lower-order linear terms.

    p holds the random parameters:
        n       order of the ODE (1-3)
        linear  1 = build a linear ODE, 0 = include a nonlinear term
        homog   1 = homogeneous (no forcing term), 0 = add a forcing term
        s       seed for all the term choices below

    Returns (lhs, rhs, moved):
        lhs, rhs  lists of (int coeff, LaTeX body, SymPy expr) terms; rhs may be empty (= 0)
        moved     True if a y-term was moved to the right-hand side (a homogeneous ODE
                  whose right-hand side isn't zero; the solution notes mention it)

    The same p always gives the same ODE. That matters because prompt_fn and
    solver each call this separately and must agree.
    """
    # A private RNG seeded from p["s"], so the term choices are reproducible.
    rng = random.Random(p["s"])
    n = p["n"]
    n_terms = rng.choice([2, 3])   # total number of terms, counting both sides

    def lin_term(k, coeffs):
        """A linear term: random coefficient * (1, x, x^2, sin(x) or e^x) * d^k y/dx^k."""
        cname, cexpr = rng.choice(_X_COEFFS)
        return (rng.choice(coeffs), cname + _DY[k], cexpr * D[k])

    # Leading term: always contains the n-th derivative, which fixes the order.
    # Half of nonlinear ODEs square the highest derivative itself: order is unchanged
    squared_lead = not p["linear"] and rng.random() < 0.5
    if squared_lead:
        lhs = [(rng.choice([1, 1, 2]), *_squared(n))]   # small positive coefficient, 1 most likely
    else:
        lhs = [lin_term(n, [1, 1, 2])]

    # A nonlinear ODE whose leading term isn't squared gets one nonlinear term
    # elsewhere: half the time a squared lower (or equal) derivative, otherwise
    # one from _NONLINEAR that doesn't exceed order n.
    extra = []
    if not p["linear"] and not squared_lead:
        if rng.random() < 0.5:
            body, e = _squared(rng.randint(1, n))
        else:
            _, body, e = rng.choice([t for t in _NONLINEAR if t[0] <= n])
        extra.append((rng.choice(_NONZERO), body, e))

    # Use the remaining term budget for lower-order linear terms (orders 0..n-1,
    # each used at most once, highest first). The forcing term, if any, counts
    # towards the budget.
    n_lower = n_terms - len(lhs) - len(extra) - (0 if p["homog"] else 1)
    lower = rng.sample(range(n), max(0, min(n_lower, n)))
    lhs += [lin_term(k, _NONZERO) for k in sorted(lower, reverse=True)]
    # Put the nonlinear term at a random position after the leading term, so it
    # isn't always in the same place.
    for t in extra:
        lhs.insert(rng.randint(1, len(lhs)), t)

    rhs, moved = [], False
    if not p["homog"]:
        # Inhomogeneous: the right-hand side is a forcing term (function of x only).
        body, e = rng.choice(_FORCING)
        rhs = [(rng.choice(_NONZERO), body, e)]
    elif len(lhs) > 1 and rng.random() < 0.5:   # move a y-term across: still homogeneous
        # Moving the last term to the right-hand side (sign flipped) gives a
        # homogeneous ODE with a non-zero right-hand side. A common trap is to
        # call it inhomogeneous because "the right-hand side isn't 0".
        c, body, e = lhs.pop()
        rhs, moved = [(-c, body, e)], True
    return lhs, rhs, moved


def _classify_ode_str(lhs, rhs):
    """LaTeX for 'lhs = rhs' from the term lists built above (an empty rhs is shown as 0)."""
    return (poly(*[(c, b) for c, b, _ in lhs]) + " = "
            + (poly(*[(c, b) for c, b, _ in rhs]) if rhs else "0"))


def _classify_solver(p):
    """Rebuild the ODE, classify it from its SymPy expressions, and explain each part."""
    lhs, rhs, moved = _classify_build(p)
    terms = lhs + rhs   # which side a term is on doesn't affect any of the tests below

    # Order: the highest k with d^k y/dx^k present in any term.
    order = max(k for k in range(4) if any(e.has(D[k]) for _, _, e in terms))
    # Nonlinear terms: a term is linear in (y, y', y'', y''') exactly when all its
    # second partial derivatives with respect to those symbols vanish.
    # coef(c, b) gives the term's display form, used in the notes.
    nonlinear = [coef(c, b) for c, b, e in terms
                 if any(diff(e, u, v) != 0 for u in D for v in D)]
    # Forcing terms: terms that contain no y or derivative of y at all.
    free = [coef(c, b) for c, b, e in terms if not any(e.has(d) for d in D)]
    linear, homog = not nonlinear, not free
    # Safety net: the classification derived from the maths must match what the
    # builder was asked to produce. If not, the builder has a bug, so fail
    # loudly rather than show a wrong answer.
    if (linear, homog) != (bool(p["linear"]), bool(p["homog"])):
        raise ValueError(f"classification mismatch for {p}")

    # Explanatory notes shown under each step.
    if linear:
        lin_note = (r"$y$ and its derivatives appear only to the first power, never multiplied "
                    r"together or inside other functions, and their coefficients depend on $x$ only.")
    else:
        # Name the first offending term, so the student can see why.
        lin_note = (rf"The term ${nonlinear[0]}$ is nonlinear in $y$. In a linear ODE, $y$ and its "
                    r"derivatives appear only to the first power, multiplied by functions of $x$.")

    if homog:
        hom_note = r"Every term contains $y$ or a derivative of $y$."
        if moved:
            hom_note += (r" The right-hand side contains $y$ too: moving it across leaves zero on the "
                         r"right.")
    else:
        hom_note = rf"The term ${free[0]}$ contains no $y$."
    if not linear:
        # Homogeneity is mainly a linear-ODE idea; flag that for nonlinear ODEs.
        hom_note += r" (Strictly, homogeneous/inhomogeneous is mostly used to describe linear ODEs.)"

    # If any derivative appears squared, add a note to the order step that order
    # counts the highest derivative, not the power it's raised to.
    squared = [k for k in range(1, 4) if any(e == D[k]**2 for _, _, e in terms)]
    order_note = ""
    if squared:
        order_note = (r"The derivative $" + _DY[max(squared)] + r"$ appears squared, but the order "
                      r"counts the highest derivative, not the power it is raised to.")

    lin_word = "linear" if linear else "nonlinear"
    hom_word = "homogeneous" if homog else "inhomogeneous"
    return [
        Step("Order: the highest derivative that appears",
             rf"{_DY[order]} \implies \text{{order }} {order}", order_note),
        Step("Linear or nonlinear?",
             rf"\text{{{lin_word.capitalize()}}}", lin_note),
        Step("Homogeneous or inhomogeneous?",
             rf"\text{{{hom_word.capitalize()}}}", hom_note),
        Step("Classification",
             rf"\text{{{_ORDINALS[order].capitalize()}-order, {lin_word}, {hom_word} ODE}}"),
    ]


classify_template = QuestionTemplate(
    id="ode-classify",
    topic="ODEs",
    subtopic="Classifying ODEs",
    difficulty=1,
    # The ODE is shown as centred display maths ($$ on their own lines).
    # [:2] drops the `moved` flag; the prompt only needs the two sides.
    prompt_fn=lambda p: (
        "Classify the ODE\n\n$$\n" + _classify_ode_str(*_classify_build(p)[:2]) + "\n$$\n\n"
        "State its order, whether it is linear, and whether it is homogeneous."
    ),
    solver=_classify_solver,
    # linear/homog are 0/1 flags (so each case comes up about half the time);
    # s seeds the term choices, giving many distinct ODEs per (n, linear, homog).
    parameters={"n": (1, 3), "linear": (0, 1), "homog": (0, 1), "s": (0, 99999)},
)


# ── Separable First-Order ─────────────────────────────────────────────────────
# Four types: dy/dx = f(x), g(y), f(x)g(y), and f(x,y) homogeneous (y = ux).
#
# Every solver here computes its answer with SymPy and then checks it with
# _check() (differentiate the solution and substitute it back into the ODE)
# before returning any steps.

Y = Symbol('y')          # y as a plain symbol, for writing right-hand sides
A, C = symbols('A C')    # arbitrary constants

# LaTeX snippets reused below. Logs are always written ln|x| (with modulus),
# as is conventional when integrating 1/x. SymPy itself would write log(x).
_LN_ABS_X = r'\ln|x|'
_X_LN_ABS_X = r'x\ln|x|'
_X2_LN_ABS_X = r'x^2\ln|x|'
_Y_OVER_X = r'\frac{y}{x}'


def _check(ysol, rhs):
    """SymPy verification that y = ysol satisfies dy/dx = rhs(x, y).

    ysol is an expression in x (and the arbitrary constants A or C); rhs is an
    expression in x and the symbol Y. Raises ValueError if
    d(ysol)/dx - rhs(x, ysol) doesn't simplify to 0. The generator turns that
    into "Generation failed", so a wrong solution is never shown.
    """
    if simplify(diff(ysol, x) - rhs.subs(Y, ysol)) != 0:
        raise ValueError(f"y = {ysol} does not satisfy dy/dx = {rhs}")


def _sep_prompt(rhs_latex):
    """Prompt text shared by all four separable templates."""
    return rf"Find the general solution of $\dfrac{{dy}}{{dx}} = {rhs_latex}$"


def _signed_frac(n, num_var, den):
    """' + |n|num_var/den' or ' - ...' for appending to an expression.

    E.g. _signed_frac(-2, 'x', 'y') -> ' - \\frac{2x}{y}'. The sign goes outside
    the fraction, never inside it.
    """
    return f" {'-' if n < 0 else '+'} " + frac(coef(abs(n), num_var), den)


def _integrand(expr):
    """LaTeX for expr inside an integral, bracketed if it starts with a minus.

    So '\\int \\left(-x^2\\right) dx' rather than the ambiguous '\\int -x^2 dx'.
    """
    body = latex(expr)
    return rf"\left({body}\right)" if expr.could_extract_minus_sign() else body


# Type 1: dy/dx = f(x)
# The right-hand side has no y, so the answer is just y = integral of f(x) dx + C.
# f(x) = a x^m + b * (cos, sin or exp)(k x): a power plus one transcendental
# term, so students practise integrating both kinds.

_TYPE1_FUNCS = (cos, sin, exp)   # chosen by the "kind" parameter (0, 1, 2)

def _type1_rhs(p):
    """f(x) as a SymPy expression."""
    return p["a"] * x**p["m"] + p["b"] * _TYPE1_FUNCS[p["kind"]](p["k"] * x)


def _type1_solver(p):
    f = _type1_rhs(p)
    F = integrate(f, x)   # antiderivative (SymPy adds no constant)
    _check(F + C, f)

    def _integral(t):
        """'\\int (term) dx = (its antiderivative)' for one term of f."""
        return rf"\int {_integrand(t)}\,dx = {latex(integrate(t, x))}"

    return [
        Step(r"The right-hand side depends on $x$ only, so this is type $\frac{dy}{dx} = f(x)$: integrate both sides with respect to $x$",
             rf"y = \int \left({latex(f)}\right) dx"),
        # Each term of f integrated separately, side by side.
        Step("Integrate term by term",
             r",\qquad ".join(_integral(t) for t in f.as_ordered_terms())),
        Step("General solution",
             rf"y = {latex(F)} + C"),
    ]


sep_fx_template = QuestionTemplate(
    id="ode-separable-fx",
    topic="ODEs",
    subtopic="Separable: dy/dx = f(x)",
    difficulty=1,
    prompt_fn=lambda p: _sep_prompt(latex(_type1_rhs(p))),
    solver=_type1_solver,
    # a: coefficient of x^m; m: power (1-3); b: coefficient of the trig/exp term;
    # k: frequency/rate inside it; kind: 0 = cos, 1 = sin, 2 = exp.
    parameters={"a": (-6, 6), "m": (1, 3), "b": (-4, 4), "k": (1, 3), "kind": (0, 2)},
    constraints=[lambda p: p["a"] != 0 and p["b"] != 0],
)


# Type 2: dy/dx = g(y), with g linear: dy/dx = a + by
# Divide by g(y) and integrate: (1/b) ln|a + by| = x + C. Exponentiating gives
# a + by = A e^{bx}, so y = -a/b + A e^{bx}. This covers exponential growth and
# decay (a = 0) and Newton's law of cooling (a != 0).

def _type2_solver(p):
    a, b = p["a"], p["b"]
    g = a + b * Y
    # Inside the log: just y when a = 0 (ln|y|, not ln|by|, which differs only by a constant).
    lhs = 'y' if a == 0 else latex(g)
    # Final answer, with any constant factor absorbed into A.
    ysol = Rational(-a, b) + A * exp(b * x)
    _check(ysol, g)

    ln_lhs = r'\ln|' + lhs + '|'
    steps = [
        Step(r"The right-hand side depends on $y$ only, so this is type $\frac{dy}{dx} = g(y)$: divide by $g(y)$ and integrate",
             rf"\int \frac{{dy}}{{{latex(g)}}} = \int dx"),
        # integral dy/(a + by) = (1/b) ln|a + by|
        Step("Integrate both sides",
             rf"{coef(Rational(1, b), ln_lhs)} = x + C"),
        # Multiply through by b and exponentiate. The +- from removing the
        # modulus, and e^{bC}, combine into one arbitrary constant A.
        Step("Exponentiate",
             rf"{lhs} = Ae^{{{latex(b * x)}}}",
             rf"where $A = \pm e^{{{coef(b, 'C')}}}$ is a new arbitrary constant."),
    ]
    # When a = 0 the previous step already reads y = A e^{bx}. Otherwise
    # rearrange for y explicitly.
    if a != 0:
        steps.append(Step("Solve for $y$", rf"y = {latex(ysol)}",
                          "" if b == 1 else rf"where $A$ has absorbed the factor ${latex(Rational(1, b))}$."))
    return steps


sep_gy_template = QuestionTemplate(
    id="ode-separable-gy",
    topic="ODEs",
    subtopic="Separable: dy/dx = g(y)",
    difficulty=1,
    prompt_fn=lambda p: _sep_prompt(latex(p["a"] + p["b"] * Y)),
    solver=_type2_solver,
    # a may be 0 (pure exponential); b must not be (otherwise there's no y on the right).
    parameters={"a": (-6, 6), "b": (-3, 3)},
    constraints=[lambda p: p["b"] != 0],
)


# Type 3: dy/dx = f(x)g(y), with f = a x^m and g in {y, y^2, 1/y}
# Separate: dy/g(y) = f(x) dx, then integrate both sides. The three choices of
# g give three different answer shapes:
#   g = y    ->  ln|y| = F + C     ->  y = A e^{F}
#   g = y^2  ->  -1/y  = F + C     ->  y = 1 / (C - F)
#   g = 1/y  ->  y^2/2 = F + C     ->  y^2 = 2F + C   (left in implicit form)
# where F is the antiderivative of f.

_TYPE3_G = (Y, Y**2, 1 / Y)                                  # g(y), chosen by "kind"
_TYPE3_DY = (r"\frac{dy}{y}", r"\frac{dy}{y^2}", r"y\,dy")   # dy / g(y), as written on screen

def _type3_solver(p):
    a, m, kind = p["a"], p["m"], p["kind"]
    f, g = a * x**m, _TYPE3_G[kind]
    F = integrate(f, x)   # = a x^{m+1} / (m+1)

    # Each g has its own left-hand-side integral and final rearrangement.
    # ysol is an explicit solution used only for the SymPy check.
    if kind == 0:    # dy/y = f dx  ->  ln|y| = F + C
        lhs = r"\ln|y|"
        ysol = A * exp(F)
        solve_step = Step("Exponentiate", rf"y = Ae^{{{latex(F)}}}",
                          r"where $A = \pm e^{C}$ is a new arbitrary constant.")
    elif kind == 1:  # dy/y^2 = f dx  ->  -1/y = F + C
        lhs = r"-\frac{1}{y}"
        # -1/y = F + C  =>  y = -1/(F + C) = 1/(-C - F); rename -C as C.
        ysol = 1 / (C - F)
        solve_step = Step("Solve for $y$", rf"y = \frac{{1}}{{{latex(C - F)}}}",
                          r"where $C$ has absorbed a minus sign.")
    else:            # y dy = f dx  ->  y^2/2 = F + C
        lhs = r"\frac{y^2}{2}"
        ysol = sqrt(2 * F + C)   # the positive branch, for the check
        solve_step = Step("Multiply by 2", rf"y^2 = {latex(2 * F)} + C",
                          r"where $C$ has absorbed the factor of 2.")
    _check(ysol, f * g)

    return [
        # Identify f and g explicitly: this is the skill being practised.
        Step(r"The right-hand side is a product of a function of $x$ and a function of $y$, so this is type $\frac{dy}{dx} = f(x)g(y)$",
             rf"f(x) = {latex(f)},\qquad g(y) = {latex(g)}"),
        Step(r"Separate the variables: $\dfrac{dy}{g(y)} = f(x)\,dx$",
             rf"\int {_TYPE3_DY[kind]} = \int {_integrand(f)}\,dx"),
        Step("Integrate both sides",
             rf"{lhs} = {latex(F)} + C"),
        solve_step,
    ]


sep_fxgy_template = QuestionTemplate(
    id="ode-separable-fxgy",
    topic="ODEs",
    subtopic="Separable: dy/dx = f(x)g(y)",
    difficulty=2,
    prompt_fn=lambda p: _sep_prompt(latex(p["a"] * x**p["m"] * _TYPE3_G[p["kind"]])),
    solver=_type3_solver,
    # a: coefficient; m: power of x (1-3); kind: 0 = y, 1 = y^2, 2 = 1/y.
    parameters={"a": (-4, 4), "m": (1, 3), "kind": (0, 2)},
    constraints=[lambda p: p["a"] != 0],
)


# Type 4: dy/dx = f(x,y), homogeneous (a function of y/x only), via y = ux
#   kind 0: (ax + y)/x       = a + u    ->  y   = a x ln|x| + Cx
#   kind 1: (ax^2 + y^2)/(xy) = a/u + u  ->  y^2 = 2a x^2 ln|x| + Cx^2
#
# Method: substitute y = u(x) x, so dy/dx = u + x du/dx. Because f depends only
# on y/x = u, the ODE becomes u + x du/dx = F(u), i.e. x du/dx = F(u) - u, which
# is separable in u and x. In both kinds used here the u on each side cancels,
# leaving a one-line integral. (u, not v, to match the lecture notes.)
#
# NB: "homogeneous" here means "a function of y/x", which is a different meaning
# from "homogeneous" in the classification template (no forcing term).

def _type4_rhs_latex(a, kind):
    """The right-hand side as it appears in the question, e.g. (y - 3x)/x."""
    if kind == 0:
        return rf"\frac{{{poly((1, 'y'), (a, 'x'))}}}{{x}}"
    return rf"\frac{{{poly((1, 'y^2'), (a, 'x^2'))}}}{{xy}}"


def _type4_solver(p):
    a, kind = p["a"], p["kind"]
    # Each kind needs its own LaTeX for every stage of the working:
    #   in_yx       the right-hand side rewritten in terms of y/x
    #   in_u        the same after substituting u = y/x
    #   reduced     the separable equation left once the u's cancel
    #   integrated  after integrating both sides
    #   back        the solution after substituting u = y/x back
    # The SymPy check verifies the final answer against the original ODE.
    if kind == 0:
        rhs = (a * x + Y) / x
        in_yx = f"{_Y_OVER_X} {'-' if a < 0 else '+'} {abs(a)}"
        in_u = f"u {'-' if a < 0 else '+'} {abs(a)}"
        reduced = rf"x\frac{{du}}{{dx}} = {a} \implies \int du = \int {_integrand(a / x)}\,dx"
        integrated = rf"u = {coef(a, _LN_ABS_X)} + C"
        back = rf"y = {coef(a, _X_LN_ABS_X)} + Cx"
        _check(a * x * log(x) + C * x, rhs)
    else:
        rhs = (a * x**2 + Y**2) / (x * Y)
        in_yx = _Y_OVER_X + _signed_frac(a, 'x', 'y')
        in_u = "u" + _signed_frac(a, '', 'u')
        reduced = rf"x\frac{{du}}{{dx}} = {latex(a / Symbol('u'))} \implies \int u\,du = \int {_integrand(a / x)}\,dx"
        integrated = rf"\frac{{u^2}}{{2}} = {coef(a, _LN_ABS_X)} + C"
        back = rf"y^2 = {coef(2 * a, _X2_LN_ABS_X)} + Cx^2"
        # Check using the positive branch y = sqrt(...); log(x) assumes x > 0, which is fine for the check.
        _check(sqrt(2 * a * x**2 * log(x) + C * x**2), rhs)

    steps = [
        Step(r"Not separable as written, but the right-hand side is a function of $y/x$ only:",
             rf"\frac{{dy}}{{dx}} = {in_yx}"),
        Step(r"Substitute $y = u(x)x$. Because we're lazy we can write $u(x)$ as just $u$. Using the chain rule we know that: $\dfrac{dy}{dx} = u + x\dfrac{du}{dx}$."
             "\n\n"   # paragraph break (the label is Markdown)
             r"So, sub that as the left hand side of the equation we want to solve above ($\dfrac{dy}{dx}$), and then sub $y=ux$ for $y$ into the right hand side:",
             rf"u + x\frac{{du}}{{dx}} = {in_u}"),
        Step(r"Cancel $u$: the equation in $u$ and $x$ is now separable",
             reduced),
        Step("Integrate both sides",
             integrated),
        Step(r"Back-substitute $u = y/x$ (multiply through by $x$)" if kind == 0 else
             r"Back-substitute $u = y/x$ (multiply through by $2x^2$)",
             back,
             "" if kind == 0 else r"where $C$ has absorbed the factor of 2."),
    ]
    return steps


sep_fxy_template = QuestionTemplate(
    id="ode-separable-fxy-homogeneous",
    topic="ODEs",
    subtopic="Separable: dy/dx = f(x, y)",
    difficulty=3,
    prompt_fn=lambda p: _sep_prompt(_type4_rhs_latex(p["a"], p["kind"])),
    solver=_type4_solver,
    # kind 0 = (y + ax)/x, kind 1 = (y^2 + ax^2)/(xy).
    parameters={"a": (-4, 4), "kind": (0, 1)},
    constraints=[lambda p: p["a"] != 0],
)


# ── Second-Order, Constant Coefficients: shared machinery ─────────────────────
# The homogeneous subtopic is split into three root cases, which the student
# picks on the main page (the non-homogeneous one picks its root case at random):
#
#   case                    characteristic roots   discriminant   y_c (homogeneous solution)
#   Distinct real roots     m1 != m2, both real    > 0            C1 e^{m1 x} + C2 e^{m2 x}
#   Repeated root           m1 = m2 = m            = 0            (C1 + C2 x) e^{m x}
#   Complex roots           alpha +- beta i        < 0            e^{alpha x}(C1 cos(beta x) + C2 sin(beta x))
#
# Each case becomes its own QuestionTemplate (built by the factories further
# down), sharing the subtopic but with a different `case` label. The app's case
# selector then just filters templates by that label.
#
# Equations are generated *from their roots*, not from random coefficients, so
# every question in a case has clean roots (integers, or alpha +- beta i with
# integer alpha and beta), with no surds. The monic characteristic polynomial is
#   (m - m1)(m - m2) = m^2 - (m1 + m2) m + m1 m2,
# so the ODE's coefficients follow directly from the chosen roots.
#
# In the homogeneous subtopic the student is told the case up front, and the
# first worked steps use the discriminant to confirm it, then solve. Each solver also checks with SymPy
# that the discriminant really has the expected sign, that the roots are right,
# and that the final answer satisfies the ODE.

m = symbols('m')                   # variable of the characteristic (auxiliary) equation, m as in the lecture notes
C1, C2 = symbols('C_1 C_2')        # arbitrary constants in the general solution


def _paren(n):
    """An integer in brackets for substituting into a formula, e.g. (-4) or (3)."""
    return f"({n})"


# The three cases. For each:
#   params, constraints  the random root parameters and what's excluded
#   bc(p)                (b, c) of the monic characteristic polynomial m^2 + b m + c
#   roots(p)             the two roots (one entry for a repeated root)
#   y_c(p)               the homogeneous solution with arbitrary constants C1, C2
#   sign                 the expected comparison of the discriminant with 0
#   phrase               how the prompt describes the roots
#   disc_note            note under the discriminant step, confirming the case
#   disc_found           the same note when the case wasn't given in the question
#   form_note            note under the general-solution step, giving the standard form
_ROOT_CASES = [
    dict(
        name="Distinct real roots",
        slug="distinct",
        # Two different non-zero integer roots; p < q stops the same pair coming up in both orders.
        params={"p": (-4, 4), "q": (-4, 4)},
        constraints=[lambda p: p["p"] != 0 and p["q"] != 0 and p["p"] < p["q"]],
        bc=lambda p: (-(p["p"] + p["q"]), p["p"] * p["q"]),
        roots=lambda p: [Integer(p["p"]), Integer(p["q"])],
        y_c=lambda p: C1 * exp(p["p"] * x) + C2 * exp(p["q"] * x),
        sign=">",
        phrase="two distinct real roots",
        disc_note=r"$\Delta > 0$, confirming two distinct real roots.",
        disc_found=r"$\Delta > 0$, so there are two distinct real roots.",
        form_note=r"Two distinct real roots $m_1, m_2$ give $y = C_1e^{m_1x} + C_2e^{m_2x}$.",
    ),
    dict(
        name="Repeated root",
        slug="repeated",
        params={"p": (-4, 4)},
        constraints=[lambda p: p["p"] != 0],
        bc=lambda p: (-2 * p["p"], p["p"]**2),
        roots=lambda p: [Integer(p["p"])],
        y_c=lambda p: (C1 + C2 * x) * exp(p["p"] * x),
        sign="=",
        phrase="a repeated real root",
        disc_note=r"$\Delta = 0$, confirming a single repeated real root.",
        disc_found=r"$\Delta = 0$, so there is a single repeated real root.",
        form_note=(r"A repeated root $m$ gives $y = (C_1 + C_2x)e^{mx}$: the extra factor of $x$ "
                   r"supplies the second independent solution."),
    ),
    dict(
        name="Complex roots",
        slug="complex",
        # alpha = real part, beta = imaginary part. alpha != 0 keeps a damping
        # (or growth) factor e^{alpha x}, matching the form note below.
        params={"al": (-3, 3), "be": (1, 3)},
        constraints=[lambda p: p["al"] != 0],
        bc=lambda p: (-2 * p["al"], p["al"]**2 + p["be"]**2),
        roots=lambda p: [p["al"] - p["be"] * I, p["al"] + p["be"] * I],
        y_c=lambda p: exp(p["al"] * x) * (C1 * cos(p["be"] * x) + C2 * sin(p["be"] * x)),
        sign="<",
        phrase="a pair of complex roots",
        disc_note=r"$\Delta < 0$, confirming a complex conjugate pair of roots $m = \alpha \pm \beta i$.",
        disc_found=r"$\Delta < 0$, so there is a complex conjugate pair of roots $m = \alpha \pm \beta i$.",
        form_note=(r"Complex roots $\alpha \pm \beta i$ give $y = e^{\alpha x}\left(C_1\cos(\beta x) + "
                   r"C_2\sin(\beta x)\right)$: oscillations that decay or grow (unless $\alpha = 0$)."),
    ),
]


def _root_steps(A_, B_, C_, case, p, told=True):
    """Steps shared by both subtopics, for A_ y'' + B_ y' + C_ y = (...):
    characteristic equation, discriminant (confirming the case), and roots.

    Also checks, with SymPy, that the discriminant has the sign the case
    expects and that the roots match the ones the equation was built from.
    told=False words the discriminant step as finding the case rather than
    confirming one the question gave.
    """
    char_eq = A_ * m**2 + B_ * m + C_
    disc = B_**2 - 4 * A_ * C_
    rel = ">" if disc > 0 else "=" if disc == 0 else "<"
    roots = case["roots"](p)
    if rel != case["sign"] or set(solve(char_eq, m)) != set(roots):
        raise ValueError(f"{case['name']}: equation {char_eq} doesn't have the expected roots {roots}")

    # Discriminant with the numbers substituted, e.g. (-4)^2 - 4(1)(13) = -36 < 0.
    disc_latex = (rf"\Delta = b^2 - 4ac = {_paren(B_)}^2 - 4{_paren(A_)}{_paren(C_)} = {disc}"
                  + ("" if rel == "=" else rf" {rel} 0"))

    # Quadratic formula with the numbers substituted. sqrt(Delta) is an integer
    # for real roots, or an integer multiple of i for complex roots.
    if len(roots) == 1:
        roots_latex = rf"m = \frac{{-b}}{{2a}} = \frac{{{-B_}}}{{{2 * A_}}} = {latex(roots[0])}"
    else:
        roots_latex = (rf"m = \frac{{-b \pm \sqrt{{\Delta}}}}{{2a}} = \frac{{{-B_} \pm {latex(sqrt(Integer(disc)))}}}{{{2 * A_}}}"
                       rf" \implies m_1 = {latex(roots[0])},\quad m_2 = {latex(roots[1])}")

    return [
        Step(r"Try $y = e^{mx}$: substituting gives the characteristic (auxiliary) equation",
             rf"{latex(char_eq)} = 0"),
        Step(r"Confirm the case with the discriminant $\Delta = b^2 - 4ac$" if told else
             r"Find the type of roots with the discriminant $\Delta = b^2 - 4ac$",
             disc_latex, case["disc_note"] if told else case["disc_found"]),
        Step("Find the roots",
             roots_latex),
    ]


# ── Second-Order Homogeneous ──────────────────────────────────────────────────
# a y'' + b y' + c y = 0. Trying y = e^{mx} gives the characteristic equation
# a m^2 + b m + c = 0, and the type of its roots fixes the form of the solution.
# The leading coefficient a (1 or 2) is a common factor on top of the monic
# polynomial built from the roots, so the roots stay clean even when a = 2.

def _make_second_order_solver(case):
    def solver(p):
        b, c = case["bc"](p)
        A_, B_, C_ = p["a"], p["a"] * b, p["a"] * c   # full ODE coefficients
        y_c = case["y_c"](p)
        # SymPy check: the general solution must satisfy the ODE identically.
        if simplify(A_ * diff(y_c, x, 2) + B_ * diff(y_c, x) + C_ * y_c) != 0:
            raise ValueError(f"{case['name']}: y = {y_c} does not satisfy the ODE")
        return _root_steps(A_, B_, C_, case, p) + [
            Step("Write down the general solution for this case",
                 rf"y = {latex(y_c)}", case["form_note"]),
        ]
    return solver


def _make_second_order_template(case):
    def prompt(p):
        b, c = case["bc"](p)
        return ("Find the general solution of\n\n$$\n"
                + _ode_str(p["a"], p["a"] * b, p["a"] * c) + " = 0\n$$\n\n"
                + f"Its characteristic equation has **{case['phrase']}**: confirm this using the "
                  "discriminant, then find the general solution.")

    return QuestionTemplate(
        id=f"ode-2nd-order-homogeneous-{case['slug']}",
        topic="ODEs",
        subtopic="2nd-Order Homogeneous",
        case=case["name"],
        difficulty=2,
        prompt_fn=prompt,
        solver=_make_second_order_solver(case),
        parameters={"a": (1, 2), **case["params"]},
        constraints=case["constraints"],
    )


second_order_templates = [_make_second_order_template(case) for case in _ROOT_CASES]


# ── 2nd-Order Non-Homogeneous (Undetermined Coefficients) ─────────────────────
# d2y/dx2 + b dy/dx + c y = f(x), worked as in PHYS108 lecture 16:
#   1. solve the complementary equation (f(x) = 0) for y_c, as in the homogeneous subtopic
#   2. guess a particular solution y_p from the form of f(x), with unknown coefficients
#   3. differentiate y_p, sub into the equation, and equate coefficients
#   4. y = y_c + y_p
# Split into the lecture's four cases for f(x), which the student picks on the main page:
#   Polynomial     f = a degree 1-2 polynomial   y_p = a polynomial of the same degree
#   Constant       f = k                         y_p = C; or Cx when there's no y term (c = 0)
#   Exponential    f = k e^{gamma x}             y_p = Ae^{gamma x}, Axe^{gamma x} or Ax^2e^{gamma x},
#                                                as gamma equals neither, one or both roots
#   Trigonometric  f = k1 sin(gamma x) and/or k2 cos(gamma x)
#                                                y_p = A cos(gamma x) + B sin(gamma x), times x when
#                                                the roots are +-gamma i (y_c has the same form)
# The root type of the complementary equation is random within each case. A
# builder per case picks the equation and the trial function; the shared solver
# then does the substitution and coefficient matching with SymPy, and checks
# that y_c + y_p satisfies the equation.

_ROOT_CASE = {case["slug"]: case for case in _ROOT_CASES}   # root cases by slug
_UA, _UB, _UC = symbols('A B C')   # unknown coefficients in y_p, named as in the lecture

# Placeholders for sin(gamma x), cos(gamma x) and e^{gamma x}, so the
# coefficient of each kind of term can be read off with Poly (see _coeffs).
_S, _CO, _E = symbols('S Co E')

# Coefficients for f(x): -5..5 excluding 0.
_RHS_COEFFS = [k for k in range(-5, 6) if k != 0]


def _root_params(rng, kind, nonzero=True):
    """(root case, its params) for a random complementary equation of the given
    kind ('distinct', 'repeated' or 'complex'). Real roots are in -3..3, and
    non-zero unless nonzero=False. Complex roots are alpha +- beta i with alpha
    in -2..2 (0 allowed: pure oscillation) and beta in 1..3."""
    case = _ROOT_CASE[kind]
    span = [v for v in range(-3, 4) if v or not nonzero]
    if kind == "distinct":
        p, q = sorted(rng.sample(span, 2))
        return case, {"p": p, "q": q}
    if kind == "repeated":
        return case, {"p": rng.choice(span)}
    return case, {"al": rng.randint(-2, 2), "be": rng.randint(1, 3)}


# Each builder returns a dict:
#   case, rp    the root case of the complementary equation and its params
#   f           the right-hand side
#   gamma       the gamma in e^{gamma x}, sin(gamma x), cos(gamma x) (None for polynomials)
#   trial       the trial y_p, in the unknowns
#   unknowns    the unknown coefficients in the trial
#   label, note the step explaining the choice of trial function

def _polynomial_rhs(rng):
    """Case 1: f is a polynomial of degree 1 or 2; try the same degree.
    No zero root, so there's always a y term and the trial works as it is."""
    case, rp = _root_params(rng, rng.choice(["distinct", "distinct", "repeated", "complex"]))
    deg = rng.choice([1, 2, 2])
    f = rng.choice([-3, -2, -1, 1, 1, 2, 3]) * x**deg + sum(rng.randint(-5, 5) * x**k for k in range(deg))
    unknowns = [_UA, _UB, _UC][:deg + 1]
    trial = sum(u * x**(deg - i) for i, u in enumerate(unknowns))
    return dict(case=case, rp=rp, f=f, gamma=None, trial=trial, unknowns=unknowns,
                label=(rf"$f(x) = {latex(f)}$ is a polynomial of degree {deg}, so try a polynomial "
                       rf"of degree {deg} with unknown coefficients:"),
                note="")


def _constant_rhs(rng):
    """Case 2: f is a constant. Half the time there's no y term (one root is 0),
    so y_p = C can't work and the trial is multiplied by x, as in the lecture."""
    k = rng.choice(_RHS_COEFFS)
    if rng.random() < 0.5:
        q = rng.choice([v for v in range(-3, 4) if v])
        return dict(case=_ROOT_CASE["distinct"], rp={"p": min(0, q), "q": max(0, q)},
                    f=Integer(k), gamma=None, trial=_UC * x, unknowns=[_UC],
                    label=(rf"$f(x) = {k}$ is a constant, so we'd try $y_p = C$. But there's no $y$ term, "
                           rf"and the derivatives of $y_p = C$ are zero, so subbing it in gives $0 = {k}$. "
                           r"Multiply the trial function by $x$:"),
                    note=r"Generally, if a trial function doesn't work, multiply it by $x$.")
    case, rp = _root_params(rng, rng.choice(["distinct", "repeated", "complex"]))
    return dict(case=case, rp=rp, f=Integer(k), gamma=None, trial=_UC, unknowns=[_UC],
                label=rf"$f(x) = {k}$ is a constant, so try a constant:", note="")


def _exponential_rhs(rng):
    """Case 3: f = k e^{gamma x}. The trial depends on how many roots equal gamma:
    none (half the time), one, or both (a repeated root)."""
    gamma, k = rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice(_RHS_COEFFS)
    match = rng.choice(["none", "none", "one", "both"])
    if match == "both":
        case, rp = _ROOT_CASE["repeated"], {"p": gamma}
    elif match == "one":
        other = rng.choice([v for v in range(-3, 4) if v != gamma])
        case, rp = _ROOT_CASE["distinct"], {"p": min(gamma, other), "q": max(gamma, other)}
    else:
        while True:
            case, rp = _root_params(rng, rng.choice(["distinct", "repeated", "complex"]))
            if gamma not in case["roots"](rp):
                break
    trial = _UA * x**{"none": 0, "one": 1, "both": 2}[match] * exp(gamma * x)
    reason = {"none": r"neither root equals $\gamma$", "one": r"one root equals $\gamma$",
              "both": r"both roots equal $\gamma$"}[match]
    f = k * exp(gamma * x)
    return dict(case=case, rp=rp, f=f, gamma=gamma, trial=trial, unknowns=[_UA],
                label=(rf"$f(x) = {latex(f)}$ is of the form $Ae^{{\gamma x}}$ with $\gamma = {gamma}$. "
                       rf"Here {reason}, so try:"),
                note=(r"The trial function depends on the roots: if $m_1, m_2 \neq \gamma$, try $Ae^{\gamma x}$; "
                      r"if $m_1 = \gamma$, $m_2 \neq \gamma$, try $Axe^{\gamma x}$; "
                      r"if $m_1 = m_2 = \gamma$, try $Ax^2e^{\gamma x}$."))


def _trig_rhs(rng):
    """Case 4: f = k1 sin(gamma x) and/or k2 cos(gamma x). A quarter of the time
    the roots are +-gamma i, so y_c has the same form as the usual trial and
    the trial is multiplied by x, as in the lecture's y'' + 4y = cos(2x)."""
    gamma = rng.randint(1, 3)
    which = rng.choice(["sin", "cos", "cos", "both"])
    f = ((rng.choice(_RHS_COEFFS) * sin(gamma * x) if which in ("sin", "both") else 0)
         + (rng.choice(_RHS_COEFFS) * cos(gamma * x) if which in ("cos", "both") else 0))
    base = _UA * cos(gamma * x) + _UB * sin(gamma * x)
    common = rf"$f(x) = {latex(f)}$ is sinusoidal with $\gamma = {gamma}$, so "
    if rng.random() < 0.25:
        return dict(case=_ROOT_CASE["complex"], rp={"al": 0, "be": gamma}, f=f, gamma=gamma,
                    trial=x * base, unknowns=[_UA, _UB],
                    label=(common + rf"we'd use $y_p = {latex(base)}$. But this is the same form as $y_c$, "
                           r"so multiply by $x$ and try:"),
                    note=(r"If the solutions of the complementary equation are of the same form as the "
                          r"particular solution, the particular solution must be modified."))
    while True:
        case, rp = _root_params(rng, rng.choice(["distinct", "repeated", "complex"]))
        if rp != {"al": 0, "be": gamma}:
            break
    return dict(case=case, rp=rp, f=f, gamma=gamma, trial=base, unknowns=[_UA, _UB],
                label=common + "try:", note="")


def _coeffs(expr, gamma):
    """{powers: (term, coefficient)} for expr written as a sum of terms in powers
    of x, sin(gamma x), cos(gamma x) and e^{gamma x}, with any A, B, C left in
    the coefficients. powers are the exponents of (x, sin, cos, exp), so sorting
    them in reverse puts the highest power of x first.
    E.g. -2Ax^2 + (2A - 2B)x + 1  ->  {(2,0,0,0): (x**2, -2A), (1,0,0,0): (x, 2A - 2B), (0,0,0,0): (1, 1)}."""
    e = expand(expr)
    if gamma is not None:
        e = e.subs({sin(gamma * x): _S, cos(gamma * x): _CO, exp(gamma * x): _E})
    gens = (x, _S, _CO, _E)
    back = {_S: sin(gamma * x), _CO: cos(gamma * x), _E: exp(gamma * x)} if gamma is not None else {}
    return {powers: (Mul(*[g**n for g, n in zip(gens, powers)]).subs(back), c)
            for powers, c in Poly(e, *gens).terms() if c != 0}


def _nonhom_build(p, rhs_case):
    """The equation and its full solution for one of the four cases, redrawn
    until the coefficients come out with small denominators (at most 60)."""
    rng = random.Random(p["s"])
    for _ in range(100):
        d = _RHS_BUILDERS[rhs_case](rng)
        b, c = d["case"]["bc"](d["rp"])
        trial = d["trial"]
        d1, d2 = expand(diff(trial, x)), expand(diff(trial, x, 2))
        lhs = d2 + b * d1 + c * trial
        lc, rc = _coeffs(lhs, d["gamma"]), _coeffs(d["f"], d["gamma"])
        # One equation per kind of term, highest power of x first; terms whose
        # coefficients cancel on both sides (e.g. -4A + 4A = 0) are left out.
        eqs = []    # (term, its coefficient on the left, on the right)
        for k in sorted(set(lc) | set(rc), reverse=True):
            term, l = lc.get(k, (None, 0))
            term, r_ = rc.get(k, (term, 0))
            if expand(l - r_) != 0:
                eqs.append((term, l, r_))
        sol = solve([l - r_ for _, l, r_ in eqs], d["unknowns"], dict=True)
        if not sol or any(u not in sol[0] for u in d["unknowns"]):
            continue
        sol = sol[0]
        if any(Rational(v).q > 60 for v in sol.values()):
            continue
        y_p = expand(trial.subs(sol))
        y_c = d["case"]["y_c"](d["rp"])
        # SymPy check: y_c + y_p must satisfy the full equation.
        y_full = y_c + y_p
        if simplify(diff(y_full, x, 2) + b * diff(y_full, x) + c * y_full - d["f"]) != 0:
            raise ValueError(f"{rhs_case}: y = {y_full} does not satisfy the equation")
        d.update(b=b, c=c, d1=d1, d2=d2, lc=lc, eqs=eqs, sol=sol, y_p=y_p, y_c=y_c)
        return d
    raise RuntimeError(f"no suitable {rhs_case} equation found")


_RHS_BUILDERS = {"Polynomial": _polynomial_rhs, "Constant": _constant_rhs,
                 "Exponential": _exponential_rhs, "Trigonometric": _trig_rhs}


def _signed_join(parts):
    """Join LaTeX terms with ' + ', or ' - ' when a term starts with a minus."""
    out = []
    for s_ in parts:
        if out and s_.startswith('-'):
            out.append(f"- {s_[1:].strip()}")
        else:
            out.append(f"+ {s_}" if out else s_)
    return ' '.join(out) or '0'


def _collected_term(coeff, term):
    """LaTeX for coeff times term, e.g. '-2Ax^2' or '(2A - 2B)x'; a constant term is just its coefficient."""
    if term == 1:
        return latex(coeff)
    if coeff.is_Add:
        return rf"\left({latex(coeff)}\right){latex(term)}"
    return latex(coeff * term)


def _make_nonhom_solver(rhs_case):
    def solver(p):
        d = _nonhom_build(p, rhs_case)
        b, c, f, trial = d["b"], d["c"], d["f"], d["trial"]

        def br(e):
            """e in brackets when it's a sum or negative, as in 2(2Ax + B) or -5(-2Ae^{-2x});
            a single positive term as is, as in -3C."""
            return rf"\left({latex(e)}\right)" if e.is_Add or e.could_extract_minus_sign() else latex(e)

        # Sub into the equation, each derivative in brackets as in the lecture;
        # a derivative that is zero is left out.
        subbed = poly((1 if d["d2"] != 0 else 0, br(d["d2"])), (b if d["d1"] != 0 else 0, br(d["d1"])),
                      (c, br(trial)))
        collected = _signed_join([_collected_term(co, t) for _, (t, co) in sorted(d["lc"].items(), reverse=True)])
        equations = r" \\ ".join(
            rf"{latex(l)} = {latex(r_)} \qquad " +
            (r"\text{(constants)}" if t == 1 else rf"\text{{(coefficients of }} {latex(t)}\text{{)}}")
            for t, l, r_ in d["eqs"])
        values = r",\quad ".join(f"{latex(u)} = {latex(d['sol'][u])}" for u in d["unknowns"])
        # y_p written as the lecture does, number in front of each term:
        # -\frac{1}{2}x^2 - \frac{1}{2}x - \frac{3}{4}, not -x^2/2 - x/2 - 3/4.
        # The trial is linear in the unknowns, so d(trial)/dA is the term A multiplies.
        yp_terms = [(d["sol"][u], diff(trial, u)) for u in d["unknowns"]]
        yp_latex = poly(*[(v, "" if t == 1 else latex(t)) for v, t in yp_terms])

        return [
            Step(r"First solve the complementary equation ($f(x) = 0$)",
                 rf"{_ode_str(1, b, c)} = 0"),
        ] + _root_steps(1, b, c, d["case"], d["rp"], told=False) + [
            Step("This gives the solution to the complementary equation",
                 rf"y_c = {latex(d['y_c'])}", d["case"]["form_note"]),
            Step(d["label"], rf"y_p = {latex(trial)}", d["note"]),
            Step("Get the derivatives of the particular solution",
                 rf"\frac{{dy_p}}{{dx}} = {latex(d['d1'])}, \qquad \frac{{d^2y_p}}{{dx^2}} = {latex(d['d2'])}"),
            Step("Sub into the original equation, and collect the terms",
                 rf"{subbed} = {latex(f)}"
                 # The collected line, unless it's the same as the line above (e.g. -3C = 1).
                 + ("" if subbed.replace(" ", "") == collected.replace(" ", "")
                    else rf" \\ \therefore\; {collected} = {latex(f)}")),
            Step(r"For this to hold for all $x$, the coefficients of each term must match",
                 rf"{equations} \\ \implies {values}"),
            Step("So the particular solution is",
                 rf"y_p = {yp_latex}"),
            Step("Put the complementary and particular solutions together to get the general solution",
                 rf"y = y_c + y_p = {_signed_join([latex(d['y_c']), yp_latex])}"),
        ]
    return solver


# The four cases, in the lecture's order: (case name, id slug).
_NONHOM_CASES = [("Polynomial", "polynomial"), ("Constant", "constant"),
                 ("Exponential", "exponential"), ("Trigonometric", "trig")]

# Shown on the main page before a case is picked (plain text: it sits inside HTML, so no $ maths).
_NONHOM_INTRO = ("Choose the type of right-hand side, f(x), you want to work with above. "
                 "Solve the complementary equation (f(x) = 0) first, then use the form of f(x) to guess a "
                 "particular solution, and add the two together for the general solution.")


def _make_nonhom_template(rhs_case, slug):
    def prompt(p):
        d = _nonhom_build(p, rhs_case)
        return ("Find the general solution of\n\n$$\n"
                + _ode_str(1, d["b"], d["c"]) + f" = {latex(d['f'])}" + "\n$$")

    return QuestionTemplate(
        id=f"ode-2nd-order-nonhom-{slug}",
        topic="ODEs",
        subtopic="2nd-Order Non-Homogeneous",
        case=rhs_case,
        case_intro=_NONHOM_INTRO,
        difficulty=3,
        prompt_fn=prompt,
        solver=_make_nonhom_solver(rhs_case),
        parameters={"s": (0, 99999)},
    )


nonhom_templates = [_make_nonhom_template(name, slug) for name, slug in _NONHOM_CASES]


# # ── Laplace Transform ─────────────────────────────────────────────────────────
# Not yet enabled. To switch it on, uncomment this block and add laplace_template
# to ALL_TEMPLATES below. Its strings still use prime notation and would need
# converting to Leibniz form (_DY / _ode_str) to match the other templates.

# def _laplace_solver(params):
#     a, b = params["a"], params["b"]
#     # Solve y'' + a*y' + b*y = 0, y(0)=1, y'(0)=0 via Laplace
#     # L{y''} = s²Y - s, L{y'} = sY - 1
#     Y = symbols('Y')
#     s_sym = symbols('s')

#     # Algebraic equation for Y(s)
#     # (s² + as + b)Y - s - a = 0
#     Y_s_expr = (s_sym + a) / (s_sym**2 + a * s_sym + b)
#     Y_s_simplified = apart(Y_s_expr, s_sym)

#     # Use dsolve with ICs for ground truth
#     ode = y(x).diff(x, 2) + a * y(x).diff(x) + b * y(x)
#     ics = {y(0): 1, y(x).diff(x).subs(x, 0): 0}
#     sol = dsolve(ode, y(x), ics=ics)

#     return [
#         Step("ODE with ICs",
#              rf"y'' + {a}y' + {b}y = 0,\quad y(0)=1,\quad y'(0)=0"),
#         Step("Apply Laplace transform",
#              rf"\mathcal{{L}}\{{y''\}} = s^2 Y - s \cdot 1 - 0,\quad \mathcal{{L}}\{{y'\}} = sY - 1"),
#         Step("Algebraic equation for $Y(s)$",
#              rf"(s^2 + {a}s + {b})Y = s + {a}"),
#         Step("Solve for $Y(s)$",
#              rf"Y(s) = {latex(Y_s_expr)}"),
#         Step("Partial fractions / complete the square",
#              latex(Y_s_simplified)),
#         Step("Invert via Laplace table",
#              latex(sol.rhs)),
#     ]


# laplace_template = QuestionTemplate(
#     id="ode-laplace-ivp",
#     topic="ODEs",
#     subtopic="Laplace Transforms",
#     difficulty=3,
#     prompt_fn=lambda p: (
#         rf"Solve using Laplace transforms: $y'' + {p['a']}y' + {p['b']}y = 0$, $y(0)=1$, $y'(0)=0$"
#     ),
#     solver=_laplace_solver,
#     parameters={"a": (1, 4), "b": (2, 8)},
#     constraints=[
#         lambda p: p["a"]**2 - 4 * p["b"] < 0,  # complex roots for interesting Laplace inversion
#     ],
# )


# Order here = order of subtopics in the sidebar dropdown.
ALL_TEMPLATES = [
    classify_template,
    sep_fx_template,
    sep_gy_template,
    sep_fxgy_template,
    sep_fxy_template,
    *second_order_templates,   # one per root case
    *nonhom_templates,         # one per case of f(x)
    # laplace_template,
]
