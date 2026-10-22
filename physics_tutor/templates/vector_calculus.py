"""
Vector calculus question templates.
All solvers return list[Step] with LaTeX strings.
SymPy is the sole mathematical authority.

Templates in this module (fields built from random terms):
    Gradient        grad of a scalar f with 2-3 terms, e.g. 3x^2z - xy + 2yz^2
    Divergence      div  of F, each component 1-2 terms
    Curl            curl of F, each component 1-2 terms
    Line integrals  of a scalar field phi(x, y) or a vector field E(x, y), along a
                    straight line or a parabola (four subtopics), worked by
                    eliminating a variable 
    Surface integrals  flux of a 3D vector field through a flat rectangle in a
                    coordinate plane

Each template has a single parameter s, which seeds a private RNG. A builder
function uses it to pick monomials (x^i y^j z^k) and integer coefficients, so
the fields vary in shape as well as in their numbers. Gradient, divergence and
curl fields sometimes swap one term for a sin, cos or exponential of two
variables (e.g. 2sin(xy)), which needs the chain rule. Line integrals stay
polynomial, because e.g. sin(xy) along y = x gives the non-elementary
integral of sin(x^2).

The same s always gives the same field, which matters because prompt_fn and
solver each call the builder separately and must agree.

Each solver builds the field with SymPy, differentiates or integrates it, and
writes each intermediate result into a Step. Every number on screen therefore
comes from SymPy, never from hand-written formulas.
"""
import random
from collections import namedtuple
from itertools import product, combinations
import numpy as np
from sympy import symbols, diff, latex, integrate, expand, degree, Mul, Integer, sin, cos, exp, lambdify
from ..core.types import QuestionTemplate, Step
from ..core.fmt import poly
from ..core.plot import field_path_map, surface_flux_sketch


# Cartesian coordinates.
x, y, z = symbols('x y z')
_VARS = (x, y, z)
_NAMES = ('x', 'y', 'z')

# LaTeX for the unit vectors, in component order.
_HATS = (r'\hat{i}', r'\hat{j}', r'\hat{k}')

# Integer coefficients for every term: -4..4 excluding 0.
_NONZERO = [c for c in range(-4, 5) if c != 0]

# Seed parameter shared by every template here (see the module docstring).
_SEED_PARAM = {"s": (0, 99999)}

# How many seeded redraws a builder tries before giving up on its checks. The
# checks reject only a small fraction of draws, so this is never reached in practice.
_MAX_DRAWS = 100


def _powers(n_vars, degrees, max_pow=2):
    """Every exponent tuple (i, j, k) with total degree in `degrees` and no
    exponent above max_pow. With n_vars=2 the z exponent is always 0.

    E.g. _powers(2, (2,)) -> [(0, 2, 0), (1, 1, 0), (2, 0, 0)]  i.e. y^2, xy, x^2
    """
    return [pw + (0,) * (3 - n_vars)
            for pw in product(range(max_pow + 1), repeat=n_vars)
            if sum(pw) in degrees]


# Monomial banks the builders draw from.
#   _MONOS_3D  degree 2-3 in x, y, z: x^2, xy, ..., x^2y, xyz, yz^2, ... (13 shapes)
#   _MONOS_2D  degree 1-2 in x, y:    x, y, x^2, xy, y^2                  (5 shapes, line integrals)
# No single exponent above 2, so the derivatives stay short.
_MONOS_3D = _powers(3, (2, 3))
_MONOS_2D = _powers(2, (1, 2))


# Non-polynomial terms: sin, cos or exp of a product of two variables, as
# (LaTeX body, SymPy expr) pairs: \sin(xy), \cos(xy), e^{xy}, \sin(xz), ... (9 shapes).
# Differentiating them needs the chain rule, e.g. d/dx sin(xy) = y cos(xy),
# and the factor that comes out front is the thing students forget.
_FUNCS_3D = [(body.format(n1 + n2), f(v1 * v2))
             for (n1, v1), (n2, v2) in combinations(zip(_NAMES, _VARS), 2)
             for body, f in ((r'\sin({})', sin), (r'\cos({})', cos), ('e^{{{}}}', exp))]

# Chance that a gradient, divergence or curl field includes one function term.
_P_FUNC = 0.25


def _mono(pw):
    """(LaTeX body, SymPy expr) for the monomial x^i y^j z^k.

    E.g. _mono((2, 0, 1)) -> ('x^2z', x**2*z)
    """
    body = ''.join(n if p == 1 else f'{n}^{p}' for n, p in zip(_NAMES, pw) if p)
    expr = Mul(*[v**p for v, p in zip(_VARS, pw)])
    return body, expr


def _sorted(pws):
    """Exponent tuples in the usual written order: highest degree first, then
    x before y before z (x^2y, xy^2, xyz, ..., x^2, xy, ...)."""
    return sorted(pws, key=lambda pw: (-sum(pw), [-e for e in pw]))


def _term(rng, pw, coeffs=_NONZERO):
    """A term (int coeff, LaTeX body, SymPy expr) with a random coefficient from coeffs."""
    return (rng.choice(coeffs), *_mono(pw))


def _expr(terms):
    """The SymPy expression for a list of terms."""
    return sum(c * e for c, _, e in terms)


def _terms_str(terms):
    """LaTeX for a list of terms, e.g. '3x^2z - xy'."""
    return poly(*[(c, b) for c, b, _ in terms])


def _field_str(comps):
    """LaTeX for a vector field from its component term lists (i, j[, k] order).

    A single-term component is written with its coefficient in front
    ('-3xy\\,\\hat{i}'); a multi-term one is bracketed ('(x^2 - 2yz)\\,\\hat{j}').
    poly() does the joining, so signs come out as ' - ' rather than '+ -'.
    """
    parts = []
    for terms, hat in zip(comps, _HATS):
        if len(terms) == 1:
            c, b, _ = terms[0]
            parts.append((c, rf'{b}\,{hat}'))
        else:
            parts.append((1, rf'\left({_terms_str(terms)}\right){hat}'))
    return poly(*parts)


def _add_func(rng, term_lists):
    """With probability _P_FUNC, swap one randomly chosen term (keeping its
    coefficient) for a term from _FUNCS_3D. Edits term_lists in place; the
    function term goes last in its list, after the polynomial terms."""
    if rng.random() < _P_FUNC:
        terms = rng.choice(term_lists)
        c, _, _ = terms.pop(rng.randrange(len(terms)))
        terms.append((c, *rng.choice(_FUNCS_3D)))


def _components(rng, monos, n_comps, coeffs=_NONZERO):
    """n_comps field components, each 1 or 2 distinct terms drawn from monos."""
    return [[_term(rng, pw, coeffs) for pw in _sorted(rng.sample(monos, rng.choice([1, 2])))]
            for _ in range(n_comps)]


# The del operator, written out under every gradient, divergence and curl question.
_NABLA_DEF = (r"\nabla = \frac{\partial }{\partial x}\hat{i} + \frac{\partial }{\partial y}\hat{j}"
              r" + \frac{\partial }{\partial z}\hat{k}")


def _prompt(what, symbol, field_lhs, field_str):
    """Question text shared by gradient, divergence and curl: the field and the
    del operator, each in display maths.

    E.g. _prompt("scalar field, $f$, find its gradient", r"\nabla f", "f(x,y,z)", "3x^2y - yz")
    """
    return (rf"For a {what} ${symbol}$ where:" "\n\n$$\n"
            rf"{field_lhs} = {field_str}" "\n$$\n\n"
            "and\n\n$$\n" + _NABLA_DEF + "\n$$")


def _partial(name, var, body, result):
    """Full working for one partial derivative, with the function written out so
    it can be read without scrolling back to the question:
        df/dx = d/dx (3x^2y + ...) = 6xy + ...
    name is the function's LaTeX name ('f', 'F_x'), body its LaTeX as written
    in the question, result the SymPy derivative."""
    return (rf"\frac{{\partial {name}}}{{\partial {var}}} = \frac{{\partial}}{{\partial {var}}}"
            rf"\left({body}\right) = {latex(result)}")


# ── Gradient ─────────────────────────────────────────────────────────────────
# grad f = (df/dx, df/dy, df/dz) for a scalar field f of 2-3 terms.

def _gradient_build(p):
    """The terms of f. All three partial derivatives are non-zero, and at
    least one term mixes variables, so there is a partial derivative that
    treats another variable as a constant."""
    rng = random.Random(p["s"])
    for _ in range(_MAX_DRAWS):
        terms = [_term(rng, pw) for pw in _sorted(rng.sample(_MONOS_3D, rng.choice([2, 3, 3])))]
        _add_func(rng, [terms])
        f = _expr(terms)
        covers_all = all(diff(f, v) != 0 for v in _VARS)
        mixed = any(len(e.free_symbols) >= 2 for _, _, e in terms)
        if covers_all and mixed:
            return terms
    raise RuntimeError("no suitable scalar field found")


def _gradient_solver(params):
    terms = _gradient_build(params)
    f = _expr(terms)
    f_str = _terms_str(terms)   # f as written in the question

    # The three partial derivatives become the i, j, k components of grad f.
    fx = diff(f, x)
    fy = diff(f, y)
    fz = diff(f, z)

    return [
        # Step 1: what the gradient is, and the plan (three partials, then assemble).
        Step(r"$f$ is a scalar, $\nabla$ is a vector, so $\nabla f$ is a vector defined by:",
             rf"\nabla f = \frac{{\partial f}}{{\partial x}}\hat{{i}} + \frac{{\partial f}}{{\partial y}}\hat{{j}} + \frac{{\partial f}}{{\partial z}}\hat{{k}}",
             rf"which means we have to calculate the partial derivative of $f$ with respect to $x$, $y$, then $z$, then write these as the components along the respective directions $\hat{{i}}$, $\hat{{j}}$, and $\hat{{k}}$."),
        # Steps 2-4: one partial derivative each
        Step(rf"Compute $\partial f/\partial x$ for the component along $\hat{{i}}$",
             _partial('f', 'x', f_str, fx)),
        Step(r"Compute $\partial f/\partial y$ for the component along $\hat{j}$",
             _partial('f', 'y', f_str, fy)),
        Step(r"Compute $\partial f/\partial z$ for the component along $\hat{k}$",
             _partial('f', 'z', f_str, fz)),

        Step(r"Write each as the components along the respective directions $\hat{i}$, $\hat{j}$, and $\hat{k}$ to build the gradient vector field.",
             rf"\nabla f = \left({latex(fx)}\right)\hat{{i}} + \left({latex(fy)}\right)\hat{{j}} + \left({latex(fz)}\right)\hat{{k}}"),
    ]


gradient_template = QuestionTemplate(
    id="vc-gradient-polynomial",
    topic="Vector Calculus",
    subtopic="Gradient",
    difficulty=2,
    prompt_fn=lambda p: _prompt("scalar field, $f$, find its gradient", r"\nabla f",
                                "f(x,y,z)", _terms_str(_gradient_build(p))),
    solver=_gradient_solver,
    parameters=_SEED_PARAM,
)


# ── Divergence ────────────────────────────────────────────────────────────────
# div F = dFx/dx + dFy/dy + dFz/dz, each component 1-2 terms.

def _divergence_build(p):
    """Component term lists [F_x, F_y, F_z]. Each component's first term
    contains that component's own variable, so its derivative is non-zero.
    An optional second term may not, and then differentiates to 0, which is a
    useful thing to spot. Draws are rejected if a component's own derivative
    is 0 (possible once a function term is swapped in) or the divergence
    cancels to 0."""
    rng = random.Random(p["s"])
    for _ in range(_MAX_DRAWS):
        comps = []
        for i in range(3):
            first = rng.choice([pw for pw in _MONOS_3D if pw[i]])
            pws = [first]
            if rng.random() < 0.5:
                pws.append(rng.choice([pw for pw in _MONOS_3D if pw != first]))
            comps.append([_term(rng, pw) for pw in _sorted(pws)])
        _add_func(rng, comps)
        own = [diff(_expr(c), v) for c, v in zip(comps, _VARS)]
        if all(d != 0 for d in own) and expand(sum(own)) != 0:
            return comps
    raise RuntimeError("no suitable vector field found")


def _divergence_solver(params):
    comps = _divergence_build(params)
    Fx, Fy, Fz = (_expr(c) for c in comps)
    Fx_str, Fy_str, Fz_str = (_terms_str(c) for c in comps)   # as written in the question

    # Each component is differentiated only with respect to its own direction.
    dFx = diff(Fx, x)
    dFy = diff(Fy, y)
    dFz = diff(Fz, z)
    div = expand(dFx + dFy + dFz)

    return [
        # Step 1: what the divergence is, and the plan (three partials, then add).
        Step(r"$\mathbf{F}$ is a vector, $\nabla$ is a vector, so $\nabla \cdot \mathbf{F}$ is a scalar defined by:",
             r"\nabla \cdot \mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}",
             r"which means we have to calculate the partial derivative of each component of $\mathbf{F}$ with respect to its own direction: $F_x$ with respect to $x$, $F_y$ with respect to $y$, and $F_z$ with respect to $z$, then add them together."),
        # Steps 2-4: one partial derivative each
        Step(r"Compute $\partial F_x/\partial x$, from the component along $\hat{i}$",
             _partial('F_x', 'x', Fx_str, dFx)),
        Step(r"Compute $\partial F_y/\partial y$, from the component along $\hat{j}$",
             _partial('F_y', 'y', Fy_str, dFy)),
        Step(r"Compute $\partial F_z/\partial z$, from the component along $\hat{k}$",
             _partial('F_z', 'z', Fz_str, dFz)),
        # The divergence is a scalar: just the sum of the three derivatives.
        Step(r"Add the three together to get the divergence, which is a scalar field.",
             rf"\nabla \cdot \mathbf{{F}} = \left({latex(dFx)}\right) + \left({latex(dFy)}\right) + \left({latex(dFz)}\right) = {latex(div)}"),
    ]


divergence_template = QuestionTemplate(
    id="vc-divergence-polynomial",
    topic="Vector Calculus",
    subtopic="Divergence",
    difficulty=2,
    prompt_fn=lambda p: _prompt(r"vector field, $\mathbf{F}$, find its divergence", r"\nabla \cdot \mathbf{F}",
                                r"\mathbf{F}(x,y,z)", _field_str(_divergence_build(p))),
    solver=_divergence_solver,
    parameters=_SEED_PARAM,
)


# ── Curl ──────────────────────────────────────────────────────────────────────
# curl F = (dFz/dy - dFy/dz) i + (dFx/dz - dFz/dx) j + (dFy/dx - dFx/dy) k,
# each component 1-2 terms. Terms that don't involve the other
# variables (e.g. x^2 in F_x) drop out of the curl, so students see zeros too.

def _curl_parts(comps):
    """(curl_i, curl_j, curl_k) of the field given by the component term lists."""
    Fx, Fy, Fz = (_expr(c) for c in comps)
    # The three components of the curl, from expanding the determinant
    #   | i    j    k   |
    #   | d/dx d/dy d/dz|
    #   | Fx   Fy   Fz  |
    # expand, not simplify: simplify may factorise, e.g. -2y - 2yz -> 2y(-z - 1).
    return (expand(diff(Fz, y) - diff(Fy, z)),
            expand(diff(Fx, z) - diff(Fz, x)),
            expand(diff(Fy, x) - diff(Fx, y)))


def _curl_build(p):
    """Component term lists [F_x, F_y, F_z], redrawn until at least two
    components of the curl are non-zero (so it's never close to a gradient field)."""
    rng = random.Random(p["s"])
    for _ in range(_MAX_DRAWS):
        comps = _components(rng, _MONOS_3D, 3)
        _add_func(rng, comps)
        if sum(1 for c in _curl_parts(comps) if c != 0) >= 2:
            return comps
    raise RuntimeError("no suitable vector field found")


def _curl_solver(params):
    comps = _curl_build(params)
    curl_i, curl_j, curl_k = _curl_parts(comps)
    # Each component as (LaTeX name, SymPy expr, LaTeX as written in the question).
    F = {v: (f'F_{v}', _expr(c), _terms_str(c)) for v, c in zip('xyz', comps)}

    def component(a, u, b, v, result):
        """Working for one curl component, dF_a/du - dF_b/dv, on three lines
        (\\ breaks the line): each partial derivative in full, then the difference."""
        (na, ea, sa), (nb, eb, sb) = F[a], F[b]
        da, db = diff(ea, u), diff(eb, v)
        return (_partial(na, u, sa, da) + r" \\ "
                + _partial(nb, v, sb, db) + r" \\ "
                + rf"\therefore\; \frac{{\partial {na}}}{{\partial {u}}} - \frac{{\partial {nb}}}{{\partial {v}}}"
                + rf" = \left({latex(da)}\right) - \left({latex(db)}\right) = {latex(result)}")

    return [
        # Step 1: what the curl is, and the plan (two partials per component, then assemble).
        Step(r"$\mathbf{F}$ is a vector, $\nabla$ is a vector, so $\nabla \times \mathbf{F}$ is a vector defined by:",
             r"\nabla \times \mathbf{F} = \begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ \frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\ F_x & F_y & F_z \end{vmatrix}"
             r" \\ = \left(\frac{\partial F_z}{\partial y} - \frac{\partial F_y}{\partial z}\right)\hat{i}"
             r" + \left(\frac{\partial F_x}{\partial z} - \frac{\partial F_z}{\partial x}\right)\hat{j}"
             r" + \left(\frac{\partial F_y}{\partial x} - \frac{\partial F_x}{\partial y}\right)\hat{k}",
             r"which means for each of the directions $\hat{i}$, $\hat{j}$, and $\hat{k}$ we have to calculate two partial derivatives of the *other* two components of $\mathbf{F}$, then take their difference."),
        # Steps 2-4: one curl component each
        Step(r"Compute the component along $\hat{i}$: $\partial F_z/\partial y - \partial F_y/\partial z$",
             component('z', 'y', 'y', 'z', curl_i)),
        Step(r"Compute the component along $\hat{j}$: $\partial F_x/\partial z - \partial F_z/\partial x$",
             component('x', 'z', 'z', 'x', curl_j)),
        Step(r"Compute the component along $\hat{k}$: $\partial F_y/\partial x - \partial F_x/\partial y$",
             component('y', 'x', 'x', 'y', curl_k)),
        Step(r"Write each as the components along the respective directions $\hat{i}$, $\hat{j}$, and $\hat{k}$ to build the curl.",
             rf"\nabla \times \mathbf{{F}} = \left({latex(curl_i)}\right)\hat{{i}} + \left({latex(curl_j)}\right)\hat{{j}} + \left({latex(curl_k)}\right)\hat{{k}}"),
    ]


curl_template = QuestionTemplate(
    id="vc-curl-polynomial",
    topic="Vector Calculus",
    subtopic="Curl",
    difficulty=2,
    prompt_fn=lambda p: _prompt(r"vector field, $\mathbf{F}$, find its curl", r"\nabla \times \mathbf{F}",
                                r"\mathbf{F}(x,y,z)", _field_str(_curl_build(p))),
    solver=_curl_solver,
    parameters=_SEED_PARAM,
)


# ── Line Integrals ────────────────────────────────────────────────────────────
# use the path's equation to
# eliminate one variable (and its differential), so the integral is over the
# other variable alone, with that variable's limits. No parametrisation in t.
#
# Two kinds of field, each with two kinds of path (four subtopics):
#   scalar field phi(x, y) 
#       int_C phi dr = int phi dx i + int phi dy j     a vector: an i part and a j part
#   vector field E(x, y)  
#       int_C E . dr = int (E_x dx + E_y dy)           a scalar
#   straight lines  y = mx + c (dy = m dx), or y = c (dy = 0), or x = c (dx = 0)
#   parabolas       y = x^2 (dy = 2x dx), or y^2 = x (dx = 2y dy, integrate over y)
# Each question comes with a plot of the field and the path (core.plot.field_path_map).

# A path between two points. var is the variable we integrate over; other is
# the one we eliminate, using other = expr(var) and d(other)/d(var) = deriv
# (0 on a line where other is constant). var runs from lo to hi. eq is the
# path's equation as written in the question ('y = x + 1', 'y^2 = x').
_Path = namedtuple("_Path", "var other expr deriv lo hi start end eq")

# Field coefficients for line integrals: -3..3 excluding 0. Smaller than
# elsewhere, since the path substitution already makes the numbers grow.
_LINE_COEFFS = [c for c in range(-3, 4) if c != 0]


def _pt(p):
    """LaTeX for a point, e.g. (1, -2)."""
    return f"({p[0]}, {p[1]})"


def _point(path, u):
    """The (x, y) point on the path where var = u."""
    o = path.expr.subs(path.var, u)
    return (u, o) if path.var == x else (o, u)


def _make_path(var, expr, lo, hi, eq):
    """A _Path along other = expr(var), for var from lo to hi."""
    other = y if var == x else x
    expr = Integer(expr) if isinstance(expr, int) else expr
    p = _Path(var, other, expr, diff(expr, var), lo, hi, None, None, eq)
    return p._replace(start=_point(p, lo), end=_point(p, hi))


def _straight_path(rng):
    """A straight line: sloped (y = mx + c, most often), horizontal (y = c) or
    vertical (x = c). Coordinates stay small: the varying coordinate runs
    between two values in 0..3 (or 0..4 for flat lines), always increasing,
    and a sloped line's y values stay within -4..4."""
    kind = rng.choice(["sloped", "sloped", "horizontal", "vertical"])
    if kind == "sloped":
        while True:
            lo, hi = sorted(rng.sample(range(0, 4), 2))
            m, c = rng.choice([-1, 1, 1, 2]), rng.randint(0, 2)
            if all(abs(m * u + c) <= 4 for u in (lo, hi)):
                return _make_path(x, m * x + c, lo, hi, f"y = {poly((m, 'x'), (c, ''))}")
    lo, hi = sorted(rng.sample(range(0, 5), 2))
    c = rng.randint(0, 3)
    if kind == "horizontal":
        return _make_path(x, c, lo, hi, f"y = {c}")
    return _make_path(y, c, lo, hi, f"x = {c}")


def _parabola_path(rng):
    """y = x^2 (eliminate y) or y^2 = x (eliminate x),
    between two of the points where the free variable is 0, 1 or 2."""
    lo, hi = sorted(rng.sample(range(0, 3), 2))
    if rng.random() < 0.5:
        return _make_path(x, x**2, lo, hi, "y = x^2")
    return _make_path(y, y**2, lo, hi, "y^2 = x")


def _paren(e):
    """LaTeX for e, bracketed if it's a sum (so '\\int (x + 3)\\,dx', not '\\int x + 3\\,dx')."""
    return rf"\left({latex(e)}\right)" if e.is_Add else latex(e)


def _times(a, m):
    """LaTeX for a times m, left unexpanded so the substitution is visible:
    '2x^{3}\\,(2x)' or '\\left(x^{2} + x\\right)(2x)'. m = 1 is left out."""
    return _paren(a) if m == 1 else rf"{_paren(a)}\,\left({latex(m)}\right)"


def _join(parts):
    """Join LaTeX terms with ' + ', or ' - ' for a term starting with a minus,
    so there's never a '+ -'. Empty parts are skipped; nothing left gives '0'."""
    out = []
    for s in parts:
        if not s:
            continue
        if out and s.startswith('-'):
            out.append(f"- {s[1:].strip()}")
        else:
            out.append(f"+ {s}" if out else s)
    return ' '.join(out) or '0'


def _vec2(ci, cj):
    """LaTeX for the 2D vector ci i + cj j (SymPy values), skipping a zero
    component and writing 1 and -1 as just the sign; a multi-term component
    is bracketed."""
    return _join([_scaled(_paren(c), hat) for c, hat in ((ci, _HATS[0]), (cj, _HATS[1])) if c != 0])


def _scaled(s, hat):
    """LaTeX for the number or expression s times a unit vector, e.g.
    '3\\,\\hat{i}', with 1 and -1 written as just the sign: '\\hat{i}', '-\\hat{i}'."""
    return {"1": hat, "-1": f"-{hat}"}.get(s, rf"{s}\,{hat}")


def _approx(v):
    """A number to 3 significant figures, e.g. 24.8."""
    return f"{float(v):.3g}"


def _evaluate(G, var, lo, hi):
    """'(G(hi) terms) - (G(lo) terms)', term by term, e.g. '\\left(8 + \\frac{16}{3}\\right) - \\left(\\frac{1}{2} + \\frac{2}{3}\\right)'.
    The lower bracket is left out when the lower limit gives 0."""
    terms = G.as_ordered_terms()
    at_hi = poly(*[(e.subs(var, hi), '') for e in terms])
    at_lo = poly(*[(e.subs(var, lo), '') for e in terms])
    return rf"\left({at_hi}\right)" + ("" if at_lo == "0" else rf" - \left({at_lo}\right)")


def _eliminate_step(path, scalar):
    """Step 2: use the path to eliminate one variable, and state the new limits."""
    v, o = latex(path.var), latex(path.other)
    limits = (f"After we eliminate ${o}$ we only need ${v}$ limits: "
              f"${v}$ goes from ${path.lo}$ to ${path.hi}$.")
    if path.deriv == 0:
        lost = (rf"the $\hat{{{'i' if o == 'x' else 'j'}}}$ part is zero" if scalar
                else f"the $E_{o}\\,d{o}$ part is zero")
        return Step(f"Along ${path.eq}$, ${o}$ doesn't change, so $d{o} = 0$",
                    rf"{o} = {latex(path.expr)}, \quad d{o} = 0",
                    f"So {lost}. {limits}")
    return Step(f"Eliminate ${o}$ and $d{o}$ by subbing in the path:",
                rf"{o} = {latex(path.expr)}, \quad \frac{{d{o}}}{{d{v}}} = {latex(path.deriv)}"
                rf" \;\rightarrow\; d{o} = " + (rf"d{v}" if path.deriv == 1 else rf"{latex(path.deriv)}\,d{v}"),
                limits)


def _multipliers(path):
    """(dx/d var, dy/d var): what dx and dy become in terms of d var."""
    return (1, path.deriv) if path.var == x else (path.deriv, 1)


def _scalar_line_solver(params, path_fn):
    path, terms = _line_build(params, path_fn, scalar=True)
    phi = _expr(terms)
    phi_str = _terms_str(terms)
    v, lo, hi, A, B = path.var, path.lo, path.hi, _pt(path.start), _pt(path.end)
    dv = rf"\,d{latex(v)}"

    phi_sub = phi.subs(path.other, path.expr)          # phi on the path, in terms of var only
    mults = _multipliers(path)
    # The i and j integrands, unexpanded (to show the substitution) and expanded.
    shown = [_times(phi_sub, m) if m != 0 else "" for m in mults]
    expanded = [expand(phi_sub * m) for m in mults]
    G = [integrate(e, v) for e in expanded]            # antiderivatives of the i and j parts
    result = [g.subs(v, hi) - g.subs(v, lo) for g in G]

    def integral(s, hat):
        return rf"\int_{{{lo}}}^{{{hi}}} {s}{dv}\,{hat}" if s else ""

    # The substituted integrals, then (only if expanding changes anything) expanded.
    lines = [_join([integral(s, h) for s, h in zip(shown, _HATS)])]
    if any(m not in (0, 1) for m in mults) or phi_sub != expand(phi_sub):
        lines.append(_join([integral(_paren(e), h) if s else "" for e, s, h in zip(expanded, shown, _HATS)]))

    final = rf"= {_vec2(*result)}"
    if any(not r.is_integer for r in result):
        final += r" \approx " + _join([_scaled(_approx(r), h) for r, h in zip(result, _HATS) if r != 0])

    return [
        Step(r"The field is a scalar, so the line integral is a vector. Write it as an $\hat{i}$ part and a $\hat{j}$ part:",
             rf"\int_{{{A}}}^{{{B}}} \phi(x,y)\,(dx\,\hat{{i}} + dy\,\hat{{j}}) = "
             rf"\int_{{{A}}}^{{{B}}} \left({phi_str}\right) dx\,\hat{{i}} + \int_{{{A}}}^{{{B}}} \left({phi_str}\right) dy\,\hat{{j}}"),
        _eliminate_step(path, scalar=True),
        Step(rf"Sub in, so each part is an integral over ${latex(v)}$ only:",
             "= " + r" \\ = ".join(lines)),
        Step("Integrate, then put in the limits:",
             rf"= \left[{_vec2(*G)}\right]_{{{lo}}}^{{{hi}}} \\ {final}"),
    ]


def _vector_line_solver(params, path_fn):
    path, comps = _line_build(params, path_fn, scalar=False)
    Ex, Ey = (_expr(c) for c in comps)
    Ex_str, Ey_str = (_terms_str(c) for c in comps)
    v, lo, hi, A, B = path.var, path.lo, path.hi, _pt(path.start), _pt(path.end)
    dv = rf"\,d{latex(v)}"

    subs = [e.subs(path.other, path.expr) for e in (Ex, Ey)]   # E_x, E_y on the path
    mults = _multipliers(path)
    integrand = expand(sum(s * m for s, m in zip(subs, mults)))
    # The substituted integrand: E_x(dx/d var) + E_y(dy/d var), dropping parts
    # that are zero. A lone part with nothing to multiply needs no square brackets.
    parts = [(s, m) for s, m in zip(subs, mults) if m != 0 and s != 0]
    if len(parts) == 1 and parts[0][1] == 1:
        shown = _paren(parts[0][0])
    else:
        shown = rf"\left[{_join([_times(s, m) for s, m in parts])}\right]"
    lines = [rf"L = \int_{{{lo}}}^{{{hi}}} {shown}{dv}"]
    if shown != _paren(integrand):   # expanding changes something
        lines.append(rf"= \int_{{{lo}}}^{{{hi}}} {_paren(integrand)}{dv}")
    G = integrate(integrand, v)
    result = G.subs(v, hi) - G.subs(v, lo)

    final = rf"= {latex(result)}" + ("" if result.is_integer else rf" \approx {_approx(result)}")

    return [
        Step(r"The field is a vector, and we take its dot product with the path, so the line integral is a scalar:",
             rf"L = \int_C \bar{{E}}\cdot d\bar{{r}} = \int_{{{A}}}^{{{B}}} \bar{{E}}\cdot(dx\,\hat{{i}} + dy\,\hat{{j}}) "
             rf"= \int_{{{A}}}^{{{B}}} (E_x\,dx + E_y\,dy) \\ "
             rf"= \int_{{{A}}}^{{{B}}} \left[\left({Ex_str}\right) dx + \left({Ey_str}\right) dy\right]"),
        _eliminate_step(path, scalar=False),
        Step(rf"Sub in, so the integral is over ${latex(v)}$ only:",
             r" \\ ".join(lines)),
        Step("Integrate, then put in the limits:",
             rf"L = \left[{latex(G)}\right]_{{{lo}}}^{{{hi}}} \\ "
             rf"= {_evaluate(G, v, lo, hi)} \\ {final}"),
    ]


def _line_build(p, path_fn, scalar):
    """(path, field) with field the terms of phi (scalar) or [E_x terms, E_y terms]
    (vector), redrawn until the line integral isn't zero."""
    rng = random.Random(p["s"])
    for _ in range(_MAX_DRAWS):
        path = path_fn(rng)
        if scalar:
            field = [_term(rng, pw, _LINE_COEFFS)
                     for pw in _sorted(rng.sample(_MONOS_2D, rng.choice([1, 2])))]
            phi = _expr(field).subs(path.other, path.expr)
            mults = _multipliers(path)
            nonzero = any(integrate(phi * m, (path.var, path.lo, path.hi)) != 0 for m in mults)
        else:
            field = _components(rng, _MONOS_2D, 2, _LINE_COEFFS)
            on_path = [_expr(c).subs(path.other, path.expr) for c in field]
            integrand = sum(s * m for s, m in zip(on_path, _multipliers(path)))
            nonzero = integrate(integrand, (path.var, path.lo, path.hi)) != 0
        if nonzero:
            return path, field
    raise RuntimeError("no suitable field found")


def _path_desc(path):
    what = "curve" if path.eq in ("y = x^2", "y^2 = x") else "line"
    return rf"the {what} ${path.eq}$, starting at ${_pt(path.start)}$ and ending at ${_pt(path.end)}$"


def _line_prompt(p, path_fn, scalar):
    path, field = _line_build(p, path_fn, scalar)
    if scalar:
        return ("For the scalar field\n\n$$\n"
                rf"\phi(x,y) = {_terms_str(field)}" "\n$$\n\n"
                rf"what is the line integral $\int_C \phi(x,y)\,d\bar{{r}}$ along {_path_desc(path)}? " "\n\n"
                "The plot shows the field and the path from A to B:")
    return ("For the vector field\n\n$$\n"
            rf"\bar{{E}}(x,y) = {_field_str(field)}" "\n$$\n\n"
            rf"what is the line integral $L = \int_C \bar{{E}}\cdot d\bar{{r}}$ along {_path_desc(path)}? ""\n\n"
            "The plot shows the field (colour for strength, arrows for direction) and the path from A to B:")


def _line_figure(p, path_fn, scalar):
    """The field around the path: phi itself for a scalar field; |E| with
    direction arrows for a vector field."""
    path, field = _line_build(p, path_fn, scalar)
    uu = np.linspace(float(path.lo), float(path.hi), 200)
    oo = np.broadcast_to(lambdify(path.var, path.expr, "numpy")(uu), uu.shape).astype(float)
    xs, ys = (uu, oo) if path.var == x else (oo, uu)
    pts = [tuple(int(c) for c in pt) for pt in (path.start, path.end)]
    if scalar:
        phi = lambdify((x, y), _expr(field), "numpy")
        return field_path_map(phi, (xs, ys), *pts, r"$\phi(x,y)$")
    Ex, Ey = (lambdify((x, y), _expr(c), "numpy") for c in field)
    mag = lambda gx, gy: np.hypot(np.broadcast_to(Ex(gx, gy), np.shape(gx)),
                                  np.broadcast_to(Ey(gx, gy), np.shape(gx)))
    return field_path_map(mag, (xs, ys), *pts, r"$|\bar{E}|$", arrows=(Ex, Ey))


def _line_template(id, subtopic, difficulty, path_fn, scalar):
    """A line integral template for one kind of field and one kind of path."""
    return QuestionTemplate(
        id=id,
        topic="Vector Calculus",
        subtopic=subtopic,
        difficulty=difficulty,
        prompt_fn=lambda p: _line_prompt(p, path_fn, scalar),
        solver=lambda p: (_scalar_line_solver if scalar else _vector_line_solver)(p, path_fn),
        figure_fn=lambda p: _line_figure(p, path_fn, scalar),
        parameters=_SEED_PARAM,
    )


line_scalar_straight_template = _line_template(
    "vc-line-integral-scalar-straight", "Line Integrals: scalar fields, straight lines", 2, _straight_path, True)
line_scalar_parabola_template = _line_template(
    "vc-line-integral-scalar-parabola", "Line Integrals: scalar fields, parabolas", 3, _parabola_path, True)
line_vector_straight_template = _line_template(
    "vc-line-integral-vector-straight", "Line Integrals: vector fields, straight lines", 2, _straight_path, False)
line_vector_parabola_template = _line_template(
    "vc-line-integral-vector-parabola", "Line Integrals: vector fields, parabolas", 3, _parabola_path, False)


# ── Surface Integrals ─────────────────────────────────────────────────────────
# Flux of a vector field through a flat rectangle lying in a coordinate plane
#   dS = dS n,  dS = dx dz,  n = 0i + 1j + 0k        (for a surface in the xz plane)
#   v . dS = v . (dS n) = (only the j component) dx dz
#   Phi = int_S v . dS = int_0^a x dx int_0^b z dz     (each term split into two 1D integrals)
# The plane, its position, the limits, which way the normal points, and the
# field all vary. The field is 3D, each component 1-2 terms of degree 1-2.

_MONOS_3D_LOW = _powers(3, (1, 2))   # x, y, z, x^2, xy, ..., z^2 (9 shapes)

# A flat surface: perpendicular to axis n (0, 1, 2 = x, y, z), in the plane
# (that axis) = c, spanning axes u and w (u < w) over lims[u] and lims[w];
# sign = +1 or -1 is the direction of its normal along axis n.
_Surface = namedtuple("_Surface", "n c u w lims sign comps")


def _surface_build(p):
    """A surface and field, redrawn until the flux is non-zero."""
    rng = random.Random(p["s"])
    for _ in range(_MAX_DRAWS):
        n = rng.randrange(3)
        u, w = (i for i in range(3) if i != n)
        lims = {}
        for i in (u, w):
            lo = rng.choice([0, 0, 1])      # usually from 0
            lims[i] = (lo, lo + rng.randint(1, 2))
        s = _Surface(n, rng.choice([0, 0, 1, 2]), u, w, lims, rng.choice([1, 1, 1, -1]),
                     _components(rng, _MONOS_3D_LOW, 3, _LINE_COEFFS))
        if _surface_flux(s) != 0:
            return s
    raise RuntimeError("no suitable surface found")


def _surface_through(s):
    """The part of v that flows through the surface, on the surface: the
    normal component, times the normal's sign, with the plane's coordinate put in."""
    return expand(s.sign * _expr(s.comps[s.n]).subs(_VARS[s.n], s.c))


def _surface_flux(s):
    (lo_u, hi_u), (lo_w, hi_w) = s.lims[s.u], s.lims[s.w]
    return integrate(_surface_through(s), (_VARS[s.u], lo_u, hi_u), (_VARS[s.w], lo_w, hi_w))


def _surface_solver(params):
    s = _surface_build(params)
    nu, nw, nn = _NAMES[s.u], _NAMES[s.w], _NAMES[s.n]
    dS = rf"d{nu}\,d{nw}"
    nhat = _join([f"{s.sign if i == s.n else 0}{h}" for i, h in enumerate(_HATS)])
    vn = _expr(s.comps[s.n])                 # the component through the surface
    through = _surface_through(s)
    sign_note = ("" if s.sign == 1 else
                 rf" It points in the $-{nn}$ direction, so $\hat{{n}}$ has a $-1$, which flips the sign of the flux.")

    # Step 2: the dot product, then (if the plane's coordinate appears) the plane put in.
    dot = [rf"\bar{{v}}\cdot d\bar{{S}} = \bar{{v}}\cdot(dS\,\hat{{n}}) = "
           rf"\left({_field_str(s.comps)}\right)\cdot\left({nhat}\right) {dS}",
           rf"= {_paren(expand(s.sign * vn))}\,{dS}"]
    dot_note = (f"The other two components flow along the surface, so only the "
                f"${_HATS[s.n]}$ component flows through it.")
    if _VARS[s.n] in vn.free_symbols:
        dot.append(rf"= {_paren(through)}\,{dS}")
        dot_note += f" On the surface ${nn} = {s.c}$, so we put ${nn} = {s.c}$ in."

    # Step 3: split each term into a product of two 1D integrals.
    (lo_u, hi_u), (lo_w, hi_w) = s.lims[s.u], s.lims[s.w]
    terms = []      # (coefficient, power of u, power of w) for each term
    for term in through.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()
        terms.append((coeff, degree(rest, _VARS[s.u]), degree(rest, _VARS[s.w])))

    def one_d(var, name, lo, hi, p):
        body = latex(var**p) if p else ""
        return rf"\int_{{{lo}}}^{{{hi}}} {body}\,d{name}"

    split = poly(*[(c, one_d(_VARS[s.u], nu, lo_u, hi_u, pu) + one_d(_VARS[s.w], nw, lo_w, hi_w, pw))
                   for c, pu, pw in terms])

    # Step 4: antiderivatives in brackets, their values, then the total.
    brackets, values = [], []
    for c, pu, pw in terms:
        Fu, Fw = integrate(_VARS[s.u]**pu, _VARS[s.u]), integrate(_VARS[s.w]**pw, _VARS[s.w])
        vu, vw = Fu.subs(_VARS[s.u], hi_u) - Fu.subs(_VARS[s.u], lo_u), Fw.subs(_VARS[s.w], hi_w) - Fw.subs(_VARS[s.w], lo_w)
        brackets.append((c, rf"\left[{latex(Fu)}\right]_{{{lo_u}}}^{{{hi_u}}}\left[{latex(Fw)}\right]_{{{lo_w}}}^{{{hi_w}}}"))
        values.append((c, rf"\left({latex(vu)}\right)\left({latex(vw)}\right)"))
    flux = _surface_flux(s)
    final = rf"= {latex(flux)}" + ("" if flux.is_integer else rf" \approx {_approx(flux)}")

    return [
        Step(f"The surface lies flat in the ${nu}{nw}$ plane, so:",
             rf"d\bar{{S}} = dS\,\hat{{n}}, \qquad dS = {dS}, \qquad \hat{{n}} = {nhat}",
             f"$dS$ is made of the two directions the surface spans, ${nu}$ and ${nw}$, and "
             rf"$\hat{{n}}$ is the unit vector perpendicular to it.{sign_note}"),
        Step("Take the dot product of the field with $d\\bar{S}$:",
             r" \\ ".join(dot), dot_note),
        Step(f"Integrate over the surface: ${nu}$ from ${lo_u}$ to ${hi_u}$ and ${nw}$ from ${lo_w}$ to ${hi_w}$",
             rf"\therefore\; \Phi = \int_S \bar{{v}}\cdot d\bar{{S}} = \int_S {_paren(through)}\,{dS} \\ = {split}",
             "Each term is a product of a part in only one variable and a part in only the other, "
             "so it splits into two separate integrals."),
        Step("Do each integral, then multiply:",
             rf"\Phi = {poly(*brackets)} \\ = {poly(*values)} \\ {final}"),
    ]


def _surface_prompt(p):
    s = _surface_build(p)
    nu, nw, nn = _NAMES[s.u], _NAMES[s.w], _NAMES[s.n]
    (lo_u, hi_u), (lo_w, hi_w) = s.lims[s.u], s.lims[s.w]
    return ("For the vector field\n\n$$\n"
            rf"\bar{{v}}(x,y,z) = {_field_str(s.comps)}" "\n$$\n\n"
            rf"what is the flux $\Phi = \int_S \bar{{v}}\cdot d\bar{{S}}$ through the flat surface in the plane "
            rf"${nn} = {s.c}$ that spans from ${nu} = {lo_u}$ to ${nu} = {hi_u}$ and from ${nw} = {lo_w}$ to "
            rf"${nw} = {hi_w}$, with its normal pointing in the ${'+' if s.sign == 1 else '-'}{nn}$ direction? "
            "The sketch shows the surface, its normal, and the direction of the field.")


def _surface_figure(p):
    s = _surface_build(p)
    field = [lambdify((x, y, z), _expr(c), "numpy") for c in s.comps]
    return surface_flux_sketch(field, s.n, s.c, s.lims, s.sign)


surface_integral_template = QuestionTemplate(
    id="vc-surface-integral-flat",
    topic="Vector Calculus",
    subtopic="Surface Integrals",
    difficulty=2,
    prompt_fn=_surface_prompt,
    solver=_surface_solver,
    figure_fn=_surface_figure,
    parameters=_SEED_PARAM,
)


# Order here = order of subtopics in the sidebar dropdown.
ALL_TEMPLATES = [
    gradient_template,
    divergence_template,
    curl_template,
    line_scalar_straight_template,
    line_scalar_parabola_template,
    line_vector_straight_template,
    line_vector_parabola_template,
    surface_integral_template,
]
