"""
Fourier analysis question templates.

Templates in this module:
    Sketching Periodic Functions  sketch a wave given on one interval, over N periods
    Fourier Series                the Fourier series of the same waves (sawtooth, square,
                                  triangle, parabola, ramp, pulse, rectified sine)
    Fourier Transforms            top hat (real transform) or odd pair of steps
                                  (imaginary transform), worked as in PHYS108 lecture 27

Fourier series convention (period T, as in the lectures):
    f(t) = a_0/2 + sum_{n>=1} [ a_n cos(2n pi t / T) + b_n sin(2n pi t / T) ]
    a_n  = (2/T) integral over one period of f(t) cos(2n pi t / T) dt
    b_n  = (2/T) integral over one period of f(t) sin(2n pi t / T) dt
Fourier transform convention:
    f~(omega) = integral_{-inf}^{inf} f(x) e^{-i omega x} dx
"""
import numpy as np
from sympy import (symbols, integrate, diff, expand, factor, pi, sin, cos, latex, simplify,
                   exp, I, re, im, Integer, S, Symbol, lambdify)
from ..core.types import QuestionTemplate, Step
from ..core.fmt import coef, poly
from ..core.plot import periodic_sketch, transform_plot

# real=True lets SymPy simplify things like |x| and cos(n pi) more readily.
x, n, t, omega = symbols('x n t omega', real=True)


# ── Sketching Periodic Functions ──────────────────────────────────────────────
# Practice for the first step of any Fourier series question: picturing the
# periodic function. The student gets f(t) on one interval, e.g. -2 < t < 2,
# and sketches it repeated over N periods. The worked solution:
#   1. finds the period T (the length of the interval),
#   2. sketches one period, noting the values at the two ends,
#   3. sketches N periods, with the original interval shaded, dashed lines at
#      every period boundary, a "T = ..." arrow, and a vertical line (with open
#      circles) at each jump,
#   4. states the symmetry (odd / even / neither) and what it means for the
#      Fourier series.
# End values, jumps and symmetry are all computed from the function, not written
# in by hand per shape.

def _sketch_shape(p):
    """The function on its base interval, as (pieces, display).

    pieces   list of (SymPy expr in t, start, end), in order, covering one period
    display  LaTeX for the right-hand side of "f(t) = ...", or None to build it
             from the pieces (a single expression, or a cases block)

    kind: 0 sawtooth a t        on -L < t < L      (odd)
          1 square wave +-a     on -L < t < L      (odd)
          2 triangle a|t|       on -L < t < L      (even)
          3 parabola a t^2      on -L < t < L      (even)
          4 ramp a t            on  0 < t < L      (neither; period L)
          5 pulse 0 / a         on -L < t < L      (neither)
          6 rectified a sin(t)  on  0 < t < pi     (even; period pi, L unused)
    """
    a, L, kind = p["a"], p["L"], p["kind"]
    if kind == 0:
        return [(a * t, S(-L), S(L))], None
    if kind == 1:
        return [(Integer(-a), S(-L), S(0)), (Integer(a), S(0), S(L))], None
    if kind == 2:
        return [(-a * t, S(-L), S(0)), (a * t, S(0), S(L))], coef(a, '|t|')
    if kind == 3:
        return [(a * t**2, S(-L), S(L))], None
    if kind == 4:
        return [(a * t, S(0), S(L))], None
    if kind == 5:
        return [(Integer(0), S(-L), S(0)), (Integer(a), S(0), S(L))], None
    return [(a * sin(t), S(0), pi)], None


def _sketch_definition(pieces, display):
    """LaTeX 'f(t) = ...', as a cases block when there is more than one piece."""
    if display:
        body = display
    elif len(pieces) == 1:
        body = latex(pieces[0][0])
    else:
        body = (r"\begin{cases} "
                + r" \\ ".join(rf"{latex(e)} & {latex(s)} < t < {latex(f)}" for e, s, f in pieces)
                + r" \end{cases}")
    return "f(t) = " + body


def _sketch_parts(p):
    """Everything the prompt and solver share: pieces, display, interval and period."""
    pieces, display = _sketch_shape(p)
    t0, t1 = pieces[0][1], pieces[-1][2]   # the base interval t0 < t < t1
    return pieces, display, t0, t1, t1 - t0


def _sketch_prompt(p):
    pieces, display, t0, t1, T = _sketch_parts(p)
    return ("Sketch the function\n\n$$\n" + _sketch_definition(pieces, display) + "\n$$\n\n"
            + rf"defined for ${latex(t0)} < t < {latex(t1)}$ and repeated periodically, "
            + f"over **{p['N']} periods**.")


def _symmetry(num, t0, T):
    """("Odd" / "Even" / "Neither odd nor even", explanatory note) for the
    periodic extension of the pieces, tested numerically at generic points
    (offset so they never land exactly on a breakpoint). num holds numeric
    pieces (f, start, end), as built in _sketch_solver."""
    def g(tv):
        u = float(t0) + np.mod(tv - float(t0), float(T))   # shift each point into the base period
        out = np.empty_like(u)
        for f, s, e in num:
            m = (u >= s) & (u <= e)
            out[m] = f(u[m])
        return out
    ts = 0.1234 + np.linspace(0, 3 * float(T), 997)
    if np.allclose(g(-ts), -g(ts)):
        return "Odd", (r"$f(-t) = -f(t)$, so the function is odd: its Fourier series contains "
                       r"only sine terms ($a_0 = a_n = 0$).")
    if np.allclose(g(-ts), g(ts)):
        return "Even", (r"$f(-t) = f(t)$, so the function is even: its Fourier series contains "
                        r"only cosine terms and a constant ($b_n = 0$).")
    return "Neither odd nor even", (r"$f(-t)$ is neither $f(t)$ nor $-f(t)$, so the Fourier "
                                    r"series needs both sine and cosine terms.")


def _sketch_solver(p):
    pieces, display, t0, t1, T = _sketch_parts(p)
    N = p["N"]
    # N copies of the base period, centred on it: e.g. N = 3 -> shifts -1, 0, 1.
    shifts = list(range(-((N - 1) // 2), N - (N - 1) // 2))

    # Numeric versions of the pieces for plotting and the symmetry test.
    num = [(lambdify(t, e, "numpy"), float(s), float(f)) for e, s, f in pieces]

    # Tick marks: every breakpoint for the one-period sketch; every period
    # boundary for the N-period sketch. Labels use SymPy's LaTeX, e.g. "-2" or "\pi".
    def _tick(v):
        return (float(v), f"${latex(v)}$")
    one_ticks = [_tick(v) for v in sorted({v for _, s, f in pieces for v in (s, f)}, key=float)]
    many_ticks = [_tick(t0 + k * T) for k in range(min(shifts), max(shifts) + 2)]

    one_png = periodic_sketch(num, float(t0), float(T), [0], one_ticks)
    many_png = periodic_sketch(num, float(t0), float(T), shifts, many_ticks,
                               period_label=f"$T = {latex(T)}$")

    # Values approached at the two ends of the interval, and any jumps between pieces.
    v0, v1 = pieces[0][0].subs(t, t0), pieces[-1][0].subs(t, t1)
    inner_jumps = [pieces[i][2] for i in range(len(pieces) - 1)
                   if pieces[i][0].subs(t, pieces[i][2]) != pieces[i + 1][0].subs(t, pieces[i][2])]

    ends_note = (rf"At the ends of the interval, $f(t) \to {latex(v0)}$ as $t \to {latex(t0)}^+$ "
                 rf"and $f(t) \to {latex(v1)}$ as $t \to {latex(t1)}^-$.")
    if inner_jumps:
        ends_note += " Inside the interval it jumps at " + ", ".join(f"$t = {latex(j)}$" for j in inner_jumps) + "."

    k = Symbol('k')
    if v0 != v1:
        repeat_note = (rf"The two end values differ, so the periodic function jumps at every period boundary, "
                       rf"$t = {latex(t0 + k * T)}$ for integer $k$ (vertical lines, with open circles at the ends).")
    else:
        repeat_note = r"The two end values match, so the copies join up: no jump at the period boundaries."
    if inner_jumps:
        repeat_note += " The jumps inside the interval repeat every period too."
    repeat_note += " The shaded band is the original interval; dashed lines mark the period boundaries."

    sym, sym_note = _symmetry(num, t0, T)

    t0_sub = latex(t0) if t0 >= 0 else rf"\left({latex(t0)}\right)"   # bracket a negative start
    return [
        Step("Find the period: the length of the interval the function is defined on",
             rf"T = {latex(t1)} - {t0_sub} = {latex(T)}"),
        Step(rf"Sketch one period, on ${latex(t0)} < t < {latex(t1)}$",
             _sketch_definition(pieces, display), ends_note, image=one_png),
        Step(rf"Repeat the same shape every $T = {latex(T)}$, for {N} periods",
             rf"f(t + {latex(T)}) = f(t)", repeat_note, image=many_png),
        Step("Check the symmetry (this decides which Fourier terms appear)",
             rf"\text{{{sym}}}", sym_note),
    ]


sketch_template = QuestionTemplate(
    id="fourier-sketch-periodic",
    topic="Fourier Analysis",
    subtopic="Sketching Periodic Functions",
    difficulty=1,
    prompt_fn=_sketch_prompt,
    solver=_sketch_solver,
    # a: amplitude; L: half-width of the interval (or width, for the ramp);
    # kind: which shape (see _sketch_shape); N: number of periods to sketch.
    parameters={"a": (1, 3), "L": (1, 3), "kind": (0, 6), "N": (2, 4)},
)


# ── Fourier Series ────────────────────────────────────────────────────────────
# The same waves as the sketching subtopic (_sketch_shape): sawtooth, square,
# triangle, parabola, ramp, pulse and rectified sine, with random amplitude and
# width. The question shows the wave sketched over three periods. With period T,
#   f(t) = a_0/2 + sum_{n>=1} [ a_n cos(2n pi t / T) + b_n sin(2n pi t / T) ]
#   a_n  = (2/T) integral over one period of f(t) cos(2n pi t / T) dt   (a_0: n = 0)
#   b_n  = (2/T) integral over one period of f(t) sin(2n pi t / T) dt
# The worked solution finds T, uses the symmetry to say which coefficients
# vanish, integrates piece by piece (skipping pieces where f = 0), gives the
# general a_n and b_n (SymPy uses sin(n pi) = 0 and cos(n pi) = (-1)^n, as n
# is a positive integer), then writes out the first few terms.

nn = symbols('n', integer=True, positive=True)   # the harmonic number in a_n, b_n
_N_TERMS = 5                                      # harmonics n = 1..5 written out at the end


def _numeric(pieces):
    """Numeric versions (f, start, end) of the pieces, for plots and the symmetry test."""
    return [(lambdify(t, e, "numpy"), float(s), float(f)) for e, s, f in pieces]


def _series_parts(p):
    """Everything the prompt, solver and figure share: the pieces, how f is
    written, the interval, the period, and the argument 2n pi t / T."""
    pieces, display, t0, t1, T = _sketch_parts(p)
    return pieces, display, t0, t1, T, 2 * nn * pi * t / T


def _series_figure(p):
    """The wave over three periods, as in the sketching subtopic's final sketch."""
    pieces, _, t0, _, T, _ = _series_parts(p)
    ticks = [(float(t0 + k * T), f"${latex(t0 + k * T)}$") for k in range(-1, 3)]
    return periodic_sketch(_numeric(pieces), float(t0), float(T), [-1, 0, 1], ticks,
                           period_label=f"$T = {latex(T)}$")


# n for the working of the integrals: positive, but not known to be an integer,
# so SymPy leaves sin(n pi) and cos(n pi) as they are. That lets the solution
# show them before using sin(n pi) = 0 and cos(n pi) = (-1)^n (with nn).
_n_work = Symbol('n', positive=True)


def _bracket(e):
    """LaTeX for e, bracketed if it's a sum or negative: (-2), (2t - 1)."""
    return rf"\left({latex(e)}\right)" if e.is_Add or e.could_extract_minus_sign() else latex(e)


def _coefficient_steps(name, trig, pieces, T, arg, result):
    """The working for one coefficient, as a list of Steps.

    name is 'a_0', 'a_n' or 'b_n'; trig is cos or sin (None for a_0).
    a_0 is one step: set up, integrate, put in the limits. a_n and b_n get
    three steps: set up the integral; do each piece's integral, showing the
    method (by parts, product to sum, or directly); then put in the limits,
    with sin(n pi) and cos(n pi) still visible, and simplify them.
    """
    prefix = "" if 2 / T == 1 else latex(2 / T)   # 2/T in front, unless it's 1
    nonzero = [(e, s, f) for e, s, f in pieces if e != 0]
    trig_tex = rf"\{trig.__name__}\left({latex(arg)}\right)" if trig else ""
    integrals = poly(*[(1, rf"\int_{{{latex(s)}}}^{{{latex(f)}}} {_bracket(e)}{trig_tex}\,dt")
                       for e, s, f in nonzero])
    setup = rf"{name} = {prefix}\left[{integrals}\right]"
    zero_note = (r"$f(t) = 0$ on part of the period, so that part of the integral is left out."
                 if len(nonzero) < len(pieces) else "")

    argw = arg.subs(nn, _n_work)

    def antiderivative(e):
        """The antiderivative of e trig(arg), expanded term by term (not combined
        into one fraction), as it comes out of the method shown. For the
        rectified sine it's worked from the product-to-sum form."""
        if trig is not None and e.has(sin):
            c = e.coeff(sin(t))
            pair = (sin(t + argw) + sin(t - argw)) if trig is cos else (cos(t - argw) - cos(t + argw))
            return expand(c / 2 * integrate(pair, t, conds='none'))
        return expand(integrate(e * (trig(argw) if trig else 1), t, conds='none'))

    def at_limits(parts):
        """'[G_1]_{s}^{f} + ...' for each piece's antiderivative."""
        return poly(*[(1, rf"\left[{latex(G)}\right]_{{{latex(s)}}}^{{{latex(f)}}}") for G, s, f in parts])

    parts = [(antiderivative(e), s, f) for e, s, f in nonzero]
    if trig is None:   # a_0: a single step
        return [Step(rf"Find $a_0$ (the $n = 0$ coefficient)",
                     rf"{setup} \\ = {prefix}\left[{at_limits(parts)}\right] = {latex(result)}", zero_note)]

    # Step 2: each piece's integral, with its method.
    V = integrate(trig(argw), t)                  # the integral of trig(arg) on its own
    method_lines, methods = [], set()
    for (e, s, f), (G, _, _) in zip(nonzero, parts):
        lhs = rf"\int {_bracket(e)}\{trig.__name__}\left({latex(argw)}\right)dt"
        if e.has(sin):                            # rectified sine: product to sum
            methods.add("sum")
            half = coef(e.coeff(sin(t)) / 2, r"\int")
            plus, minus = factor(t + argw), factor(t - argw)
            pair = (rf"\sin\left({latex(plus)}\right) + \sin\left({latex(minus)}\right)" if trig is cos else
                    rf"\cos\left({latex(minus)}\right) - \cos\left({latex(plus)}\right)")
            method_lines.append(rf"{lhs} = {half}\left[{pair}\right]dt = {latex(G)}")
        elif e.has(t):                            # a power of t: by parts, u = e, dv = trig dt
            methods.add("parts")
            rest = simplify(diff(e, t) * V)
            method_lines.append(rf"{lhs} = {_bracket(e)}{_bracket(V)} - \int {_bracket(rest)}\,dt = {latex(G)}")
        else:                                     # a constant times trig: integrate directly
            method_lines.append(rf"{lhs} = {latex(G)}")
    method_note = []
    if "parts" in methods:
        method_note.append(r"Integrate by parts, $\int u\,dv = uv - \int v\,du$, with $u$ the power of $t$ "
                           r"and $dv$ the trig function" + (" (twice, for $t^2$)." if any(e.has(t**2) for e, _, _ in nonzero) else "."))
    if "sum" in methods:
        method_note.append(r"Use $\sin A\cos B = \tfrac{1}{2}\left[\sin(A + B) + \sin(A - B)\right]$ "
                           r"(or $\sin A\sin B = \tfrac{1}{2}\left[\cos(A - B) - \cos(A + B)\right]$) "
                           r"to turn the product into a sum first.")

    # Step 3: put in the limits, (upper) - (lower) for each piece as in the
    # lecture solutions, then use sin(n pi) = 0 and cos(n pi) = (-1)^n.
    def upper_minus_lower(G, s, f):
        hi, lo = G.subs(t, f), G.subs(t, s)
        return rf"\left({latex(hi)}\right)" + ("" if lo == 0 else rf" - \left({latex(lo)}\right)")
    evaluated = " + ".join(upper_minus_lower(*pc) for pc in parts)
    limits_note = (r"$n$ is a positive integer, so $\sin(n\pi) = 0$ and $\cos(n\pi) = (-1)^n$ "
                   r"(so $\sin(2n\pi) = 0$ and $\cos(2n\pi) = 1$).")

    return [
        Step(rf"Find ${name}$: set up the integral", setup, zero_note),
        Step("Do each integral", r" \\ ".join(method_lines), " ".join(method_note)),
        Step(r"Put in the limits, then simplify",
             rf"{name} = {prefix}\left[{at_limits(parts)}\right] \\ "
             rf"= {prefix}\left[{evaluated}\right] \\ = {latex(result)}",
             limits_note),
    ]


def _signed_terms(parts):
    """Join LaTeX terms with ' + ', or ' - ' when a term starts with a minus."""
    out = []
    for s_ in parts:
        if out and s_.startswith('-'):
            out.append(f"- {s_[1:].strip()}")
        else:
            out.append(f"+ {s_}" if out else s_)
    return ' '.join(out) or '0'


def _series_solver(p):
    pieces, display, t0, t1, T, arg = _series_parts(p)
    sym, sym_note = _symmetry(_numeric(pieces), t0, T)
    odd, even = sym == "Odd", sym == "Even"

    def coeff(trig):
        return simplify(2 / T * sum(integrate(e * trig(arg), (t, s, f)) for e, s, f in pieces))
    a0 = simplify(2 / T * sum(integrate(e, (t, s, f)) for e, s, f in pieces))
    an = Integer(0) if odd else coeff(cos)
    bn = Integer(0) if even else coeff(sin)

    # SymPy check: the function's symmetry really does kill the coefficients it should.
    if (odd and a0 != 0) or (even and coeff(sin) != 0):
        raise ValueError(f"symmetry {sym} doesn't match the coefficients for {p}")

    t0_sub = latex(t0) if t0 >= 0 else rf"\left({latex(t0)}\right)"
    steps = [
        Step("Find the period, and write down the Fourier series for that period",
             rf"T = {latex(t1)} - {t0_sub} = {latex(T)} \\ "
             rf"f(t) = \frac{{a_0}}{{2}} + \sum_{{n=1}}^{{\infty}}\left[a_n\cos\left({latex(arg)}\right) "
             rf"+ b_n\sin\left({latex(arg)}\right)\right]",
             rf"In general the arguments are $\frac{{2n\pi t}}{{T}}$, and "
             rf"$a_n = \frac{{2}}{{T}}\int_{{\text{{period}}}} f(t)\cos\left(\frac{{2n\pi t}}{{T}}\right)dt$, "
             rf"$b_n = \frac{{2}}{{T}}\int_{{\text{{period}}}} f(t)\sin\left(\frac{{2n\pi t}}{{T}}\right)dt$."),
        Step("Check the symmetry (this decides which coefficients we need)",
             rf"\text{{{sym}}}", sym_note),
    ]
    if not odd:
        steps += _coefficient_steps("a_0", None, pieces, T, arg, a0)
        steps += _coefficient_steps("a_n", cos, pieces, T, arg, an)
    if not even:
        steps += _coefficient_steps("b_n", sin, pieces, T, arg, bn)

    # The general series, then its first few terms.
    general = []
    if a0 != 0:
        general.append(latex(a0 / 2))
    for c, trig in ((an, r"\cos"), (bn, r"\sin")):
        if c != 0:
            # A minus sign goes in front of the sum: -\sum 8(-1)^n/(pi n) ..., not \sum -8(-1)^n/(pi n) ...
            neg = c.could_extract_minus_sign()
            general.append(("-" if neg else "")
                           + rf"\sum_{{n=1}}^{{\infty}} {latex(-c if neg else c)}{trig}\left({latex(arg)}\right)")
    first = [latex(a0 / 2)] if a0 != 0 else []
    for k in range(1, _N_TERMS + 1):
        for c, trig in ((an.subs(nn, k), r"\cos"), (bn.subs(nn, k), r"\sin")):
            if c != 0:
                first.append(rf"{latex(c)}{trig}\left({latex(arg.subs(nn, k))}\right)")
    steps.append(Step("Put the coefficients into the series",
                      rf"f(t) = {_signed_terms(general)} \\ = {_signed_terms(first)} + \cdots",
                      f"Writing out the terms up to $n = {_N_TERMS}$ (terms whose coefficient is zero are left out)."))
    return steps


def _series_prompt(p):
    pieces, display, t0, t1, T, _ = _series_parts(p)
    return ("Find the Fourier series of the periodic function\n\n$$\n" + _sketch_definition(pieces, display)
            + "\n$$\n\n" + rf"defined for ${latex(t0)} < t < {latex(t1)}$ and repeated periodically "
            "(sketched below).")


series_template = QuestionTemplate(
    id="fourier-series-periodic",
    topic="Fourier Analysis",
    subtopic="Fourier Series",
    difficulty=2,
    prompt_fn=_series_prompt,
    solver=_series_solver,
    figure_fn=_series_figure,
    # The same shapes as the sketching subtopic (see _sketch_shape).
    parameters={"a": (1, 3), "L": (1, 3), "kind": (0, 6)},
)


# ── Fourier Transforms ────────────────────────────────────────────────────────
# 
#   f~(omega) = integral_{-inf}^{inf} f(x) e^{-i omega x} dx
#   modify the limits so only the parts where f(x) is not 0 are included,
#   do the integral, evaluate e^{-i omega x} between the limits, and simplify
#   with e^{i theta} - e^{-i theta} = 2i sin(theta) (or + ... = 2 cos(theta)).
# Two shapes, with random height c and half-width a:
#   top hat     c on -a < x < a                    (even: real transform, 2c sin(a omega)/omega)
#   odd pair    -c on -a < x < 0, c on 0 < x < a   (odd: purely imaginary transform)
# The question shows f(x); the last step plots the transform's real and imaginary parts.

def _eiw(k):
    """LaTeX for e^{k i omega}, e.g. _eiw(-2) -> 'e^{-2i\\omega}', _eiw(1) -> 'e^{i\\omega}'."""
    return rf"e^{{{coef(k, r'i\omega')}}}"


def _over_iw(k):
    """LaTeX for k / (i omega), sign out front: '\\frac{2}{i\\omega}', '-\\frac{1}{i\\omega}'."""
    return ("-" if k < 0 else "") + rf"\frac{{{abs(k)}}}{{i\omega}}"


def _ft_parts(p):
    """(pieces, display, transform) for the chosen shape. pieces are (value, lo, hi)
    where f is non-zero; transform is the SymPy result in omega."""
    c, a = p["c"], p["a"]
    if p["kind"] < 2:   # top hat (two thirds of the time)
        return [(c, -a, a)], (rf"\begin{{cases}} {c}, & {-a} < x < {a} \\ 0, & \text{{otherwise}} \end{{cases}}"), \
            2 * c * sin(a * omega) / omega
    return ([(-c, -a, 0), (c, 0, a)],
            (rf"\begin{{cases}} {-c}, & {-a} < x < 0 \\ {c}, & 0 < x < {a} \\ 0, & \text{{otherwise}} \end{{cases}}"),
            -2 * I * c * (1 - cos(a * omega)) / omega)


def _ft_solver(p):
    pieces, display, F = _ft_parts(p)
    c, a = p["c"], p["a"]
    w = symbols('w', positive=True)   # a positive stand-in for omega in the check (avoids omega = 0)
    exact = sum(integrate(v * exp(-I * w * x), (x, lo, hi)) for v, lo, hi in pieces)
    if simplify((exact - F.subs(omega, w)).rewrite(exp)) != 0:
        raise ValueError(f"transform check failed for {p}")

    def piece_integral(v, lo, hi):
        return rf"\int_{{{lo}}}^{{{hi}}} ({v})\,e^{{-i\omega x}}\,dx"

    def bracket(v, lo, hi):
        """v/(-i omega) e^{-i omega x} evaluated from lo to hi, as in the lecture."""
        return rf"{_over_iw(-v)}\,e^{{-i\omega x}}\Big|_{{x={lo}}}^{{x={hi}}}"

    steps = [
        Step("The Fourier transform is:",
             r"\tilde{f}(\omega) = \int_{-\infty}^{\infty} f(x)\,e^{-i\omega x}\,dx"),
        Step(r"Modify the limits so that we only include the parts where $f(x)$ is not 0:",
             r"\tilde{f}(\omega) = " + " + ".join(piece_integral(*pc) for pc in pieces)),
    ]
    if len(pieces) == 1:    # top hat
        k = -c   # c / (-i omega) = -c / (i omega)
        steps += [
            Step("Do the integral:",
                 rf"= {bracket(c, -a, a)} = {_over_iw(k)}\left[{_eiw(-a)} - {_eiw(a)}\right]"),
            Step(r"Use $e^{i\theta} - e^{-i\theta} = 2i\sin(\theta)$:",
                 rf"= {_over_iw(k)}\left[-2i\sin({coef(a, r'\omega')})\right] = {latex(F)}",
                 "The top hat is an even function, so its Fourier transform is real."),
        ]
    else:                   # odd pair
        steps += [
            Step("Do each integral:",
                 rf"= {_signed_terms([bracket(-c, -a, 0), bracket(c, 0, a)])} \\ "
                 rf"= {_signed_terms([rf'{_over_iw(c)}\left[1 - {_eiw(a)}\right]', rf'{_over_iw(-c)}\left[{_eiw(-a)} - 1\right]'])} \\ "
                 rf"= {_over_iw(c)}\left[2 - \left({_eiw(a)} + {_eiw(-a)}\right)\right]"),
            Step(r"Use $e^{i\theta} + e^{-i\theta} = 2\cos(\theta)$, and $\frac{1}{i} = -i$:",
                 rf"= {_over_iw(c)}\left[2 - 2\cos({coef(a, r'\omega')})\right] = {latex(F)}",
                 "This function is odd, so its Fourier transform is purely imaginary."),
        ]
    # re_fn = lambdify(omega, re(F.subs(omega, w)).subs(w, omega), "numpy")
    # im_fn = lambdify(omega, im(F.subs(omega, w)).subs(w, omega), "numpy")
    # steps.append(Step("Plot the transform",
    #                   rf"\tilde{{f}}(\omega) = {latex(F)}",
    #                   image=transform_plot(re_fn, im_fn, wmax=20 / a)))
    return steps


def _ft_prompt(p):
    _, display, _ = _ft_parts(p)
    return ("Find the Fourier transform of the function\n\n$$\n"
            rf"f(x) = {display}" "\n$$\n\n"
            "(sketched below).")


def _ft_figure(p):
    """f(x) over a window a little wider than where it's non-zero."""
    pieces, _, _ = _ft_parts(p)
    a = p["a"]
    edge = a + 2
    full = [(Integer(0), -edge, -a)] + [(Integer(v), lo, hi) for v, lo, hi in pieces] + [(Integer(0), a, edge)]
    num = [(lambdify(x, e, "numpy"), float(lo), float(hi)) for e, lo, hi in full]
    ticks = [(float(v), f"${v}$") for v in sorted({lo for _, lo, _ in full[1:]} | {a})]
    return periodic_sketch(num, float(-edge), float(2 * edge), [0], ticks, xlabel="x", ylabel="f(x)")


ft_template = QuestionTemplate(
    id="fourier-transform-top-hat",
    topic="Fourier Analysis",
    subtopic="Fourier Transforms",
    difficulty=2,
    prompt_fn=_ft_prompt,
    solver=_ft_solver,
    figure_fn=_ft_figure,
    # kind 0-1: top hat, 2: odd pair; c: height (non-zero); a: half-width.
    parameters={"kind": (0, 2), "c": (-3, 3), "a": (1, 3)},
    constraints=[lambda p: p["c"] != 0],
)


# Order here = order of subtopics (and of templates within a subtopic) in the sidebar.
ALL_TEMPLATES = [
    sketch_template,
    series_template,
    ft_template,
]
