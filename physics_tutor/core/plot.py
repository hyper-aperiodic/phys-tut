"""
Figures for worked solutions, rendered with matplotlib and returned as PNG
bytes for Step.image.

Figures are rendered to bytes straight away (not kept as live matplotlib
objects) so a Question stays lightweight in Streamlit's session state, and no
pyplot figures are left open between reruns.

Styling matches the app's dark theme (colours from app.py's CSS): one accent
colour for the curve, light text, and muted axes and guides. The background
is painted in the step card's own colour (not transparent), so the image
blends into the card but stays legible if viewed anywhere else.
"""
import io

import matplotlib
matplotlib.use("Agg")  # no GUI backend: draw straight to an image (safe inside Streamlit)
import matplotlib.pyplot as plt
import numpy as np

_CURVE = "#4f8ef7"   # the function itself (the app's accent blue)
_INK = "#e8e8e8"     # primary text: the period label and arrow
_MUTED = "#9ca3af"   # axes, tick labels, axis names
_GUIDE = "#6b7280"   # dashed period-boundary lines
_CARD = "#13161f"    # step-card background: the figure background, and the fill of open circles


def _png(fig):
    """Render a figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_CARD)
    plt.close(fig)
    return buf.getvalue()


def periodic_sketch(pieces, t0, T, shifts, xticks, period_label=None, xlabel="t", ylabel="f(t)"):
    """Sketch a function defined piecewise on one period, repeated over several periods.

    pieces        list of (f, start, end) on the base period [t0, t0 + T]. Each f is a
                  numpy-callable continuous on its closed piece, e.g. from sympy.lambdify.
    t0, T         start of the base period and the period length (floats)
    shifts        which copies to draw: k means the base period moved by k*T.
                  [0] draws one period; [-1, 0, 1] draws three, centred on the base one.
    xticks        list of (position, mathtext label) for the t-axis, e.g. (6.28, r"$2\\pi$")
    period_label  if given (e.g. r"$T = 4$"), the base period is shaded, every period
                  boundary gets a dashed line, and a double-headed arrow labelled with
                  this text spans one period above the curve.

    Jumps (where one piece's end value differs from the next piece's start value,
    including across the period boundary) get a vertical line joining the two
    sides, with open circles at both ends, matching definitions on open intervals
    like -2 < t < 2.
    """
    n = len(pieces)
    # Value of each piece at its own start and end, to detect jumps between pieces.
    ends = [(float(f(s)), float(f(e))) for f, s, e in pieces]
    jump_after = [not np.isclose(ends[i][1], ends[(i + 1) % n][0]) for i in range(n)]   # jump at piece i's end?

    wide = len(shifts) > 1
    fig, ax = plt.subplots(figsize=(8.5 if wide else 5, 3.2))
    ax.set_facecolor(_CARD)

    # Sample each piece once; the copies are the same samples shifted along t.
    # broadcast_to handles constant pieces, where f returns a single number.
    samples = []
    for f, s, e in pieces:
        tt = np.linspace(s, e, 400)
        samples.append((tt, np.broadcast_to(f(tt), tt.shape).astype(float)))

    for k in shifts:
        off = k * T
        for i, (f, s, e) in enumerate(pieces):
            tt, yy = samples[i]
            ax.plot(tt + off, yy, color=_CURVE, lw=2, solid_capstyle="round", zorder=3)
            # Vertical line across each jump, joining this piece's end to the next
            # piece's start, as in a hand-drawn square wave or sawtooth. Skipped
            # after the very last piece drawn, where there's nothing to join to.
            last_drawn = (k == max(shifts) and i == n - 1)
            if jump_after[i] and not last_drawn:
                ax.plot([e + off, e + off], [ends[i][1], ends[(i + 1) % n][0]],
                        color=_CURVE, lw=2, zorder=3)
            # Open circles on both sides of each jump.
            if jump_after[i - 1]:
                ax.plot(s + off, ends[i][0], "o", ms=6, mfc=_CARD, mec=_CURVE, mew=1.5, zorder=4)
            if jump_after[i]:
                ax.plot(e + off, ends[i][1], "o", ms=6, mfc=_CARD, mec=_CURVE, mew=1.5, zorder=4)

    # Vertical extent from the sampled curve (not just the end values, which miss
    # interior peaks like sin's), always including the t-axis, with headroom above
    # for the period arrow.
    lo = min(0.0, min(yy.min() for _, yy in samples))
    hi = max(0.0, max(yy.max() for _, yy in samples))
    span = (hi - lo) or 1.0
    arrow_y = hi + 0.18 * span
    ax.set_ylim(lo - 0.12 * span, hi + (0.38 if period_label else 0.12) * span)
    left, right = t0 + min(shifts) * T, t0 + (max(shifts) + 1) * T
    pad = 0.04 * (right - left)
    ax.set_xlim(left - pad, right + pad)

    if period_label:
        # Shade the base period, dash every period boundary, and label one period with T.
        ax.axvspan(t0, t0 + T, color=_CURVE, alpha=0.08, lw=0, zorder=1)
        for k in range(min(shifts), max(shifts) + 2):
            ax.axvline(t0 + k * T, color=_GUIDE, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.annotate("", xy=(t0, arrow_y), xytext=(t0 + T, arrow_y),
                    arrowprops=dict(arrowstyle="<->", color=_INK, lw=1.3, shrinkA=0, shrinkB=0))
        # The label gets a card-coloured box so the y-axis line doesn't run through it.
        ax.text(t0 + T / 2, arrow_y + 0.04 * span, period_label, color=_INK,
                ha="center", va="bottom", fontsize=12, zorder=5,
                bbox=dict(boxstyle="round,pad=0.25", fc=_CARD, ec="none"))

    # Textbook-style axes: lines through the origin when it's in view, otherwise along the edges.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_position(("data", 0) if left - pad < 0 < right + pad else ("axes", 0))
    ax.spines["bottom"].set_position(("data", 0))
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_xticks([p for p, _ in xticks], [lab for _, lab in xticks])
    # y ticks only within the curve's range (none up in the arrow's headroom), and
    # no 0, which would sit on top of the t-axis's own 0.
    # Whole-number ticks, since the generated functions reach whole-number values
    # (amplitudes, end values). Every integer in range if there are few enough,
    # otherwise an evenly spaced subset. Rounding first lets a sampled peak like
    # 0.99998 (no sample lands exactly on the top of sin) still count as 1.
    lo_r, hi_r = round(lo, 3), round(hi, 3)
    yt = list(range(int(np.ceil(lo_r)), int(np.floor(hi_r)) + 1))
    if len(yt) > 7:
        yt = matplotlib.ticker.MaxNLocator(nbins=4, integer=True).tick_values(lo, hi)
    ax.set_yticks([v for v in yt if lo_r <= v <= hi_r and v != 0])
    ax.text(right + pad, 0, f"  ${xlabel}$", color=_MUTED, va="center", ha="left", fontsize=11)
    ax.text(0 if left - pad < 0 < right + pad else left - pad, ax.get_ylim()[1], f"${ylabel}$",
            color=_MUTED, va="bottom", ha="center", fontsize=11)

    return _png(fig)


def field_path_map(values, path, start, end, label, arrows=None):
    """Colour map of a 2D field around a path, with the path drawn over it.

    values      numpy-callable of (x, y) giving the colour: a scalar field itself,
                or the magnitude of a vector field
    path        (xs, ys) arrays of points along the path, in direction of travel
    start, end  the path's end points as (x, y), labelled A and B on the plot
    label       colour bar label (mathtext), e.g. r"$\\phi(x,y)$"
    arrows      optional (Fx, Fy) numpy-callables of a vector field, drawn as
                arrows of equal length on top: the arrows show the direction,
                the colour shows the strength

    Values that change sign (a scalar field like 2xy around the origin) get a
    diverging colour map centred on 0, so positive and negative regions look
    different; otherwise a sequential one.

    The view is a square around the path with a margin, so straight lines along
    an axis still get a 2D region of field around them. The path is drawn in
    light ink (not the accent blue, which would vanish into the colour map),
    with an arrowhead halfway along showing the direction of travel.
    """
    xs, ys = path
    # Square view centred on the path, at least 2 units across, plus a margin.
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min(), 2.0) / 2 * 1.3
    extent = (cx - half, cx + half, cy - half, cy + half)
    gx, gy = np.meshgrid(np.linspace(cx - half, cx + half, 300),
                         np.linspace(cy - half, cy + half, 300))
    # broadcast_to handles fields that come out as a single number.
    vals = np.broadcast_to(values(gx, gy), gx.shape).astype(float)

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.set_facecolor(_CARD)
    if vals.min() < 0 < vals.max():
        lim = np.abs(vals).max()
        im = ax.imshow(vals, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="bilinear", extent=extent)
    else:
        im = ax.imshow(vals, origin="lower", cmap="magma", interpolation="bilinear", extent=extent)

    if arrows:
        # A coarse grid of unit arrows, faint enough not to fight the path.
        qx, qy = np.meshgrid(np.linspace(cx - half, cx + half, 13)[1:-1],
                             np.linspace(cy - half, cy + half, 13)[1:-1])
        u, v = (np.broadcast_to(f(qx, qy), qx.shape).astype(float) for f in arrows)
        size = np.hypot(u, v)
        size[size == 0] = 1.0   # zero vectors stay zero instead of dividing by 0
        ax.quiver(qx, qy, u / size, v / size, color=_INK, alpha=0.55, pivot="mid",
                  scale=16, width=0.005, headwidth=4, zorder=2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, color=_MUTED, fontsize=11)
    cbar.ax.tick_params(colors=_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(_GUIDE)

    # The path: a dark outline under a light line keeps it visible on every colour.
    ax.plot(xs, ys, color=_CARD, lw=4.5, solid_capstyle="round", zorder=3)
    ax.plot(xs, ys, color=_INK, lw=2.2, solid_capstyle="round", zorder=4)
    # Direction arrow halfway along: an arrowhead marker turned to the path's
    # direction there. The axes have equal aspect, so the angle in data
    # coordinates is also the angle on screen. (A plain triangle marker won't
    # do: an equilateral triangle has no single tip, so it can't show direction.)
    mid = len(xs) // 2
    a = np.arctan2(ys[mid + 1] - ys[mid - 1], xs[mid + 1] - xs[mid - 1])
    head = np.array([(1, 0), (-0.7, 0.6), (-0.35, 0), (-0.7, -0.6)])   # pointing along +x
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ax.plot(xs[mid], ys[mid], marker=head @ rot.T, ms=14, color=_INK,
            mec=_CARD, mew=1.2, zorder=5)
    for (px, py), name in ((start, "A"), (end, "B")):
        ax.plot(px, py, "o", ms=7, mfc=_INK, mec=_CARD, mew=1.5, zorder=6)
        # Label on whichever side of the point has more room, so it isn't cut off at the edge.
        right_half = px > cx
        ax.annotate(f"{name} $({px:g}, {py:g})$", (px, py), xytext=(-6 if right_half else 6, 6),
                    ha="right" if right_half else "left",
                    textcoords="offset points", color=_INK, fontsize=9, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=_CARD, ec="none", alpha=0.75))

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, integer=True))
    ax.tick_params(colors=_MUTED, labelsize=8)
    for side in ax.spines.values():
        side.set_color(_GUIDE)
    ax.set_xlabel("$x$", color=_MUTED, fontsize=11)
    ax.set_ylabel("$y$", color=_MUTED, fontsize=11, rotation=0, labelpad=10)

    return _png(fig)


_NORMAL = "#f87171"  # the surface normal arrow (red, as in the lecture sketches)


def surface_flux_sketch(field, n, c, lims, sign):
    """3D sketch of a flat rectangular surface in a vector field, for flux questions.

    field   (Fx, Fy, Fz) numpy-callables of (x, y, z)
    n       index of the axis the surface is perpendicular to (0 = x, 1 = y, 2 = z)
    c       the surface lies in the plane (that axis) = c
    lims    {axis index: (lo, hi)} for the two axes the surface spans
    sign    +1 or -1: which way the normal points along axis n

    Draws the surface as a translucent gridded rectangle (like the lecture's
    blue mesh), its normal as a red arrow from the centre, and the field as
    faint arrows of equal length on a coarse 3D grid (direction only).
    """
    # Plot ranges: a margin around the surface, and +-1.2 either side of the plane.
    ranges = [None] * 3
    for i, (lo, hi) in lims.items():
        ranges[i] = (min(lo, 0) - 0.3, hi + 0.3)
    ranges[n] = (c - 1.2, c + 1.2)

    def place(coords_by_axis):
        """Order a {axis: array} dict as (x, y, z)."""
        return [coords_by_axis[i] for i in range(3)]

    fig = plt.figure(figsize=(4.8, 4.2))
    ax = fig.add_subplot(projection="3d")
    ax.set_facecolor(_CARD)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0.03))
        a._axinfo["grid"]["color"] = (1, 1, 1, 0.08)
        a.line.set_color(_GUIDE)
    ax.tick_params(colors=_MUTED, labelsize=7)

    # Field arrows: unit length, so they show direction; zero vectors are skipped.
    g = [np.linspace(lo, hi, 4) for lo, hi in ranges]
    gx, gy, gz = np.meshgrid(*g, indexing="ij")
    u, v, w = (np.broadcast_to(f(gx, gy, gz), gx.shape).astype(float) for f in field)
    keep = np.hypot(np.hypot(u, v), w) > 0
    ax.quiver(gx[keep], gy[keep], gz[keep], u[keep], v[keep], w[keep], length=0.35,
              normalize=True, color=_INK, alpha=0.45, linewidth=0.8, arrow_length_ratio=0.35)

    # The surface: a gridded rectangle in the plane (axis n) = c.
    (i, (lo_i, hi_i)), (j, (lo_j, hi_j)) = sorted(lims.items())
    si, sj = np.meshgrid(np.linspace(lo_i, hi_i, 9), np.linspace(lo_j, hi_j, 9))
    X, Y, Z = place({i: si, j: sj, n: np.full_like(si, float(c))})
    ax.plot_surface(X, Y, Z, color=_CURVE, alpha=0.35, edgecolor=_CURVE, linewidth=0.4, shade=False)

    # The normal, from the centre of the surface.
    centre = place({i: (lo_i + hi_i) / 2, j: (lo_j + hi_j) / 2, n: float(c)})
    direction = place({i: 0.0, j: 0.0, n: float(sign)})
    ax.quiver(*centre, *direction, length=1.0, color=_NORMAL, linewidth=2.5, arrow_length_ratio=0.25)
    tip = [p + 1.1 * d for p, d in zip(centre, direction)]
    ax.text(*tip, r"$\hat{n}$", color=_NORMAL, fontsize=12)

    ax.set_xlim(*ranges[0]); ax.set_ylim(*ranges[1]); ax.set_zlim(*ranges[2])
    # zoom < 1 leaves room for the z-axis label, which is otherwise cut off.
    ax.set_box_aspect([hi - lo for lo, hi in ranges], zoom=0.85)
    for setter, name in ((ax.set_xlabel, "x"), (ax.set_ylabel, "y"), (ax.set_zlabel, "z")):
        setter(f"${name}$", color=_MUTED, fontsize=11, labelpad=2)

    return _png(fig)


_IMAG = "#f59e0b"   # imaginary part of a transform (orange, as in the lecture plots)


def transform_plot(re_fn, im_fn, wmax, label=r"$\tilde{f}(\omega)$"):
    """Real and imaginary parts of a Fourier transform against omega, like the
    lecture's "Hat transform" plot.

    re_fn, im_fn  numpy-callables of omega (they are never called at omega = 0,
                  where transforms like 2 sin(omega)/omega need a limit)
    wmax          omega runs from -wmax to wmax
    """
    w = np.linspace(-wmax, wmax, 2000)   # an even count, so 0 itself isn't sampled
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.set_facecolor(_CARD)
    ax.axhline(0, color=_GUIDE, lw=1, zorder=1)
    ax.plot(w, np.broadcast_to(re_fn(w), w.shape), color=_CURVE, lw=2, label="Real", zorder=3)
    ax.plot(w, np.broadcast_to(im_fn(w), w.shape), color=_IMAG, lw=2, label="Imaginary", zorder=2)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(_MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
    ax.tick_params(colors=_MUTED, labelsize=9)
    ax.set_xlabel(r"$\omega$", color=_MUTED, fontsize=11)
    ax.set_ylabel(label, color=_MUTED, fontsize=11)
    return _png(fig)
