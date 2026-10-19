"""Board rendering (PIL) and matplotlib charts."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import board as B

# Palette (validated categorical slots 1-2 on a light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

LIGHT_SQ = (238, 232, 213)
DARK_SQ = (181, 136, 99)
HILITE = (246, 214, 92)
CHECK = (232, 106, 96)

_FONT_DIR = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def _font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(_FONT_DIR, name), size)


# ------------------------------------------------------- real-board replay --

def replay(tb, traj):
    """Trajectories store canonical (symmetry-reduced) ids. Convert them back
    into a continuous sequence of real board positions for display."""
    kind, wid = traj[0]
    pos = tuple(int(x) for x in tb.w_pos[wid])
    real = [pos]
    for kind, sid in traj[1:]:
        if kind == "B":   # white just moved
            cands = B.white_moves(*pos)
            pos = next(q for q in cands if tb.b_id(*q) == sid)
        else:              # black just moved
            cands, _ = B.black_moves(*pos)
            pos = next(q for q in cands if tb.w_id(*q) == sid)
        real.append(pos)
    return real


def draw_board(pos, prev=None, sq=56, title="", status="", status_color=INK):
    wk, wr, bk = pos   # wr may be None once the rook has been captured
    margin_top, margin_bot, pad = 46, 44, 22
    W = pad * 2 + sq * 8
    H = margin_top + sq * 8 + margin_bot
    img = Image.new("RGB", (W, H), SURFACE)
    d = ImageDraw.Draw(img)
    moved = set()
    if prev is not None:
        moved = {x for a, b in zip(prev, pos) if a != b for x in (a, b) if x is not None}
    checked = wr is not None and B.in_check(wk, wr, bk)
    coord = _font(11)
    for s in range(64):
        f, r = s & 7, s >> 3
        x0, y0 = pad + f * sq, margin_top + (7 - r) * sq
        col = LIGHT_SQ if (f + r) % 2 else DARK_SQ
        if s in moved:
            col = tuple(int(0.55 * c + 0.45 * h) for c, h in zip(col, HILITE))
        if checked and s == bk:
            col = CHECK
        d.rectangle([x0, y0, x0 + sq - 1, y0 + sq - 1], fill=col)
    for i in range(8):
        d.text((pad + i * sq + sq / 2, margin_top + 8 * sq + 4), "abcdefgh"[i],
               font=coord, fill=INK_2, anchor="mt")
        d.text((pad / 2, margin_top + (7 - i) * sq + sq / 2), str(i + 1),
               font=coord, fill=INK_2, anchor="mm")
    piece_font = _font(int(sq * 0.82))
    for s, glyph in ((wk, "♔"), (wr, "♖"), (bk, "♚")):
        if s is None:
            continue
        f, r = s & 7, s >> 3
        cx, cy = pad + f * sq + sq / 2, margin_top + (7 - r) * sq + sq / 2
        if glyph != "♚":  # white pieces: fill then outline
            d.text((cx, cy), "♚" if glyph == "♔" else "♜",
                   font=piece_font, fill=(255, 255, 255), anchor="mm")
        d.text((cx, cy), glyph, font=piece_font, fill=INK, anchor="mm")
    d.text((W / 2, 14), title, font=_font(16, bold=True), fill=INK, anchor="mt")
    d.text((W / 2, H - 14), status, font=_font(14, bold=True), fill=status_color, anchor="mb")
    return img


def side_by_side_gif(left, right, path, fps=2.5, hold=8):
    """left/right: dicts with keys positions, title, statuses, colors."""
    n = max(len(left["positions"]), len(right["positions"]))
    frames = []
    for i in range(n + hold):
        panels = []
        for side in (left, right):
            ps = side["positions"]
            j = min(i, len(ps) - 1)
            prev = ps[j - 1] if j > 0 else None
            panels.append(draw_board(ps[j], prev, title=side["title"],
                                     status=side["statuses"][j],
                                     status_color=side["colors"][j]))
        w, h = panels[0].size
        frame = Image.new("RGB", (w * 2 + 10, h), SURFACE)
        frame.paste(panels[0], (0, 0))
        frame.paste(panels[1], (w + 10, 0))
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


# ------------------------------------------------------------------ charts --

def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def learning_curve(curve, path):
    ep = np.array([c["episode"] for c in curve]) / 1000
    mate = np.array([c["mate_rate"] for c in curve]) * 100
    opt = np.array([c["optimal_rate"] for c in curve]) * 100
    exc = np.array([c["mean_excess_moves"] for c in curve], dtype=float)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    _style(a1)
    a1.plot(ep, mate, color=BLUE, lw=2)
    a1.plot(ep, opt, color=ORANGE, lw=2)
    a1.text(ep[-1], mate[-1] + 2, f"{mate[-1]:.0f}%",
            color=INK, fontsize=9, ha="right", va="bottom")
    a1.text(ep[-1], opt[-1] - 5, f"{opt[-1]:.0f}%",
            color=INK, fontsize=9, ha="right", va="top")
    a1.set_ylim(0, 108)
    a1.set_xlabel("Training games (thousands)", color=INK_2, fontsize=9)
    a1.set_ylabel("% of test positions", color=INK_2, fontsize=9)
    a1.set_title("Learning to mate a perfect defender", color=INK, fontsize=12,
                 loc="left", fontweight="bold")
    a1.legend(["Mates within 50 moves", "Plays the shortest mate"], frameon=False,
              fontsize=9, loc="center right", labelcolor=INK_2)

    _style(a2)
    ok = ~np.isnan(exc) & (mate >= 50)   # only once most games end in mate
    a2.plot(ep[ok], exc[ok], color=BLUE, lw=2)
    a2.set_ylim(bottom=0)
    a2.set_xlim(a1.get_xlim())
    a2.set_xlabel("Training games (thousands)", color=INK_2, fontsize=9)
    a2.set_ylabel("Extra moves vs. perfect play", color=INK_2, fontsize=9)
    a2.set_title("Wasted moves per checkmate", color=INK, fontsize=12, loc="left",
                 fontweight="bold")
    a2.text(ep[ok][-1], exc[ok][-1] + 0.3, f"{exc[ok][-1]:.2f}", color=INK,
            fontsize=9, ha="right", va="bottom")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def value_vs_truth(true_moves, learned, path):
    xs = np.unique(true_moves)
    mean = np.array([learned[true_moves == x].mean() for x in xs])
    lo = np.array([np.percentile(learned[true_moves == x], 10) for x in xs])
    hi = np.array([np.percentile(learned[true_moves == x], 90) for x in xs])
    fig, ax = plt.subplots(figsize=(6.5, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.plot([0, xs.max()], [0, xs.max()], color=INK_2, lw=1, ls="--")
    ax.fill_between(xs, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(xs, mean, color=BLUE, lw=2, marker="o", ms=5)
    ax.text(xs.max(), xs.max() - 0.6, "perfect agreement", color=INK_2,
            fontsize=8, ha="right", va="top")
    ax.set_xlabel("True moves to mate (from the tablebase)", color=INK_2, fontsize=9)
    ax.set_ylabel("Agent's learned estimate", color=INK_2, fontsize=9)
    ax.set_title("The agent rediscovered the tablebase", color=INK, fontsize=12,
                 loc="left", fontweight="bold")
    ax.text(0.02, 0.97, "line = mean, band = 10th–90th percentile\n"
            "(positions the agent actually visited)", transform=ax.transAxes,
            color=INK_2, fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
