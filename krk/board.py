"""Minimal King + Rook vs King chess rules.

Squares are 0..63 (a1=0, b1=1, ..., h8=63). A position is the triple
(white king, white rook, black king) plus whose turn it is. We never need a
full chess engine: these few rules are all the endgame requires.
"""
from __future__ import annotations

import numpy as np

FILES = "abcdefgh"


def sq_name(s: int) -> str:
    return f"{FILES[s & 7]}{(s >> 3) + 1}"


def _king_moves(s: int) -> list[int]:
    f, r = s & 7, s >> 3
    out = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == dr == 0:
                continue
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                out.append(nr * 8 + nf)
    return out


KING_MOVES = [_king_moves(s) for s in range(64)]
ADJ = [frozenset(m) for m in KING_MOVES]
ROOK_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _between(a: int, b: int, c: int) -> bool:
    """Is square c strictly between a and b on a shared rank or file?"""
    af, ar, bf, br, cf, cr = a & 7, a >> 3, b & 7, b >> 3, c & 7, c >> 3
    if ar == br == cr:
        return min(af, bf) < cf < max(af, bf)
    if af == bf == cf:
        return min(ar, br) < cr < max(ar, br)
    return False


def rook_attacks(wr: int, t: int, wk: int) -> bool:
    """Does a rook on wr attack t? Only the white king can block (the black
    king is the piece being attacked / moving, so it never blocks)."""
    if wr == t:
        return False
    if (wr & 7) != (t & 7) and (wr >> 3) != (t >> 3):
        return False
    return not _between(wr, t, wk)


def legal(wk: int, wr: int, bk: int, white_to_move: bool) -> bool:
    if wk == wr or wk == bk or wr == bk or bk in ADJ[wk]:
        return False
    # With white to move, black cannot already be in check.
    if white_to_move and rook_attacks(wr, bk, wk):
        return False
    return True


def in_check(wk: int, wr: int, bk: int) -> bool:
    return rook_attacks(wr, bk, wk)


def white_moves(wk: int, wr: int, bk: int) -> list[tuple[int, int, int]]:
    """All positions reachable by one white move (black to move after)."""
    out = []
    for t in KING_MOVES[wk]:
        if t != wr and t != bk and t not in ADJ[bk]:
            out.append((t, wr, bk))
    f, r = wr & 7, wr >> 3
    for df, dr in ROOK_DIRS:
        nf, nr = f + df, r + dr
        while 0 <= nf < 8 and 0 <= nr < 8:
            t = nr * 8 + nf
            if t == wk or t == bk:
                break
            out.append((wk, t, bk))
            nf, nr = nf + df, nr + dr
    return out


def black_moves(wk: int, wr: int, bk: int):
    """Returns (successor positions, can_capture_rook)."""
    out, capture = [], False
    for t in KING_MOVES[bk]:
        if t == wk or t in ADJ[wk]:
            continue
        if t == wr:
            capture = True  # rook is undefended -> black grabs it, draw
            continue
        if rook_attacks(wr, t, wk):
            continue
        out.append((wk, wr, t))
    return out, capture


# ---------------------------------------------------------------- symmetry --
# KRK has no pawns or castling, so all 8 board symmetries (dihedral group)
# preserve the game. We map every position to the smallest index among its
# 8 images, shrinking the state space ~8x.

def encode(wk, wr, bk):
    return (wk * 64 + wr) * 64 + bk


def _square_transforms() -> np.ndarray:
    s = np.arange(64)
    f, r = s & 7, s >> 3
    maps = []
    for flip_f in (False, True):
        for flip_r in (False, True):
            for transpose in (False, True):
                ff = 7 - f if flip_f else f
                rr = 7 - r if flip_r else r
                if transpose:
                    ff, rr = rr, ff
                maps.append(rr * 8 + ff)
    return np.array(maps)  # (8, 64)


SYM = _square_transforms()


def canonical_table() -> np.ndarray:
    """canon[idx] = canonical index of the position encoded by idx."""
    idx = np.arange(64 ** 3)
    wk, wr, bk = idx // 4096, (idx // 64) % 64, idx % 64
    best = idx.copy()
    for m in SYM:
        best = np.minimum(best, (m[wk] * 64 + m[wr]) * 64 + m[bk])
    return best
