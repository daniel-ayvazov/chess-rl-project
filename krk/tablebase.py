"""Perfect-play K+R vs K tablebase via retrograde analysis.

We enumerate every legal position (up to symmetry), build the move graph,
then repeatedly apply the minimax Bellman update in vectorized NumPy:

    dtm_W(w) = 1 + min over white moves  dtm_B(b)
    dtm_B(b) = 1 + max over black moves  dtm_W(w)   (0 if checkmated)

DTM ("distance to mate") is measured in plies. Any position where black can
capture the rook, or is stalemated, is a draw (DTM = INF). Iteration stops
when nothing changes. The resulting table is the ground truth the RL agent
is compared against, and it powers the perfect defender it trains against.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

from . import board as B

INF = 10_000


@dataclass
class Tablebase:
    w_pos: np.ndarray        # (nW, 3) canonical white-to-move positions
    b_pos: np.ndarray        # (nB, 3) canonical black-to-move positions
    w_off: np.ndarray        # CSR offsets into w_succ (len nW+1)
    w_succ: np.ndarray       # successor B ids
    b_off: np.ndarray        # CSR offsets into b_succ (len nB+1)
    b_succ: np.ndarray       # successor W ids
    b_capture: np.ndarray    # bool: black can take the rook
    b_check: np.ndarray      # bool: black king in check
    dtm_w: np.ndarray        # plies to mate, white to move
    dtm_b: np.ndarray        # plies to mate, black to move
    w_index: np.ndarray      # raw idx -> W id (-1 if not canonical/legal)
    b_index: np.ndarray      # raw idx -> B id
    canon: np.ndarray        # raw idx -> canonical raw idx

    # ---- convenience -------------------------------------------------------
    def w_id(self, wk, wr, bk) -> int:
        return int(self.w_index[self.canon[B.encode(wk, wr, bk)]])

    def b_id(self, wk, wr, bk) -> int:
        return int(self.b_index[self.canon[B.encode(wk, wr, bk)]])

    def mate_in(self, w: int) -> int:
        """Moves (not plies) to mate for a white-to-move position."""
        return (int(self.dtm_w[w]) + 1) // 2


def _segment_reduce(vals, off, ufunc, empty_value):
    n = len(off) - 1
    out = np.full(n, empty_value, dtype=np.int64)
    nonempty = off[1:] > off[:-1]
    if vals.size:
        red = ufunc.reduceat(vals, off[:-1][nonempty])
        out[nonempty] = red
    return out


def build(verbose: bool = True) -> Tablebase:
    t0 = time.time()
    canon = B.canonical_table()
    raw = np.arange(64 ** 3)
    is_canon = canon == raw

    w_list, b_list = [], []
    for idx in raw[is_canon]:
        wk, wr, bk = int(idx) // 4096, (int(idx) // 64) % 64, int(idx) % 64
        if B.legal(wk, wr, bk, True):
            w_list.append((wk, wr, bk))
        if B.legal(wk, wr, bk, False):
            b_list.append((wk, wr, bk))

    w_index = np.full(64 ** 3, -1, dtype=np.int64)
    b_index = np.full(64 ** 3, -1, dtype=np.int64)
    for i, p in enumerate(w_list):
        w_index[B.encode(*p)] = i
    for i, p in enumerate(b_list):
        b_index[B.encode(*p)] = i

    # White move graph (W -> B)
    w_off, w_succ = [0], []
    for p in w_list:
        for q in B.white_moves(*p):
            w_succ.append(b_index[canon[B.encode(*q)]])
        w_off.append(len(w_succ))
    # Black move graph (B -> W)
    b_off, b_succ, b_capture, b_check = [0], [], [], []
    for p in b_list:
        moves, cap = B.black_moves(*p)
        for q in moves:
            b_succ.append(w_index[canon[B.encode(*q)]])
        b_off.append(len(b_succ))
        b_capture.append(cap)
        b_check.append(B.in_check(*p))

    w_off, w_succ = np.array(w_off), np.array(w_succ, dtype=np.int64)
    b_off, b_succ = np.array(b_off), np.array(b_succ, dtype=np.int64)
    b_capture, b_check = np.array(b_capture), np.array(b_check)
    assert (w_succ >= 0).all() and (b_succ >= 0).all()

    # Retrograde / minimax value iteration
    n_b_moves = b_off[1:] - b_off[:-1]
    mated = (n_b_moves == 0) & b_check & ~b_capture
    dtm_b = np.full(len(b_list), INF, dtype=np.int64)
    dtm_b[mated] = 0
    dtm_w = np.full(len(w_list), INF, dtype=np.int64)
    can_lose = (n_b_moves > 0) & ~b_capture  # black has only non-drawing moves
    for it in range(200):
        new_w = np.minimum(1 + _segment_reduce(dtm_b[w_succ], w_off, np.minimum, INF), INF)
        best_b = np.minimum(1 + _segment_reduce(new_w[b_succ], b_off, np.maximum, INF), INF)
        new_b = np.where(can_lose, best_b, dtm_b)
        if np.array_equal(new_w, dtm_w) and np.array_equal(new_b, dtm_b):
            break
        dtm_w, dtm_b = new_w, new_b

    tb = Tablebase(np.array(w_list), np.array(b_list), w_off, w_succ, b_off,
                   b_succ, b_capture, b_check, dtm_w, dtm_b, w_index, b_index,
                   canon)
    if verbose:
        won = dtm_w < INF
        print(f"[tablebase] {len(w_list):,} white-to-move / {len(b_list):,} "
              f"black-to-move positions (after symmetry)")
        print(f"[tablebase] converged after {it} sweeps in {time.time()-t0:.1f}s; "
              f"{won.mean():.1%} of WTM positions are wins; "
              f"longest forced mate = {(int(dtm_w[won].max()) + 1) // 2} moves")
    return tb


def load_or_build(path: str = "artifacts/tablebase.npz", verbose=True) -> Tablebase:
    fields = Tablebase.__dataclass_fields__.keys()
    if os.path.exists(path):
        data = np.load(path)
        if verbose:
            print(f"[tablebase] loaded from {path}")
        return Tablebase(**{k: data[k] for k in fields})
    tb = build(verbose)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, **{k: getattr(tb, k) for k in fields})
    return tb
