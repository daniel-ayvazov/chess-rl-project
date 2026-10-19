"""Sanity checks against known K+R vs K endgame theory.  Run: python -m pytest"""
import random

import numpy as np
import pytest

from krk import board as B
from krk.agent import MATE, Env
from krk.tablebase import INF, load_or_build


def sq(name):
    return "abcdefgh".index(name[0]) + 8 * (int(name[1]) - 1)


@pytest.fixture(scope="module")
def tb():
    return load_or_build()


def test_position_count_matches_theory(tb):
    # 28,056 legal black-to-move KRK positions up to symmetry (classic figure)
    assert len(tb.b_pos) == 28_056


def test_every_white_to_move_position_is_won(tb):
    assert (tb.dtm_w < INF).all()


def test_longest_mate_is_16_moves(tb):
    assert (tb.dtm_w.max() + 1) // 2 == 16


def test_mate_in_one(tb):
    # White Kb6, Rh1 vs black Ka8: Rh8#
    assert tb.mate_in(tb.w_id(sq("b6"), sq("h1"), sq("a8"))) == 1


def test_checkmate_detected(tb):
    env = Env(tb)
    b = tb.b_id(sq("b6"), sq("h8"), sq("a8"))
    assert env.reply(b, random.Random(0)) == MATE


def test_undefended_rook_can_be_captured():
    _, capture = B.black_moves(sq("a1"), sq("e5"), sq("e6"))
    assert capture


def test_defended_rook_cannot_be_captured():
    _, capture = B.black_moves(sq("d4"), sq("e5"), sq("e6"))
    assert not capture


def test_symmetry_is_consistent(tb):
    # Mirrored positions must have identical distance to mate.
    rng = np.random.default_rng(0)
    for w in rng.integers(0, len(tb.w_pos), 200):
        wk, wr, bk = (int(x) for x in tb.w_pos[w])
        for m in B.SYM:
            assert tb.dtm_w[tb.w_id(m[wk], m[wr], m[bk])] == tb.dtm_w[w]
