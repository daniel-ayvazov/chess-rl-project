"""Environment + tabular TD agent that learns to checkmate from scratch.

The agent (White) knows only the legal moves. It learns a value V(b) for every
position *after* its own move ("afterstate"): the number of further white moves
needed to mate. Initialising V to 0 is wildly optimistic (it believes every
move mates immediately), which drives systematic exploration: as soon as a
line fails to mate, its value rises and the agent tries something else.

The opponent is a perfect defender that reads the tablebase and always picks
the reply that delays mate the longest (and grabs the rook if it can).
"""
from __future__ import annotations

import random

import numpy as np

from .tablebase import INF, Tablebase

MATE, DRAW = -1, -2
DRAW_COST = 100.0      # losing the rook / stalemate: far worse than any mate
MAX_MOVES = 50         # the fifty-move rule


class Env:
    def __init__(self, tb: Tablebase):
        self.tb = tb
        self.w_succ = [tb.w_succ[tb.w_off[i]:tb.w_off[i + 1]].tolist()
                       for i in range(len(tb.w_off) - 1)]
        self.b_succ = [tb.b_succ[tb.b_off[i]:tb.b_off[i + 1]].tolist()
                       for i in range(len(tb.b_off) - 1)]
        self.mated = (np.diff(tb.b_off) == 0) & tb.b_check & ~tb.b_capture
        self.mated = self.mated.tolist()
        dtm_w = tb.dtm_w
        # Perfect defender: replies that maximise distance to mate.
        self.best = []
        for i, succ in enumerate(self.b_succ):
            if tb.b_capture[i] or not succ:
                self.best.append(None)          # capture or stalemate -> draw
                continue
            vals = dtm_w[succ]
            m = vals.max()
            self.best.append([s for s, v in zip(succ, vals) if v == m])

    def reply(self, b: int, rng: random.Random, random_prob: float = 0.0) -> int:
        """Black's move from afterstate b. Returns a W id, MATE or DRAW."""
        if self.mated[b]:
            return MATE
        best = self.best[b]
        if best is None:
            return DRAW
        if random_prob and rng.random() < random_prob:
            return rng.choice(self.b_succ[b])
        return rng.choice(best)


class Agent:
    def __init__(self, n_afterstates: int, values=None):
        self.V = list(values) if values is not None else [0.0] * n_afterstates

    def greedy(self, succ: list[int], rng: random.Random) -> int:
        V = self.V
        m = min(V[s] for s in succ)
        return rng.choice([s for s in succ if V[s] == m])

    def act(self, succ, rng, eps=0.0):
        if eps and rng.random() < eps:
            return rng.choice(succ)
        return self.greedy(succ, rng)


def play_episode(env: Env, agent: Agent, start_w: int, rng: random.Random, *,
                 learn=False, eps=0.0, alpha=0.3, black_random=0.0,
                 record=False):
    """Play one game from a white-to-move position.

    Returns (result, n_white_moves, trajectory) where result is
    'mate', 'draw' (rook lost / stalemate) or 'timeout' (fifty-move rule).
    """
    V = agent.V
    traj = [("W", start_w)] if record else None
    b = agent.act(env.w_succ[start_w], rng, eps)
    for move in range(1, MAX_MOVES + 1):
        if record:
            traj.append(("B", b))
        r = env.reply(b, rng, black_random)
        if r == MATE:
            return "mate", move, traj
        if r == DRAW:
            if learn:
                V[b] += alpha * (DRAW_COST - V[b])
            return "draw", move, traj
        if record:
            traj.append(("W", r))
        succ = env.w_succ[r]
        if learn:
            target = 1.0 + V[agent.greedy(succ, rng)]
            V[b] += alpha * (target - V[b])
        b = agent.act(succ, rng, eps)
    return "timeout", MAX_MOVES, traj


def evaluate(env: Env, agent: Agent, starts, seed=0):
    """Greedy play vs the perfect defender on fixed start positions."""
    rng = random.Random(seed)
    mates, optimal, excess = 0, 0, []
    for w in starts:
        res, n, _ = play_episode(env, agent, w, rng)
        if res == "mate":
            mates += 1
            extra = n - env.tb.mate_in(w)
            excess.append(extra)
            optimal += extra == 0
    k = len(starts)
    return {
        "mate_rate": mates / k,
        "optimal_rate": optimal / k,
        "mean_excess_moves": float(np.mean(excess)) if excess else float("nan"),
    }
