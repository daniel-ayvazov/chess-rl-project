"""Evaluate a trained agent against the perfect defender and render visuals.

    python evaluate.py               # after train.py
"""
import argparse
import random

import numpy as np

from krk.agent import Agent, Env, evaluate, play_episode
from krk.board import KING_MOVES, sq_name
from krk.render import BLUE, INK, ORANGE, replay, side_by_side_gif, value_vs_truth
from krk.tablebase import INF, load_or_build


def describe(pos):
    wk, wr, bk = pos
    return f"K{sq_name(wk)} R{sq_name(wr)} vs k{sq_name(bk)}"


def statuses(real, result, n_moves, optimal):
    out, colors = [], []
    for i in range(len(real)):
        out.append(f"Move {min((i + 1) // 2, n_moves)}" if i else "Start")
        colors.append(INK)
    if result == "mate":
        out[-1] = f"Checkmate in {n_moves}  (perfect: {optimal})"
        colors[-1] = BLUE
    elif result == "draw":
        out[-1] = f"Rook lost after {n_moves} moves: draw"
        colors[-1] = ORANGE
    else:
        out[-1] = "Fifty-move rule: draw"
        colors[-1] = ORANGE
    return out, colors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="random test positions")
    args = ap.parse_args()

    tb = load_or_build()
    env = Env(tb)
    trained = Agent(len(tb.b_pos), np.load("artifacts/values.npy").tolist())
    untrained = Agent(len(tb.b_pos))

    rng = random.Random(777)
    starts = [rng.randrange(len(tb.w_pos)) for _ in range(args.n)]
    hardest = np.flatnonzero((tb.dtm_w + 1) // 2 == 16).tolist()

    print(f"{'vs. perfect defender':<42} mate%   perfect%   extra moves/mate")
    for name, ag in (("untrained (random)", untrained), ("trained", trained)):
        for label, s in ((f"{args.n} random starts", starts),
                         (f"{len(hardest)} mate-in-16 starts", hardest)):
            m = evaluate(env, ag, s)
            print(f"{name:<19} {label:<22} {m['mate_rate']:>6.1%} {m['optimal_rate']:>9.1%} "
                  f"{m['mean_excess_moves']:>10.2f}")

    # Learned value vs. ground truth, over afterstates the agent has learned.
    V = np.array(trained.V)
    true_moves = tb.dtm_b / 2
    mask = (tb.dtm_b < INF) & (tb.dtm_b > 0) & (V > 0) & (V < 60)
    corr = np.corrcoef(true_moves[mask], V[mask])[0, 1]
    print(f"\nlearned value vs. true moves-to-mate: r = {corr:.3f} "
          f"over {mask.sum():,} positions")
    value_vs_truth(true_moves[mask].astype(int), V[mask], "assets/value_vs_truth.png")

    # Side-by-side GIF from a hardest (mate-in-16) position.
    chosen = None
    for w in hardest:
        for seed in range(5):
            res, n, traj = play_episode(env, trained, w, random.Random(seed), record=True)
            if res == "mate" and n == 16:
                chosen = (w, res, n, traj)
                break
        if chosen:
            break
    w, res_t, n_t, traj_t = chosen
    best_u = None
    for seed in range(200):
        res, n, traj = play_episode(env, untrained, w, random.Random(seed), record=True)
        if res == "draw" and 4 <= n <= 20:
            best_u = (res, n, traj)
            break
    if best_u is None:
        best_u = play_episode(env, untrained, w, random.Random(0), record=True)
    res_u, n_u, traj_u = best_u

    real_t, real_u = replay(tb, traj_t), replay(tb, traj_u)
    if res_u == "draw":
        wk, wr, bk = real_u[-1]
        if wr in KING_MOVES[bk]:          # show black king taking the rook
            real_u.append((wk, None, wr))
    st_t, col_t = statuses(real_t, res_t, n_t, 16)
    st_u, col_u = statuses(real_u, res_u, n_u, 16)
    side_by_side_gif(
        {"positions": real_u, "title": "Before training", "statuses": st_u, "colors": col_u},
        {"positions": real_t, "title": "After 300k self-taught games", "statuses": st_t, "colors": col_t},
        "assets/before_after.gif")
    print(f"GIF start position: {describe(real_t[0])} (white to move, mate in 16)")
    print("saved assets/value_vs_truth.png, assets/before_after.gif")


if __name__ == "__main__":
    main()
