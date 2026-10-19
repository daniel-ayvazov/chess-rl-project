"""Train an agent to checkmate with King + Rook vs King, from scratch.

    python train.py                  # ~30s on a laptop CPU
    python train.py --episodes 100000 --seed 1
"""
import argparse
import json
import os
import random
import time

import numpy as np

from krk.agent import Agent, Env, evaluate, play_episode
from krk.render import learning_curve
from krk.tablebase import load_or_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300_000)
    ap.add_argument("--alpha", type=float, default=0.3, help="TD learning rate")
    ap.add_argument("--eps-start", type=float, default=0.10)
    ap.add_argument("--eps-end", type=float, default=0.01)
    ap.add_argument("--black-random", type=float, default=0.10,
                    help="chance the defender plays a random move during training "
                         "(widens coverage; evaluation is always vs perfect defence)")
    ap.add_argument("--eval-every", type=int, default=10_000)
    ap.add_argument("--eval-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    tb = load_or_build()
    env = Env(tb)
    agent = Agent(len(tb.b_pos))
    n_w = len(tb.w_pos)

    eval_rng = random.Random(12345)
    eval_starts = [eval_rng.randrange(n_w) for _ in range(args.eval_size)]
    rng = random.Random(args.seed)

    curve = [{"episode": 0, **evaluate(env, agent, eval_starts)}]
    t0 = time.time()
    print(f"{'games':>8} {'mate%':>7} {'perfect%':>9} {'extra moves':>12} {'time':>6}")
    for ep in range(1, args.episodes + 1):
        frac = ep / args.episodes
        eps = args.eps_start * (1 - frac) + args.eps_end * frac
        play_episode(env, agent, rng.randrange(n_w), rng, learn=True, eps=eps,
                     alpha=args.alpha, black_random=args.black_random)
        if ep % args.eval_every == 0:
            m = evaluate(env, agent, eval_starts)
            curve.append({"episode": ep, **m})
            print(f"{ep:>8,} {m['mate_rate']:>7.1%} {m['optimal_rate']:>9.1%} "
                  f"{m['mean_excess_moves']:>12.2f} {time.time()-t0:>5.0f}s")

    np.save("artifacts/values.npy", np.array(agent.V, dtype=np.float32))
    with open("artifacts/curve.json", "w") as f:
        json.dump(curve, f, indent=1)
    learning_curve(curve, "assets/learning_curve.png")
    print("saved artifacts/values.npy, artifacts/curve.json, assets/learning_curve.png")


if __name__ == "__main__":
    main()
