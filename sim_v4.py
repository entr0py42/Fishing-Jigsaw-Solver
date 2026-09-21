import os

import numpy as np
from sb3_contrib import MaskablePPO

from train_v4 import MODEL_NAME, GridPackerEnv, clear_multiplier


def choose_action(model, obs, mask):
    if model is not None:
        action, _ = model.predict(obs, deterministic=True, action_masks=mask)
        return action
    return np.random.choice(np.flatnonzero(mask))


def play_round(env, model, piece_budget):
    """Play boards until the piece budget runs out; returns per-tier counts and Xs earned."""
    clears = {3.0: 0, 2.0: 0, 1.0: 0}
    cutoffs = 0
    pieces_left = piece_budget

    while pieces_left > 0:
        obs, _ = env.reset()
        done = False

        while not done and pieces_left > 0:
            action = choose_action(model, obs, env.action_masks())
            obs, _, terminated, truncated, _ = env.step(action)
            pieces_left -= 1
            done = terminated or truncated

        if env.is_board_full:
            clears[clear_multiplier(env.pieces_used)] += 1
        else:
            cutoffs += 1

    xs_earned = sum(multiplier * count for multiplier, count in clears.items())
    return clears[3.0], clears[2.0], clears[1.0], cutoffs, xs_earned


def run_budget_benchmark(num_rounds=10, pieces_per_round=1000, model_path=None):
    env = GridPackerEnv()

    if model_path and os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model = MaskablePPO.load(model_path)
    else:
        print("Model file not found. Running random masked policy baseline.")
        model = None

    tier_3x, tier_2x, tier_1x, cutoffs, xs_per_round = [], [], [], [], []

    print(f"\nStarting benchmark: {num_rounds} rounds of {pieces_per_round} pieces each...")

    for round_index in range(num_rounds):
        fast, standard, slow, cut, xs = play_round(env, model, pieces_per_round)
        tier_3x.append(fast)
        tier_2x.append(standard)
        tier_1x.append(slow)
        cutoffs.append(cut)
        xs_per_round.append(xs)

        print(
            f"Round {round_index + 1:2d}/{num_rounds} | "
            f"<11 (3x): {fast:3d} | "
            f"11-24 (2x): {standard:3d} | "
            f"25+ (1x): {slow:3d} | "
            f"Earned: {xs:5.1f} Xs"
        )

    total_pieces = num_rounds * pieces_per_round
    print("\n" + "=" * 65)
    print(f"      {total_pieces:,} TOTAL PIECES ({num_rounds} x {pieces_per_round} ROUNDS) BENCHMARK      ")
    print("=" * 65)
    print(f"Average Xs Earned per Round:         {np.mean(xs_per_round):.2f} Xs  (±{np.std(xs_per_round):.2f})")
    print(f"Average Tier <11 (3x) Boards:        {np.mean(tier_3x):.2f}  (±{np.std(tier_3x):.2f})")
    print(f"Average Tier 11-24 (2x) Boards:      {np.mean(tier_2x):.2f}  (±{np.std(tier_2x):.2f})")
    print(f"Average Tier 25+ (1x) Boards:        {np.mean(tier_1x):.2f}  (±{np.std(tier_1x):.2f})")
    print(f"Average Mid-board Cutoffs per Round: {np.mean(cutoffs):.2f}")
    print("-" * 65)
    print(f"Total Xs Earned Across All Rounds:   {sum(xs_per_round):.1f} Xs")
    print("=" * 65)


if __name__ == "__main__":
    run_budget_benchmark(
        num_rounds=10,
        pieces_per_round=1000,
        model_path=f"{MODEL_NAME}.zip",
    )
