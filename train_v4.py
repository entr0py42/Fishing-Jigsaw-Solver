import argparse
import os

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

BOARD_ROWS, BOARD_COLS = 4, 6
BOARD_CELLS = BOARD_ROWS * BOARD_COLS

MODEL_NAME = "puzzle_packer"
CHECKPOINT_DIR = "checkpoints"
TENSORBOARD_DIR = "tb_logs/"
NUM_ENVS = 8
MAX_EPISODE_STEPS = 60

# Piece cells as (row, col) offsets from the anchor cell.
PIECE_SHAPES = {
    0: [(0, 0), (0, 1), (1, 1), (1, 2)],  # Z
    1: [(0, 0), (0, 1), (1, 0), (1, 1)],  # square
    2: [(0, 0), (1, 0), (1, 1)],          # L
    3: [(0, 0), (0, 1), (1, 1)],          # reversed L
    4: [(0, 0), (1, 0), (2, 0)],          # stick
    5: [(0, 0)],                          # dot
}
NUM_PIECE_TYPES = len(PIECE_SHAPES)

DISCARD_ACTION = BOARD_CELLS
ACTION_COUNT = BOARD_CELLS + 1
OBSERVATION_SIZE = 2 * BOARD_CELLS + NUM_PIECE_TYPES + 1

INVALID_ACTION_PENALTY = -10.0

# A cleared board pays 3x under 11 pieces, 2x up to 24 pieces, 1x beyond that.
FAST_CLEAR_LIMIT = 11
STANDARD_CLEAR_LIMIT = 24


def clear_multiplier(pieces_used: int) -> float:
    if pieces_used < FAST_CLEAR_LIMIT:
        return 3.0
    if pieces_used <= STANDARD_CLEAR_LIMIT:
        return 2.0
    return 1.0


class GridPackerEnv(gym.Env):
    """Fill a 4x6 board with random pieces; the reward is multiplier / pieces used."""

    def __init__(self, max_steps=MAX_EPISODE_STEPS):
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_COUNT)
        self.max_steps = max_steps
        self.reset()

    @property
    def pieces_used(self) -> int:
        return self.pieces_placed + self.pieces_discarded

    @property
    def is_board_full(self) -> bool:
        return bool(np.all(self.grid == 1))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.step_count = 0
        self.pieces_placed = 0
        self.pieces_discarded = 0
        self._draw_next_piece()
        return self._get_obs(), {}

    def _draw_next_piece(self):
        self.current_piece = int(self.np_random.integers(0, NUM_PIECE_TYPES))

    def _valid_placements(self) -> np.ndarray:
        valid = np.zeros(BOARD_CELLS, dtype=bool)
        for cell in range(BOARD_CELLS):
            row, col = divmod(cell, BOARD_COLS)
            valid[cell] = self.can_place(self.current_piece, row, col)
        return valid

    def _get_obs(self):
        piece_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
        piece_one_hot[self.current_piece] = 1.0
        usage = np.array([self.pieces_used / self.max_steps], dtype=np.float32)

        return np.concatenate(
            [
                self.grid.flatten().astype(np.float32),
                self._valid_placements().astype(np.float32),
                piece_one_hot,
                usage,
            ]
        ).astype(np.float32)

    def action_masks(self) -> np.ndarray:
        mask = np.append(self._valid_placements(), True)
        return mask

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = False
        reward = 0.0

        if action == DISCARD_ACTION:
            self.pieces_discarded += 1
            self._draw_next_piece()
        else:
            row, col = divmod(action, BOARD_COLS)
            if self.can_place(self.current_piece, row, col):
                self.place_piece(self.current_piece, row, col)
                self.pieces_placed += 1
                self._draw_next_piece()
            else:
                reward = INVALID_ACTION_PENALTY

        if self.is_board_full:
            terminated = True
            reward += clear_multiplier(self.pieces_used) / float(self.pieces_used)

        if self.step_count >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def can_place(self, piece_id, row, col):
        for d_row, d_col in PIECE_SHAPES[piece_id]:
            r, c = row + d_row, col + d_col
            if not (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS) or self.grid[r, c] == 1:
                return False
        return True

    def place_piece(self, piece_id, row, col):
        for d_row, d_col in PIECE_SHAPES[piece_id]:
            self.grid[row + d_row, col + d_col] = 1


def mask_fn(env: GridPackerEnv) -> np.ndarray:
    return env.action_masks()


def make_single_env():
    return ActionMasker(GridPackerEnv(), mask_fn)


class RatioEvalCallback(BaseCallback):
    """Periodically plays greedy episodes and logs the mean reward on cleared boards."""

    def __init__(self, eval_env, eval_interval=100_000, episodes_per_eval=500, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.episodes_per_eval = episodes_per_eval
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_interval:
            self._last_eval_step = self.num_timesteps
            self._evaluate()
        return True

    def _evaluate(self):
        env = self.eval_env
        fast_clears = standard_clears = slow_or_failed = 0
        cleared_rewards = []

        for _ in range(self.episodes_per_eval):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action, _ = self.model.predict(
                    obs, deterministic=True, action_masks=env.action_masks()
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

            if env.is_board_full:
                cleared_rewards.append(episode_reward)

            if env.pieces_used < FAST_CLEAR_LIMIT:
                fast_clears += 1
            elif env.pieces_used <= STANDARD_CLEAR_LIMIT:
                standard_clears += 1
            else:
                slow_or_failed += 1

        n = self.episodes_per_eval
        mean_ratio = np.mean(cleared_rewards) if cleared_rewards else 0.0

        if self.verbose:
            print(
                f"\n[eval @ {self.num_timesteps} steps] "
                f"Avg X/N Ratio: {mean_ratio:.4f} | "
                f"<11: {fast_clears / n * 100:.1f}% | "
                f"11-24: {standard_clears / n * 100:.1f}% | "
                f"25+: {slow_or_failed / n * 100:.1f}%\n"
            )

        self.logger.record("eval/xn_ratio", mean_ratio)
        self.logger.record("eval/under_11_rate", fast_clears / n)
        self.logger.record("eval/under_25_rate", standard_clears / n)
        self.logger.record("eval/rest_rate", slow_or_failed / n)


def find_resume_path(final_model_path: str) -> str:
    """Prefer the newest checkpoint if it is more recent than the saved final model."""
    if not os.path.isdir(CHECKPOINT_DIR):
        return final_model_path

    checkpoints = [
        os.path.join(CHECKPOINT_DIR, name)
        for name in os.listdir(CHECKPOINT_DIR)
        if name.endswith(".zip")
    ]
    if not checkpoints:
        return final_model_path

    newest = max(checkpoints, key=os.path.getmtime)
    if not os.path.exists(final_model_path) or os.path.getmtime(newest) > os.path.getmtime(
        final_model_path
    ):
        return newest
    return final_model_path


def resolve_device(use_cuda: int) -> str:
    """Return "cuda" only when requested and available; otherwise fall back to CPU."""
    if use_cuda:
        if torch.cuda.is_available():
            return "cuda"
        print("Warning: --cuda 1 was requested but CUDA is not available. Falling back to CPU.")
    return "cpu"


def run_training(total_steps=1_000_000, use_cuda=0):
    device = resolve_device(use_cuda)
    print(f"--- Using device: {device} ---")
    vec_env = make_vec_env(make_single_env, n_envs=NUM_ENVS, vec_env_cls=SubprocVecEnv)
    eval_env = GridPackerEnv()
    final_model_path = f"{MODEL_NAME}.zip"

    resume_path = find_resume_path(final_model_path)

    if os.path.exists(resume_path):
        print(f"--- Loading existing model: {resume_path} ---")
        model = MaskablePPO.load(
            resume_path,
            env=vec_env,
            tensorboard_log=TENSORBOARD_DIR,
            device=device,
        )
    else:
        print(f"--- Creating new MaskablePPO model on {device}... ---")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            device=device,
            tensorboard_log=TENSORBOARD_DIR,
        )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    callbacks = CallbackList(
        [
            RatioEvalCallback(eval_env, eval_interval=100_000, episodes_per_eval=50),
            CheckpointCallback(
                save_freq=max(50_000 // NUM_ENVS, 1),
                save_path=f"{CHECKPOINT_DIR}/",
                name_prefix="ppo_ratio_ckpt",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
        ]
    )

    try:
        model.learn(
            total_timesteps=total_steps,
            progress_bar=True,
            callback=callbacks,
            tb_log_name="run_ratio_max",
            reset_num_timesteps=False,
        )
        model.save(MODEL_NAME)
        print(f"--- Completed & Saved: {final_model_path} ---")
    except KeyboardInterrupt:
        print("\n--- Interrupted, saving progress... ---")
        model.save(MODEL_NAME)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the MaskablePPO grid packer agent.")
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        choices=[0, 1],
        help="Set to 1 to train on the GPU (CUDA). Defaults to 0 (CPU).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(1_000_000, use_cuda=args.cuda)
