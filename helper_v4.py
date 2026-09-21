import numpy as np
import pygame
import torch
from sb3_contrib import MaskablePPO

from train_v4 import (
    ACTION_COUNT,
    BOARD_CELLS,
    BOARD_COLS,
    BOARD_ROWS,
    DISCARD_ACTION,
    MAX_EPISODE_STEPS,
    MODEL_NAME,
    NUM_PIECE_TYPES,
    PIECE_SHAPES,
)

WINDOW_SIZE = (950, 560)
CELL_SIZE = 55
GRID_ORIGIN = (40, 40)

PIECE_NAMES = {
    0: "Z-Piece", 1: "Square", 2: "L-Small",
    3: "r-L Small", 4: "I-Stick", 5: "Dot",
}

PIECE_COLORS = {
    0: (255, 50, 50),
    1: (255, 255, 50),
    2: (50, 255, 50),
    3: (50, 255, 255),
    4: (200, 50, 255),
    5: (255, 255, 255),
}

WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
GREEN = (60, 220, 60)
RED = (220, 60, 60)

PIECE_BUTTONS = [
    pygame.Rect(40 + (i % 3) * 125, 290 + (i // 3) * 45, 120, 35)
    for i in range(NUM_PIECE_TYPES)
]
COUNTER_DOWN_BUTTON = pygame.Rect(260, 385, 35, 30)
COUNTER_UP_BUTTON = pygame.Rect(305, 385, 35, 30)
RESET_BUTTON = pygame.Rect(40, 435, 180, 35)
PRINT_BUTTON = pygame.Rect(230, 435, 180, 35)

PANEL_RECT = pygame.Rect(440, 20, 490, 510)
PANEL_TEXT_X = PANEL_RECT.x + 15
BAR_X = PANEL_RECT.x + 270
BAR_MAX_WIDTH = 180
TOP_ACTIONS_SHOWN = 5
MIN_DISPLAY_PROBABILITY = 0.0001

MODEL_PATH = f"{MODEL_NAME}.zip"
try:
    policy_model = MaskablePPO.load(MODEL_PATH)
    print(f"Successfully loaded {MODEL_PATH}")
except Exception as error:
    policy_model = None
    print(f"Warning: Model load failed or file missing! ({error})")


def describe_action(action_id):
    if action_id == DISCARD_ACTION:
        return "DISCARD"
    return f"Place (R:{action_id // BOARD_COLS}, C:{action_id % BOARD_COLS})"


class PolicyDebugger:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("AI Visual Solver - X/N Ratio Maximiser Debugger")
        self.font = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 13)

        self.grid = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=int)
        self.current_piece = 0
        self.pieces_used = 0
        self.max_steps = MAX_EPISODE_STEPS

        self.chosen_action = None
        self.action_probs = np.zeros(ACTION_COUNT)
        self.state_value = 0.0
        self.top_actions = []

        self.running = True
        self.update_ai()

    def can_place(self, piece_id, row, col):
        for d_row, d_col in PIECE_SHAPES[piece_id]:
            r, c = row + d_row, col + d_col
            if not (0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS) or self.grid[r, c] == 1:
                return False
        return True

    def valid_placements(self):
        valid = np.zeros(BOARD_CELLS, dtype=bool)
        for cell in range(BOARD_CELLS):
            row, col = divmod(cell, BOARD_COLS)
            valid[cell] = self.can_place(self.current_piece, row, col)
        return valid

    def get_obs(self):
        piece_one_hot = np.zeros(NUM_PIECE_TYPES, dtype=np.float32)
        piece_one_hot[self.current_piece] = 1.0
        usage = np.array([self.pieces_used / float(self.max_steps)], dtype=np.float32)

        return np.concatenate(
            [
                self.grid.flatten().astype(np.float32),
                self.valid_placements().astype(np.float32),
                piece_one_hot,
                usage,
            ]
        ).astype(np.float32)

    def get_action_mask(self):
        return np.append(self.valid_placements(), True)

    def update_ai(self):
        if policy_model is None:
            return

        obs = self.get_obs()
        mask = self.get_action_mask()

        action, _ = policy_model.predict(obs, deterministic=True, action_masks=mask)
        self.chosen_action = int(action)

        obs_tensor = torch.as_tensor(obs, device=policy_model.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, device=policy_model.device).unsqueeze(0)

        with torch.no_grad():
            distribution = policy_model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
            probs = distribution.distribution.probs.cpu().numpy().flatten()
            self.action_probs = probs

            values = policy_model.policy.predict_values(obs_tensor)
            self.state_value = float(values.cpu().numpy().flatten()[0])

        ranked = np.argsort(probs)[::-1]
        self.top_actions = [
            (idx, probs[idx])
            for idx in ranked[:TOP_ACTIONS_SHOWN]
            if probs[idx] > MIN_DISPLAY_PROBABILITY
        ]

    def print_game_state(self):
        print("\n" + "=" * 50)
        print("           CURRENT GAME STATE & POLICY LOGITS     ")
        print("=" * 50)
        print(f"Pieces Consumed So Far (N): {self.pieces_used}")
        print("\n--- Board Grid State (4x6) ---")
        print(self.grid)

        print("\n--- Current Piece ---")
        print(f"ID: {self.current_piece} ({PIECE_NAMES[self.current_piece]})")

        print("\n--- Model Predictions ---")
        print(f"Critic Estimated Value V(s): {self.state_value:+.4f}")

        if self.chosen_action == DISCARD_ACTION:
            print("Selected Action: DISCARD SHAPE")
        elif self.chosen_action is not None:
            row, col = divmod(self.chosen_action, BOARD_COLS)
            print(f"Selected Action: PLACE at Row {row}, Col {col} (Action ID {self.chosen_action})")

        print("\n--- Top Action Probabilities ---")
        for action_id, prob in self.top_actions:
            print(f"  - Action {action_id:2d} | {describe_action(action_id):<20} : {prob * 100:6.2f}%")
        print("=" * 50 + "\n")

    def draw_text(self, text, color, pos, font=None):
        self.screen.blit((font or self.font).render(text, True, color), pos)

    def draw_board(self):
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                cell = pygame.Rect(
                    GRID_ORIGIN[0] + col * CELL_SIZE,
                    GRID_ORIGIN[1] + row * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                fill = (60, 255, 60) if self.grid[row, col] == 1 else (50, 50, 50)
                pygame.draw.rect(self.screen, fill, cell)
                pygame.draw.rect(self.screen, (100, 100, 100), cell, 1)

        if self.chosen_action is not None and self.chosen_action < DISCARD_ACTION:
            anchor_row, anchor_col = divmod(self.chosen_action, BOARD_COLS)
            outline = PIECE_COLORS[self.current_piece]
            for d_row, d_col in PIECE_SHAPES[self.current_piece]:
                ghost = pygame.Rect(
                    GRID_ORIGIN[0] + (anchor_col + d_col) * CELL_SIZE,
                    GRID_ORIGIN[1] + (anchor_row + d_row) * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(self.screen, outline, ghost.inflate(-4, -4), 4)

    def draw_controls(self):
        for piece_id, button in enumerate(PIECE_BUTTONS):
            selected = self.current_piece == piece_id
            pygame.draw.rect(self.screen, PIECE_COLORS[piece_id] if selected else (60, 60, 60), button)
            label_color = (0, 0, 0) if selected else (200, 200, 200)
            self.draw_text(PIECE_NAMES[piece_id], label_color, (button.x + 8, button.y + 8))

        self.draw_text(f"Total Pieces Used (N): {self.pieces_used}", WHITE, (40, 390))
        pygame.draw.rect(self.screen, (80, 80, 80), COUNTER_DOWN_BUTTON)
        pygame.draw.rect(self.screen, (80, 80, 80), COUNTER_UP_BUTTON)
        self.draw_text("-1", WHITE, (268, 390))
        self.draw_text("+1", WHITE, (312, 390))

        pygame.draw.rect(self.screen, (150, 50, 50), RESET_BUTTON)
        pygame.draw.rect(self.screen, (50, 120, 180), PRINT_BUTTON)
        self.draw_text("RESET GRID", WHITE, (80, 443))
        self.draw_text("PRINT STATE", WHITE, (265, 443))

    def draw_panel(self):
        pygame.draw.rect(self.screen, (20, 20, 20), PANEL_RECT)
        pygame.draw.rect(self.screen, (70, 70, 70), PANEL_RECT, 2)

        self.draw_text("MODEL EVALUATION & POLICY LOGITS", GOLD, (PANEL_TEXT_X, 30))
        self.draw_text(f"Critic Estimated Value V(s): {self.state_value:+.4f}", WHITE, (PANEL_TEXT_X, 60))

        if self.chosen_action == DISCARD_ACTION:
            decision, decision_color = "DISCARD SHAPE", (255, 80, 80)
        elif self.chosen_action is not None:
            row, col = divmod(self.chosen_action, BOARD_COLS)
            decision, decision_color = f"PLACE SHAPE at Row {row}, Col {col}", (80, 255, 80)
        else:
            decision, decision_color = "NO MODEL", (150, 150, 150)
        self.draw_text(f"Decision: {decision}", decision_color, (PANEL_TEXT_X, 85))

        pygame.draw.line(
            self.screen, (60, 60, 60), (PANEL_TEXT_X, 115), (PANEL_RECT.x + 475, 115), 1
        )
        self.draw_text("Action Probability Distribution:", (200, 200, 200), (PANEL_TEXT_X, 125))

        y = 150
        for action_id, prob in self.top_actions:
            marker = ">> " if action_id == self.chosen_action else "   "
            label = f"{marker}DISCARD ACTION" if action_id == DISCARD_ACTION else f"{marker}{describe_action(action_id)}"
            bar_color = RED if action_id == DISCARD_ACTION else GREEN

            self.draw_text(f"{label:<20} {prob * 100:6.2f}%", WHITE, (PANEL_TEXT_X, y), self.font_small)
            pygame.draw.rect(self.screen, (40, 40, 40), (BAR_X, y + 2, BAR_MAX_WIDTH, 12))
            bar_width = int(prob * BAR_MAX_WIDTH)
            if bar_width > 0:
                pygame.draw.rect(self.screen, bar_color, (BAR_X, y + 2, bar_width, 12))
            y += 24

    def draw(self):
        self.screen.fill((40, 20, 20) if self.chosen_action == DISCARD_ACTION else (30, 30, 30))
        self.draw_board()
        self.draw_controls()
        self.draw_panel()
        pygame.display.flip()

    def paint_cell_under_mouse(self):
        pressed = pygame.mouse.get_pressed()
        if not (pressed[0] or pressed[2]):
            return

        mouse_x, mouse_y = pygame.mouse.get_pos()
        col = (mouse_x - GRID_ORIGIN[0]) // CELL_SIZE
        row = (mouse_y - GRID_ORIGIN[1]) // CELL_SIZE
        if 0 <= col < BOARD_COLS and 0 <= row < BOARD_ROWS:
            value = 1 if pressed[0] else 0
            if self.grid[row, col] != value:
                self.grid[row, col] = value
                self.update_ai()

    def handle_click(self, mouse_pos):
        for piece_id, button in enumerate(PIECE_BUTTONS):
            if button.collidepoint(mouse_pos):
                self.current_piece = piece_id
                self.update_ai()

        if COUNTER_DOWN_BUTTON.collidepoint(mouse_pos):
            self.pieces_used = max(0, self.pieces_used - 1)
            self.update_ai()
        if COUNTER_UP_BUTTON.collidepoint(mouse_pos):
            self.pieces_used = min(self.max_steps, self.pieces_used + 1)
            self.update_ai()

        if RESET_BUTTON.collidepoint(mouse_pos):
            self.grid.fill(0)
            self.pieces_used = 0
            self.update_ai()
        if PRINT_BUTTON.collidepoint(mouse_pos):
            self.print_game_state()

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                self.paint_cell_under_mouse()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())

            self.draw()
        pygame.quit()


if __name__ == "__main__":
    PolicyDebugger().run()
